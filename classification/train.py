import argparse
import glob
from math import gamma
import re
import os
import random
import numpy as np
from importlib import import_module


from torch.optim import optimizer
from torchvision import transforms
from torchvision.transforms.transforms import ToTensor
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts

from dataset import BigDataset, SmallDataset
from model import efficientnet_b4, efficientnet_b0

import wandb

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # np.random.seed(seed)
    random.seed(seed)

def convert_model_to_torchscript(
    model: nn.Module, path
) -> torch.jit.ScriptModule:
    """Convert PyTorch Module to TorchScript.

    Args:
        model: PyTorch Module.

    Return:
        TorchScript module.
    """
    model.eval()
    jit_model = torch.jit.script(model)

    if path:
        jit_model.save(path)

    return jit_model

def save_model(model, path, device, ckp):
    """save model to torch script, onnx."""

    torch.save(ckp, f=path)
    ts_path = os.path.splitext(path)[:-1][0] + ".ts"
    convert_model_to_torchscript(model, ts_path)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def increment_path(path, exist_ok=False, sep='', mkdir=True):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def train(train_dir, val_dir, model_dir, args):
    seed_everything(args.seed)

    wandb.init(project='Final_Project', entity='yoorichae', name=f'{args.name}')
    
    if args.save:
        save_dir = increment_path(os.path.join(model_dir, args.name)) # 모델 저장 경로
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: BaseAugmentation
    train_set = dataset_module(
        data_dir=train_dir,
        mode='train'
    )
    val_set = dataset_module(
        data_dir=val_dir,
        mode='valid'
    )

    num_classes = len(os.listdir(train_dir))
    print(f'num_classes = {num_classes}')

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=use_cuda,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True
    )

    # -- model
    model = efficientnet_b0(num_classes=num_classes)
    model = model.to(device)
    # model = nn.DataParallel(model)

    # -- loss & optim
    criterion = nn.CrossEntropyLoss()
    # criterion = F1_Loss(classes=args.num_classes)

    # optimizer = SGD(
    #     params=model.parameters(),
    #     lr=args.lr,
    #     momentum=0.9,
    #     weight_decay=0.01    
    # )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    
    # T_0 = int(len(train_loader) * args.epochs//5)
    # scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=T_0)


    best_val_acc = 0
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        #train loop
        model.train()
        loss_value = 0
        matches = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()
            # scheduler.step()

            loss_value += loss.item()
            matches += (preds==labels).sum().item()

            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                
                wandb.log({
                    'train/loss': train_loss,
                    'train/lr': current_lr,
                    'train/acc':train_acc,
                    'train/epoch':epoch
                })
                loss_value = 0
                matches = 0
        # pbar.close()
        scheduler.step()

        #val loop
        with torch.no_grad():
            print("Calculating validation results...")  
            model.eval()
            val_loss_items = []
            val_acc_items = []
            val_acc_list = np.zeros(num_classes)

            for idx, val_batch in enumerate(val_loader):
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                acc_item = (labels==preds).sum().item()    
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                for label, pred in zip(labels, preds):
                    if label==pred:
                        val_acc_list[pred.cpu()] += 1
                

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)

            # save 
            if args.save:
                if val_acc > best_val_acc:
                    print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                    checkpoint = {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    save_model(
                            model=model,
                            path=f"{save_dir}/best.pt",
                            device=device,
                            ckp=checkpoint,
                        )
                    best_val_acc = val_acc
                checkpoint = {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }
                save_model(
                        model=model,
                        path=f"{save_dir}/last.pt",
                        device=device,
                        ckp=checkpoint,
                    )

            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            for i, cls in enumerate(sorted(os.listdir(val_dir))):
                acc_by_class = val_acc_list[i] / val_set.get_nums_by_class(i)                
                print(f'{i} : {acc_by_class:4.2%}', end=' ')

            print()
            # wandb logging
            img_log = []
            for input, pred, label in zip(inputs, preds, labels):
                pred, label = pred.cpu(), label.cpu()
                if pred != label:
                    caption = f'pred : {pred}, label : {label}'
                    pred_img = cv2.imread(val_set.get_samples(pred))
                    pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
                    
                    img_log.append(wandb.Image(pred_img, caption=f'pred : {pred}'))
                    img_log.append(wandb.Image(input.cpu(), caption=f'label : {label}'))
            wandb.log({
                    'val/acc': val_acc,
                    'val/loss': val_loss,
                    'val/epoch': epoch,
                    'val_img': img_log                    
            })
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # from dotenv import load_dotenv
    # load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=2021, help='random seed (default: 2021)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 1)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 1e-3)')
    parser.add_argument('--lr_decay_step', type=int, default=5, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--num_classes', type=int, default=12, help='Class Number')
    parser.add_argument('--save', dest='save', default=False, action='store_true')
    parser.add_argument('--dataset', type=str, default='SmallDataset', help='dataset type (default: SmallDataset)')

    # Container environment
    parser.add_argument('--train_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', 'new_data/train'))
    parser.add_argument('--val_dir', type=str, default=os.environ.get('SM_CHANNEL_VALID', 'new_data/valid'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', 'model2'))

    args = parser.parse_args()
    print(args)

    train_dir = args.train_dir
    val_dir = args.val_dir
    model_dir = args.model_dir
    # print('num_classes:', args.num_classes)
    train(train_dir, val_dir, model_dir, args)