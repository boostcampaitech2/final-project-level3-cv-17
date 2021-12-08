import argparse
import glob
from math import gamma
import re
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.optim.lr_scheduler import StepLR

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ClsDataset
from model import ClsModel

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
    save_dir = increment_path(os.path.join(model_dir, args.name)) # 모델 저장 경로
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_set = ClsDataset(train_dir)
    num_classes = 5
    val_set = ClsDataset(val_dir)

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
        shuffle=False,
        pin_memory=use_cuda,
    )

    # -- model
    model = ClsModel(num_classes=num_classes)
    model = model.to(device)
    model = nn.DataParallel(model)
    # -- loss & optim
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)


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

            loss_value += loss.item()
            matches += (preds==labels).sum().item()
            
            
            # train_loss = loss_value / args.log_interval
            # train_acc = matches / args.batch_size / args.log_interval
            # current_lr = get_lr(optimizer)

            # pbar.update()
            # pbar.set_description(
            #     f"Train: [{epoch + 1:03d}] "
            #     f"Loss: {train_loss:.3f}, "
            #     f"Acc: {train_acc * 100:.2f}% "
            #     f"Lr: {current_lr}"
            # )
            # loss_value = 0
            # matches = 0
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )

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
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                acc_item = (labels==preds).sum().item()    
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)
            if val_acc > best_val_acc:
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            print()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # from dotenv import load_dotenv
    # load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 1)')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=1, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--train_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', './data/train'))
    parser.add_argument('--val_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', './data/valid'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    train_dir = args.train_dir
    val_dir = args.val_dir
    model_dir = args.model_dir

    train(train_dir, val_dir, model_dir, args)