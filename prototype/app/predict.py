import os
import cv2
from pathlib import Path
import numpy as np
from typing import List, Dict, Any

import torch
from model import efficientnet_b0
from utils import transform_image, get_config, set_logging, check_img_size, time_sync, non_max_suppression, scale_coords
import yaml
from collections import defaultdict
from typing import Tuple
import pandas as pd

config = get_config()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_class_model(cls) -> efficientnet_b0:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = pd.read_csv(config['classes'])
    num_class = classes[classes['EN']==cls]

    model = efficientnet_b0(num_classes=num_class).to(device)
    model.load_state_dict(torch.load(config['model_path'][cls], map_location=device)['state_dict'])
    # model = torch.jit.load(torch.load(config['model_path']))
    return model

def load_det_model():
    model = torch.jit.load(config['model_path']['detection'])
    return model

def load_big_model():
    model = torch.jit.load(config['model_path']['bigclass'], map_location=device)
    model.eval()
    return model

def load_small_model(bigclass):
    model = torch.jit.load(config['model_path']['smallclass'][bigclass], map_location=device)
    model.eval()
    return model

def load_quantity_model():
    model = torch.jit.load('models/quantity.ts', map_location=device)
    model.eval()
    return model

def get_big_prediction(model, img):
    with torch.no_grad():
        img = img.unsqueeze(0)
        out = model(img)
        preds = torch.argmax(out, dim=-1)
        pred = config['class_lst'][preds.item()]

    return pred

def get_small_predicitions(bigclass, model, img):
    preds=[]
    food_info = pd.read_csv(config['classes'])
    with torch.no_grad():
        img = img.unsqueeze(0)
        out = model(img)
        pred = torch.argmax(out, dim=-1)
        small_classes = food_info[food_info['EN']==bigclass]
        preds.append(small_classes.iloc[pred]['소분류'].item())
        preds.append(small_classes.iloc[pred]['탄수화물'].item())
        preds.append(small_classes.iloc[pred]['단백질'].item())
        preds.append(small_classes.iloc[pred]['지방'].item())
        preds.append(small_classes.iloc[pred]['당'].item())
        preds.append(small_classes.iloc[pred]['칼로리(kcal)'].item())
    return preds

def get_quantity_prediction(model, img):
    with torch.no_grad():
        img = img.unsqueeze(0)
        out = model(img)
        pred = torch.argmax(out, dim=-1)
    return int(pred.item())

def predict_from_image_byte(model: efficientnet_b0, image_bytes: bytes, config: Dict[str, Any]) -> List[str]:
    transformed_image = transform_image(image_bytes)
    model.eval()
    outputs = model(transformed_image)
    # _, y_hat = outputs.max(1)
    y_hat = torch.argmax(outputs, dim=-1)
    print(outputs)
    food_info=[]
    classes = pd.read_csv(config['classes'])
    vegetable = classes[classes['Custom']=="김치류"]
    food_info.append(vegetable.iloc[y_hat]['소분류'])
    # return list(food_info.loc[:,['소분류','1회제공량(g)','칼로리(kcal)','kcal per g','탄수화물','단백질','지방','당 (총당류)']])
    return food_info

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

@torch.no_grad()
def run(model,
        img0=None,  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=0,  # filter by class: --class 0, or --class 0 2 3
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        half=False,  # use FP16 half-precision inference
        ):
    # Initialize
    set_logging()

    # Load model

    pt = True  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    # if pt:
    # model = torch.jit.load(w)
    # if half:
    #     model.half()  # to FP16

    imgsz = check_img_size(imgsz, s=stride)  # check image size

    dt, seen = [0.0, 0.0, 0.0], 0

    img = letterbox(im=img0)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    t1 = time_sync()

    img = torch.from_numpy(img).to(device)

    img = img.half() if half else img.float()  # uint8 to fp16/32
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1

    pred = model(img)[0]

    t3 = time_sync()
    dt[1] += t3 - t2

    # NMS
    pred = non_max_suppression(pred, 0.25, 0.45, 0, False, max_det=1000)
    dt[2] += time_sync() - t3

    # Process predictions
    det = pred[0]

    seen += 1
    s, im0 = '', img0.copy()
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class

        # Write results
        xyxys = []
        for *xyxy, conf, cls in reversed(det):
            xyxy = ((torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            xyxys.append(xyxy)
    return xyxys
                