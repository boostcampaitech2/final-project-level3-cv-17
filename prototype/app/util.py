import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import *
import io
from PIL import Image

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    w, h = 302, 464
    y[0] = (w * (x[0] - x[2] / 2))
    y[1] = (h * (x[1] - x[3] / 2))
    y[2] = (w * (x[0] + x[2] / 2))
    y[3] = (h * (x[1] + x[3] / 2))
    return y.round(4)

def transform(image_bytes):
    transform = transforms.Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)),
        ])
    im = Image.open(io.BytesIO(image_bytes))
    im = im.convert('RGB')
    # image_array = np.array(im)
    return transform(im)
