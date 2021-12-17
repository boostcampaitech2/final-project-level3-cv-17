import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def xywhn2xyxy(x, img_size):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    h = img_size[0]
    w = img_size[1]
    y[0] = w * (x[0] - x[2] / 2)
    y[1] = h * (x[1] - x[3] / 2)
    y[2] = w * (x[0] + x[2] / 2)
    y[3] = h * (x[1] + x[3] / 2)
    return y

def pil_draw_rect(image, point1, point2):
    draw = ImageDraw.Draw(image)
    draw.rectangle((point1, point2), outline=(0, 0, 255), width=5)
    return image

def pil_draw_text(image, point1, point2, txt, color):
    draw = ImageDraw.Draw(image)
    fnt = ImageFont.truetype("utils/FreeMono.ttf", 30)
    draw.text((point1, point2), txt, color, fnt)
    return image