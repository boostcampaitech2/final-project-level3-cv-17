# label, score, xmin, ymin, xmax, ymax
from glob import glob
import os
from PIL import Image

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

IMG_SIZE = 1024
LABEL_DIR = "/opt/ml/detection/yolov5/runs/detect/Ensemble/labels"
PRED_LABEL_FMT = os.path.join(LABEL_DIR, "*.txt")

########################
label_files = sorted(glob(PRED_LABEL_FMT))
imageid_predstring = {
    "image_id": [],
    "PredictionString": []
}

for label_file in label_files:
    label = label_file.split("/")[-1].split(".")[0]
    image_id = os.path.join("test/", f"{label}.jpg")
    imageid_predstring["image_id"].append(image_id)
    
    with open(label_file, "r") as f:
        lines = f.read().strip().split("\n")
    
    pred_string = ""
    if lines:
        for line in lines:
            pred_label, x_center, y_center, w, h, conf_score = line.split(" ")
            x_min = str( (float(x_center) - float(w)/2) * IMG_SIZE )
            y_min = str( (float(y_center) - float(h)/2) * IMG_SIZE )

            w = str(float(w) * IMG_SIZE)
            h = str(float(h) * IMG_SIZE)

            x_max = str( (float(x_min) + float(w)) )
            y_max = str( (float(y_min) + float(h)) )

            pred_string += " ".join([pred_label, conf_score, x_min, y_min, x_max, y_max]) + " "

        imageid_predstring["PredictionString"].append(pred_string)

having_label_file_names = imageid_predstring["image_id"]
no_label_files = []
for idx in range(len(glob("./exp/*.jpg"))):
    file_name = f"test/{idx:04d}.jpg"
    if file_name not in having_label_file_names:
        imageid_predstring["image_id"].append(file_name)
        imageid_predstring["PredictionString"].append("")
        no_label_files.append(file_name)

df = pd.DataFrame(imageid_predstring)
df = df.sort_values("image_id")
df.index = range(len(df))
df.to_csv("../submissions_for_single_model/submission_yolov5s.csv")

print(f"{df[df['PredictionString'] == ''].shape[0]} : 모델이 레이블을 안매긴 이미지 개수")