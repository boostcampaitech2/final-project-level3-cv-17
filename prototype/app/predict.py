import os
import torch
# import streamlit as st
import pandas as pd

from collections import defaultdict
from util import transform
from model import efficientnet_b0
# from confirm_button_hack import cache_on_button_press

class_lst = ['deopbab', 'dumpling', 'fried', 'herbs', 'kimchi', 'meat', 
'noodle', 'rice', 'seafood', 'stew', 'sushi', 'vegetable']
food_info = pd.read_csv('Food_info.csv')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# @st.cache
def load_big_model():
    # model = efficientnet_b0(num_classes=12).to(device)
    # model.load_state_dict(torch.load('models/big.pt', map_location=device)['state_dict'])
    model = torch.jit.load('models/big.ts', map_location=device)
    model.eval()
    return model

# @st.cache
def load_quantity_model():
    model = torch.jit.load('models/quantity.ts', map_location=device)
    model.eval()
    return model

# @st.cache
def load_small_model():
    model_info = defaultdict(dict)
    for k in class_lst:                                            # 소분류 모델 load
        model_info['path'][k] = f'models/{k}.ts' 
        model_info['cls'][k]= sorted(list(food_info[food_info['EN']==k]['folder_name']))
        cls_len = len(model_info['cls'][k])
        print( f'{k} : {cls_len}') 
        # model_info['model'][k] = efficientnet_b0(num_classes=len(model_info['cls'][k]))
        # model_info['model'][k].load_state_dict(torch.load(model_info['path'][k], map_location=device)['state_dict'])
        model_info['model'][k] = torch.jit.load(model_info['path'][k], map_location=device)
        model_info['model'][k] = model_info['model'][k].to(device)
        model_info['model'][k].eval() 
    return model_info

# @cache_on_button_press('Rough Prediction')
def get_big_prediction(model, img):
    img = transform(img)
    img = img.to(device)
    with torch.no_grad():
        img = img.unsqueeze(0)
        out = model(img)
        preds = torch.argmax(out, dim=-1)
        pred = class_lst[preds.item()]

    return pred


# @cache_on_button_press('Detailied prediction')
def get_small_prediction(img, model_info, cls):
    labels = []
    img = transform(img)
    img = img.to(device) 
    with torch.no_grad():
        img = img.unsqueeze(0)
        out = model_info['model'][cls](img)
        preds = torch.argmax(out, dim=-1)
        pred = model_info['cls'][cls][preds.item()]
    labels.append(food_info[food_info['folder_name']==int(pred)]['소분류'].item())
    labels.append(float(food_info[food_info['folder_name']==int(pred)]['탄수화물'].item()))
    labels.append(food_info[food_info['folder_name']==int(pred)]['단백질'].item())
    labels.append(food_info[food_info['folder_name']==int(pred)]['지방'].item())
    labels.append(food_info[food_info['folder_name']==int(pred)]['당'].item())
    labels.append(food_info[food_info['folder_name']==int(pred)]['칼로리(kcal)'].item())
    print(labels)
    return labels

def get_quantity_prediction(model, img):
    img = transform(img)
    img = img.to(device)
    with torch.no_grad():
        img = img.unsqueeze(0)
        out = model(img)
        pred = torch.argmax(out, dim=-1)
    return int(pred.item())