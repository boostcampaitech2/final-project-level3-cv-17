import io
import os
from pathlib import Path
from numpy.lib.type_check import imag

import requests
from PIL import Image
import numpy as np
import json

from torchvision.transforms.functional import crop

import streamlit as st
from confirm_button_hack import cache_on_button_press
from utils.rect_draw import xywhn2xyxy, pil_draw_rect, pil_draw_text


st.set_page_config(layout='wide')

root_password = '123'

def image_to_byte_array(image:Image):
  imgByteArr = io.BytesIO()
  image.save(imgByteArr, format='png')
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr

def main():
    st.title('Food Detection Model')
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        print(len(image_bytes), type(image_bytes))
        image = Image.open(io.BytesIO(image_bytes))
        print(type(image))
        image_np = np.array(image)

        st.image(image, caption='Uploaded Image')
        st.write("Detecting...")

        # xywh = run(MODEL_DIR_PATH + 'best.pt', image, True, True)
        
        files = [
                ('files', (uploaded_file.name, image_bytes,
                            uploaded_file.type))
                ]

        response = requests.post("http://localhost:8000/detect", files=files)
        xyxys = response.json()["xywhs"][0]["result"]
        for xyxy in xyxys:
            h, w, c = image_np.shape
            x1, y1 = int(w*xyxy[0]), int(h*xyxy[1])
            x2, y2 = int(w*xyxy[2]), int(h*xyxy[3])

            cropped_img = image.crop((x1, y1, x2, y2))            
            cropped_img_bytes = image_to_byte_array(cropped_img)
            
            print(len(cropped_img_bytes))
            cr = Image.open(io.BytesIO(cropped_img_bytes))

            print(len(cropped_img_bytes), type(cropped_img_bytes))
            response = requests.post('http://localhost:8000/order', files = {'file' : cropped_img_bytes})
            big_label = response.json()['products'][0]['result']

            response = requests.post('http://localhost:8000/quant', files = {'file' : cropped_img_bytes})
            quantity = response.json()['products'][0]['result'] + 1

            print(quantity, type(quantity))
            response = requests.post(f'http://localhost:8000/order/{big_label}', files = {'file' : cropped_img_bytes})            
            name, carbohydrate, protein, fat, sugar, kcal= response.json()['products'][0]['result']
            carbohydrate, protein, fat, sugar, kcal = [round(float(v) * quantity * 0.2, 2) for v in [carbohydrate, protein, fat, sugar, kcal]]
            
            st.image(cr, caption = name)
            st.write(f'탄수화물 = {carbohydrate}g, 단백질 = {protein}g, 지방 = {fat}g, 당 = {sugar}g')
            st.write(f'칼로리 = {kcal} kcal')

            image = pil_draw_rect(image, (x1, y1), (x2, y2))
            image = pil_draw_text(image, x1+10, y1+10, big_label, (255,0,0))


        st.image(image, caption='Detected Image')

@cache_on_button_press('Authenticate')
def authenticate(password) -> bool:
    return password == root_password

password = st.text_input('password', type="password")

if authenticate(password):
    st.success('You are authenticated!')
    main()
else:
    st.error('The password is invalid.')