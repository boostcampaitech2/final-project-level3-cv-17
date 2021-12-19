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

        st.image(image, caption='Uploaded Image')
        st.write("Detecting...")

        files = [
                ('files', (uploaded_file.name, image_bytes,
                            uploaded_file.type))
                ]

        response = requests.post("http://localhost:8000/order", files=files)
        for food in response.json()['Foods']:
            id, big_label, name, xyxy, info = food.values()
            x1, y1, x2, y2 = xyxy
            q, carbohydrate, protein, fat, sugar, kcal = info.values()
            cropped_img = image.crop((x1, y1, x2, y2))
            
            st.image(cropped_img, caption = name)
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