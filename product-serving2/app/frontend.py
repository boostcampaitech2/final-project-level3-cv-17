import io
import os
import numpy as np
from pathlib import Path

import requests
from PIL import Image

import streamlit as st
from confirm_button_hack import cache_on_button_press
from utils import pil_draw_rect, pil_draw_text

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

def image_to_byte_array(image:Image):
  imgByteArr = io.BytesIO()
  image.save(imgByteArr, format='png')
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr

def main():
    st.image('../assets/headerbg.jpg', use_column_width  = True)
    st.title("Welcome to DoYouKnowKimchi!")
    st.write(" ------ ")
    st.header('DYKKC')

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        st.write("Classifying...")

        # 기존 stremalit 코드
        # _, y_hat = get_prediction(model, image_bytes)
        # label = config['classes'][y_hat.item()]

        files = [
            ('files', (uploaded_file.name, image_bytes,
                       uploaded_file.type))
        ]
        response = requests.post("http://localhost:8000/detect", files=files)

        foods = response.json()
        for food in foods['products']:
            xyxy = food['xywh']
            h, w, c = image_np.shape
            x1, y1 = int(w*xyxy[0]), int(h*xyxy[1])
            x2, y2 = int(w*xyxy[2]), int(h*xyxy[3])

            cropped_img = image.crop((x1, y1, x2, y2))            
            cropped_img_bytes = image_to_byte_array(cropped_img)
            
            print(len(cropped_img_bytes))
            cr = Image.open(io.BytesIO(cropped_img_bytes))

            # print(len(cropped_img_bytes), type(cropped_img_bytes))

            response = requests.post('http://localhost:8000/order', files = {'file' : cropped_img_bytes})
            big_label = response.json()['name']

            image = pil_draw_rect(image, (x1, y1), (x2, y2))
            image = pil_draw_text(image, x1+10, y1+10, big_label, (255,255,255))

        st.image(image, caption='Detected Image')

main()
# root_password = "password"

# password = st.text_input("Password", type="password")

# @cache_on_button_press('Authenticate')
# def authenticate(password) ->bool:
#     return password==root_password

# if authenticate(password):
#     st.success("You are authenticated!")
#     main()
# else: st.error("The password is invalid")