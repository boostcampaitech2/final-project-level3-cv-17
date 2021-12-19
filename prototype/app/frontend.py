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
st.set_page_config(page_title = "DoYouKnowKimchi", page_icon="random", layout="wide")

def image_to_byte_array(image:Image):
  imgByteArr = io.BytesIO()
  image.save(imgByteArr, format='png')
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr

activity_dic = {'Light':30,'Moderate':35,'Active':40}

def main():
    st.image('../assets/headerbg.jpg', use_column_width  = True)
    st.title("Welcome to DoYouKnowKimchi!")
    st.write(" ------ ")

    st.sidebar.title("User Info")

    with st.sidebar.form(key='sidebar form'):
        st.subheader("Gender")
        gender = st.selectbox('Select', ['Male','Female'])
        st.subheader("Height")
        height = st.slider('Height(cm)', min_value=101, max_value=200, value=150)
        st.subheader("Weight")
        weight = st.slider('Weight(kg)', min_value=20, max_value=200, value=70)
        st.subheader("Activity")
        activity = st.radio('Activity', ['Light','Moderate','Active'])
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        avg_weight = (height-100) * 0.9
        kcal = avg_weight * activity_dic[activity]
        with st.form(key='kcal'):
            st.header(f"Suggested calories intake: {kcal}")

            want_kcal = st.slider('Calories intake setting (kcal)', min_value=kcal*0.5, value=kcal, max_value = kcal*1.5)
            kcal_submit = st.form_submit_button(label='Submit')

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))

        files = [
            ('files', (uploaded_file.name, image_bytes,
                    uploaded_file.type))
        ]

        with st.spinner('Wait for it...'):
            response = requests.post("http://localhost:8000/detect", files=files)

            for food in response.json()['Foods']:
                id, big_label, xyxy  = food.values()
                x1, y1, x2, y2 = xyxy

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