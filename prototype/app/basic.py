
import streamlit as st

import io
import os

import cv2
from collections import defaultdict

import requests
from PIL import Image

from predict import load_big_model, load_small_model, get_big_prediction, get_small_prediction
from confirm_button_hack import cache_on_button_press


# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")


root_password = 'password'


def main():
    st.title("Food Classification Model")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg","png"])

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        # print(image_bytes)
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize((224,224))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.image(image, caption='Uploaded Image')

        st.write("Big Classifying...")

        files = [
            ('files', (uploaded_file.name, image_bytes,
                       uploaded_file.type))
        ]

        # big_label = get_big_prediction(big_model, image)
        response = requests.post('http://127.0.0.1:8000/order', files=files)
        print(response)
        big_label = response.json()['products'][0]['result']
        st.write(f'big label is {big_label}')

        st.write("Small Classifying...")
        # name, carbohydrate, protein, fat, sugar, kcal= get_small_prediction(image, small_model, big_label)
        response2 = requests.post(f'http://127.0.0.1:8000/order/{big_label}', files=files)
        # print(response2)
        name, carbohydrate, protein, fat, sugar, kcal= response2.json()['products'][0]['result']

        st.write(f'{name}, 탄수화물 = {carbohydrate}g, 단백질 = {protein}g, 지방 = {fat}g, 당 = {sugar}g')
        st.write(f'칼로리 = {kcal} kcal')

        


@cache_on_button_press('Authenticate')
def authenticate(password):
    print(type(password))
    return password == root_password

# main()

password = st.text_input('password', type="password")

if authenticate(password):
    st.success('You are authenticated!')
    main()
else:
    st.error('The password is invalid.')