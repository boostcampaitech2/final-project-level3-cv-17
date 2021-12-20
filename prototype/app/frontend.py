import io
import os
import numpy as np
from pathlib import Path

import requests
from PIL import Image
import pandas as pd
import streamlit as st
from confirm_button_hack import cache_on_button_press
from utils import pil_draw_rect, pil_draw_text, get_concat_h

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(page_title = "DoYouKnowKimchi", page_icon="random", layout="wide")

def image_to_byte_array(image:Image):
  imgByteArr = io.BytesIO()
  image.save(imgByteArr, format='png')
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr

activity_dic = {'Light':30,'Moderate':35,'Active':40}

def goodnbad(compare, mode, s):
    if mode:
        if compare:
            st.success(f'Good😊 {s}')
        else:
            st.error(f'Bad😢 {s}')
    else:        
        if compare:
            st.error(f'Bad😢 {s}')
        else:
            st.success(f'Good😊 {s}')
                    

def main():
    st.image('../assets/headerbg.jpg', use_column_width  = True)
    st.title("Welcome to DoYouKnowKimchi!")
    st.write(" ------ ")

    st.sidebar.title("User Info")

    if 'side_submit' not in st.session_state:
        st.session_state.side_submit = 0
    if 'kcal_submit' not in st.session_state:
        st.session_state.kcal_submit = 0

    with st.sidebar.form(key='sidebar form'):
        st.subheader("Gender")
        gender = st.selectbox('Select', ['Male','Female'])
        st.subheader("Height")
        height = st.slider('Height(cm)', min_value=101, max_value=250, value=150)
        st.subheader("Weight")
        weight = st.slider('Weight(kg)', min_value=20, max_value=200, value=70)
        st.subheader("Activity")
        activity = st.radio('Activity', ['Light','Moderate','Active'])
        with st.expander("Activity란?"):
            st.subheader("활동지수")
            st.write("Light: 앉아서 주로 생활하거나 매일 가벼운 움직임만 하며 활동량이 적은 경우")
            st.write("Moderate: 규칙적인 생활로 보통의 활동량을 가진 경우")
            st.write("Active: 육체노동 등 평소 신체 활동량이 많은 경우")
        submit_button = st.form_submit_button(label='Submit')
        if submit_button:
            st.session_state.side_submit = 1

    if st.session_state.side_submit:
        avg_weight = (height-100) * 0.9
        kcal = avg_weight * activity_dic[activity]
        with st.form(key='kcal'):
            st.header(f"Suggested calories intake: {round(kcal, 2)}")
            want_kcal = st.slider('Calories intake setting (kcal)', min_value=kcal*0.5, value=kcal, max_value = kcal*1.5)
            # 탄수화물 
            want_car = st.slider('Set the carbohydrate ratio (%)', min_value=0, value=33, max_value=100)
            want_car = want_car * want_kcal / 100
            # 단백질
            want_pro = st.slider('Set the protain ratio (%)', min_value=0, value=33, max_value=100)
            want_pro = want_pro * want_kcal / 100
            # 지방
            want_fat = st.slider('Set the fat ratio (%)', min_value=0, value=33, max_value=100)
            want_fat = want_fat * want_kcal / 100

            kcal_submit = st.form_submit_button(label='Submit')
            if kcal_submit:
                st.session_state.kcal_submit = 1
    
    if st.session_state.kcal_submit:
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image_bytes = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(image_bytes))

            files = [
                ('files', (uploaded_file.name, image_bytes,
                        uploaded_file.type))
            ]

            with st.spinner('Wait for it...'):
                response = requests.post("http://localhost:8000/intake", files=files)

                food_list = []
                for food in response.json()['Foods']:
                    id, big_label, name, xyxy, info = food.values()
                    x1, y1, x2, y2 = xyxy
                    q, carbohydrate, protein, fat, sugar, kcal = info.values()
                    food_info = {'소분류':name, 'kcal': kcal, '탄수화물': carbohydrate, '단백질':protein, '지방': fat, '당': sugar}
                    food_list.append(food_info)
                    image = pil_draw_rect(image, (x1, y1), (x2, y2))
                    image = pil_draw_text(image, x1+10, y1+10, name, (255,255,255))

                T_kcal = response.json()['Total']['kcal']
                T_car = response.json()['Total']['carbohydrate'] * 4
                T_pro = response.json()['Total']['protein'] * 4
                T_fat = response.json()['Total']['fat'] * 9
                KC = int(T_kcal//19)

                if T_kcal <= want_kcal:
                    st.success(f'Good😊 Total kcal : {T_kcal}, goal_kcal : {want_kcal}')
                    image_kimchi = get_concat_h(Image.open('../assets/김치맨2.png'), KC)
                else:
                    st.error(f'Bad😢 Total kcal : {T_kcal}, goal_kcal : {want_kcal}')
                    image_kimchi = get_concat_h(Image.open('../assets/김치맨1.jpg'), KC)
                
                goodnbad(T_car <= want_car, True, 'carbohydrate')
                goodnbad(T_pro <= want_pro, True, 'protein')
                goodnbad(T_fat <= want_fat, True, 'fat')

                st.image(image_kimchi, caption=f'This is {KC} Kimchi')
                st.subheader(f"You have consumed {KC} Kimchi!")
                st.image(image, caption='Detected Image') 
                st.table(pd.DataFrame(food_list))                


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