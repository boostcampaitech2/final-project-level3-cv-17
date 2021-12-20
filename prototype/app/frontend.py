import io
import os
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from pandas.core.reshape.reshape import stack

import requests
from PIL import Image
import pandas as pd
import streamlit as st
from confirm_button_hack import cache_on_button_press
from utils import pil_draw_rect, pil_draw_text, get_concat_h

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(page_title = "DoYouKnowKimchi", page_icon="🌶", layout="wide")

def image_to_byte_array(image:Image):
  imgByteArr = io.BytesIO()
  image.save(imgByteArr, format='png')
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr

activity_dic = {'Light':30,'Moderate':35,'Active':40}

def goodnbad(f, g):
    if f/g > 1.2:
        st.error(f'Too Much😢')
    elif g/f > 1.2:        
        st.warning(f'Too Little😢')
    else:
        st.success(f'Good😊')
                    

def main():
    st.image('../assets/unnamed.jpg', use_column_width  = True)
    st.title("Do You Know Kimchi?🥬🌶")
    st.write(" ------ ")
    st.subheader("What is Kimchi?")
    st.write("Kimchi, a staple food in Korean cuisine, is a traditional side dish of salted and fermented vegetables, such as napa cabbage and Korean radish, made with a widely varying selection of seasonings, including gochugaru (Korean chili powder), spring onions, garlic, ginger, and jeotgal (salted seafood), etc.")

    st.sidebar.title("User Info")

    if 'side_submit' not in st.session_state:
        st.session_state.side_submit = 0
    if 'kcal_submit' not in st.session_state:
        st.session_state.kcal_submit = 0

    with st.sidebar.form(key='sidebar form'):
        st.subheader("키")
        height = st.slider('Height(cm)', min_value=101, max_value=250, value=150)
        st.subheader("몸무게")
        weight = st.slider('Weight(kg)', min_value=20, max_value=200, value=70)
        st.subheader("활동지수")
        activity = st.radio('Activity', ['Light','Moderate','Active'])
        with st.expander("활동지수란?"):
            st.subheader("활동지수")
            st.write("Light: 앉아서 주로 생활하거나 매일 가벼운 움직임만 하며 활동량이 적은 경우")
            st.write("Moderate: 규칙적인 생활로 보통의 활동량을 가진 경우")
            st.write("Active: 육체노동 등 평소 신체 활동량이 많은 경우")
        submit_button = st.form_submit_button(label='Submit')
        if submit_button:
            st.session_state.side_submit = 1

    if st.session_state.side_submit:
        avg_weight = (height-100) * 0.9
        kcal = round(avg_weight * activity_dic[activity])
        with st.sidebar.form(key='kcal'):
            st.header(f"Suggested calories intake: {kcal}")
            want_kcal = st.slider('칼로리 설정 (kcal)', min_value=round(kcal*0.5), max_value = round(kcal*1.5),value=(round(kcal*0.9), round(kcal*1.1)))
            # 탄수화물 
            want_car = st.slider('탄수화물 비율 (%)', min_value=0, value=33, max_value=100)
            want_car = want_car * (want_kcal[0]+want_kcal[1])/2 / 100
            # 단백질
            want_pro = st.slider('단백질 비율 (%)', min_value=0, value=33, max_value=100)
            want_pro = want_pro * (want_kcal[0]+want_kcal[1])/2 / 100
            # 지방
            want_fat = st.slider('지방 비율 (%)', min_value=0, value=33, max_value=100)
            want_fat = want_fat * (want_kcal[0]+want_kcal[1])/2 / 100

            kcal_submit = st.form_submit_button(label='Submit')
            if kcal_submit:
                st.session_state.kcal_submit = 1
    
    if st.session_state.kcal_submit:
        st.subheader(f"User's Calories intake goal: {want_kcal[0]}~{want_kcal[1]}kcal")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image_bytes = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(image_bytes))

            files = [
                ('files', (uploaded_file.name, image_bytes,
                        uploaded_file.type))
            ]

            with st.spinner('Calculating...'):
                response = requests.post("http://localhost:8000/intake", files=files)

                food_list = []
                for food in response.json()['Foods']:
                    id, big_label, name, xyxy, info = food.values()
                    x1, y1, x2, y2 = xyxy
                    q, carbohydrate, protein, fat, sugar, kcal = info.values()
                    food_info = {'음식':name, 'kcal': kcal, '탄수화물': carbohydrate, '단백질':protein, '지방': fat, '당': sugar}
                    food_list.append(food_info)
                    image = pil_draw_rect(image, (x1, y1), (x2, y2))
                    image = pil_draw_text(image, x1+10, y1+10, name, (255,255,255))

                T_kcal = response.json()['Total']['kcal']
                T_car = response.json()['Total']['carbohydrate'] * 4
                T_pro = response.json()['Total']['protein'] * 4
                T_fat = response.json()['Total']['fat'] * 9

                Total = {'음식': 'Total', 'kcal': T_kcal, '탄수화물': response.json()['Total']['carbohydrate'], 
                        '단백질':response.json()['Total']['protein'], '지방': response.json()['Total']['fat'], 
                        '당': response.json()['Total']['sugar']}

                KC = int(T_kcal//19)

                if want_kcal[0]<=T_kcal<=want_kcal[1]:
                    image_kimchi = get_concat_h(Image.open('../assets/김치맨2.png'), KC)
                else:
                    image_kimchi = get_concat_h(Image.open('../assets/김치맨1.jpg'), KC)
            
            st.subheader(f"{KC} Kimchi!")
            st.image(image_kimchi, caption=f'This is {KC} Kimchi')
            st.image(image, caption='Detected Image') 
            food_list.append(Total)

            # CSS to inject contained in a string
            hide_table_row_index = """
                        <style>
                        tbody th {display:none}
                        .blank {display:none}
                        </style>
                        """
            # Inject CSS with Markdown
            st.markdown(hide_table_row_index, unsafe_allow_html=True)
            # Display a static table
            st.table(pd.DataFrame(food_list))

            st.subheader("섭취 칼로리")
            if want_kcal[0]<=T_kcal<=want_kcal[1]:
                st.success(f'Good!😊 Total kcal : {T_kcal}')
            elif T_kcal < want_kcal[0]:
                st.error(f'Not Enough!🤤 Total kcal : {T_kcal}')
            else:
                st.error(f"Too Much!😭 Total kcal : {T_kcal}")
            
            ###################### 그래프
            tot = T_car + T_fat + T_pro
            avg = (want_kcal[0]+want_kcal[1])/2
            names = ['Food', 'Goal']

            want_tot = want_car + want_fat + want_pro

            car = [T_car/tot*100, want_car/want_tot*100]
            pro = [T_pro/tot*100, want_fat/want_tot*100]
            fat = [T_fat/tot*100, want_pro/want_tot*100]

            fig = go.Figure(data=[
                go.Bar(name='Carbohydrate', y=names, x=car,orientation='h'),
                go.Bar(name='Protein', y=names, x=pro,orientation='h'),
                go.Bar(name='Fat', y=names, x=fat,orientation='h')
            ])
            # Change the bar mode
            fig.update_layout(barmode='stack')
            st.subheader("Nutrients")
            st.plotly_chart(fig, use_container_width=True)

            ######################
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("탄수화물")
                goodnbad(T_car/tot*100, want_car/want_tot*100)
            with col2:
                st.subheader("단백질")
                goodnbad(T_pro/tot*100, want_fat/want_tot*100)
            with col3:
                st.subheader("지방")
                goodnbad(T_fat/tot*100, want_pro/want_tot*100)               


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