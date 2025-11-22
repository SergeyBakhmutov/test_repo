import streamlit as st
import pandas as pd
import os

from src.utils import load_model, prepare_input_data

st.set_page_config(
    page_title="House Price Prediction App",
)
st.title("House Price Prediction")

model = load_model('models/catboost_model.cbm')
df = pd.read_csv('data/realty_data_cleaned.csv')

districts = df['district'].dropna().unique().tolist()
selected_district = st.selectbox(
    'Выберите район:', districts
)
df_district = df[df["district"] == selected_district]

street_houses = df_district['street_house'].dropna().unique().tolist()
selected_street_house = st.selectbox(
    'Выберите улицу и дом:', street_houses
)

df_city = df['city'].dropna().unique().tolist()
selected_city = st.selectbox(
    'Выберите город:', df_city
)

df_house = df_district[df_district["street_house"] == selected_street_house]

if not df_house["postcode"].isna().all():
    postcode_value = df_house["postcode"].mode().iloc[0]
else:
    postcode_value = ""

st.write(f"Почтовый индекс: **{postcode_value}**")

total_square = st.number_input("Площадь (м²)")
rooms = st.number_input("Комнаты")
floor = st.number_input("Этаж")

if st.button("Предсказать цену"):
    input_df = prepare_input_data(
        city=selected_city,
        district=selected_district,
        street_house=selected_street_house,
        postcode=postcode_value,
        total_square=total_square,
        rooms=int(rooms),
        floor=int(floor)
    )

    prediction = model.predict(input_df)[0]
    st.success(f"Предполагаемая цена: {prediction:,.0f} руб.")

