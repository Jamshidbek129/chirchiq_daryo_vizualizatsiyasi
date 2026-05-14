import streamlit as st
import joblib

st.set_page_config(page_title="Chirchiq suv sifati modeli", layout="wide")

st.title("Chirchiq daryosi suv sifati bashorati va 2D dispersion modeli")

water_model = joblib.load("water_models.pkl")

future_year = st.sidebar.number_input("Yilni kiriting", min_value=2026, max_value=2050, value=2030)

future_month = st.sidebar.selectbox(
    "Oyni tanlang",
    list(water_model.month_display.values()),
    index=4
)

param = st.sidebar.selectbox(
    "Parametrni tanlang",
    water_model.parameters,
    index=1
)

if st.sidebar.button("Bashorat qilish"):
    fig, predicted_value = water_model.predict_and_plot(future_year, future_month, param)

    st.success(f"Bashorat qilingan downstream {param} qiymati: {predicted_value:.4f}")
    st.pyplot(fig)