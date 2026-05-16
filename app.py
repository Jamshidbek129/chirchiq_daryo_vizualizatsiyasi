import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Chirchiq suv sifati modeli", layout="wide")

st.title("Chirchiq daryosi suv sifati bashorati va 2D dispersion modeli")

river_params_df = pd.read_excel(
    "Chirchiq_gidro_uzel_uchun_mothly_discharge,_kengik,_eni,_chuqurligi.xlsx"
)

df_down = pd.read_csv("downstream.csv")

parameters = ['Temp', 'COD', 'BOD', 'TSS', 'P', 'NH4', 'NO2', 'NO3']

month_num_map = {
    'January': 1,
    'February': 2,
    'March': 3,
    'April': 4,
    'May': 5,
    'June': 6,
    'July': 7,
    'August': 8,
    'September': 9,
    'October': 10,
    'November': 11,
    'December': 12
}

decay_rates = {
    'COD': 5.0e-5,
    'BOD': 1.0e-4,
    'NH4': 5.0e-5,
    'NO2': 5.0e-5,
    'NO3': 0.0,
    'P': 0.0,
    'TSS': 0.0,
    'Temp': 0.0
}

colors = ["#010048", "#056CF2", "#08F7FE", "#AEFD9B", "#F9F871"]
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

river_params_df['Months'] = (
    river_params_df['Months']
    .astype(str)
    .str.strip()
    .str.lower()
)

month_map_excel = {
    'jan': 1,
    'feb': 2,
    'mar': 3,
    'apr': 4,
    'may': 5,
    'jun': 6,
    'jul': 7,
    'aug': 8,
    'sep': 9,
    'oct': 10,
    'nov': 11,
    'dec': 12,
    'january': 1,
    'february': 2,
    'march': 3,
    'april': 4,
    'june': 6,
    'july': 7,
    'august': 8,
    'september': 9,
    'october': 10,
    'november': 11,
    'december': 12
}

river_params_df['Month_num'] = river_params_df['Months'].map(month_map_excel)

df_down['Month'] = df_down['Month'].astype(str).str.strip()
df_down['Month_num'] = df_down['Month'].map(month_num_map)


@st.cache_resource
def train_models(df):
    models = {}

    for p in parameters:
        data = df.dropna(subset=['Year', 'Month_num', p]).copy()

        X = data[['Year', 'Month_num']]
        y = data[p]

        model = RandomForestRegressor(
            n_estimators=300,
            random_state=42
        )

        model.fit(X, y)
        models[p] = model

    return models


models = train_models(df_down)

st.sidebar.header("Bashorat sozlamalari")

future_year = st.sidebar.number_input(
    "Yilni kiriting",
    min_value=2026,
    max_value=2050,
    value=2030
)

future_month = st.sidebar.selectbox(
    "Oyni tanlang",
    list(month_num_map.keys()),
    index=4
)

param = st.sidebar.selectbox(
    "Parametrni tanlang",
    parameters,
    index=1
)

if st.sidebar.button("Bashorat qilish"):

    future_month_num = month_num_map[future_month]

    model = models[param]

    future_input = pd.DataFrame(
        [[future_year, future_month_num]],
        columns=['Year', 'Month_num']
    )

    predicted_value = model.predict(future_input)[0]

    river_row = river_params_df[
        river_params_df['Month_num'] == future_month_num
    ]

    if river_row.empty:
        st.error("Bu oy uchun gidrologik ma'lumot topilmadi.")

    else:
        river_row = river_row.iloc[0]

        U = river_row['Velocity (m/s)']
        depth = river_row['Depth(m)']
        width = river_row['Width(m)']

        x = np.linspace(1, 20000, 500)
        y = np.linspace(-width / 2, width / 2, 50)

        X_grid, Y_grid = np.meshgrid(x, y)

        D_L = 0.6 * U * depth
        D_T = 0.1 * D_L

        decay_rate = decay_rates[param]

        C = predicted_value * np.exp(
            -decay_rate * X_grid / U
        ) * np.exp(
            -Y_grid**2 / (4 * D_T * X_grid)
        )

        st.success(
            f"Bashorat qilingan downstream {param}: {predicted_value:.4f}"
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        contour = ax.contourf(
            X_grid / 1000,
            Y_grid,
            C,
            levels=25,
            cmap=custom_cmap
        )

        fig.colorbar(contour, ax=ax, label=param)

        ax.set_title(
            f"{future_year}-yil | {future_month} | {param} dispersion modeli"
        )

        ax.set_xlabel("Masofa (km)")
        ax.set_ylabel("Daryo kengligi (m)")

        ax.scatter(
            0,
            0,
            color="red",
            marker="*",
            label="Boshlang‘ich nuqta"
        )

        ax.legend()
        ax.grid(True)

        st.pyplot(fig)