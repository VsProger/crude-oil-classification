import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================
# Загрузка модели
# =========================
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Oil Classifier", layout="centered")

st.title("🛢️ Oil Classification App")
st.markdown("Введите параметры нефти (можно оставить пустыми)")

# =========================
# Ввод (разрешаем пустые)
# =========================
col1, col2 = st.columns(2)

with col1:
    density = st.text_input("Density (g/mL)", "0.85")
    viscosity = st.text_input("Viscosity (mPa·s)", "10")
    pour_point = st.text_input("Pour Point (°C)", "-10")

with col2:
    flash_point = st.text_input("Flash Point (°C)", "50")
    sulfur = st.text_input("Sulfur (%)", "1")

st.markdown("---")

# =========================
# Парсинг значений
# =========================
def parse_input(value):
    try:
        return float(value)
    except:
        return np.nan


# =========================
# Кнопка предсказания
# =========================
if st.button("🔍 Predict"):

    input_data = pd.DataFrame({
        "density": [parse_input(density)],
        "viscosity": [parse_input(viscosity)],
        "pour_point": [parse_input(pour_point)],
        "flash_point": [parse_input(flash_point)],
        "saturates": [50],
        "aromatics": [20],
        "resins": [10],
        "asphaltenes": [5],
        "sulfur": [parse_input(sulfur)]
    })

    # =========================
    # Обработка пропусков
    # =========================
    input_data = input_data.fillna(input_data.mean(numeric_only=True))

    # =========================
    # Предсказание
    # =========================
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    st.subheader(f"Результат: {prediction.upper()}")

    # =========================
    # Вероятности (ПРАВИЛЬНО)
    # =========================
    proba_dict = dict(zip(model.classes_, proba))

    st.write("Вероятности классов:")
    st.write(proba_dict)

    # =========================
    # График (красиво)
    # =========================
    st.bar_chart(proba_dict)