
import streamlit as st
import pickle
import numpy as np

# Загружаем модель
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

# Заголовок приложения
st.title("FinScore.ai — MVP скоринга")
st.markdown("Введите данные клиента для оценки риска")

# Форма ввода
age = st.number_input("Возраст", min_value=18, max_value=100, value=30)
income = st.number_input("Доход", min_value=0, max_value=1000000, value=50000)
score = st.slider("Кредитный скор", min_value=300, max_value=850, value=600)

# Кнопка запуска модели
if st.button("Оценить риск"):
    features = np.array([[age, income, score]])
    prediction = model.predict(features)
    st.write(f"📊 Предсказанный риск: **{prediction[0]}**")
