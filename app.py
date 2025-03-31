
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
income = st.number_input("Ежемесячный доход (в сомах)", min_value=0, step=1000, value=30000)
experience = st.number_input("Стаж (в годах)", min_value=0, max_value=50, value=5)
claims = st.number_input("Кол-во страховых случаев в прошлом", min_value=0, max_value=20, value=1)
insurance_type = st.selectbox("Тип страхования", ["авто", "здоровье", "имущество"])

# Преобразование категориального признака
insurance_map = {"авто": 0, "здоровье": 1, "имущество": 2}
insurance_code = insurance_map[insurance_type]

# Кнопка запуска
if st.button("Оценить риск"):
    input_data = np.array([[age, income, experience, claims, insurance_code]])
    score = model.predict_proba(input_data)[0][1]  # Предсказываем вероятность

    # Интерпретация
    if score > 0.75:
        category = "Высокий риск"
        recommendation = "Отказать или назначить высокий тариф"
    elif score > 0.4:
        category = "Средний риск"
        recommendation = "Ручная проверка или стандартный тариф"
    else:
        category = "Низкий риск"
        recommendation = "Выдать полис"

    # Вывод результата
    st.success(f"Score: {score:.2f} — {category}")
    st.info(f"Рекомендация: {recommendation}")
