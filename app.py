import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="finscore.ai", layout="centered")
st.title("finscore.ai — Страховой скоринг")

st.markdown("""
### 💡 Демоверсия скоринговой модели
Это демо показывает, как AI может предсказывать вероятность страхового случая на основе клиентских данных.
""")

# === Ввод данных ===
st.subheader("Введите данные клиента:")
age = st.slider("Возраст клиента (AGE)", 18, 100, 35)
income = st.number_input("Доход (INCOME)", min_value=0, value=50000)
yoj = st.slider("Стаж на последнем месте работы (YOJ)", 0, 40, 5)
car_age = st.slider("Возраст автомобиля (CAR_AGE)", 0, 20, 5)
oldclaim = st.number_input("Сумма предыдущих выплат (OLDCLAIM)", min_value=0.0, value=0.0)
clm_freq = st.slider("Частота обращений (CLM_FREQ)", 0, 10, 0)
car_type = st.selectbox("Тип автомобиля (CAR_TYPE)", options=["Minivan", "SUV", "Sports Car", "Pickup", "Panel Truck", "Van"])
red_car = st.selectbox("Машина красного цвета (RED_CAR)", options=["yes", "no"])
revoked = st.selectbox("Права были отобраны (REVOKED)", options=["yes", "no"])
urbanicity = st.selectbox("Место жительства (URBANICITY)", options=["Urban", "Highly Urban"])
mstatus = st.selectbox("Семейное положение (MSTATUS)", options=["Yes", "No"])
parent1 = st.selectbox("Есть ли дети (PARENT1)", options=["Yes", "No"])

# === Преобразуем категориальные признаки ===
car_type_map = {k: i for i, k in enumerate(["Minivan", "SUV", "Sports Car", "Pickup", "Panel Truck", "Van"])}
red_car_map = {"yes": 1, "no": 0}
revoked_map = {"yes": 1, "no": 0}
urbanicity_map = {"Urban": 0, "Highly Urban": 1}
mstatus_map = {"Yes": 1, "No": 0}
parent1_map = {"Yes": 1, "No": 0}

# Преобразование признаков в массив
X_input = np.array([[age, income, yoj, car_age, oldclaim, clm_freq,
                     car_type_map[car_type], red_car_map[red_car], revoked_map[revoked],
                     urbanicity_map[urbanicity], mstatus_map[mstatus], parent1_map[parent1]]])

# === Модель (встроенная, обучена ранее) ===
# Демонстрационная обучающая выборка
np.random.seed(42)
X_demo = np.random.randint(0, 100, size=(500, 12))
y_demo = np.random.choice([0, 1], size=500, p=[0.75, 0.25])

model = RandomForestClassifier()
model.fit(X_demo, y_demo)

# === Предсказание ===
if st.button("Оценить риск"):
    prob = model.predict_proba(X_input)[0][1]  # вероятность страхового случая

    if prob > 0.7:
        level = "🟥 Высокий риск"
        advice = "Рекомендуется ручная проверка и пересмотр условий."
    elif prob > 0.4:
        level = "🟨 Средний риск"
        advice = "Можно рассматривать с осторожностью."
    else:
        level = "🟩 Низкий риск"
        advice = "Можно одобрить без ограничений."

    st.subheader("📊 Результат оценки")
    st.write(f"**Вероятность страхового случая:** {prob:.2f}")
    st.write(f"**Категория риска:** {level}")
    st.info(advice)
