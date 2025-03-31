import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Заголовок и описание
st.title("finscore.ai — MVP скоринга")
st.markdown("""
### 💡 О проекте:
Платформа **finscore.ai** позволяет автоматически оценивать риск клиента
на основе демографических и финансовых данных. Ниже — демо-прототип скоринговой системы.
""")

# === Ввод данных ===
st.subheader("Введите данные клиента:")
age = st.slider("Возраст", 18, 75, 30)
income = st.slider("Доход (в сомах)", 10000, 500000, 100000)
experience = st.slider("Стаж работы (лет)", 0, 50, 5)
claims = st.slider("Количество страховых случаев в прошлом", 0, 10, 0)
insurance_type = st.selectbox("Тип страхования", ["авто", "здоровье", "имущество"])

# Кодировка типа страхования
insurance_map = {"авто": 0, "здоровье": 1, "имущество": 2}
insurance_code = insurance_map[insurance_type]

# === Модель ===
# Генерация фейковых данных для обучения
np.random.seed(42)
X_fake = np.random.randint(18, 60, size=(100, 5))
y_fake = np.random.choice([0, 1], size=100, p=[0.7, 0.3])  # 0 — низкий риск, 1 — высокий

model = RandomForestClassifier()
model.fit(X_fake, y_fake)

# === Предсказание ===
if st.button("Оценить риск"):
    X_input = np.array([[age, income, experience, claims, insurance_code]])
    prediction = model.predict_proba(X_input)[0][1]  # вероятность "высокого риска"

    if prediction > 0.7:
        level = "🟥 Высокий риск"
        recommendation = "Требуется ручная проверка и/или отказ в оформлении"
    elif prediction > 0.4:
        level = "🟨 Средний риск"
        recommendation = "Можно одобрить, но с повышенным тарифом"
    else:
        level = "🟩 Низкий риск"
        recommendation = "Клиент выглядит надёжно. Можно оформлять."

    st.subheader("📊 Результат")
    st.write(f"**Оценка риска:** {prediction:.2f}")
    st.write(f"**Категория:** {level}")
    st.info(f"Рекомендация: {recommendation}")
