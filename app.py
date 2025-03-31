import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Пример данных
X = np.array([[20, 50000], [30, 60000], [40, 70000], [50, 80000]])
y = [0, 0, 1, 1]

# Обучаем модель
model = RandomForestClassifier()
model.fit(X, y)

# Streamlit интерфейс
st.title("finscore.ai — MVP скоринга")
age = st.slider("Возраст", 18, 70, 30)
income = st.slider("Доход", 10000, 200000, 50000)

prediction = model.predict([[age, income]])[0]
st.write(f"Предсказание риска: {'Высокий' if prediction else 'Низкий'}")
