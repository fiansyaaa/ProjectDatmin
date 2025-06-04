import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("Prediksi Anemia")

# Load data dan model
df = pd.read_csv("anemia_dataset.csv")
X = df.drop(columns=["Anemia"])
y = df["Anemia"]

model = RandomForestClassifier()
model.fit(X, y)

# Input form
st.markdown("Masukkan data pasien untuk memprediksi apakah mengalami anemia.")

input_data = {}
for col in X.columns:
    if df[col].dtype in ['float64', 'int64']:
        input_data[col] = st.number_input(f"{col}", value=float(df[col].mean()))
    else:
        input_data[col] = st.selectbox(f"{col}", options=list(df[col].unique()))

if st.button("Prediksi"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"Hasil Prediksi: {'Anemia' if prediction == 1 else 'Tidak Anemia'}")
