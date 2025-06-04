import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

st.title("Prediksi Anemia")

try:
    df = pd.read_csv("data/anemia_dataset.csv")  # ← diperbaiki path-nya

    # Pisah fitur dan label
    X = df.drop(columns=["Anemia"])
    y = df["Anemia"]

    # Preprocessing numerik
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Pilih model
    model_name = st.selectbox("Pilih Metode", ["K-Nearest Neighbors (KNN)", "Naive Bayes"])

    if model_name == "K-Nearest Neighbors (KNN)":
        model = KNeighborsClassifier(n_neighbors=3)
    else:
        model = GaussianNB()

    model.fit(X_scaled, y)

    st.markdown("### Masukkan Data Baru")
    input_data = {}
    for col in X.columns:
        input_data[col] = st.number_input(f"{col}", value=float(df[col].mean()))

    if st.button("Prediksi"):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        st.success(f"Hasil Prediksi: {'Anemia' if pred == 1 else 'Tidak Anemia'}")

except FileNotFoundError:
    st.error("File 'anemia_dataset.csv' tidak ditemukan di folder 'data/'.")
