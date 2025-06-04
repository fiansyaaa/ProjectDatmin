import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

st.title("Prediksi Anemia")

try:
    # Baca dataset
    df = pd.read_csv("data/anemia_dataset.csv")

    # Bersihkan kolom: hapus yang tidak perlu
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    df = df.drop(columns=["Name"], errors="ignore")  # hapus 'Name' kalau ada

    # Rename agar konsisten (opsional)
    df.columns = df.columns.str.strip()

    # Pastikan kolom target ada
    if "Anaemic" not in df.columns:
        st.error("Kolom 'Anaemic' tidak ditemukan.")
    else:
        X = df.drop(columns=["Anaemic"])
        y = df["Anaemic"]

        # Normalisasi
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Pilih model
        model_choice = st.selectbox("Pilih Metode Klasifikasi", ["K-Nearest Neighbors (KNN)", "Naive Bayes"])
        if model_choice == "K-Nearest Neighbors (KNN)":
            model = KNeighborsClassifier(n_neighbors=3)
        else:
            model = GaussianNB()

        # Latih model
        model.fit(X_scaled, y)

        st.subheader("Masukkan Data Baru untuk Prediksi")
        input_data = {}

        # Input angka sesuai fitur
        for col in X.columns:
            val = st.number_input(f"{col}", value=float(df[col].mean()))
            input_data[col] = val

        if st.button("Prediksi"):
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)[0]

            hasil = "Anemia" if pred == 1 else "Tidak Anemia"
            st.success(f"Hasil Prediksi: {hasil}")

except FileNotFoundError:
    st.error("File anemia_dataset.csv tidak ditemukan di folder 'data/'.")
