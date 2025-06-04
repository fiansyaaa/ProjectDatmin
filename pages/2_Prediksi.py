import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

st.title("Prediksi Anemia")

try:
    df = pd.read_csv("data/anemia_dataset.csv")

    # Bersihkan nama kolom dari spasi
    df.columns = df.columns.str.strip()

    st.write("Kolom dalam data:", df.columns.tolist())

    # Pastikan target tersedia
    if "Anemia" not in df.columns:
        st.error("Kolom 'Anemia' tidak ditemukan. Cek penamaan kolom di CSV.")
    else:
        X = df.drop(columns=["Anemia"])
        y = df["Anemia"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model_name = st.selectbox("Pilih Metode", ["K-Nearest Neighbors (KNN)", "Naive Bayes"])
        model = KNeighborsClassifier(n_neighbors=3) if model_name == "K-Nearest Neighbors (KNN)" else GaussianNB()
        model.fit(X_scaled, y)

        st.subheader("Input Data Baru")
        input_data = {col: st.number_input(col, value=float(df[col].mean())) for col in X.columns}

        if st.button("Prediksi"):
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)[0]
            st.success(f"Hasil Prediksi: {'Anemia' if pred == 1 else 'Tidak Anemia'}")

except FileNotFoundError:
    st.error("File anemia_dataset.csv tidak ditemukan di folder data/.")
