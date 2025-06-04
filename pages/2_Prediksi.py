import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

st.title("🩸 Prediksi Anemia Berdasarkan Data Pasien")

try:
    # Baca dataset
    df = pd.read_csv("data/anemia_dataset.csv")

    # Bersihkan kolom tidak penting
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    df = df.drop(columns=["Name"], errors="ignore")  # jika ada kolom Name
    df.columns = df.columns.str.strip()  # hilangkan spasi

    # Cek kolom target
    if "Anaemic" not in df.columns:
        st.error("❌ Kolom 'Anaemic' tidak ditemukan di dataset.")
    else:
        # Pisahkan fitur dan target
        X = df.drop(columns=["Anaemic"])
        y = df["Anaemic"]

        # Deteksi kolom numerik dan kategorik
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # Encode kategorikal (jika ada)
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        # Normalisasi numerik
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)

        # Pilih model
        model_choice = st.selectbox("📌 Pilih Metode Klasifikasi", ["K-Nearest Neighbors (KNN)", "Naive Bayes"])
        if model_choice == "K-Nearest Neighbors (KNN)":
            model = KNeighborsClassifier(n_neighbors=3)
        else:
            model = GaussianNB()

        # Latih model
        model.fit(X_scaled, y)

        # ───────────────────────────────────────────────
        st.subheader("📝 Masukkan Data Baru untuk Prediksi")

        input_data = {}

        # Input numerik
        for col in numeric_cols:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())
            input_data[col] = st.number_input(
                f"{col}", min_value=min_val, max_value=max_val, value=mean_val
            )

        # Input kategorik
        for col in categorical_cols:
            options = df[col].dropna().unique().tolist()
            selected = st.selectbox(f"{col}", options)
            for cat in pd.get_dummies(df[[col]], drop_first=True).columns:
                input_data[cat] = 1.0 if selected in cat else 0.0

        # Tombol prediksi
        if st.button("🔮 Prediksi"):
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            hasil = "Anemia" if prediction == 1 else "Tidak Anemia"
            st.success(f"✅ Hasil Prediksi: **{hasil}**")

except FileNotFoundError:
    st.error("❌ File 'anemia_dataset.csv' tidak ditemukan di folder 'data/'.")
