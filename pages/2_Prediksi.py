import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

st.title("🔬 Prediksi Anemia")

try:
    # Baca dataset
    df = pd.read_csv("data/anemia_dataset.csv")

    # Hilangkan spasi di nama kolom
    df.columns = df.columns.str.strip()

    # Tampilkan kolom untuk debug
    st.write("📌 Kolom tersedia di dataset:", df.columns.tolist())

    # Pastikan kolom target ada
    if "Anaemic" not in df.columns:
        st.error("Kolom 'Anaemic' tidak ditemukan di dataset.")
    else:
        # Pisahkan fitur dan target
        X = df.drop(columns=["Anaemic"])
        y = df["Anaemic"]

        # Ubah kolom kategorikal ke numerik jika ada (otomatis)
        X_encoded = pd.get_dummies(X, drop_first=True)

        # Normalisasi
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)

        # Pilih model
        model_choice = st.selectbox("Pilih Metode Klasifikasi", ["K-Nearest Neighbors (KNN)", "Naive Bayes"])
        if model_choice == "K-Nearest Neighbors (KNN)":
            model = KNeighborsClassifier(n_neighbors=3)
        else:
            model = GaussianNB()

        # Latih model
        model.fit(X_scaled, y)

        st.subheader("📥 Masukkan Data Baru untuk Prediksi")
        input_data = {}

        # Input fitur sesuai X_encoded.columns (yang sudah one-hot encoded)
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                val = st.number_input(f"{col}", value=float(df[col].mean()))
                input_data[col] = val
            else:
                unique_vals = df[col].unique().tolist()
                val = st.selectbox(f"{col}", unique_vals)
                input_data[col] = val

        # Proses input data sesuai transformasi .get_dummies
        input_df = pd.DataFrame([input_data])
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

        # Tombol prediksi
        if st.button("🔍 Prediksi"):
            input_scaled = scaler.transform(input_encoded)
            prediction = model.predict(input_scaled)[0]
            hasil = "Anemia" if prediction == 1 else "Tidak Anemia"
            st.success(f"✅ Hasil Prediksi: **{hasil}**")

except FileNotFoundError:
    st.error("❌ File anemia_dataset.csv tidak ditemukan di folder 'data/'.")
except Exception as e:
    st.error(f"❌ Terjadi error: {e}")
