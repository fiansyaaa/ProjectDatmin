import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

st.title("🩸 Prediksi Anemia Berdasarkan Data Pasien")

try:
    # Baca dataset
    df = pd.read_csv("data/anemia_dataset.csv")

    # Bersihkan kolom tidak penting
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    df = df.drop(columns=["Name", "Number"], errors="ignore")  # Hapus kolom 'Name' & 'Number' jika ada
    df.columns = df.columns.str.strip()

    if "Anaemic" not in df.columns:
        st.error("❌ Kolom 'Anaemic' tidak ditemukan di dataset.")
    else:
        # Pisahkan fitur dan target
        X = df.drop(columns=["Anaemic"])
        y = df["Anaemic"]

        # Deteksi kolom numerik dan kategorikal
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # Encoding kategorikal
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        # Normalisasi
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)

        # Pilihan model
        model_choice = st.selectbox(
            "📌 Pilih Metode Klasifikasi",
            ["Decision Tree", "K-Nearest Neighbors (KNN)", "Naive Bayes"]
        )

        if model_choice == "K-Nearest Neighbors (KNN)":
            model = KNeighborsClassifier(n_neighbors=3)
        elif model_choice == "Naive Bayes":
            model = GaussianNB()
        else:  # Decision Tree default
            model = DecisionTreeClassifier(random_state=42)

        # Latih model
        model.fit(X_scaled, y)

        st.subheader("📝 Masukkan Data Baru untuk Prediksi")
        input_data = {}

        # Input numerik
        for col in numeric_cols:
            max_val = float(df[col].max())
            input_data[col] = st.number_input(
                f"{col}", min_value=0.0, max_value=max_val, value=0.0
            )

        # Input kategorikal
        for col in categorical_cols:
            options = df[col].dropna().unique().tolist()
            selected = st.selectbox(f"{col}", options)
            dummies = pd.get_dummies(df[[col]], drop_first=True)
            for dummy_col in dummies.columns:
                input_data[dummy_col] = 1.0 if selected in dummy_col else 0.0

        # Prediksi
        if st.button("🔮 Prediksi"):
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            hasil = "Anemia" if prediction == 1 else "Tidak Anemia"
            st.success(f"✅ Hasil Prediksi: **{hasil}**")

except FileNotFoundError:
    st.error("❌ File 'anemia_dataset.csv' tidak ditemukan di folder 'data/'.")
