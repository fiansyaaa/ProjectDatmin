import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

st.title("🩸 Prediksi Anemia Berdasarkan Data Pasien")

try:
    # Baca dataset
    df = pd.read_csv("data/anemia_dataset.csv")
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    df = df.drop(columns=["Name", "Number"], errors="ignore")
    df.columns = df.columns.str.strip()

    if "Anaemic" not in df.columns:
        st.error("❌ Kolom 'Anaemic' tidak ditemukan di dataset.")
    else:
        # Target dan fitur
        X = df.drop(columns=["Anaemic"])
        y = df["Anaemic"]

        # Deteksi kolom numerik dan kategorikal
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # Encoding dan normalisasi
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)

        # Tampilkan distribusi label
        st.subheader("🔍 Distribusi Label")
        st.write(y.value_counts())

        # Split untuk evaluasi
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Pilihan model
        model_choice = st.selectbox(
            "📌 Pilih Metode Klasifikasi",
            ["Decision Tree", "K-Nearest Neighbors (KNN)", "Naive Bayes"]
        )

        if model_choice == "K-Nearest Neighbors (KNN)":
            model = KNeighborsClassifier(n_neighbors=3)
        elif model_choice == "Naive Bayes":
            model = GaussianNB()
        else:
            model = DecisionTreeClassifier(random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluasi model
        st.subheader("📊 Evaluasi Model")
        st.text(classification_report(y_test, y_pred))

        # Input Data Baru
        st.subheader("📝 Masukkan Data Baru untuk Prediksi")
        input_data = {}

        for col in numeric_cols:
            max_val = float(df[col].max())
            input_data[col] = st.number_input(
                f"{col}", min_value=0.0, max_value=max_val, value=0.0
            )

        for col in categorical_cols:
            options = df[col].dropna().unique().tolist()
            selected = st.selectbox(f"{col}", options)
            dummies = pd.get_dummies(df[[col]], drop_first=True)
            for dummy_col in dummies.columns:
                input_data[dummy_col] = 1.0 if selected in dummy_col else 0.0

        if st.button("🔮 Prediksi"):
            input_df = pd.DataFrame([input_data])

            # Pastikan kolom sama persis dengan X_encoded
            for col in X_encoded.columns:
                if col not in input_df.columns:
                    input_df[col] = 0.0
            input_df = input_df[X_encoded.columns]

            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            hasil = "Anemia" if prediction == 1 else "Tidak Anemia"
            st.success(f"✅ Hasil Prediksi: **{hasil}**")

except FileNotFoundError:
    st.error("❌ File 'anemia_dataset.csv' tidak ditemukan di folder 'data/'.")
