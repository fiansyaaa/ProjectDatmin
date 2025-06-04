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

    # Bersihkan data
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    df = df.drop(columns=["Name", "Number"], errors="ignore")
    df.columns = df.columns.str.strip()

    # Konversi target "Yes"/"No" → 1/0
    if "Anaemic" in df.columns:
        df["Anaemic"] = df["Anaemic"].map({"Yes": 1, "No": 0})
    else:
        st.error("❌ Kolom 'Anaemic' tidak ditemukan.")
        st.stop()

    # Pisahkan fitur dan target
    X = df.drop(columns=["Anaemic"])
    y = df["Anaemic"]

    # Deteksi numerik dan kategorikal
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Encoding kategorikal (tidak perlu karena semua fitur numerik)
    X_encoded = X.copy()  # karena tidak ada kolom kategorikal

    # Normalisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # Distribusi label
    st.subheader("🔍 Distribusi Label")
    st.write(y.value_counts())

    # Split dan evaluasi
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

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

    st.subheader("📊 Evaluasi Model")
    st.text(classification_report(y_test, y_pred))

    # Form input prediksi baru
    st.subheader("📝 Masukkan Data Baru untuk Prediksi")
    input_data = {}

    for col in numeric_cols:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        input_data[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=0)

    if st.button("🔮 Prediksi"):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        hasil = "Anemia" if prediction == 1 else "Tidak Anemia"
        st.success(f"✅ Hasil Prediksi: **{hasil}**")

except FileNotFoundError:
    st.error("❌ File 'anemia_dataset.csv' tidak ditemukan di folder 'data/'.")
