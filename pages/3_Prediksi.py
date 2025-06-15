import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import model dari modul terpisah
from model.knn import get_model as get_knn
from model.naive_bayes import get_model as get_nb
from model.decision_tree import get_model as get_dt

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

    # Deteksi kolom numerik dan kategorikal
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Encoding (jika ada kolom kategorikal)
    if categorical_cols:
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    else:
        X_encoded = X.copy()

    # Normalisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Pilihan model
    model_choice = st.selectbox(
        "📌 Pilih Metode Klasifikasi",
        ["Decision Tree", "K-Nearest Neighbors (KNN)", "Naive Bayes"]
    )

    # Ambil model dari file
    if model_choice == "K-Nearest Neighbors (KNN)":
        model = get_knn()
    elif model_choice == "Naive Bayes":
        model = get_nb()
    else:
        model = get_dt()

    # Latih model
    model.fit(X_train, y_train)

    # Form input prediksi baru
    st.subheader("📝 Masukkan Data Baru untuk Prediksi")
    input_data = {}

    for col in numeric_cols:
        max_val = float(df[col].max())
        input_data[col] = st.number_input(
            f"{col}", min_value=0.0, max_value=max_val, value=0.0
        )

    if st.button("🔮 Prediksi"):
        input_df = pd.DataFrame([input_data])

        # Samakan fitur input dengan X_encoded
        if categorical_cols:
            dummy_input = pd.get_dummies(input_df)
            dummy_input = dummy_input.reindex(columns=X_encoded.columns, fill_value=0)
        else:
            dummy_input = input_df

        input_scaled = scaler.transform(dummy_input)
        prediction = model.predict(input_scaled)[0]
        hasil = "Anemia" if prediction == 1 else "Tidak Anemia"
        st.success(f"✅ Hasil Prediksi: *{hasil}*")

except FileNotFoundError:
    st.error("❌ File 'anemia_dataset.csv' tidak ditemukan di folder 'data/'.")

# (Opsional) Tambahkan tombol kembali ke dashboard jika mau
if st.button("⬅️ Back"):
    st.switch_page("pages/1_Data_Anemia.py")

# Tombol Next ke halaman berikutnya
if st.button("➡️ Next"):
    st.switch_page("pages/3_Perfomance.py")
