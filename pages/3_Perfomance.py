import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.title("📈 Prediksi Anemia Berdasarkan Data Pasien")  # Ganti judul juga agar tidak menyebut evaluasi

try:
    # Load dataset
    df = pd.read_csv("data/anemia_dataset.csv")

    # Hapus kolom tidak penting
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    df = df.drop(columns=["Name", "Number"], errors="ignore")
    df.columns = df.columns.str.strip()

    # Ubah target: "Yes"/"No" → 1/0
    if "Anaemic" in df.columns:
        df["Anaemic"] = df["Anaemic"].map({"Yes": 1, "No": 0})
    else:
        st.error("❌ Kolom 'Anaemic' tidak ditemukan.")
        st.stop()

    # Pisahkan fitur dan target
    X = df.drop(columns=["Anaemic"])
    y = df["Anaemic"]

    # Normalisasi data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data latih dan uji
    X_latih, X_uji, y_latih, y_uji = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Latih model Decision Tree
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_latih, y_latih)

    # Prediksi (tanpa menampilkan apa pun)
    _ = model.predict(X_uji)

    # Info ringkas saja jika mau
    st.success("✅ Model telah dilatih dengan sukses.")

except FileNotFoundError:
    st.error("❌ File 'anemia_dataset.csv' tidak ditemukan di folder 'data/'.")
