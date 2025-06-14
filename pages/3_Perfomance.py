import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

st.title("📈 Evaluasi Kinerja Model Prediksi Anemia")

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

    # Split data latih dan data uji
    X_latih, X_uji, y_latih, y_uji = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Buat dan latih model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_latih, y_latih)

    # Prediksi
    y_pred = model.predict(X_uji)

    # Tampilkan evaluasi model
    st.subheader("📊 Laporan Klasifikasi")
    st.text(classification_report(y_uji, y_pred, target_names=["Tidak Anemia", "Anemia"]))

    st.subheader("🧾 Confusion Matrix")
    cm = confusion_matrix(y_uji, y_pred)
    df_cm = pd.DataFrame(cm, index=["Aktual: Tidak Anemia", "Aktual: Anemia"],
                         columns=["Prediksi: Tidak Anemia", "Prediksi: Anemia"])
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    st.pyplot(plt)

except FileNotFoundError:
    st.error("❌ File 'anemia_dataset.csv' tidak ditemukan di folder 'data/'.")
    
# (Opsional) Tambahkan tombol kembali ke dashboard jika mau

# Tombol Next ke halaman Prediksi
if st.button("⬅️"):
    st.switch_page("pages/2_Prediksi.py")

if st.button("➡️"):
    st.switch_page("pages/1_Data_Anemia.py")
