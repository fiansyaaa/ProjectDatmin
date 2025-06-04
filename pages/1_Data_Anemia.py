import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# Judul Halaman
st.title("📊 Dataset Anemia")

# Deskripsi
st.markdown("""
Dataset ini berisi data kondisi anemia pasien berdasarkan fitur-fitur seperti umur, jenis kelamin, dan kadar hemoglobin.
""")

# Load data
data_path = os.path.join("data", "anemia_dataset.csv")

try:
    df = pd.read_csv(data_path)

    # Tampilkan seluruh data
    st.subheader("🔍 Seluruh Data")
    st.dataframe(df)

    # Info umum
    st.subheader("📌 Info Dataset")
    st.write(f"Jumlah baris: {df.shape[0]}")
    st.write(f"Jumlah kolom: {df.shape[1]}")
    st.write("Kolom-kolom:")
    st.write(df.columns.tolist())

    # Statistik
    st.subheader("📈 Statistik Deskriptif")
    st.write(df.describe())

    # ─────────────────────────────────────────────
    # 🎨 VISUALISASI
    st.subheader("📊 Visualisasi Data")

    # 1. Histogram Kadar Hemoglobin
    if 'hemoglobin' in df.columns:
        st.markdown("#### Distribusi Kadar Hemoglobin")
        fig1, ax1 = plt.subplots()
        ax1.hist(df['hemoglobin'].dropna(), bins=20, color='skyblue', edgecolor='black')
        ax1.set_xlabel("Kadar Hemoglobin")
        ax1.set_ylabel("Frekuensi")
        st.pyplot(
