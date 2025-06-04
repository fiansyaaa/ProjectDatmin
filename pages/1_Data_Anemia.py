import streamlit as st
import pandas as pd
import os

# Judul halaman
st.title("📊 Dataset Kematangan Buah Alpukat")

# Keterangan
st.markdown("""
Dataset ini berisi informasi tentang **kematangan buah alpukat** berdasarkan beberapa parameter seperti ukuran, warna kulit, dan tekstur.
""")

# Path file
data_path = os.path.join("data", "avocado_ripeness_dataset.csv")

# Load data
try:
    df = pd.read_csv(data_path)

    # Tampilkan 5 baris pertama
    st.subheader("🔍 Contoh Data (5 Baris Pertama)")
    st.dataframe(df.head())

    # Info data
    st.subheader("📌 Informasi Dataset")
    st.write("Jumlah baris:", df.shape[0])
    st.write("Jumlah kolom:", df.shape[1])
    st.write("Kolom yang tersedia:")
    st.write(df.columns.tolist())

    # Statistik deskriptif
    st.subheader("📈 Statistik Deskriptif")
    st.write(df.describe())

except FileNotFoundError:
    st.error("❌ File avocado_ripeness_dataset.csv tidak ditemukan di folder 'data/'.")
except Exception as e:
    st.error(f"❌ Terjadi kesalahan saat memuat data: {e}")
