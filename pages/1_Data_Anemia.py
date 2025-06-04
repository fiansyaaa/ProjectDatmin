import streamlit as st
import pandas as pd
import os

# Judul Halaman
st.title("📊 Dataset Anemia")

# Deskripsi Dataset
st.markdown("""
Dataset ini berisi informasi terkait kondisi anemia berdasarkan berbagai fitur seperti umur, jenis kelamin, kadar hemoglobin, dan lainnya.
""")

# Path ke file
data_path = os.path.join("data", "anemia_dataset.csv")

# Load data
try:
    df = pd.read_csv(data_path)

    # Tampilkan 5 data pertama
    st.subheader("🔍 Contoh Data (5 Baris Pertama)")
    st.dataframe(df.head())

    # Informasi dimensi data
    st.subheader("📌 Informasi Dataset")
    st.write("Jumlah baris:", df.shape[0])
    st.write("Jumlah kolom:", df.shape[1])
    st.write("Kolom-kolom:")
    st.write(df.columns.tolist())

    # Statistik deskriptif
    st.subheader("📈 Statistik Deskriptif")
    st.write(df.describe())

except FileNotFoundError:
    st.error("❌ File anemia_dataset.csv tidak ditemukan di folder 'data/'.")
except Exception as e:
    st.error(f"❌ Terjadi kesalahan saat membaca data: {e}")
