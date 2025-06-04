import streamlit as st
import pandas as pd

st.title("📊 Data Gempa")

# Deskripsi
st.markdown("""
Halaman ini menampilkan data gempa yang telah diunggah.
Silakan cek isi dataset sebelum dilakukan proses prediksi.
""")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/katalog_gempa.csv")
    return df

df = load_data()

# Tampilkan data
st.subheader("🔍 Tabel Data Gempa")
st.dataframe(df)

# Tampilkan info ringkas
st.subheader("📌 Info Singkat Dataset")
st.write(df.describe())

# Tampilkan jumlah data
st.info(f"Jumlah baris: {df.shape[0]} | Jumlah kolom: {df.shape[1]}")

