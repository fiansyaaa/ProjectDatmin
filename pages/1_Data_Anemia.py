import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Judul Halaman
st.title("📊 Dataset Anemia")

# Deskripsi
st.markdown("""
Dataset ini berisi data kondisi anemia berdasarkan ciri-ciri gambar seperti persentase warna piksel dan kadar hemoglobin (Hb).
""")

# Load data
data_path = os.path.join("data", "anemia_dataset.csv")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)

    # Hapus kolom yang tidak berguna
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    if 'Name' in df.columns:
        df = df.drop(columns=['Name'])

    return df

try:
    df = load_data(data_path)

    # Tampilkan seluruh data
    st.subheader("🔍 Seluruh Data (Bersih)")
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

    # Visualisasi Data
    st.subheader("📊 Visualisasi Data")

    # 1. Histogram Hb
    if 'Hb' in df.columns:
        st.markdown("#### Distribusi Kadar Hemoglobin (Hb)")
        fig1, ax1 = plt.subplots()
        sns.histplot(df['Hb'].dropna(), bins=20, kde=True, color='skyblue', ax=ax1)
        st.pyplot(fig1)
    else:
        st.warning("Kolom 'Hb' tidak ditemukan di dataset.")

    # 2. Boxplot Hb vs Status Anemia
    if 'Hb' in df.columns and 'Anaemic' in df.columns:
        st.markdown("#### Distribusi Hb Berdasarkan Status Anemia")
        fig2, ax2 = plt.subplots()
        sns.boxplot(data=df, x='Anaemic', y='Hb', palette='Pastel1', ax=ax2)
        st.pyplot(fig2)
    else:
        st.warning("Kolom 'Hb' dan/atau 'Anaemic' tidak ditemukan.")

    # 3. Korelasi Antar Variabel Numerik
    if st.checkbox("Tampilkan Korelasi Antar Variabel"):
        st.subheader("🔗 Korelasi Antar Variabel Numerik")
        corr = df.corr(numeric_only=True)
        fig3, ax3 = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax3)
        st.pyplot(fig3)

except FileNotFoundError:
    st.error("Dataset tidak ditemukan. Pastikan file 'anemia_dataset.csv' ada di folder 'data/'.")
    
# Tombol Next ke halaman Prediksi
if st.button("➡️"):
    st.switch_page("pages/2_Prediksi.py")
