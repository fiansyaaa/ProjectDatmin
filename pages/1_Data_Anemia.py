import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Judul Halaman
st.title("📊 Dataset Anemia")

# Deskripsi
st.markdown("""
Dataset ini berisi informasi tentang kondisi anemia berdasarkan beberapa fitur seperti umur, jenis kelamin, kadar hemoglobin, dan lainnya.
""")

# Path ke file
data_path = os.path.join("data", "anemia_dataset.csv")

# Load data
try:
    df = pd.read_csv(data_path)

    # Tampilkan semua data
    st.subheader("🔍 Seluruh Data Anemia")
    st.dataframe(df, use_container_width=True, height=600)

    # Info dataset
    st.subheader("📌 Informasi Dataset")
    st.write(f"Jumlah baris: {df.shape[0]}")
    st.write(f"Jumlah kolom: {df.shape[1]}")
    st.write("Kolom-kolom:")
    st.write(df.columns.tolist())

    # Statistik deskriptif
    st.subheader("📈 Statistik Deskriptif")
    st.write(df.describe())

    # Visualisasi
    st.subheader("📊 Visualisasi Data")

    # 1. Histogram Kadar Hemoglobin
    if 'hemoglobin' in df.columns:
        st.markdown("#### Distribusi Kadar Hemoglobin")
        fig1, ax1 = plt.subplots()
        sns.histplot(df['hemoglobin'], bins=20, kde=True, color='skyblue', ax=ax1)
        st.pyplot(fig1)

    # 2. Barplot Anemia Berdasarkan Jenis Kelamin
    if 'gender' in df.columns and 'anemia' in df.columns:
        st.markdown("#### Jumlah Kasus Anemia Berdasarkan Jenis Kelamin")
        fig2, ax2 = plt.subplots()
        sns.countplot(data=df, x='gender', hue='anemia', palette='Set2', ax=ax2)
        st.pyplot(fig2)

    # 3. Boxplot Umur Berdasarkan Status Anemia
    if 'age' in df.columns and 'anemia' in df.columns:
        st.markdown("#### Perbandingan Umur Berdasarkan Status Anemia")
        fig3, ax3 = plt.subplots()
        sns.boxplot(data=df, x='anemia', y='age', palette='pastel', ax=ax3)
        st.pyplot(fig3)

except FileNotFoundError:
    st.error("❌ File anemia_dataset.csv tidak ditemukan di folder 'data/'.")
except Exception as e:
    st.error(f"❌ Terjadi kesalahan: {e}")
