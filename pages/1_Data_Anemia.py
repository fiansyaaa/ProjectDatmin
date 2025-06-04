import streamlit as st
import pandas as pd
import seaborn as sns
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

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

try:
    df = load_data(data_path)

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

    # Visualisasi Data
    st.subheader("📊 Visualisasi Data")

    # 1. Histogram Kadar Hemoglobin
    if 'hemoglobin' in df.columns:
        st.markdown("#### Distribusi Kadar Hemoglobin")
        fig1, ax1 = plt.subplots()
        sns.histplot(df['hemoglobin'].dropna(), bins=20, kde=True, color='skyblue', ax=ax1)
        st.pyplot(fig1)
    else:
        st.warning("Kolom 'hemoglobin' tidak ditemukan di dataset.")

    # 2. Barplot Anemia Berdasarkan Jenis Kelamin
    if 'gender' in df.columns and 'anemia' in df.columns:
        st.markdown("#### Jumlah Kasus Anemia Berdasarkan Jenis Kelamin")
        fig2, ax2 = plt.subplots()
        sns.countplot(data=df, x='gender', hue='anemia', palette='Set2', ax=ax2)
        st.pyplot(fig2)
    else:
        st.warning("Kolom 'gender' dan/atau 'anemia' tidak ditemukan.")

    # 3. Boxplot Umur dan Anemia
    if 'age' in df.columns and 'anemia' in df.columns:
        st.markdown("#### Distribusi Umur Berdasarkan Status Anemia")
        fig3, ax3 = plt.subplots()
        sns.boxplot(data=df, x='anemia', y='age', palette='Pastel1', ax=ax3)
        st.pyplot(fig3)
    else:
        st.warning("Kolom 'age' dan/atau 'anemia' tidak ditemukan.")

    # 4. Korelasi antar variabel numerik
    if st.checkbox("Tampilkan Korelasi Antar Variabel"):
        st.subheader("🔗 Korelasi Antar Variabel")
        corr = df.corr(numeric_only=True)
        fig4, ax4 = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax4)
        st.pyplot(fig4)

except FileNotFoundError:
    st.error("Dataset tidak ditemukan. Pastikan file 'anemia_dataset.csv' ada di folder 'data/'.")
