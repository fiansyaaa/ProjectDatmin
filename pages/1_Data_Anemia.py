import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

    # Bersihkan kolom penting
    cols_needed = ['hemoglobin', 'age', 'gender', 'anemia']
    for col in cols_needed:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') if col in ['hemoglobin', 'age'] else df[col]
    df.dropna(subset=cols_needed, inplace=True)

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
        sns.histplot(df['hemoglobin'], bins=20, kde=True, color='skyblue', ax=ax1)
        st.pyplot(fig1)

    # 2. Barplot Anemia Berdasarkan Jenis Kelamin
    if 'gender' in df.columns and 'anemia' in df.columns:
        st.markdown("#### Jumlah Kasus Anemia Berdasarkan Jenis Kelamin")
        fig2, ax2 = plt.subplots()
        sns.countplot(data=df, x='gender', hue='anemia', palette='Set2', ax=ax2)
        st.pyplot(fig2)

    # 3. Boxplot Umur dan Anemia
    if 'age' in df.columns and 'anemia' in df.columns:
        st.markdown("#### Distribusi Umur Berdasarkan Status Anemia")
        fig3, ax3 = plt.subplots()
        sns.boxplot(data=df, x='anemia', y='age', palette='Pastel1', ax=ax3)
        st.pyplot(fig3)

except FileNotFoundError:
    st.error("❌ Dataset tidak ditemukan. Pastikan file 'anemia_dataset.csv' ada di folder 'data/'.")
except Exception as e:
    st.error(f"❌ Terjadi error: {e}")
