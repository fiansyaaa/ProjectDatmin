import streamlit as st
import pandas as pd

# Judul halaman
st.title("📊 Data Anemia")

# Load dataset
data = pd.read_csv("data/anemia_dataset.csv")

# Tampilkan seluruh isi data
st.dataframe(data, use_container_width=True)
