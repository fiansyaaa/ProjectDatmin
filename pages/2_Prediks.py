import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Judul halaman
st.title("Prediksi Anemia")

# Load data
df = pd.read_csv("anemia_dataset.csv")
X = df.drop(columns=["Anemia"])
y = df["Anemia"]

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Pilih metode
model_choice = st.selectbox("Pilih Metode Prediksi", ["K-Nearest Neighbors (KNN)", "Naive Bayes"])

if model_choice == "K-Nearest Neighbors (KNN)":
    model = KNeighborsClassifier(n_neighbors=3)
elif model_choice == "Naive Bayes":
    model = GaussianNB()

# Latih model
model.fit(X_scaled, y)

# Form input
st.markdown("Masukkan data pasien untuk memprediksi apakah mengalami anemia.")
input_data = {}
for col in X.columns:
    if df[col].dtype in ['float64', 'int64']:
        input_data[col] = st.number_input(f"{col}", value=float(df[col].mean()))
    else:
        input_data[col] = st.selectbox(f"{col}", options=list(df[col].unique()))

# Prediksi
if st.button("Prediksi"):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    st.success(f"Hasil Prediksi: {'Anemia' if prediction == 1 else 'Tidak Anemia'}")
