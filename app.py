# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# Fungsi load dan latih model
# -------------------------------
@st.cache_resource
def train_model():
    # Load dataset
    df = pd.read_csv("Kelulusan Train.csv")

    # Hapus kolom yang tidak dibutuhkan
    if 'NAMA' in df.columns:
        df.drop(columns=["NAMA"], inplace=True)

    # Label Encoding
    label_cols = ['JENIS KELAMIN', 'STATUS MAHASISWA', 'STATUS NIKAH', 'STATUS KELULUSAN']
    encoders = {}
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Split fitur dan target
    X = df.drop(columns=["STATUS KELULUSAN"])
    y = df["STATUS KELULUSAN"]

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, encoders, X.columns.tolist()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Prediksi Status Kelulusan Mahasiswa")
st.write("Aplikasi ini menggunakan algoritma Random Forest untuk memprediksi apakah mahasiswa lulus atau tidak.")

model, encoders, fitur = train_model()

# Form input data mahasiswa
st.header("Masukkan Data Mahasiswa")

input_data = {}
for col in fitur:
    if col in encoders:
        options = encoders[col].classes_
        input_data[col] = st.selectbox(f"Pilih {col}:", options)
    else:
        input_data[col] = st.number_input(f"Masukkan nilai {col}:", step=1.0)

# Tombol prediksi
if st.button("Prediksi Kelulusan"):
    # Siapkan input untuk model
    data_input_df = pd.DataFrame([input_data])
    for col in encoders:
        if col in data_input_df:
            data_input_df[col] = encoders[col].transform(data_input_df[col])

    prediction = model.predict(data_input_df)[0]
    result = encoders["STATUS KELULUSAN"].inverse_transform([prediction])[0]

    st.success(f"Prediksi: Mahasiswa akan '{result}'.")

