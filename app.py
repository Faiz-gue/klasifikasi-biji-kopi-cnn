import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
import numpy as np
import json
from PIL import Image
import os

# ===============================
# KONFIGURASI HALAMAN
# ===============================

st.set_page_config(
    page_title="Klasifikasi Biji Kopi",
    page_icon="☕",
    layout="centered"
)

# ===============================
# LOAD MODEL DAN CLASS NAMES
# ===============================

MODEL_PATH = "model_biji_kopi.keras"
CLASS_PATH = "class_names.json"

@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    with open(CLASS_PATH, "r") as f:
        return json.load(f)

st.title("☕ Klasifikasi Citra Biji Kopi Menggunakan CNN")
st.write(
    "Aplikasi ini digunakan untuk mengklasifikasikan citra biji kopi berdasarkan gambar "
    "yang diunggah oleh pengguna. Model yang digunakan adalah Convolutional Neural Network (CNN)."
)

st.markdown("---")

# Cek file penting
if not os.path.exists(MODEL_PATH):
    st.error(
        "File model_biji_kopi.keras belum ditemukan. "
        "Pastikan file model hasil training sudah diupload ke GitHub dengan nama yang sama persis."
    )
    st.stop()

if not os.path.exists(CLASS_PATH):
    st.error(
        "File class_names.json belum ditemukan. "
        "Pastikan file class_names.json sudah diupload ke GitHub."
    )
    st.stop()

model = load_trained_model()
class_names = load_class_names()

uploaded_file = st.file_uploader(
    "Upload gambar biji kopi",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    if st.button("Prediksi"):
        img = image.resize((150, 150))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = np.max(prediction) * 100

        if confidence < 70:
            st.warning(f"Model belum terlalu yakin. Prediksi sementara: {predicted_class}")
            st.info(f"Confidence: {confidence:.2f}%")
        else:
            st.success(f"Hasil Prediksi: {predicted_class}")
            st.info(f"Confidence: {confidence:.2f}%")

        st.subheader("Probabilitas Setiap Kelas")
        for i, class_name in enumerate(class_names):
            st.write(f"{class_name}: {prediction[0][i] * 100:.2f}%")

else:
    st.warning("Silakan upload gambar biji kopi terlebih dahulu.")

st.markdown("---")
st.caption("Project Data Science - Klasifikasi Citra Biji Kopi Berbasis Streamlit")
