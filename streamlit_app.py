import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utility import load_tokenizer_from_json
import numpy as np

# Fungsi untuk memproses input teks
def preprocess_input(text, tokenizer, maxlen=100):
    sequences = tokenizer.texts_to_sequences([text])  # Tokenisasi teks
    padded = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')
    return padded

# Fungsi untuk menginterpretasikan hasil prediksi dengan threshold
def interpret_prediction_with_threshold(prediction, confidence, threshold=0.5):
    if confidence < threshold:
        return "Tidak Diketahui"
    elif prediction == 0:
        return "Teks Normal"
    elif prediction == 1:
        return "Teks Negatif"
    else:
        return "Tidak Diketahui"

# Memuat model
model = tf.keras.models.load_model("best_model.keras")

# Memuat tokenizer
tokenizer = load_tokenizer_from_json("tokenizer.json")

# Streamlit app
st.title("Uji Model Deteksi Kecemasan")
st.write("Aplikasi ini mendeteksi kecemasan pada teks Bahasa Inggris. Fitur deteksi Bahasa Indonesia dan Bahasa lainnya masih dalam pengembangan.")
user_input = st.text_area("Masukkan teks di bawah ini (Bahasa Inggris):")

if st.button("Prediksi"):
    if user_input.strip():
        try:
            # Preprocessing input teks
            processed_text = preprocess_input(user_input, tokenizer)

            # Melakukan prediksi dengan model
            raw_prediction = model.predict(processed_text)

            # Mengambil nilai prediksi untuk kelas dengan probabilitas tertinggi
            prediction = raw_prediction[0].argmax()
            confidence = max(raw_prediction[0])

            # Interpretasi hasil prediksi dengan threshold
            result = interpret_prediction_with_threshold(prediction, confidence, threshold=0.5)

            # Menampilkan hasil prediksi
            st.subheader("Hasil Prediksi")
            st.write(f"**Klasifikasi**: {result}")
            st.write(f"**Confidence Score**: {confidence:.2f}")

            # Menambahkan informasi tambahan
            if result == "Teks Negatif":
                st.warning("Teks ini menunjukkan kemungkinan adanya indikasi kecemasan.")
            elif result == "Teks Normal":
                st.success("Teks ini tidak menunjukkan indikasi kecemasan.")
            else:
                st.info("Model tidak yakin dengan prediksi. Harap masukkan teks yang lebih jelas.")

        except Exception as e:
            # Menampilkan pesan kesalahan
            st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")
    else:
        # Peringatan jika input kosong
        st.warning("Harap masukkan teks terlebih dahulu.")
