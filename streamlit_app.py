import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = tf.keras.models.load_model("best_model-2.keras")

# Load the tokenizer information
tokenizer_data = np.load("tokenizer.npz", allow_pickle=True)

# List all keys in the .npz file
print(tokenizer_data.keys())

word_index = tokenizer_data['vocab_size'].item()  # Word to index mapping
max_sequence_length = int(tokenizer_data['maxlen'])  # Max sequence length

# Function to preprocess text
def preprocess_input(text):
    # Convert text to sequences using the word index
    sequences = [[word_index.get(word, 0) for word in text.split()]]  # Tokenize text
    # Pad the sequences to the max sequence length
    padded = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    return padded

# Predict function
def predict(text):
    preprocessed_text = preprocess_input(text)
    prediction = model.predict(preprocessed_text)
    return prediction

# Streamlit App
st.title("Text Classification App")
st.write("This application classifies your input text using the trained model.")

# Text input area
user_input = st.text_area("Enter your text here:")

# Button to trigger prediction
if st.button("Predict"):
    if user_input.strip():
        prediction = predict(user_input)
        st.write("### Prediction:")
        st.write(prediction)
    else:
        st.warning("Please enter some text for prediction.")
