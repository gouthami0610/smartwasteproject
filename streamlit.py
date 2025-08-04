import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('waste_classifier_models.h5')

# Define class labels
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Title and instructions
st.title("🗑️ Smart Waste Classification")
st.write("Upload an image of waste to classify its type using AI.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Prediction function
def predict_waste(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)[0]
    prediction_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100
    probs = {cls: f"{float(prob)*100:.2f}%" for cls, prob in zip(class_names, prediction)}
    
    return prediction_class, confidence, probs

# Run prediction if image is uploaded
if uploaded_file is not None: 
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("Classify"):
        pred_class, confidence, probabilities = predict_waste(img)

        st.success(f"**Prediction:** {pred_class}")
        st.info(f"**Confidence:** {confidence:.2f}%")

        st.subheader("Class Probabilities:")
        st.json(probabilities)
