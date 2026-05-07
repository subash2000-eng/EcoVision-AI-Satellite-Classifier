import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


st.set_page_config(page_title="EcoVision AI", page_icon="🌍", layout="centered")

st.title("🌍 Eco-Vision: Satellite Image Classifier")
st.write("Upload a satellite image, and our AI will predict the land cover type (e.g., Forest, Highway, Residential).")


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('ecosat_model_final.keras')
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure 'ecosat_model.h5' is in the same folder.")
    st.stop()

class_names = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 
    'Highway', 'Industrial', 'Pasture', 
    'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

uploaded_file = st.file_uploader("Choose a satellite image (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
   
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Satellite Image', use_container_width=True)
    
    st.write("🔍 Analyzing...")
    
    image_resized = image.resize((64, 64))
    img_array = tf.keras.utils.img_to_array(image_resized)
    img_array = tf.expand_dims(img_array, 0) # Create a batch of 1
    

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    confidence_score = np.max(predictions[0]) * 100
    
    predicted_label = class_names[predicted_class_index]
    
    st.success(f"**Prediction:** {predicted_label}")
    st.info(f"**Confidence Score:** {confidence_score:.2f}%")