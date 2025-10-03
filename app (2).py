import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json
from PIL import Image

# --- App Configuration ---
st.set_page_config(
    page_title="Plant Disease Diagnosis",
    page_icon="🌿",
    layout="centered"
)

# --- Model and Class Names Loading ---
# Use caching to load the model only once
@st.cache_resource
def load_model_and_classes():
    """Loads the trained model and class names."""
    try:
        model = tf.keras.models.load_model('saved_model/plant_disease_model.h5')
        with open('saved_model/class_names.json', 'r') as f:
            class_names = json.load(f)
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model or class names: {e}")
        return None, None

model, class_names = load_model_and_classes()

# --- Image Preprocessing Function ---
def preprocess_image(image_bytes, img_size):
    """Preprocesses the uploaded image for prediction."""
    try:
        # Decode the image bytes to a NumPy array
        image = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize and normalize
        image = cv2.resize(image, (img_size, img_size))
        image = image / 255.0
        # Expand dimensions to create a batch of 1
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# --- Streamlit App UI ---
st.title("🌿 Plant Disease Diagnosis")
st.write(
    "Upload an image of a plant leaf, and the AI will predict the disease. "
    "This model is trained on the PlantVillage dataset and can identify 38 different classes."
)

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the file bytes
    file_bytes = uploaded_file.getvalue()
    
    # Display the uploaded image
    st.image(Image.open(uploaded_file), caption='Uploaded Image.', use_column_width=True)
    
    if st.button('Diagnose Disease'):
        if model is not None and class_names is not None:
            with st.spinner('Analyzing the image...'):
                processed_image = preprocess_image(file_bytes, 224)
                
                if processed_image is not None:
                    # Make prediction
                    prediction = model.predict(processed_image)
                    predicted_class_index = np.argmax(prediction)
                    predicted_class_name = class_names[str(predicted_class_index)]
                    confidence = np.max(prediction) * 100
                    
                    st.success(f"**Prediction:** {predicted_class_name.replace('___', ' ')}")
                    st.info(f"**Confidence:** {confidence:.2f}%")
                else:
                    st.error("Could not process the image. Please try another one.")
        else:
            st.error("Model is not loaded. Please check the application logs.")