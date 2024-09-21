from tensorflow.keras.models import load_model
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image


model_deploy = load_model('emotion_recognition_model.h5')

class_names=['angry','disgust','fear','happy','neutral','sad','surprise']

def predict_model(image_data):
    image = image_data.convert('L')  # Convert image to grayscale
    image = image.resize((48, 48))  # Resize image to 48x48 pixels
    image = np.array(image)  # Convert image to array
    image = np.stack((image,)*3, axis=-1)  # Convert to 3 channels
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 48, 48, 3)
    
    predictions = model_deploy.predict(image)
    predicted_class = class_names[np.argmax(predictions)]
    confidence_score = np.round(np.max(predictions) * 100, 3)

    return predicted_class, confidence_score


st.title("Image Classification for Emotions")
uploaded_image=st.file_uploader("Choose an image ....",type=['jpg','jpeg','png'])

if uploaded_image is not None:
    image=Image.open(uploaded_image)
    pred_class, pred_confidence = predict_model(image)
    st.header(f"{pred_class} ({pred_confidence}%)")
    st.image(image,caption="Uploaded image")




