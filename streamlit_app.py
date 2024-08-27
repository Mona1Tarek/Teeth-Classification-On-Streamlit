import streamlit as st

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from tensorflow.keras import layers, models


st.title('Teeth Classification')

st.write('This app builds an image classification model')


st.info("Upload an image of teeth to classify them")




# Function to preprocess the image for the model
def preprocess_image(image):
    # Resize the image to match the input shape of the model
    image = image.resize((128, 128))  # Example size, adjust to your model's input size
    image = np.array(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Upload an image
uploaded_file = st.file_uploader("Choose a teeth image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)
    class_idx = np.argmax(prediction, axis=1)[0]

    # Map the predicted class index to a class label
    class_labels = {0: "Cos", 1: "Gum", 2: "MC", 3: "OT", 4: "Cas", 5: "OLP", 6: "OC"}  # Example labels
    predicted_class = class_labels[class_idx]

    # Display the prediction
    st.write(f"Prediction: {predicted_class}")
