import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image  # Corrected import

model = tf.keras.models.load_model('trained_model.h5')

# Reduced font size for the heading
st.markdown(
    "<h1 style='text-align: center; font-size: 32px;'>Character Recognition: 'V' or Not 'V'</h1>",
    unsafe_allow_html=True
)
st.write("Upload an image to predict whether it's the character 'V' or not.")

def predict_image(img):
    img = img.convert("RGB")  # Ensure the image is in RGB mode
    img = np.array(img)
    img = tf.image.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    if prediction[0][0] > 0.5:
        predicted_class = 'V'
        background_color = 'green'
    else:
        predicted_class = 'Not V'
        background_color = 'red'

    return predicted_class, background_color

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    predicted_class, background_color = predict_image(Image.open(uploaded_image))
    
    st.markdown(
        f"<div style='background-color: {background_color}; padding: 10px; text-align: center;'>"
        f"<h3>The image is more likely to be '{predicted_class}'.</h3>"
        "</div>",
        unsafe_allow_html=True
    )
