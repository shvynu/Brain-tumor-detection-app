import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('main.h5')

def predict_tumor(image_path):
    # Load and preprocess the image
    test_image = image.load_img(image_path, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255
    test_image = np.expand_dims(test_image, axis=0)

    # Make prediction
    result = model.predict(test_image)

    return result

# Streamlit app
st.title('Brain Tumor Detection')

# File uploader
uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg'])

if uploaded_file is not None:
    # Display the uploaded image
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption='Uploaded Image', use_column_width=True)

    # Make prediction
    result = predict_tumor(uploaded_file)

    # Display the result
    if result[0] <= 0.5:
        st.write('No Brain Tumor')
    else:
        st.write('Brain Tumor')
