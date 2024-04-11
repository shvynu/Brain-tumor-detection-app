import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import requests
from io import BytesIO

# Function to load the model from GitHub
def load_model_from_github(url):
    response = requests.get(url)
    model_file = BytesIO(response.content)
    model = load_model(model_file)
    return model

# URL of the model file in your GitHub repository
model_url = 'https://github.com/shvynu/Brain-tumor-detection-app/raw/main/main.h5'

# Load the trained model from GitHub
model = load_model_from_github(model_url)

def predict_tumor(image_path):
    # Load and preprocess the image
    test_image = image.load_img(image_path, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255
    test_image = np.expand_dims(test_image, axis=0)

    # Make prediction
    result = model.predict(test_image)

    return result

def main():
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

if __name__ == '__main__':
    main()
