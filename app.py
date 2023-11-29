import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load pre-trained model
model = tf.keras.models.load_model('model.h5')

# Function to preprocess the image for the model
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict metal presence
def predict_metal(image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    return prediction

def main():
    st.title("Metal Detection App")

    # Provide the path to the image directly
    image_path = "path/to/your/image.jpg"  # Replace with the actual path to your image file

    st.image(image_path, caption="Uploaded Image", use_column_width=True)

    if st.button("Click here to Predict"):
        prediction = predict_metal(image_path)
        st.write(f"Prediction Result: {prediction}")

if __name__ == "__main__":
    main()
