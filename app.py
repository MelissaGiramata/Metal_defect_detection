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
    
    # Check if the image has the right shape
    if img_array.shape == (1, 224, 224, 3):
        return img_array
    else:
        st.error("Invalid Image Shape. Please upload an image with shape (224, 224, 3).")
        st.stop()

# Function to predict metal presence
def predict_metal(image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    return prediction

def main():
    st.title("Metal Detection App")

    uploaded_file = st.file_uploader("Upload an image for metal detection", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_path = f"uploads/{uploaded_file.name}"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        if st.button("Click here to Predict"):
            prediction = predict_metal(image_path)

            # Display result based on the prediction threshold
            result = "Defective" if prediction >= 0.5 else "Okay"
            st.write(f"Prediction Result: {result} (Confidence: {prediction[0][0]:.2f})")

if __name__ == "__main__":
    main()
