import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Load the pre-trained model
model_path = "C:\\Users\\Hp\\OneDrive\\Sem7\\Project\\Model.hdf5"
model = load_model(model_path)

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    return img_array

# Streamlit UI
st.title("Image Classification App")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    
    # Preprocess the uploaded image
    img_array = preprocess_image(uploaded_file)
    
    # Make predictions
    predictions = model.predict(img_array)
    
    # Decode and display the top prediction
    predicted_class = tf.argmax(predictions, axis=1)
    st.write(f"Prediction: {predicted_class[0]}")
