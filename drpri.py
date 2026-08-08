import streamlit as st
import cv2
import numpy as np
from tensorflow import keras
from PIL import Image



import gdown
gdown.download( id= "1-KWRtzskATpsRA0aGhU-cW9vUOYZPQ1", output="DR_model.keras", quiet=False)
# ----------------------
# Load the trained model
# ----------------------
model = keras.models.load_model("DR_model.keras")



# ----------------------
# Streamlit App
# ----------------------
st.title("Diabetic Retinopathy Detection")

st.write("Upload a retina image to predict the DR stage.")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img=cv2.imread(uploaded_file)                          #read images
    img = cv2.resize(img, (224, 224))

    

    prd=np.argmax(model.predict(img.reshape(1,224,224,3)),axis=1)[0]
    # Class names
    classes = ["Mild", "Moderate", "Severe", "Proliferate", "No"]
    st.success("Predicted class:", classes[prd])