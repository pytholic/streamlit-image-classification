import os
import sys

current = os.path.dirname(os.path.realpath(__file__))

parent = os.path.dirname(current)

sys.path.append(parent)

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from model import Classifier

# Load the model
model = Classifier.load_from_checkpoint("./models/checkpoint.ckpt")
model.eval()

# Define labels
labels = [
    "dog",
    "horse",
    "elephant",
    "butterfly",
    "chicken",
    "cat",
    "cow",
    "sheep",
    "spider",
    "squirrel",
]

# Preprocess function
def preprocess(image):
    image = np.array(image)
    resize = A.Resize(224, 224)
    normalize = A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    to_tensor = ToTensorV2()
    transform = A.Compose([resize, normalize, to_tensor])
    image = transform(image=image)["image"]
    return image


# Define the sample images
sample_images = {
    "dog": "./test_images/dog.jpeg",
    "cat": "./test_images/cat.jpeg",
    "butterfly": "./test_images/butterfly.jpeg",
    "elephant": "./test_images/elephant.jpg",
    "horse": "./test_images/horse.jpeg",
}

# Define the function to make predictions on an image
def predict(image):
    image = preprocess(image).unsqueeze(0)

    # Prediction
    # Make a prediction on the image
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1).item()

    # Return the predicted label
    return labels[pred]


# Define the Streamlit app
def app():
    st.title("Animal-10 Image Classification")

    # Add a file uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # # Add a selectbox to choose from sample images
    sample = st.selectbox("Or choose from sample images:", list(sample_images.keys()))

    # Create an empty placeholder for the label
    label_placeholder = st.empty()

    # If an image is uploaded, make a prediction on it
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        pred = predict(image)

    # If a sample image is chosen, make a prediction on it
    elif sample:
        image = Image.open(sample_images[sample])
        st.image(image, caption=sample.capitalize() + " Image.", use_column_width=True)
        pred = predict(image)

    # Update the label placeholder with the predicted label
    label_placeholder.text(f"The predicted label is {pred}.")


# Run the app
if __name__ == "__main__":
    app()
