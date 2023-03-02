import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from config.config import logger
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


# Define your test function
def test_image(model, image_path, labels):
    # Load the image
    image = Image.open(image_path)

    # Preprocess
    logger.info("Preprocessing the image ...")
    image = preprocess(image)
    image = torch.unsqueeze(image, 0)

    # Prediction
    # Generate the prediction
    logger.info("Running prediction ...")
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output)
        pred_mapped = labels[pred.item()]

    # Results
    logger.info(f"Prediction result: {pred_mapped}")

    # Plot the image and show the label
    plt.imshow(Image.open(image_path))
    plt.axis("off")
    plt.title(pred_mapped)
    plt.show()


# Call the test function
test_image(model, "./test.jpeg", labels=labels)
