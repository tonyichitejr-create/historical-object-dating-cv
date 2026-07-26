import cv2
import numpy as np

IMAGE_SIZE = 224


def preprocess_image(path):

    image = cv2.imread(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    image = image.astype("float32") / 255.0

    return image
