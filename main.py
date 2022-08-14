import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import numpy as np


CATEGORIES = ['adidas', 'converse', 'nike']

def prepare(path):
    IMG_SIZE = 50
    img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = keras.models.load_model("model.h5")

prediction = model.predict([prepare("C:/Users/Vaibh/Downloads/61KQyNxAVwL._AC_UL1500_.jpg")])

print(prediction)

prediction = prediction.flatten()

print(CATEGORIES[int(np.argmax(prediction).__index__())])
