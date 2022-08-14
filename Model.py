from typing import Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

DATADIR_TRAIN = 'shoes_data/train'
DATADIR_TEST = 'shoes_data/test'

IMG_SIZE = 50

CATEGORIES = ['adidas', 'converse', 'nike']

training_data = []



def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR_TRAIN, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                print(e)

create_training_data()

random.shuffle(training_data)

X = []
y = []

for features, labels in training_data:
    X.append(features)
    y.append(labels)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X = X/255.0
y = np.array(y)

model = keras.Sequential([
    layers.Conv2D(64, (3, 3), input_shape=X.shape[1:]),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(64, (3, 3)),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),

    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer='adam',
metrics=['accuracy']
)

model.fit(X, y, epochs=19, batch_size=64, validation_split=0.1)

model.save("model.h5")


