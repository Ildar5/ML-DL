import cv2
import numpy as np
import os
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import preprocessing
import pickle
from sklearn.model_selection import train_test_split

model = Sequential()

model.add(Conv2D(128, 3, activation = 'relu', input_shape=(100, 100, 1)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(32, 3, activation = 'relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(128, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(
    optimizer = 'adam',
    loss = BinaryCrossentropy(),
    metrics = ['accuracy']
)

X_train = pickle.load(open('X_train.pkl', 'rb'))
y_train = pickle.load(open('y_train.pkl', 'rb'))

X_train = X_train / 255
X_train = X_train.reshape(-1, 100, 100, 1)

model.fit(X_train, y_train, epochs = 10, batch_size = 32)

model.save('best_model2')
model.save_weights('best_model_w2')



