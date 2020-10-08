import numpy as np
import os
import cv2
import pickle
import random


DIRECTORY = r'dogs-vs-cats'
CATEGORIES = ['cat', 'dog']

data = []


path = os.path.join(DIRECTORY, 'train')
for img in os.listdir(path):
    img_path = os.path.join(path, img)
    label = img.split('.')[0]
    arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    new_arr = cv2.resize(arr, (100, 100))
    data.append([new_arr, label])


random.shuffle(data)

X_train = []
y_train = []

for features, label in data:
    X_train.append(features)
    if label == 'cat':
        y_train.append(0)
    else:
        y_train.append(1)

X_train = np.array(X_train)
y_train = np.array(y_train)

pickle.dump(X_train, open('X_train.pkl', 'wb'))
pickle.dump(y_train, open('y_train.pkl', 'wb'))


test_data = []

test_path = os.path.join(DIRECTORY, 'test')
for img in os.listdir(test_path):
    img_path = os.path.join(test_path, img)
    arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    new_arr = cv2.resize(arr, (100, 100))
    test_data.append(new_arr)

X_test = np.array(test_data)

pickle.dump(X_test, open('X_test.pkl', 'wb'))
