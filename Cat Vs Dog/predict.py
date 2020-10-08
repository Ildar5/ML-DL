import cv2
import numpy as np
import os
import pickle
import tensorflow


CATEGORIES = ['cat', 'dog']

loaded_model = tensorflow.keras.models.load_model('best_model2')
loaded_model.load_weights('best_model_w2')

loaded_model.compile(
    optimizer = 'adam',
    loss = tensorflow.keras.losses.BinaryCrossentropy(),
    metrics = ['accuracy']
)

X_test = pickle.load(open('X_test.pkl', 'rb'))


test_path = r'dogs-vs-cats/test'

for img in os.listdir(test_path):
    img_path = os.path.join(test_path, img)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    new_arr = cv2.resize(image, (100, 100))
    new_arr = np.array(new_arr)
    new_arr = new_arr.reshape(-1, 100, 100, 1)
    new_arr = new_arr.astype('float32')

    prediction = loaded_model.predict([new_arr], batch_size=128)
    print("{0} {1} {2}" . format(CATEGORIES[int(prediction)], prediction, img))