import keras.models as models
from uuid import uuid4

import numpy as np
import cv2

SCALE = 2
WINDOW_SIZE = 196
BEGIN = (800, 200)
END = (BEGIN[0] + WINDOW_SIZE * SCALE, BEGIN[1] + WINDOW_SIZE * SCALE)
MIN_VALUE = 30
CATEGORIES = ("2 fingers", "3 fingers", "4 fingers", "5 fingers", "fist", "", "ok", "stop")

def name_tracking(new_name):
    print(new_name)

def main():
    network = load_model()

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 1280, 960)

    cap.set(3, 1280)
    cap.set(4, 960)

    category = ""
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        roi = frame[BEGIN[1]:END[1],BEGIN[0]:END[0]]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.GaussianBlur(roi,(5,5),5)
        roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        _, roi = cv2.threshold(roi, MIN_VALUE, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        input = cv2.resize(roi, (WINDOW_SIZE, WINDOW_SIZE))
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        result = predict(network, input)
        category = CATEGORIES[np.argmax(result)]

        if category != "":
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        # frame[BEGIN[1]:END[1],BEGIN[0]:END[0]] = roi
        cv2.rectangle(frame, BEGIN, END, color, 8)
        cv2.rectangle(frame, (BEGIN[0]-4, BEGIN[1] - 60), (END[0]+4, BEGIN[1]), color, -1)
        cv2.putText(frame, category, (BEGIN[0] + 20, BEGIN[1] - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))
        cv2.imshow('img', frame)

        key = cv2.waitKey(10) & 0xff
        if key == 27:
            break

def load_model():
    return models.load_model("2_2_1_1_2f.h5")


def predict(network, image):
    image = np.array(image).flatten()
    image = image.reshape(1, WINDOW_SIZE, WINDOW_SIZE)

    # float32
    image = image.astype('float32')

    # normalize it
    image = image / 255

    # reshape for NN
    image = image.reshape(1, WINDOW_SIZE, WINDOW_SIZE, 1)
    return network.predict(image)[0]


if __name__ == '__main__':
    main()
