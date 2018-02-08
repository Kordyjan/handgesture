from uuid import uuid4
from os import mkdir
import cv2

SCALE = 2
WINDOW_SIZE = 196
BEGIN = (800, 200)
END = (BEGIN[0] + WINDOW_SIZE * SCALE, BEGIN[1] + WINDOW_SIZE * SCALE)
MIN_VALUE = 30

EXAMPLES_NUMBER = 1000
PREFIX = "data/learning"
folders = (f"{PREFIX}/{folder}" for folder in
           ("nothing", "5_fingers", "4_fingers", "3_fingers", "2_fingers", "ok", "stop", "fist"))


def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 1280, 960)

    cap.set(3, 1280)
    cap.set(4, 960)

    counter = None
    folder = next(folders)
    # mkdir(folder)
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        roi = frame[BEGIN[1]:END[1], BEGIN[0]:END[0]]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.GaussianBlur(roi, (5, 5), 5)
        roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        _, roi = cv2.threshold(roi, MIN_VALUE, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        input = cv2.resize(roi, (WINDOW_SIZE, WINDOW_SIZE))
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        if counter is not None:
            cv2.imwrite(f'{folder}/{uuid4()}.png', input)
            counter += 1
            if counter == EXAMPLES_NUMBER:
                counter = None
                try:
                    folder = next(folders)
                    # mkdir(folder)
                except StopIteration:
                    break

        frame[BEGIN[1]:END[1], BEGIN[0]:END[0]] = roi
        cv2.rectangle(frame, BEGIN, END, (0, 255, 0), 8)
        cv2.putText(frame, folder, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))
        cv2.imshow('img', frame)

        key = cv2.waitKey(10) & 0xff
        if key == 27:
            break
        elif key == 32:
            counter = 0


if __name__ == '__main__':
    main()
