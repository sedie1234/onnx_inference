#!/usr/bin/python3

import cv2
import onnxruntime as ort
import argparse
import numpy as np
from boxutil import predict
import time

# ------------------------------------------------------------------------------------------------------------------------------------------------
# Face detection using UltraFace-320 onnx model
# scale current rectangle to box
def scale(box):
    width = box[2] - box[0]
    height = box[3] - box[1]
    maximum = max(width, height)
    dx = int((maximum - width)/2)
    dy = int((maximum - height)/2)

    bboxes = [box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy]
    return bboxes

# crop image
def cropImage(image, box):
    num = image[box[1]:box[3], box[0]:box[2]]
    return num

# face detection method
def faceDetector(orig_image, threshold = 0.7):
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 240))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    input_name = face_detector.get_inputs()[0].name

    start = time.time()
    confidences, boxes = face_detector.run(None, {input_name: image})
    print("time:", time.time() - start)

    boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
    return boxes, labels, probs

# ------------------------------------------------------------------------------------------------------------------------------------------------
# Main void
if __name__ == '__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default="version-RFB-320.onnx",
                        help='model file')
    parser.add_argument("-i", "--image", type=str, default="images/persons.jpg",
                        help="input image")
    args=parser.parse_args()

    model_file  = args.model
    image_file = args.image

    face_detector = ort.InferenceSession(model_file, providers=['CPUExecutionProvider'])

    color = (255, 128, 0)
    orig_image = cv2.imread(image_file)
    boxes, labels, probs = faceDetector(orig_image)

    for i in range(boxes.shape[0]):
        box = scale(boxes[i, :])
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), color, 4)

    cv2.imshow('', orig_image)
    cv2.waitKey()
