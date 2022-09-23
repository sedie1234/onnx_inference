#!/usr/bin/python3

import onnxruntime
import argparse
import cv2
import numpy as np
import math
import time

fps = ""
detectfps = ""
framecount = 0
detectframecount = 0
time1 = 0
time2 = 0

# coco class is different
# check https://github.com/openvinotoolkit/open_model_zoo/blob/master/data/dataset_classes/coco_91cl_bkgr.txt

def load_classes(path):
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))

# open and display image file

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default="ssd_mobilenet_v1_12.onnx",
                        help='model file')
    parser.add_argument('-v', '--video', type=str, default="test.mp4",
                        help='video file')
    parser.add_argument('-s', '--scale', type=float, default=1,
                        help='image scale factor(float), between 0 and 1')
    parser.add_argument('--names', type=str, default='coco_91cl_bkgr.names',
                        help='*.names path')

    args = parser.parse_args()
    print(args)
    model_file  = args.model
    video_file = args.video
    scale = args.scale

    coco_labels = load_classes(args.names)

    session = onnxruntime.InferenceSession(model_file, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(video_file)
    #img_bgr = cv2.imread(image_file)

    while True:
        t1 = time.perf_counter()

        ret, img_bgr = cap.read()
        if not ret:
          continue

        h, w, c = img_bgr.shape
        scale_h = int(h * scale)
        scale_w = int(w * scale)
        img = cv2.resize(img_bgr, (scale_w, scale_h))
        # reshape the flat array returned by img.getdata() to HWC and than add an additial dimension to make NHWC, aka a batch of images with 1 image in it
        img = img.reshape(1, scale_h, scale_w, c)

        # produce outputs in this order
        #outputs = ["num_detections:0", "detection_boxes:0", "detection_scores:0", "detection_classes:0"]

        output_name_detection_boxes = session.get_outputs()[0].name
        output_name_detection_classes = session.get_outputs()[1].name
        output_name_detection_scores = session.get_outputs()[2].name
        output_name_num_detections = session.get_outputs()[3].name


        start = time.time()
        result = session.run([output_name_detection_boxes, output_name_detection_classes, output_name_detection_scores, output_name_num_detections], {input_name: img})
        print("time:", time.time() - start)
        detection_boxes, detection_classes, detection_scores, num_detections= result

        # print number of detections
        print(num_detections)
        for i in range(int(num_detections[0])):
            print(coco_labels[int(detection_classes[0][i])], " : ", detection_scores[0][i])

        # draw boundary boxes and label for each detection
        def draw_detection(img, d, c, s):
            height, width, channel = img.shape
            # the box is relative to the image size so we multiply with height and width to get pixels
            top =    d[0] * height
            left =   d[1] * width
            bottom = d[2] * height
            right =  d[3] * width

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(height, np.floor(bottom + 0.5).astype('int32'))
            right = min(width, np.floor(right + 0.5).astype('int32'))

            color = (int((1359*(c+1))%255), int((1256*(c+1))%255), int((23424*(c+1))%255))
            cv2.rectangle(img, (left, top), (right, bottom), color, 4)

            lable = coco_labels[c]
            lable += ('({:.2f})'.format(s))
            cv2.putText(img, lable, (left, top - 2), 0, 1/2, (225,255,255),  thickness=1, lineType=cv2.LINE_AA)

        # loop over the results - each returned tensor is a batch
        batch_size = num_detections.shape[0]
        for batch in range(0, batch_size):
            for detection in range(0, int(num_detections[batch])):
                c = int(detection_classes[batch][detection])
                d = detection_boxes[batch][detection]
                s = detection_scores[batch][detection]
                draw_detection(img_bgr, d, c, s)

        cv2.imshow("test", img_bgr)
        #cv2.waitKey()
        if cv2.waitKey(1)&0xFF == ord('q'):
            break

        framecount += 1
        if framecount >= 15:
            fps       = "(Playback) {:.1f} FPS".format(time1/15)
            framecount = 0
            detectframecount = 0
            time1 = 0
            time2 = 0
        t2 = time.perf_counter()
        elapsedTime = t2-t1
        time1 += 1/elapsedTime
        time2 += elapsedTime
        print(fps)
