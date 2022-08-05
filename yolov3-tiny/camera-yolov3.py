#!/usr/bin/python3

import argparse
import math
import onnxruntime
import numpy as np
import cv2
import time

fps = ""
detectfps = ""
framecount = 0
detectframecount = 0
time1 = 0
time2 = 0

# https://cpp-learning.com/onnx-runtime_yolo/
# coco labels list
coco_labels = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
            'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear','hair drier', 'toothbrush')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default="yolov3-tiny.onnx",
            help="model file")
    parser.add_argument('-c', '--camera', type=int, default=0,
                        help='camera index(0-)')
    parser.add_argument('-x', '--thread', type=int, help="number of thread")

    args = parser.parse_args()

    is_tiny = 1

    model = args.model
    camera_index = args.camera

    #session = onnxruntime.InferenceSession(model)
    if(args.thread is None):
        session = onnxruntime.InferenceSession(model,  providers=['CPUExecutionProvider'])
    else:
        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = args.thread
        session = onnxruntime.InferenceSession(model,  sess_options, 
                providers=['CPUExecutionProvider'])

    cap = cv2.VideoCapture(camera_index)
    #img_bgr = cv2.imread(image_file)
    while True:
        t1 = time.perf_counter()

        ret, img_bgr = cap.read()
        if not ret:
          continue

        # h, w, _ = img_bgr.shape
        img = cv2.resize(img_bgr, (416, 416))
        img = img.astype('float32') / 255.
        img = img.transpose(2, 0, 1)
        img = img.reshape(1,3,416,416)

        image_size = np.array([416, 416], dtype=np.float32).reshape(1, 2)

        # 2. Get input/output name
        input_name = session.get_inputs()[0].name           # 'image'
        input_name_img_shape = session.get_inputs()[1].name # 'image_shape'

        output_name_boxes = session.get_outputs()[0].name   # 'boxes'
        output_name_scores = session.get_outputs()[1].name  # 'scores'
        output_name_indices = session.get_outputs()[2].name # 'indices'

        # 3. run
        start = time.time()
        outputs_index = session.run([output_name_boxes, output_name_scores, output_name_indices],
                                    {input_name: img, input_name_img_shape: image_size})
        print("time:", time.time() - start)

        output_boxes = outputs_index[0]
        output_scores = outputs_index[1]
        output_indices = outputs_index[2]

        if is_tiny == 1:
           index =  output_indices[0]
        else:
           index =  output_indices


        # Result
        out_boxes, out_scores, out_classes = [], [], []
        for idx_ in index:
            out_classes.append(idx_[1])
            out_scores.append(output_scores[tuple(idx_)])
            idx_1 = (idx_[0], idx_[2])
            out_boxes.append(output_boxes[idx_1])

        print(out_classes)
        print(out_scores)
        print(out_boxes)

        for i in range(0, len(out_classes)):
            box_xy = out_boxes[i]
            p1_y = int(box_xy[0] * img_bgr.shape[0]/416)
            p1_x = int(box_xy[1] * img_bgr.shape[1]/416)
            p2_y = int(box_xy[2] * img_bgr.shape[0]/416)
            p2_x = int(box_xy[3] * img_bgr.shape[1]/416)

            color = (int((1359*(out_classes[i]+1))%255), int((1256*(out_classes[i]+1))%255), int((23424*(out_classes[i]+1))%255))

            cv2.rectangle(img_bgr, (p1_x,p1_y),(p2_x,p2_y), color, 4)

            lable = coco_labels[out_classes[i]]
            lable +=('({:.2f})'.format(out_scores[i]))
            cv2.putText(img_bgr, lable, (p1_x, p1_y - 2), 0, 1/2, (225,255,255), thickness=1, lineType=cv2.LINE_AA)

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
