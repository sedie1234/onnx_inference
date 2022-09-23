#!/usr/bin/python3

import argparse
import math
import numpy as np
import onnxruntime
import cv2
import time

fps = ""
detectfps = ""
framecount = 0
detectframecount = 0
time1 = 0
time2 = 0

def sort_score_index(scores, threshold=0.0, top_k=0, descending=True):
    score_index = []
    for i, score in enumerate(scores):
        if (threshold > 0) and (score > threshold):
            score_index.append([score, i])
    # print(score_index)
    if not score_index:
        return []

    np_scores = np.array(score_index)
    if descending:
        np_scores = np_scores[np_scores[:, 0].argsort()[::-1]]
    else:
        np_scores = np_scores[np_scores[:, 0].argsort()]

    if top_k > 0:
        np_scores = np_scores[0:top_k]
    return np_scores.tolist()


def nms(boxes, scores, iou_threshold=0.5, score_threshold=0.3, top_k=0):
    if scores is not None:
        scores = sort_score_index(scores, score_threshold, top_k)
        if scores:
            order = np.array(scores, np.int32)[:, 1]
        else:
            return []
    else:
        y2 = boxes[:3]
        order = np.argsort(y2)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    keep = []

    while len(order) > 0:
        idx_self = order[0]
        idx_other = order[1:]
        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)

        inter = w * h
        over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= iou_threshold)[0]
        # over -> order
        order = order[inds + 1]

    return keep


def xywh2xyxy(x):
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def load_classes(path):
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))



def plot_boxes_cv2(img, det_result, class_names=None):
    img = np.copy(img)
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    w = img.shape[1] / 640.
    h = img.shape[0] / 640.

    tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    for det in det_result:
        c1 = (int(det[2][0] * w), int(det[2][1] * h))
        c2 = (int(det[2][2] * w), int(det[2][3] * h))

        rgb = (255, 0, 0)

        if class_names:
            cls_conf = det[1]
            cls_id = det[0]
            label = class_names[cls_id]
            print("{}: {}".format(label, cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            rgb = (red, green, blue)

            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            cc2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, cc2, rgb, -1)
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        
        img = cv2.rectangle(img, c1, c2, rgb, tl)
    return img

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default="yolov3-tiny-iitp.onnx",
                        help='model file')
    parser.add_argument('-v', '--video', type=str, default="test.mp4",
                        help='video file')
    parser.add_argument('--names', type=str, default='coco.names', 
                        help='*.names path')
    args = parser.parse_args()
    print(args)
    model_file  = args.model
    video_file = args.video

    names = load_classes(args.names)

    # session = onnxruntime.InferenceSession('weights/yolov4-tiny.onnx', None)
    session = onnxruntime.InferenceSession(model_file, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    # print('Input Name:', input_name)
    input_shape = session.get_inputs()[0].shape
    print("The model expects input shape: ", input_shape)
    input_h = input_shape[2]
    input_w = input_shape[3]

    cap = cv2.VideoCapture(video_file)
    #img_bgr = cv2.imread(image_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(1000/fps)

    while True:
        t1 = time.perf_counter()

        ret, img_bgr = cap.read()
        if not ret:
          continue

        # h, w, _ = img_bgr.shape
        img = cv2.resize(img_bgr, (input_h, input_w))
        img = img.astype('float32') / 255.
        img = img.transpose(2, 0, 1)
        img = img.reshape(*input_shape)
        #print(img.shape)

        start = time.time()
        prediction = session.run([session.get_outputs()[0].name], {input_name: img})[0]
        print("time:", time.time() - start)

        print(prediction.shape)
        xc = prediction[..., 4] > 0.3

        min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
        x = prediction[0]
        x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[0]]
        
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        boxes = x[:, :4]
        boxes = xywh2xyxy(boxes)
        classes_score = x[:, 5:]

        det_result = []
        for cls in range(80):
            scores = classes_score[:, cls].flatten()
            pick = nms(boxes, scores, 0.5, 0.3)
            for i in range(len(pick)):
                det_result.append([cls, scores[pick][i], boxes[pick][i]])
        
        img_show = plot_boxes_cv2(img_bgr, det_result, names)
        cv2.imshow("test", img_show)
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



