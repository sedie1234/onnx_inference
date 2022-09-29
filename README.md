## 개요
각 onnx 모델들은 input 및 output에 대한 형식이 제각각이기 때문에 모델별로 추론하는 코드가 각각 다르다.

따라서 모델파일(*.onnx)별로 개별적인 추론 코드(python)를 작성해야 한다.

모든 모델에 대해서 이미지, 동영상, 카메라 입력소스에 대한 추론 스크립트를 아래와 같은 형식으로 다룬다.

## 주요 옵션
-m MODEL  : 사용할 모델파일을 명시한다. default는 float를 사용하는 모델이다.  
-npu      : npu 사용시 명시 (float 모델은 이 옵션을 사용해도 cpu에서 수행된다.)  
-c CAMERA : 카메라 인덱스(PC에서는 주로 0이며, zynq보드에서는 2가 된다.)

그림은 images/dog.jpg 영상은 video/mp4가 default로 지정되어 있기 때문에 따로 지정하지 않아도 되나, 특정 이미지나 영상을 사용하기 위해서는 옵션을 명시적으로 주어 처리해야 한다. 

## 그림

```
./image-yolov3.py -h
usage: image-yolov3.py [-h] [-m MODEL] [-i INPUT] [--names NAMES] [-npu]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        model file
  -i INPUT, --input INPUT
                        input image
  --names NAMES         *.names path
  -npu                  use npu
```

```
ex) ./image-yolov3.py -m yolov3-tiny-iitp-quantization.onnx -npu

```

## 동영상

```
./video-yolov3.py -h
usage: video-yolov3.py [-h] [-m MODEL] [-v VIDEO] [--names NAMES] [-npu]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        model file
  -v VIDEO, --video VIDEO
                        video file
  --names NAMES         *.names path
  -npu                  use npu

```

```
ex) ./video-yolov3.py -m yolov3-tiny-iitp-quantization.onnx -npu
```

## 카메라

```
./camera-yolov3.py -h
usage: camera-yolov3.py [-h] [-m MODEL] [-c CAMERA] [--names NAMES] [-npu]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        model file
  -c CAMERA, --camera CAMERA
                        camera index(0-)
  --names NAMES         *.names path
  -npu                  use npu
```

```
ex) ./camera-yolov3.py -c 2 -m yolov3-tiny-iitp-quantization.onnx -npu
```

## npu 성능비교
**환경**

* dynamic quantized(unit8, convInteger) 된 모델 사용
* zynq 보드에서 테스트 (4 core)
* nput bitstream은 uint8 gemm 연산용
* 정확도는 주관적인 요소임(mAP를 구하는 방법 필요)


**CPU vs NPU 성능 비교 테이블**

|                 | 추론 | 추론 | FPS | FPS | 정확도 | 정확도 |
|-----------------|------|------|-----|-----|-----|-----|
| yolov3-tiny     | 0.7  | 0.54 | 1.3 | 1.7 | 3   | 3   |
| yolov3-tiny-iitp| 1.8  | 1.3  | 0.5 | 0.7 | 7   | 7   |
| yolov4-tiny     | 0.9  | 0.65 | 1   | 1.3 | 4   | 4   |
| yolov5s         | 0.65 | 0.6  | 1.2 | 1.2 | 7.5 | 7.5 | 
| ssd-mobilnetv1  | 0.62 | 0.62 | 1.5 | 1.5 | 7   | 7   |