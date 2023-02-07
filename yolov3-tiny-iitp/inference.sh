rm test.txt
python3 image-yolov3.py -m yolov3-tiny-quantization.onnx -i images/dog.jpg --names coco.names >> test.txt
