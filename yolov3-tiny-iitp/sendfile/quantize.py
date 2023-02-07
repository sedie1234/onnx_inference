import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'pruning.onnx'
model_quant = 'pruning_q.onnx'
quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8);
