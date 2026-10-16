import os
import sys
import numpy as np
import torch
import onnx
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

# 1. Path and configuration
ONNX_FP32_PATH      = "Models_onnx/Model_tcn/model_soh_tcn_fp32.onnx"
ONNX_FP32_PREP_PATH = "Models_onnx/Model_tcn/model_soh_tcn_fp32_prep.onnx"
ONNX_INT8_PATH      = "Models_onnx/Model_tcn/model_tcn_int8.onnx"

N_CALIBRATION_SAMPLES = 1000
NPZ_FILE_PATH        = "Dataset_mv/calibration_data.npz"

# Creation of the output directory if it does not exist
os.makedirs(os.path.dirname(ONNX_INT8_PATH), exist_ok=True)

# 2. Validation and preprocessing of the ONNX model
onnx_model = onnx.load(ONNX_FP32_PATH)
onnx.checker.check_model(onnx_model)
print(f"--> Model exported and validated: {ONNX_FP32_PATH}")

quant_pre_process(ONNX_FP32_PATH, ONNX_FP32_PREP_PATH, skip_symbolic_shape=False)
print(f"--> Preprocessing (shape inference) completed: {ONNX_FP32_PREP_PATH}")


# 3. Calibration Data Reader for Quantization
class BatteryCalibrationDataReader(CalibrationDataReader):
    def __init__(self, npz_path, input_name="input", n_samples=1000):
        npz = np.load(npz_path)
        
        # Read the input tensor from the .npz file
        if 'input_tensor' in npz:
            X_full = npz['input_tensor']
        elif 'X_train' in npz:
            X_full = npz['X_train']
        else:
            X_full = npz[npz.files[0]]
        
        np.random.seed(42)
        idx = np.random.choice(len(X_full), min(n_samples, len(X_full)), replace=False)
        X = X_full[idx].astype(np.float32)
        
        # Transpose the input tensor to (N, 2, 128) if it is in the shape (N, 128, 2)
        if X.shape[1] != 2 and X.shape[2] == 2:
            X = np.transpose(X, (0, 2, 1))

        self.input_name = input_name
        self.data = iter([{self.input_name: X[i:i+1]} for i in range(len(X))])

    def get_next(self):
        return next(self.data, None)


# 4. Static Quantization INT8 of the ONNX model
print("\n--> Starting static INT8 quantization (calibration in progress)...")

calibration_reader = BatteryCalibrationDataReader(
    NPZ_FILE_PATH, input_name="input", n_samples=N_CALIBRATION_SAMPLES
)

quantize_static(
    model_input=ONNX_FP32_PREP_PATH,
    model_output=ONNX_INT8_PATH,
    calibration_data_reader=calibration_reader,
    quant_format=QuantFormat.QDQ,        
    activation_type=QuantType.QInt8,     
    weight_type=QuantType.QInt8,        
    per_channel=True,                   
)

print(f"Model quantized succesfully and saved in: {ONNX_INT8_PATH}")
