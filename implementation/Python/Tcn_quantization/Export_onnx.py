import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
from Class_models.TCN import TCN

import onnx
import torch
import json

MODEL_PATH = "Models_fp32/model_soh_tcn_best.pth"
MODEL_PARAMS_PATH = "Models_fp32/model_soh_tcn_best_params.json"

ONNX_FP32_PATH = "Models_onnx/Model_tcn/model_soh_tcn_fp32.onnx"

# Create the directory if it doesn't exist
if not os.path.exists(os.path.dirname(ONNX_FP32_PATH)):
    os.makedirs(os.path.dirname(ONNX_FP32_PATH), exist_ok=True)

# Set the device to CPU
device = torch.device("cpu")

if __name__ == "__main__":

    with open(MODEL_PARAMS_PATH, "r") as f:
        params = json.load(f)

    print("     Parameters uploaded. ")

    # Create the model and load the state dictionary
    model = TCN(
        in_channels=2,
        num_channels=params["num_channels"],
        kernel_size=params["kernel_size"],
        dense_units=params["dense_units"],
        dropout=params["dropout"]
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("    Model uploaded successfully!")
    print("    Exporting model to ONNX format...")

    dummy_input = torch.randn(1, 2, 128).to(device)
    torch.onnx.export(
        model, 
        dummy_input, 
        ONNX_FP32_PATH, 
        opset_version=11, # Set the ONNX opset version to 11 for compatibility with stm32 CUBE-MX
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
    )

    print("    Model exported to ONNX format successfully!")
