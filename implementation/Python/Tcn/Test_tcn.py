import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import json
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from Class_models.TCN import TCN

# Percorsi dei file
TEST_FILE = "Dataset_mv/test_soh.npz"
MODEL_PATH = "Models_fp32/model_soh_tcn.pth"
BEST_PARAMS_PATH = "Models_fp32/model_soh_tcn_best_params.json"

if __name__ == "__main__":
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    # 1. Opening the JSON configuration file
    if os.path.exists(BEST_PARAMS_PATH):
        with open(BEST_PARAMS_PATH, "r") as f:
            best_params = json.load(f)
        
        num_channels = best_params.get("num_channels", [16, 32, 64])
        kernel_size = best_params.get("kernel_size", 3)
        dense_units = best_params.get("dense_units", 32)
        dropout = best_params.get("dropout", 0.0)
    else:
        print(f"[WARNING] Best parameters file not found in {BEST_PARAMS_PATH}. Using default values.")
        num_channels = [16, 32, 64]
        kernel_size = 3
        dense_units = 32
        dropout = 0.0

    # 2. Upload Test Set
    print(f"--> Loading test set from: {TEST_FILE} ...")
    data = np.load(TEST_FILE)
    X_test = data['X_test']  # Forma nativa: (N, 128, 2)
    Y_test = data['Y_test']  # Forma nativa: (N,)

    total_samples = len(X_test)
    print(f"Test samples loaded: {total_samples:,}")

    # 3. Initialization and Loading of the TCN Model
    model = TCN(
        in_channels=2,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dense_units=dense_units,
        dropout=dropout
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f" TCN model successfully loaded ({num_params:,} parameters).")

    # 4. Batch Prediction and Inference
    BATCH_SIZE = 512
    preds_list = []

    print("\n--> Block inference calculation...")
    with torch.inference_mode():
        for start_idx in range(0, total_samples, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, total_samples)
            
            batch_x = torch.tensor(X_test[start_idx:end_idx], dtype=torch.float32).permute(0, 2, 1).to(device)
            
            outputs = model(batch_x)
            preds_list.append(outputs.cpu().numpy())

    # Concatenate all predictions and flatten the array
    predictions = np.concatenate(preds_list).flatten()

    # 5. Metric Calculations
    mae = mean_absolute_error(Y_test, predictions)
    rmse = np.sqrt(mean_squared_error(Y_test, predictions))
    mape = np.mean(np.abs((Y_test - predictions) / (Y_test + 1e-8))) * 100

    print("\n" + "=" * 55)
    print("      TCN RESULTS ON UNSEEN CELLS")
    print("=" * 55)
    print(f"• MAE  (Mean Absolute Error)  : {mae:.4f}")
    print(f"• RMSE (Root Mean Square Error): {rmse:.4f}")
    print(f"• MAPE (Mean Absolute % Error)     : {mape:.2f}%")
    print("=" * 55)
