import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.signal import savgol_filter

from Class_models.TCN import TCN

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

    # 1. Upload Best Hyperparameters
    if os.path.exists(BEST_PARAMS_PATH):
        with open(BEST_PARAMS_PATH, "r") as f:
            best_params = json.load(f)
        num_channels = best_params.get("num_channels", [16, 32])
        kernel_size = best_params.get("kernel_size", 3)
        dense_units = best_params.get("dense_units", 16)
        dropout = best_params.get("dropout", 0.0)
    else:
        num_channels, kernel_size, dense_units, dropout = [16, 32], 3, 16, 0.0

    # 2. Upload Test Dataset
    data = np.load(TEST_FILE)
    X_test = data['X_test']
    Y_test = data['Y_test']

    # 3. Initialize and Load the Model
    model = TCN(
        in_channels=2,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dense_units=dense_units,
        dropout=dropout
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    BATCH_SIZE = 1024
    preds_list = []
    with torch.inference_mode():
        for i in range(0, len(X_test), BATCH_SIZE):
            batch_x = torch.tensor(X_test[i:i+BATCH_SIZE], dtype=torch.float32).permute(0, 2, 1).to(device)
            preds_list.append(model(batch_x).cpu().numpy())

    predictions = np.concatenate(preds_list).flatten()

    # Application of Moving Average or Savitzky-Golay to filter high-frequency noise
    smoothed_predictions = savgol_filter(predictions, window_length=30, polyorder=2)

    mae_smooth = mean_absolute_error(Y_test, smoothed_predictions)

    # 4. Compute Metrics
    mae = mean_absolute_error(Y_test, predictions)
    rmse = np.sqrt(mean_squared_error(Y_test, predictions))

    print(f"• MAE: {mae:.4f} (es. {mae*100:.2f}% SoH)")
    print(f"• MAE after Smoothing: {mae_smooth:.4f} (es. {mae_smooth*100:.2f}% SoH)")

    # 5. Generate Plot for Ground Truth vs Predictions
    plt.figure(figsize=(14, 6))
    
    # Ground Truth and Predictions Plot
    plt.plot(Y_test, label="Ground Truth (Actual SoH)", color='black', alpha=0.8, linewidth=1.2)
    plt.plot(smoothed_predictions, label="TCN Prediction", color='red', alpha=0.7, linewidth=1.0)
    
    plt.title(f"TCN Evaluation on Test Set | MAE: {mae*100:.2f}% -> {mae_smooth*100:.2f}% - RMSE: {rmse:.4f}", fontsize=14)
    plt.xlabel("Time Window Index", fontsize=11)
    plt.ylabel("State of Health (SoH)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=12, loc="upper right")
    plt.tight_layout()
    plt.show()

    # 6. RESIDUAL ERROR PLOT GENERATION (Residuals = Y_true - Y_pred)
    residuals = (Y_test - smoothed_predictions) * 100  # Expressed in percentage
    plt.figure(figsize=(14, 4))
    plt.plot(residuals, color='purple', alpha=0.5, linewidth=0.8)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title("Instantaneous Residual Error (Actual SoH - Predicted SoH) [%]", fontsize=12)
    plt.xlabel("Time Window Index", fontsize=10)
    plt.ylabel("Error [%]", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
