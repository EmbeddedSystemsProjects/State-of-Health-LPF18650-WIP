import os
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# 1. Model path to test
MODELS_TO_TEST = {
    "TCN_FP32": "Models_onnx/Model_tcn/model_soh_tcn_fp32.onnx",
    "TCN_INT8": "Models_onnx/Model_tcn/model_tcn_int8.onnx",
}

TEST_NPZ_FILE_PATH = "Dataset_mv/test_soh.npz"
BATCH_SIZE = 256

# Function to compute evaluation metrics
def compute_metrics(preds, targets):
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    mape = np.mean(np.abs((preds - targets) / np.clip(np.abs(targets), 1e-6, None))) * 100
    return mae, rmse, mape


def evaluate_onnx(onnx_path, X, batch_size=256):
    """
    Function to evaluate an ONNX model (works for both FP32 and INT8).
    """
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    preds = []
    for start in range(0, len(X), batch_size):
        batch = X[start:start + batch_size]
        out = session.run(None, {input_name: batch})[0]
        preds.append(out)

    return np.concatenate(preds).flatten()


if __name__ == "__main__":

    # 2. Upload Test Set
    print("--> 1/4 Uploading Test Set...")
    test_npz = np.load(TEST_NPZ_FILE_PATH)
    
    X_raw = test_npz['X_test'].astype(np.float32)
    Y_test = test_npz['Y_test'].astype(np.float32)
    y_true = Y_test.flatten()

    # Controlla ed esegui la trasposizione a (N, 2, 128) se necessario
    if X_raw.shape[1] != 2 and X_raw.shape[2] == 2:
        X_test = np.transpose(X_raw, (0, 2, 1))
    else:
        X_test = X_raw

    print(f"    Test samples loaded {len(X_test):,}")

    # 3. Evaluation ONNX (FP32 vs INT8)
    print("\n--> 2/4 Evaluation ONNX models...")
    results = {}

    for name, path in MODELS_TO_TEST.items():
        # Check if the ONNX model file exists
        if not os.path.exists(path):
            print(f"    [SALTATO] {name}: file not found in '{path}'")
            continue

        print(f"    ONNX inference on: {name} ({path})...")
        y_pred = evaluate_onnx(path, X_test, batch_size=BATCH_SIZE)
        
        # Apply Savitzky-Golay filter for smoothing the predictions
        y_pred_smooth = savgol_filter(y_pred, window_length=31, polyorder=2)

        # Compute metrics for both raw and smoothed predictions
        mae_raw, rmse_raw, mape_raw = compute_metrics(y_pred, y_true)
        mae_sm, rmse_sm, mape_sm = compute_metrics(y_pred_smooth, y_true)

        results[name] = {
            "mae_raw": mae_raw, "rmse_raw": rmse_raw, "mape_raw": mape_raw, "preds_raw": y_pred,
            "mae_sm": mae_sm,   "rmse_sm": rmse_sm,   "mape_sm": mape_sm,   "preds_sm": y_pred_smooth
        }

    if not results:
        raise RuntimeError("No models found in MODELS_TO_TEST. Check the .onnx file paths.")

    # ==========================================
    # 4. METRICS COMPARISON TABLE
    # ==========================================
    header_width = 14 + 20 * len(results)
    print("\n" + "=" * header_width)
    print(f"{'Metric':<14}" + "".join(f"{name:>20}" for name in results))
    print("-" * header_width)
    print(f"{'MAE (Raw)':<14}"      + "".join(f"{r['mae_raw']:>20.4f}" for r in results.values()))
    print(f"{'MAE (Smooth)':<14}"   + "".join(f"{r['mae_sm']:>20.4f}" for r in results.values()))
    print(f"{'RMSE (Raw)':<14}"     + "".join(f"{r['rmse_raw']:>20.4f}" for r in results.values()))
    print(f"{'MAPE % (Raw)':<14}"   + "".join(f"{r['mape_raw']:>20.2f}%" for r in results.values()))
    print(f"{'MAPE % (Smooth)':<14}"+ "".join(f"{r['mape_sm']:>20.2f}%" for r in results.values()))
    print("=" * header_width)

    # Degradation relative to the FP32 version
    if "TCN_FP32" in results and "TCN_INT8" in results:
        delta_mae = results["TCN_INT8"]["mae_raw"] - results["TCN_FP32"]["mae_raw"]
        print(f"\n[INFO] INT8 vs FP32 quantization degradation: MAE {delta_mae:+.4f} ({delta_mae*100:+.2f}% SoH)")

    # ==========================================
    # 5. PREDICTIONS COMPARISON PLOT
    # ==========================================
    print("\n--> 3/4 Generating Plots...")
    plt.figure(figsize=(14, 6))
    plt.plot(y_true, label="Ground Truth (Actual SoH)", color='black', alpha=0.8, linewidth=1.2)

    colors = {'TCN_FP32': 'blue', 'TCN_INT8': 'crimson'}
    for name, r in results.items():
        c = colors.get(name, 'green')
        plt.plot(r['preds_sm'], label=f"{name} (Filtered) - MAE: {r['mae_sm']*100:.2f}%", color=c, alpha=0.8, linewidth=1.1)

    plt.title("ONNX Predictions Comparison (FP32 vs INT8) on Test Set", fontsize=13)
    plt.xlabel("Time Window Index", fontsize=11)
    plt.ylabel("State of Health (SoH)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    plt.show()

    # INT8 vs FP32 Residual Errors Plot
    plt.figure(figsize=(14, 4))
    for name, r in results.items():
        res = (y_true - r['preds_sm']) * 100
        plt.plot(res, label=f"{name} Error [%]", alpha=0.6, linewidth=0.8)

    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title("Instantaneous Residual Error [%] (Actual SoH - Filtered Prediction)", fontsize=12)
    plt.xlabel("Time Window Index", fontsize=10)
    plt.ylabel("Error [%]", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.show()
    
    print("--> [COMPLETED] Evaluation completed successfully!")
