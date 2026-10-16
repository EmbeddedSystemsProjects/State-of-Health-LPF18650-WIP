"""
Script for the generation of calibration data for INT8 quantization.
"""
import os
import numpy as np

TRAIN_VAL_FILE = "Dataset_mv/train_val_soh.npz"
CALIB_OUTPUT = "Dataset_mv/calibration_data-2.npz"
NUM_SAMPLES = 1000

print(f"--> Uploading data from {TRAIN_VAL_FILE} ...")
data = np.load(TRAIN_VAL_FILE)

X_data = data["X_train"]

# Transpose the data to match the expected input shape
X_data = np.transpose(X_data, (0, 2, 1)).astype(np.float32)

# Extract a random subset of samples for calibration
np.random.seed(42)
NUM_SAMPLES = min(NUM_SAMPLES, len(X_data))
idx = np.random.choice(len(X_data), NUM_SAMPLES, replace=False)
calib_samples = X_data[idx]

# Saved the file with both 'X_train' and 'input_tensor' for compatibility with STM32Cube.AI
os.makedirs(os.path.dirname(CALIB_OUTPUT), exist_ok=True)
np.savez_compressed(
    CALIB_OUTPUT,
    X_train=calib_samples,
    X=calib_samples,
    input_tensor=calib_samples,
)

print(f"Calibration file succesfully generated: {CALIB_OUTPUT}")
