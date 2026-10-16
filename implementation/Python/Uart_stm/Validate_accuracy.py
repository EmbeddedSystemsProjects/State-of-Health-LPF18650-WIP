"""
Comparative benchmark of the SoH model (TCN) across three different backends, 
starting from the same test CSV dataset:
    1. PyTorch FP32 (CPU)
    2. Quantized INT8 ONNX Runtime (CPU)
    3. STM32 (real hardware, via serial)
"""
import os
import sys
import time
import json
import pandas as pd
import numpy as np
import serial
import torch
import onnxruntime as ort
from scipy.signal import savgol_filter

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from Class_models.TCN import TCN

# 1. Path and Parameters
SERIAL_PORT = "COM8"
BAUDRATE    = 115200

# Model paths
ONNX_MODEL_PATH   = "Models_onnx/Model_tcn/model_tcn_int8.onnx"
PTH_MODEL_PATH    = "Models_fp32/model_soh_tcn.pth"
MODEL_PARAMS_PATH = "Models_fp32/model_soh_tcn_best_params.json"

# Dataset for testing
CSV_FILE_PATH = "E:/Dataset/cycling_data/cycling_cell_20_part17.csv"
NPZ_FILE_PATH = "Dataset_mv/train_val_soh.npz"

WINDOW_SIZE = 128
NUM_WINDOWS = 20
STEP_SIZE   = 64  # Overlap 50%

# 2. Upload Z-Score Parameters from .npz file
print("--> Upload Z-Score parameters from .npz file...")
data = np.load(NPZ_FILE_PATH)
V_MEAN = float(data["v_mean"])
V_STD  = float(data["v_std"])
I_MEAN = float(data["i_mean"])
I_STD  = float(data["i_std"])

print("=" * 85)
print(f"Z-Score parameters: V_MEAN = {V_MEAN:.5f} | V_STD = {V_STD:.5f} | I_MEAN = {I_MEAN:.5f} | I_STD = {I_STD:.5f}")
print("=" * 85)

# ==========================================
# 3. Uploading models (PyTorch FP32 & ONNX)
# ==========================================
device = torch.device("cpu")

# --- PyTorch FP32 (TCN) ---
model_pt = None
if os.path.exists(PTH_MODEL_PATH) and os.path.exists(MODEL_PARAMS_PATH):
    with open(MODEL_PARAMS_PATH, "r") as f:
        best_params = json.load(f)
    
    # Initialization of the TCN model with parameters from the JSON
    model_pt = TCN(
        num_channels=best_params["num_channels"],
        kernel_size=best_params["kernel_size"],
        dropout=best_params["dropout"]
    )
    
    # Load the state dictionary into the model
    model_pt.load_state_dict(torch.load(PTH_MODEL_PATH, map_location=device))
    model_pt.to(device)
    model_pt.eval()
    print("TCN PyTorch FP32 models loaded succesfully.")
else:
    print(f"[WARNING] PyTorch file (.pth) or parameter JSON not found in '{PTH_MODEL_PATH}'. It will be ignored.")

# --- ONNX Runtime ---
ort_session = None
if os.path.exists(ONNX_MODEL_PATH):
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
    input_onnx_name = ort_session.get_inputs()[0].name
    print("ONNX models Runtime loaded succesfully.")
else:
    print(f"[WARN] ONNX file not found in '{ONNX_MODEL_PATH}'. It will be ignored.")


# 4. Connection and handshake STM32
print(f"--> Connection to serial port {SERIAL_PORT} @ {BAUDRATE} baud...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.5)
    time.sleep(2.0)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
except Exception as e:
    print(f"[SERIAL ERROR] Unable to open {SERIAL_PORT}: {e}")
    sys.exit(1)

# Handshake Z-Score to the STM32 board
cfg_str = f"CFG:{V_MEAN:.6f},{V_STD:.6f},{I_MEAN:.6f},{I_STD:.6f}\n"
ack_received = False

print("--> Sending Z-Score configuration to STM32 board...")
for attempt in range(1, 6):
    ser.reset_input_buffer()
    ser.write(cfg_str.encode("utf-8"))
    
    start_time = time.time()
    while time.time() - start_time < 1.0:
        if ser.in_waiting > 0:
            resp = ser.readline().decode("utf-8", errors="ignore").strip()
            if "ACK_CFG" in resp:
                print(f"[OK] Handshake complete! STM32 ready (Attempt {attempt}).")
                ack_received = True
                break
    if ack_received:
        break
    time.sleep(0.2)

if not ack_received:
    ser.close()
    raise RuntimeError("[ERROR] STM32 board is not responding to Z-Score configuration (missing ACK_CFG).")


# 5. Loading CSV data for testing

print(f"--> Reading data from: {CSV_FILE_PATH}")
df = pd.read_csv(CSV_FILE_PATH, usecols=["Voltage (V)", "Current (A)"])
v_all = df["Voltage (V)"].to_numpy(dtype=np.float32)
i_all = df["Current (A)"].to_numpy(dtype=np.float32)

print("\n" + "=" * 95)
print(f"{'Window':<10}{'CSV Row':<10}{'PyTorch FP32':<16}{'ONNX PC':<16}{'STM32 Smooth':<16}{'Difference':<10}")
print("=" * 95)

start_row = 1000

# Saving history for smoothing
onnx_raw_history = []
pt_raw_history   = []

for w in range(NUM_WINDOWS):
    row_idx = start_row + (w * STEP_SIZE)
    if row_idx + WINDOW_SIZE > len(df):
        break

    # Extract 128 raw samples
    v_win = v_all[row_idx : row_idx + WINDOW_SIZE]
    i_win = i_all[row_idx : row_idx + WINDOW_SIZE]

    # --- PC inferece 1: PyTorch FP32 ---
    soh_pt_raw = float('nan')
    soh_pt_smooth = float('nan')
    if model_pt is not None:
        v_norm = (v_win - V_MEAN) / V_STD
        i_norm = (i_win - I_MEAN) / I_STD
        inp_tensor = torch.tensor(np.stack([v_norm, i_norm], axis=0), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            soh_pt_raw = model_pt(inp_tensor.to(device)).item()
            pt_raw_history.append(soh_pt_raw)
            
            # Apply Savitzky-Golay filter for smoothing the predictions
            if len(pt_raw_history) >= 31:
                soh_pt_smooth = float(savgol_filter(pt_raw_history, window_length=31, polyorder=2)[-1])
            else:
                soh_pt_smooth = float(np.mean(pt_raw_history))

    # --- PC inference 2: ONNX Runtime ---
    soh_onnx_raw = float('nan')
    soh_onnx_smooth = float('nan')
    if ort_session is not None:
        v_norm = (v_win - V_MEAN) / V_STD
        i_norm = (i_win - I_MEAN) / I_STD
        inp_onnx = np.expand_dims(np.stack([v_norm, i_norm], axis=0), axis=0).astype(np.float32)
        out = ort_session.run(None, {input_onnx_name: inp_onnx})[0]
        soh_onnx_raw = float(out.flatten()[0])
        onnx_raw_history.append(soh_onnx_raw)

        # Apply Savitzky-Golay filter for smoothing the predictions
        if len(onnx_raw_history) >= 31:
            soh_onnx_smooth = float(savgol_filter(onnx_raw_history, window_length=31, polyorder=2)[-1])
        else:
            soh_onnx_smooth = float(np.mean(onnx_raw_history))

    # --- Hardware inference: Sending line by line to STM32 ---
    payload = "".join([f"{v:.6f},{i:.6f}\n" for v, i in zip(v_win, i_win)])
 
    # Send the payload to STM32 and measure latency
    t_start = time.perf_counter()
    ser.write(payload.encode("utf-8"))

    stm32_raw = float('nan')
    stm32_smooth = float('nan')
    response_received = False

    # Read response from STM32
    start_read = time.time()
    while time.time() - start_read < 2.0:
        if ser.in_waiting > 0:
            resp = ser.readline().decode("utf-8", errors="ignore").strip()
            if "Soh_Raw:" in resp and "Soh_Smooth:" in resp:
                try:
                    parts = resp.split(",")
                    stm32_raw = float(parts[0].split(":")[1])
                    stm32_smooth = float(parts[1].split(":")[1])
                    response_received = True
                    break
                except (ValueError, IndexError):
                    pass

    latency = (time.perf_counter() - t_start) * 1000.0

    raw_str = f"{stm32_raw:.4f}" if not np.isnan(stm32_raw) else "TIMEOUT"
    smooth_str = f"{stm32_smooth:.4f}" if not np.isnan(stm32_smooth) else "TIMEOUT"

    print(f"{w:<10}{row_idx:<10}{soh_pt_smooth:<16.4f}{soh_onnx_smooth:<16.4f}{smooth_str:<16}{latency:<10.1f}")

    time.sleep(0.01)

ser.close()
print("=" * 95)
print("Comparative benchmark completed successfully.")
