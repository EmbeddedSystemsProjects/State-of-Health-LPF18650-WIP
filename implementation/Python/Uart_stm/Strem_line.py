"""
Stream real-time voltage/current data (read from a test CSV file), 
sample by sample, to the STM32 board via serial port, where the quantized SoH model is running.
"""

import time
import numpy as np
import pandas as pd
import serial

# CONFIGURATION
SERIAL_PORT = "COM8"  # Make sure this is the correct port
BAUD_RATE = 115200

# Here are specified the test files for cycling data, you can change the index to select a different file
# ['05', '12', '18', '25', '33', '41', '48', '52', '59', '63']
TEST_CSV_PATH = "E:/Dataset/cycling_data/cycling_cell_05_part8.csv"
DOWNSAMPLE_STEP = 2
MAX_TEST_SAMPLES = 3000


# SERIAL PORT OPENING
print("--> Open serial port...")
ser = serial.Serial(
    SERIAL_PORT, BAUD_RATE, timeout=1.0, dsrdtr=False, rtscts=False
)
time.sleep(2.0)
ser.reset_input_buffer()
ser.reset_output_buffer()

# LOADING Z-SCORE PARAMETERS
print("--> Loading parameters (.npz)...")
data = np.load("Dataset_mv/train_val_soh.npz")
V_MEAN = float(data["v_mean"])
V_STD = float(data["v_std"])
I_MEAN = float(data["i_mean"])
I_STD = float(data["i_std"])

# SENDING PARAMETERS WITH ACKNOWLEDGEMENT (HANDSHAKE)
print("--> Sending normalization parameters to the STM32...")
cfg_str = f"CFG:{V_MEAN:.6f},{V_STD:.6f},{I_MEAN:.6f},{I_STD:.6f}\n"

ack_received = False
for attempt in range(1, 10):
  ser.reset_input_buffer()
  ser.write(cfg_str.encode("utf-8"))

  # Wait for a response for 1 second
  resp = ser.readline().decode("utf-8", errors="ignore").strip()
  if "ACK_CFG" in resp:
    print(f"STM32 successfully configured (Attempt {attempt})!")
    ack_received = True
    break
  time.sleep(0.3)

if not ack_received:
  raise RuntimeError(
      "The STM32 did not respond to the Z-Score configuration string."
  )

# LOADING AND PREPARING CSV DATA
print(f"--> Reading data from: {TEST_CSV_PATH}")
cols_to_load = ["Voltage (V)", "Current (A)"]
df = pd.read_csv(TEST_CSV_PATH, usecols=cols_to_load)

if DOWNSAMPLE_STEP > 1:
  df = df.iloc[::DOWNSAMPLE_STEP].reset_index(drop=True)

df = df.head(MAX_TEST_SAMPLES)
raw_values = df[["Voltage (V)", "Current (A)"]].values

print(f"--> Starting streaming of {len(raw_values)} samples...")

raw_predictions = []
smooth_predictions = []
indices = []

try:
  for i in range(len(raw_values)):
    v, curr = raw_values[i, 0], raw_values[i, 1]

    # Send the single sample in ASCII
    line = f"{v:.6f},{curr:.6f}\n"
    ser.write(line.encode("utf-8"))

    # Read responses (non-blocking for ordinary samples)
    while ser.in_waiting > 0:
      response = ser.readline().decode("utf-8", errors="ignore").strip()
      # print(f"STM32: {response}")

      # Parsing the response
      if "Soh_Raw:" in response and "Soh_Smooth:" in response:
          try:
              parts = response.split(",")
              raw_val = float(parts[0].split(":")[1])
              smooth_val = float(parts[1].split(":")[1])

              raw_predictions.append(raw_val)
              smooth_predictions.append(smooth_val)
              indices.append(i)

              print(f">>> [SAMPLE {i}] ---------> SoH: {smooth_val:.4f}")
          except (ValueError, IndexError):
              pass

    time.sleep(0.02)  # pause between samples

except KeyboardInterrupt:
  print("\n--> Streaming manually interrupted.")

finally:
  ser.close()
  print(f"--> Session ended. Received {len(raw_predictions)} predictions.")
