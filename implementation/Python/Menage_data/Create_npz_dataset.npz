"""
End-to-end preprocessing pipeline that builds the Train/Val and
Test datasets for State-of-Health (SoH) estimation, starting
from raw RPT (reference performance test) and cycling CSV files.

Pipeline:
    1. Extracts SoH ground truth from RPT files (filtering implausible
       values) and linearly interpolates it across cycling weeks
    2. Fits Z-score normalization parameters (mean/std of Voltage and
       Current) using ONLY the training cells, explicitly excluding the
       cells reserved for the test set (no data leakage)
    3. For each cycling file: applies a moving average to reduce
       high-frequency noise, merges the interpolated SoH onto each row,
       normalizes Voltage/Current, and generates fixed-size sliding
       windows (128 samples) with their corresponding SoH target
    4. Routes each window to the Train or Test set based on its cell ID
    5. Saves two compressed .npz files (train/val and test), each
       bundling the windowed data together with the normalization
       parameters (v_mean, v_std, i_mean, i_std) used to produce it
"""
import os
import glob
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. CONFIGURATION OF PARAMETERS AND PATHS
# ==========================================

RPT_PATH = "E:/Dataset/rpt_data/rpt_cell_*.csv"
CYCLING_PATH = "E:/Dataset/cycling_data/cycling_cell_*.csv"

NOMINAL_CAPACITY = 1.2  # Ah (nominal capacity of the battery)
WINDOW_SIZE = 128
STRIDE = 128            # Overlap / stride for sliding windows

WINDOW_MOVING_AVERAGE = 5  # Moving average window (to reduce high-frequency noise)

# Define test cell IDs and keys
TEST_CELL_IDS = ['05', '12', '18', '25', '33', '41', '48', '52', '59', '63']
TEST_CELL_KEYS = [str(int(x)) for x in TEST_CELL_IDS]

OUTPUT_FILE_TRAIN_VAL = "Dataset_mv/train_val_soh_processed_full.npz"
OUTPUT_FILE_TEST = "Dataset_mv/test_soh.npz"

# Creation of the output directory if it does not exist
os.makedirs(os.path.dirname(OUTPUT_FILE_TRAIN_VAL), exist_ok=True)

def extract_cell_id(filepath):
    """Estraction cell ID."""
    filename = os.path.basename(filepath)
    match = re.search(r'cell_(\d+)', filename)
    if match:
        return match.group(1)
    return None

# Function to apply moving average to reduce high-frequency noise in the data
def apply_moving_average(df, window_size=5):
    df['Voltage (V)'] = df['Voltage (V)'].rolling(window=window_size, min_periods=1).mean()
    df['Current (A)'] = df['Current (A)'].rolling(window=window_size, min_periods=1).mean()
    return df

# ==========================================
# 2. SOH Ground Truth Extraction with Interpolation
# ==========================================
def build_soh_ground_truth_map(rpt_files_pattern):
    print("--> 1/5 Extraction and Linear Interpolation of Ground Truth SoH from RPT Files...")
    soh_records = []
    
    rpt_files = glob.glob(rpt_files_pattern)
    print(f"    {len(rpt_files)} RPT files found...")
    
    for file_path in rpt_files:
        cell_id = extract_cell_id(file_path)
        if not cell_id:
            continue
            
        try:
            df_rpt = pd.read_csv(file_path, usecols=['Segment Key', 'Capacity (Ah)', 'RPT Number', 'Week Number'])
        except Exception:
            continue
            
        ref_dchg = df_rpt[df_rpt['Segment Key'] == 'ref_dchg']
        group_col = 'Week Number' if 'Week Number' in ref_dchg.columns else 'RPT Number'
        
        # --- ROBUST FILTERING OUTLIERS SOH ---
        for group_val, group in ref_dchg.groupby(group_col):
            max_capacity = group['Capacity (Ah)'].max()
            soh = max_capacity / NOMINAL_CAPACITY

            # Maintains only plausible SoH values ​​for Li-ion batteries.
            if 0.65 <= soh <= 1.10:
                soh_records.append({
                    'cell_id': str(int(cell_id)),
                    'week_number': group_val,
                    'soh': soh
                })
                
    df_soh_map = pd.DataFrame(soh_records).drop_duplicates()
    df_soh_map = df_soh_map.sort_values(by=['cell_id', 'week_number'])
    
    print(f"    Native RPT map created ({len(df_soh_map)} points). Interpolation in progress...")
    return df_soh_map


# ==========================================
# 3. Sliding Windows and Normalization
# ==========================================
def process_cycling_files(cycling_files_pattern, df_soh_map):
    print("--> 2/5 Time series processing and windowing...")
    
    cycling_files = sorted(glob.glob(cycling_files_pattern))
    print(f"    {len(cycling_files)} Cycling files found...")
    
    # -------------------------------------------------------------
    # 3A. Calculation of Z-score parameters exclusively on the training set.
    # -------------------------------------------------------------
    print("   Calculation of Z-score parameters (only on the TRAIN SET, excluding test cells)...")
    sum_v, sum_sq_v, count_v = 0.0, 0.0, 0
    sum_i, sum_sq_i, count_i = 0.0, 0.0, 0

    for f in cycling_files:
        cell_id = extract_cell_id(f)
        if not cell_id:
            continue
        cell_key = str(int(cell_id))
        
        # Skip test files when calculating the mean and standard deviation
        if cell_key in TEST_CELL_KEYS:
            continue
            
        df_temp = pd.read_csv(f, usecols=['Voltage (V)', 'Current (A)'])
        v = df_temp['Voltage (V)'].values
        i = df_temp['Current (A)'].values
        
        sum_v += v.sum();   sum_sq_v += (v**2).sum();   count_v += len(v)
        sum_i += i.sum();   sum_sq_i += (i**2).sum();   count_i += len(i)

    v_mean = sum_v / count_v
    v_std = np.sqrt(sum_sq_v / count_v - v_mean**2)

    i_mean = sum_i / count_i
    i_std = np.sqrt(sum_sq_i / count_i - i_mean**2)

    scaler = StandardScaler()
    scaler.mean_ = np.array([v_mean, i_mean], dtype=np.float32)
    scaler.scale_ = np.array([v_std, i_std], dtype=np.float32)
    scaler.var_ = scaler.scale_ ** 2
            
    print(f"    [TRAIN]  V -> Mean: {scaler.mean_[0]:.4f} V, Std: {scaler.scale_[0]:.4f} V")
    print(f"    [TRAIN]  I -> Mean: {scaler.mean_[1]:.4f} A, Std: {scaler.scale_[1]:.4f} A") 

    # -------------------------------------------------------------
    # 3B. WINDOW GENERATION AND SOH INTERPOLATION
    # -------------------------------------------------------------
    train_X, train_Y = [], []
    test_X, test_Y = [], []
    
    processed_count = 0
    for file_path in cycling_files:
        cell_id = extract_cell_id(file_path)
        if not cell_id:
            continue
            
        cell_key = str(int(cell_id))
        df_cell_soh = df_soh_map[df_soh_map['cell_id'] == cell_key]
        # If there is no SoH data for this cell, skip the file.
        if df_cell_soh.empty:
            continue
            
        try:
            df_cycle = pd.read_csv(file_path, usecols=['Week Number', 'Voltage (V)', 'Current (A)'])
        except Exception:
            continue       
        
        # SAFETY CHECK: Skip empty cycling files
        if df_cycle.empty:
            continue

        # Application of a moving average to reduce high-frequency noise
        df_cycle_mv = apply_moving_average(df_cycle, window_size=WINDOW_MOVING_AVERAGE)

        # --- MERGE AND INTERPOLATION ---
        df_merged = pd.merge_ordered(
            df_cycle_mv, 
            df_cell_soh[['week_number', 'soh']], 
            left_on='Week Number', 
            right_on='week_number', 
            how='left'
        )

        # 1. Linear interpolation between consecutive RPT tests
        df_merged['soh'] = df_merged['soh'].interpolate(method='linear')

        # 2. Fill initial/final unmapped regions (before the first RPT or after the last)
        df_merged['soh'] = df_merged['soh'].ffill().bfill()

        # 3. Complete removal of any rows with missing data
        df_merged = df_merged.dropna(subset=['Voltage (V)', 'Current (A)', 'soh'])

        # --- CRITICAL CHECK TO PREVENT (0 SAMPLES) ERROR ---
        if len(df_merged) < WINDOW_SIZE:
            # If the file has fewer than 128 rows after cleaning, it is insufficient to create even 1 window
            continue
        
        # Input normalization using statistics computed on the TRAIN set
        v_i_scaled = scaler.transform(df_merged[['Voltage (V)', 'Current (A)']].values)
        soh_targets = df_merged['soh'].values
        
        # Time Window Generation
        num_samples = len(v_i_scaled)
        for start in range(0, num_samples - WINDOW_SIZE + 1, STRIDE):
            end = start + WINDOW_SIZE
            
            x_window = v_i_scaled[start:end, :]
            y_soh = soh_targets[end - 1]
            
            if cell_key in TEST_CELL_KEYS:
                test_X.append(x_window)
                test_Y.append(y_soh)
            else:
                train_X.append(x_window)
                train_Y.append(y_soh)
                
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"    Processed {processed_count}/{len(cycling_files)} files...")
                
    return (np.array(train_X, dtype=np.float32), np.array(train_Y, dtype=np.float32),
            np.array(test_X, dtype=np.float32), np.array(test_Y, dtype=np.float32),
            scaler)


# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    print("=== START OF SOH PREPROCESSING PIPELINE ===")

    # Create output directories if they do not exist
    os.makedirs(os.path.dirname(OUTPUT_FILE_TRAIN_VAL), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_FILE_TEST), exist_ok=True)

    # --- Build SoH Ground Truth Map ---
    df_soh_map = build_soh_ground_truth_map(RPT_PATH)
    
    X_train, Y_train, X_test, Y_test, scaler = process_cycling_files(CYCLING_PATH, df_soh_map)
    
    print("\n--> 3/5 Dataset Dimension Check:")
    print(f"    Train Input (X_train): {X_train.shape}  | Target (Y_train): {Y_train.shape}")
    print(f"    Test Input (X_test)  : {X_test.shape}   | Target (Y_test) : {Y_test.shape}")
    
    print(f"\n--> 4/5 Saving compressed dataset to '{OUTPUT_FILE_TRAIN_VAL}' and '{OUTPUT_FILE_TEST}'...")
    np.savez_compressed(
        OUTPUT_FILE_TRAIN_VAL,
        X_train=X_train,
        Y_train=Y_train,
        v_mean=scaler.mean_[0],
        v_std=scaler.scale_[0],
        i_mean=scaler.mean_[1],
        i_std=scaler.scale_[1]
    )
    
    np.savez_compressed(
        OUTPUT_FILE_TEST,
        X_test=X_test,
        Y_test=Y_test,
        v_mean=scaler.mean_[0],
        v_std=scaler.scale_[0],
        i_mean=scaler.mean_[1],
        i_std=scaler.scale_[1]   
    )

    print("\n=== PIPELINE COMPLETED ===")
