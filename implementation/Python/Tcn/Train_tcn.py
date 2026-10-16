import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import numpy as np
import json
import torch
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from Class_models.TCN import TCN
from Class_models.Battery_dataset import loader_creator

MODEL_CONFIG_PATH = 'python/Tcn/model_config.json'
NPZ_FILE_PATH = 'Dataset_mv/train_val_soh.npz'
MODEL_PATH = 'Models_fp32/model_soh_tcn_best.pth'
MODEL_PARAMS_PATH = 'Models_fp32/model_soh_tcn_best_params.json'

# Creation of the output directory if it does not exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    # Opening the JSON configuration file
    with open(MODEL_CONFIG_PATH, "r") as f:
        config = json.load(f)

    # Creation of the parameter dictionary for grid search
    param_grid = {k: (v if isinstance(v, list) else [v]) for k, v in config.items()}

    # Best model and parameters (GLOBAL, across the entire grid search)
    best_params = None
    best_abs_val_loss = float('inf')

    # Training cycle for each parameter combination
    for params in ParameterGrid(param_grid):
        print("="*50)
        print("Configuration:", params)

        num_channels = params["num_channels"]
        kernel_size = params["kernel_size"]
        dense_units = params["dense_units"]
        dropout = params.get("dropout", 0.1)
        
        epochs = params["epochs"]
        batch_size = params["batch_size"]
        lr = params["learning_rate"]
        early_stop = params["early_stop"]

        # Loading the dataset
        npz_file = np.load(NPZ_FILE_PATH)
        X_train, Y_train = npz_file['X_train'], npz_file['Y_train']
        X_val, Y_val = npz_file['X_val'], npz_file['Y_val']
        print(f'Loading the dataset {NPZ_FILE_PATH}...')

        # Creating DataLoader
        train_loader = loader_creator(X_train, Y_train, batch_size=batch_size, num_workers=2)
        val_loader = loader_creator(X_val, Y_val, batch_size=batch_size, num_workers=2)

        print("Dataset successfully uploaded!")
        
        # Model,loss and optimizer initialization
        model = TCN(
            in_channels=2,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dense_units=dense_units,
            dropout=dropout
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
        )
        criterion = torch.nn.SmoothL1Loss()
        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"TCN model successfully initialized! Parameters: {num_params}")

        # Local early stopping, only for the current parameter configuration
        run_best_val_loss = float('inf')
        early_stop_counter = 0

        # Training loop
        print("\n=== TRAINING BEGINS ===")
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{epochs:02d}]", unit="batch", leave=False)
            
            for bx, by in pbar:
                bx, by = bx.to(device), by.to(device)
                
                optimizer.zero_grad()
                preds = model(bx)
                
                loss = criterion(preds, by)
                
                # Security check to prevent NaN propagation
                if torch.isnan(loss):
                    print("\n[WARNING] Loss exploded to NaN! Stopping this parameter combination.")
                    break

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item() * bx.size(0)
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})

            train_loss = train_loss / len(train_loader.dataset)

            # --- Validation phase ---
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)

            val_loss = val_loss / len(val_loader.dataset)
            scheduler.step(val_loss)

            # Early stopping + GLOBAL "live" saving
            if val_loss < run_best_val_loss:
                run_best_val_loss = val_loss
                early_stop_counter = 0
                saved = " [* Best for this configuration]"

                if val_loss < best_abs_val_loss:
                    best_abs_val_loss = val_loss
                    best_params = params
                    torch.save(model.state_dict(), MODEL_PATH)
                    with open(MODEL_PARAMS_PATH, "w") as f:
                        json.dump(best_params, f, indent=4)
                    saved += " [* NEW ALL-TIME BEST - Saved to disk]"
            else:
                saved = ""
                early_stop_counter += 1
                if early_stop_counter >= early_stop:
                    print("\nEarly stopping triggered.")
                    break

            tqdm.write(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}{saved}")

    print("\n=== GRID SEARCH COMPLETE ===")
    print(f"Best absolute Val Loss: {best_abs_val_loss:.6f}")
    print(f"Best hyperparameter configuration: {best_params}")
