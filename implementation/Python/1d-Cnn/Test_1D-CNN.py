import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from Class_models.CNN_1D import CNN_1D
from Class_models.Battery_dataset import loader_creator
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import json
import torch

NPZ_FILE_PATH = 'Dataset_mv/test_soh.npz'
MODEL_PATH = 'Models_fp32/model_soh_1dcnn.pth'
MODEL_PARAMS_PATH = 'Models_fp32/model_soh_1dcnn_best_params.json'

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    # Opening the JSON configuration file
    with open(MODEL_PARAMS_PATH, "r") as f:
        params = json.load(f)

    filters = params["filters"]
    kernel_size = params["kernel_size"]
    hidden_neurons = params["dense_units"]
    batch_size = params["batch_size"]

    # Loading the dataset
    npz_file = np.load(NPZ_FILE_PATH, mmap_mode='r')
    print(f'Loading the dataset {NPZ_FILE_PATH}...')
    # Creating DataLoaders
    test_loader = loader_creator(npz_file['X_test'], npz_file['Y_test'], batch_size=batch_size, num_workers=4)

    # Extraction of the data shape
    input_shape = test_loader.dataset.get_input_shape()

    print(f"Dataset successfully uploaded!")
    # Model,loss and optimizer initialization
    model = CNN_1D(input_shape,
                    filters=filters,
                    kernel_size=kernel_size,
                    hidden_neurons=hidden_neurons).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    print(f"Model '{MODEL_PATH}' successfully loaded!")
    print(f"Evaluating on {len(test_loader)} samples from the Test Set...")
    
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            predictions.extend(outputs.cpu().numpy().flatten())
            ground_truths.extend(targets.numpy().flatten())
            
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    # Caluclation of error metrics
    mae = mean_absolute_error(ground_truths, predictions)
    rmse = np.sqrt(mean_squared_error(ground_truths, predictions))
    
    # Calculation of Average Percentage Error
    mape = np.mean(np.abs((ground_truths - predictions) / ground_truths)) * 100
    
    print("\n" + "="*40)
    print("      TEST RESULTS ON UNSEEN CELLS")
    print("="*40)
    print(f"• MAE  (Mean Absolute Error)    : {mae:.4f}")
    print(f"• RMSE (Root Mean Square Error) : {rmse:.4f}")
    print(f"• MAPE (Mean Absolute % Error)  : {mape:.2f}%")
    print("="*40)
