# Edge AI-Based State of Health (SoH) Estimation for Li-Ion Batteries 18650

![Embedded](https://img.shields.io/badge/Embedded-STM32-blue)
![Framework](https://img.shields.io/badge/AI-ST_X--CUBE--AI-lightgrey)
![Models](https://img.shields.io/badge/Models-1D--CNN%20%7C%20TinyTCN-success)
![Deployment](https://img.shields.io/badge/Deployment-INT8_PTQ-orange)

## Project Overview
This project details the design, implementation, and validation of an Edge AI-based State of Health (SoH) estimation system for lithium-ion batteries. The primary objective is to deploy optimized deep learning architectures directly onto a resource-constrained microcontroller, enabling real-time battery degradation monitoring without relying on external cloud infrastructure.

The quantized models execute seamlessly within the microcontroller’s Flash and RAM limits while maintaining strong estimation accuracy, validating the feasibility of deploying deep temporal networks for onboard real-time Battery Management Systems (BMS).

## System Architecture
The system's core features is a validation setup that couples an **STM32F401RE microcontroller** with a host PC. 

1. **PC Host (Sensor Simulator):** Streams continuous, real-world battery cycling data (Voltage and Current) to the MCU.
2. **STM32 Edge Node:** Acts as the embedded BMS processing unit. It handles:
   - Sample-by-sample data acquisition via UART.
   - Buffering a 128-step temporal sliding window.
   - Onboard Z-score normalization of raw data.
   - Executing quantized neural network inference.
   - Applying a 31-point circular moving average filter to smooth the output.
  
<img width="3566" height="1184" alt="firmware_functionality" src="https://github.com/user-attachments/assets/ff6db0df-8899-45fc-b700-adf8da819792" />


### Communication Protocol
A custom ASCII-based synchronization protocol guarantees reliable streaming and frame alignment across the UART interface (115200 baud):
* **Handshake Phase:** The PC sends Z-score calibration parameters (`CFG:<v_mean>,<v_std>,<i_mean>,<i_std>\n`). The STM32 acknowledges with `ACK_CFG`.
* **Streaming Phase:** The PC sends continuous sample pairs (`<V>,<I>\n`).
* **Inference Phase:** Every 128 samples, the STM32 runs the inference and returns the raw and smoothed SoH predictions (`Soh_Raw:<value>,Soh_Smooth:<value>\n`).

## Machine Learning Pipeline
Two distinct deep learning architectures were designed and evaluated for sequential time-series processing:
* **1D-CNN (Convolutional Neural Network):** A lightweight baseline feature extractor.
* **TinyTCN (Temporal Convolutional Network):** An optimized causal network utilizing dilated convolutions (Receptive Field = 121) to capture long-term temporal dependencies perfectly fitted for the 128-sample window.

### Quantization & Deployment
To meet strict embedded constraints, models were trained in PyTorch (FP32), exported to ONNX, and subjected to **Post-Training Quantization (INT8 PTQ)**.
- **Toolchain:** ONNX Runtime & ARM CMSIS-NN via ST X-CUBE-AI.
- **Format:** `QuantFormat.QDQ` with Per-Channel resolution.
- **Outcome:** The footprint was drastically reduced by ~4x, achieving high computational efficiency and deterministic execution time with a marginal, acceptable trade-off in accuracy.

