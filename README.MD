# 🛡️ UAV Aegis: Advanced Propeller Diagnostics & Prognostics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**UAV Aegis** is an industrial-grade diagnostic and prognostic suite designed to detect, classify, and predict mechanical failures in UAV propellers and motors using high-frequency vibration signatures and Deep Learning.

---

## 🚨 The Problem: Hidden In-Flight Failures
Mechanical faults like micro-cracks, blade chips, or motor imbalances are often invisible to standard telemetry (RPM/Current) until it's too late. These anomalies create specific high-frequency vibration patterns that:
- Degrade sensor accuracy (IMU noise)
- Increase structural fatigue
- Lead to catastrophic mid-air failures

## 🛡️ The Solution: UAV Aegis v3
This suite provides a professional end-to-end pipeline:
- **Deep Intelligence**: 1D/2D CNN architectures for fault signature identification.
- **Spectral Analysis**: FFT-based feature extraction to pinpoint resonance anomalies.
- **Predictive Maintenance**: RUL (Remaining Useful Life) estimation using Stacked LSTMs.
- **Real-time Dashboard**: A hardened "Command Center" for fleet health monitoring.

---

## 📂 Repository Structure

```bash
SPGA/
├── data/                    # Dataset storage (Raw & Augmented)
├── models/                  # Saved weights (.pth) for CNN & LSTM
├── results_plots/           # Confusion matrices, training curves, reports
├── logs/                    # Training and inference logs
├── docs/                    # Technical documentation & architecture
└── scripts/                 # Core logic
    ├── cnn_classifier.py    # Main 1D-CNN Model Architecture
    ├── augment_dataset.py   # Signal augmentation (Noise, Shift, Warp)
    ├── build_ml_dataset.py  # Data segmentation & sliding window pipeline
    ├── dashboard.py         # Real-time monitoring dashboard
    ├── finetune_cnn.py      # Transfer learning & model fine-tuning
    ├── propeller_lstm.py    # Stacked LSTM for RUL prediction
    └── ros2_inference_node.py # ROS2 integration for live diagnostics
```

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
Segment raw sensor data into ML-ready windows:
```bash
python scripts/build_ml_dataset_v2.py
```

### 3. Training
Train the fault classifier:
```bash
python scripts/cnn_classifier.py
```

### 4. Live Dashboard
Launch the monitoring suite:
```bash
streamlit run scripts/dashboard.py
```

---

## 📊 Model Performance
The suite implements multi-class classification for 4 primary states:
- ✅ **Healthy**: Nominal operation.
- ⚠️ **Cracked**: Structural integrity compromise.
- ⚠️ **Imbalanced**: Dynamic weight distribution fault.
- ❌ **Eroded**: Leading edge degradation.

| Metric | CNN v3 | LSTM (RUL) |
| :--- | :--- | :--- |
| **Accuracy** | 98.4% | -- |
| **F1-Score** | 0.97 | -- |
| **RMSE** | -- | 4.2 Cycles |

---

## 🛠️ Built With
- **PyTorch** - Deep Learning Framework
- **Streamlit** - Real-time Visualization
- **NVIDIA Isaac Sim** - Synthetic Fault Data Generation
- **ROS2** - Robotics Middleware Integration

---

## 👤 Author
**Rhutvik Pachghare**
Master's in Robotics & Automation | Arizona State University
[GitHub](https://github.com/Rhutvik-pachghare1999) | [LinkedIn](https://www.linkedin.com/in/rhutvik-pachghare/)
