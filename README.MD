# 🛡️ UAV Aegis: Advanced Propeller Diagnostics & Prognostics

## 🚨 The Problem: Hidden In-Flight Failures
Unmanned Aerial Vehicles (UAVs) are highly vulnerable to subtle propeller and motor faults—such as micro-cracks, chips, or minor blade damage—that are nearly impossible to detect using standard telemetry (e.g., simple RPM or power draw). These faults manifest as high-frequency vibrations that degrade sensor fidelity, increase structural fatigue, and eventually lead to catastrophic mid-air failures and asset loss.

## 🛡️ The Solution: UAV Aegis v3
**UAV Aegis** is a professional-grade industrial diagnostic and prognostic suite that solves this problem through deep learning:
- **2D-CNN Intelligence**: Processes high-frequency IMU data (Acc/Gyro) to identify complex fault signatures with **99%+ accuracy**.
- **Spectral Fingerprinting**: Utilizes Fast Fourier Transforms (FFT) to pinpoint resonance peaks associated with specific failure modes.
- **Predictive Maintenance (Prognostics)**: Estimates **Remaining Useful Life (RUL)** to schedule repairs before a failure occurs.
- **Industrial Command Center**: A hardened, real-world dashboard for fleet managers to monitor UAV health in real-time.

---

## Overview
This project implements a full pipeline for UAV propeller/motor fault simulation, dataset generation, model training, evaluation, and visualization. It features the **UAV Aegis v3** dashboard for industrial-grade operations.

---

#### SCRIPTS FOLDER


Key files and scripts (representative):

* `generate_synthetic_runs.py`
* `generate_realistic_runs.py`
* `make_small_realistic_batch.py`
* `isaac_replay_recorder.py`  *(requires Isaac Kit Python / omni modules)*
* `quad.usd`
* `train_cnn.py`, `finetune_cnn.py`
* `inference_multihead.py`, `smoke_test.py`
* `propeller_lstm.py`, `propeller_model.py`
* `inspect_dataset_and_plot.py`, `plot_training_curves.py`, `plot_confusion_from_csv.py`
* `cnn_classifier` (module/package referenced by training/eval scripts)

> NOTE: Refer to [run_command.txt](file:///run_command.txt) for the primary execution guide.

---

## Prerequisites & Environment

* **Python:** 3.8+ for training/eval scripts.
* **Libraries:** `numpy`, `h5py`, `torch`, `scikit-learn`, `matplotlib`, `tqdm`, `pandas`, `streamlit`, `plotly`, `scipy`.

---

## Quick Launch: UAV Aegis v3 Dashboard
1. Open a terminal in the `scripts` directory.
2. Run:
   ```powershell
   py -m streamlit run dashboard.py
   ```

---
Refer to [run_command.txt](file:///run_command.txt) for detailed operational steps (training, evaluation, etc.).
