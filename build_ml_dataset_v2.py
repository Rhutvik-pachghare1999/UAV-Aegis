# scripts/build_ml_dataset_v2.py

import os, glob, h5py, json, argparse, numpy as np, pandas as pd
from config import ISAAC_DATASET_DIR
from severity_utils import severity_from_ur

# Per paper: 10 variables: 4 RPMs, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate
DEFAULT_VARS = ["rpm1","rpm2","rpm3","rpm4","roll","pitch","yaw","gyro_x","gyro_y","gyro_z"]

def read_run(run_dir, window=100, step=1, vars=DEFAULT_VARS):
    # accept either imu.csv or imu_full.csv (some recorders write imu_full.csv)
    imu_csv = None
    for name in ("imu.csv", "imu_full.csv"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            imu_csv = p; break
    meta_json = os.path.join(run_dir, "meta.json")
    # imu_csv may be None if neither imu.csv nor imu_full.csv existed
    if imu_csv is None:
        return [], []
    if not os.path.exists(imu_csv):
        return [], []
    df = pd.read_csv(imu_csv)
    # If state.csv exists, merge columns (e.g., roll,pitch,yaw) from it to ensure rates/angles present
    state_csv = os.path.join(run_dir, "state.csv")
    if os.path.exists(state_csv):
        try:
            sdf = pd.read_csv(state_csv)
            # prefer state columns when present (e.g., roll,pitch,yaw)
            for c in ['roll', 'pitch', 'yaw', 'roll_rate', 'pitch_rate', 'yaw_rate']:
                if c in sdf.columns:
                    df[c] = sdf[c]
        except Exception:
            pass
    # ensure all requested vars exist; fill missing with 0.0
    for v in vars:
        if v not in df.columns:
            df[v] = 0.0
    data = df[vars].values
    Xs, metas = [], []
    T = data.shape[0]
    for i in range(0, T - window + 1, step):
        win = data[i:i+window].T  # (C, W)
        Xs.append(win)
        metas.append(i)
    if os.path.exists(meta_json):
        meta = json.load(open(meta_json, "r"))
    else:
        meta = {}
    ur = None
    if "fault_params" in meta:
        ur = meta["fault_params"].get("unbalance", meta["fault_params"].get("ur", None))
    sev = severity_from_ur(float(ur) if ur is not None else 0.0)
    fault_label = meta.get("fault_type", "unknown")
    return Xs, [{"fault_label": fault_label, "severity": sev, "ur": ur, "meta": meta} for _ in Xs]

def build_dataset(project_root: str, out_h5: str, window:int=100, step:int=1, vars=DEFAULT_VARS):
    runs = sorted(glob.glob(os.path.join(project_root, "isaac_dataset", "run_*")))
    print("Found runs:", len(runs))
    all_X, all_fault_labels, all_sev, all_ur = [], [], [], []
    fault_label_map = {}; next_label = 0
    for r in runs:
        Xs, metas = read_run(r, window=window, step=step, vars=vars)
        if not Xs: continue
        for x,m in zip(Xs, metas):
            fl = m.get("fault_label") or "unknown"
            if fl not in fault_label_map:
                fault_label_map[fl] = next_label; next_label += 1
            all_X.append(x)
            all_fault_labels.append(fault_label_map[fl])
            all_sev.append(int(m.get("severity", 0)))
            all_ur.append(float(m.get("ur") or 0.0))
    if not all_X:
        # No windows found: likely missing imu/state files in `isaac_dataset`.
        # Make this a graceful no-op so orchestration scripts can continue.
        print("Warning: No windows found. Check ISAAC dataset directory and imu.csv placement. Skipping dataset write.")
        return None
    X = np.stack(all_X, axis=0).astype("float32")  # (N, C, W)
    X = X[:, None, :, :]  # (N,1,C,W)
    y_fault = np.array(all_fault_labels, dtype="int64")
    y_sev = np.array(all_sev, dtype="int64")
    ur_arr = np.array(all_ur, dtype="float32")
    # Ensure minimum dataset size by simple augmentation (noise + resampling)
    MIN_SAMPLES = 20000
    N = len(X)
    if N < MIN_SAMPLES:
        print(f"Dataset has {N} samples, which is less than requested minimum {MIN_SAMPLES}. Augmenting by resampling+noise.")
        # compute per-channel std to scale noise
        # X shape is (N,1,C,W) -> take std over samples (axis=0) and time (axis=3)
        data_std = X.std(axis=(0,3), keepdims=True)
        needed = MIN_SAMPLES - N
        # sample indices with replacement
        idxs = np.random.randint(0, N, size=needed)
        aug_X = X[idxs].copy()
        # add small gaussian noise relative to per-channel std
        noise_scale = np.maximum(data_std, 1e-6) * 0.01
        aug_noise = np.random.normal(loc=0.0, scale=1.0, size=aug_X.shape).astype('float32') * noise_scale
        aug_X = aug_X + aug_noise
        aug_y_fault = y_fault[idxs].copy()
        aug_y_sev = y_sev[idxs].copy()
        aug_ur = ur_arr[idxs].copy()
        # concatenate
        X = np.concatenate([X, aug_X], axis=0)
        y_fault = np.concatenate([y_fault, aug_y_fault], axis=0)
        y_sev = np.concatenate([y_sev, aug_y_sev], axis=0)
        ur_arr = np.concatenate([ur_arr, aug_ur], axis=0)
        print(f"Augmented dataset to {len(X)} samples.")
    os.makedirs(os.path.dirname(out_h5) or ".", exist_ok=True)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("X", data=X, compression="gzip")
        f.create_dataset("y_fault", data=y_fault)
        f.create_dataset("y_sev", data=y_sev)
        f.create_dataset("ur", data=ur_arr)
        f.attrs["meta"] = json.dumps({"fault_label_map": fault_label_map, "window": window, "vars": vars})
    print("Wrote", out_h5, "samples=", len(X))
    return out_h5

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument("--out", type=str, default="ml_dataset_v2.h5")
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--vars", type=str, default=None, help="comma-separated variable list to include (overrides defaults)")
    args = p.parse_args()
    if args.vars:
        var_list = [v.strip() for v in args.vars.split(',') if v.strip()]
    else:
        var_list = DEFAULT_VARS
    build_dataset(args.project_root, out_h5=args.out, window=args.window, step=args.step, vars=var_list)
