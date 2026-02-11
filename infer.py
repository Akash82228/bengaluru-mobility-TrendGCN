import inspect
import os
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from utils.norm import StandardScaler, MinMax01Scaler, MinMax11Scaler, NScaler
from model.generator import DAAGCN as Generator
from utils.util import get_device


# ---- Fixed hyperparameters from config/PEMS04.conf ---- #
CFG = SimpleNamespace(
    dataset="PEMS04",
    num_nodes=103,
    lag=5,
    horizon=5,
    normalizer="zscore",  # matches training config
    column_wise=False,
    default_graph=True,
    input_dim=1,
    output_dim=1,
    embed_dim=6,
    rnn_units=64,   
    num_layers=2,
    cheb_k=2,
    seed=10,
    gpu_id=0,
    train_npz_path="dataset/PEMS04/PEMS04.npz",
)


# ---- Checkpoint loading (torch 2.6+ safe) ---- #
def load_checkpoint_bundle(path: str, device: torch.device):
    def _torch_load(p: str, *, map_location, weights_only: bool):
        try:
            if "weights_only" in inspect.signature(torch.load).parameters:
                return torch.load(p, map_location=map_location, weights_only=weights_only)
        except Exception:
            pass
        return torch.load(p, map_location=map_location)

    try:
        if hasattr(torch, "serialization") and hasattr(torch.serialization, "safe_globals"):
            with torch.serialization.safe_globals([SimpleNamespace]):
                ckpt = _torch_load(path, map_location=device, weights_only=True)
        else:
            ckpt = _torch_load(path, map_location=device, weights_only=True)
    except Exception:
        ckpt = _torch_load(path, map_location=device, weights_only=False)

    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    return state_dict, cfg


# ---- CSV handling ---- #
def load_window_csv(csv_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Read a single-window CSV: first column camera name/id, next 5 columns lag timesteps.
    Returns data shaped (T=5, N, 1) and list of camera names in file order.
    """
    df = pd.read_csv(csv_path)
    if df.shape[1] < 6:
        raise ValueError("CSV must have 1 id column + 5 lag columns")
    cams = df.iloc[:, 0].astype(str).tolist()
    values = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    if values.shape[1] != CFG.lag:
        raise ValueError(f"Expected {CFG.lag} lag columns, got {values.shape[1]}")
    # values: (N, lag) -> (lag, N, 1)
    data = values.T[..., None]
    return data, cams


def map_cameras_to_indices(master_txt: str, cams: List[str]) -> List[int]:
    with open(master_txt, "r", encoding="utf-8") as f:
        master = [line.strip() for line in f if line.strip()]
    name_to_idx = {name: i for i, name in enumerate(master)}
    idxs = []
    missing = []
    for cam in cams:
        if cam in name_to_idx:
            idxs.append(name_to_idx[cam])
        else:
            missing.append(cam)
    if missing:
        raise ValueError(f"Cameras not found in master list: {missing}")
    return idxs


# ---- Training scaler (use full training stats, not per-window) ---- #
def load_training_scaler(normalizer: str, column_wise: bool, sub_idx: Optional[List[int]]):
    data = np.load(CFG.train_npz_path)["data"][:, :, 0]  # [T, N]
    if sub_idx is not None:
        data = data[:, sub_idx]
    if data.ndim == 1:
        data = data[:, None]

    if normalizer in ["std", "zscore", "z-score"]:
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
    elif normalizer == "max01":
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
    elif normalizer == "max11":
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
    elif normalizer == "None":
        scaler = NScaler()
    else:
        raise ValueError(f"Unsupported normalizer {normalizer}")
    return scaler


# ---- Model prep ---- #
def load_generator(args, state_dict, sub_idx: Optional[List[int]], device):
    gen = Generator(args).to(device)
    sd = state_dict.copy()
    if sub_idx is not None:
        if "node_embeddings" not in sd:
            raise KeyError("node_embeddings missing in checkpoint")
        sd["node_embeddings"] = sd["node_embeddings"][sub_idx]
        # slice node-wise LayerNorm params
        ln_keys = [
            k for k in sd.keys() if "layernorm_graph.weight" in k or "layernorm_graph.bias" in k
        ]
        for k in ln_keys:
            t = sd[k]
            if t.dim() == 1 and t.shape[0] >= max(sub_idx) + 1:
                sd[k] = t[sub_idx]
    gen.load_state_dict(sd, strict=False if sub_idx else True)
    return gen


# ---- Core inference ---- #
def run_inference(
    weights_path: str,
    input_csv_path: str,
    node_list_txt_path: str,
    output_csv_path: Optional[str] = None,
):
    # device & args
    args = CFG
    device = get_device(args)

    # load data window
    raw, cams_in_file = load_window_csv(input_csv_path)  # (T, N, 1)
    sub_idx = map_cameras_to_indices(node_list_txt_path, cams_in_file)
    args.num_nodes = len(sub_idx)

    # use training-data scaler to match model expectations
    scaler = load_training_scaler(args.normalizer, args.column_wise, sub_idx)
    normed = scaler.transform(raw)  # (lag, N, 1)
    # shape to batch of one: [1, lag, N, 1]
    X = normed[np.newaxis, ...]

    # model & checkpoint
    state_dict, _ = load_checkpoint_bundle(weights_path, torch.device("cpu"))
    gen = load_generator(args, state_dict, sub_idx, device)
    gen.eval()

    # identity norm_dis (adj unused)
    norm_dis = torch.eye(args.num_nodes, device=device)

    # inference
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)  # [1, lag, N, 1]
        y_pred = gen(X_tensor, norm_dis)  # [1, horizon, N, 1]

    # denormalize
    # Model was trained with real_value=True (for PEMS*), so outputs are already in real scale.
    # Do NOT inverse-transform, or values will blow up.
    y_pred_real = y_pred.cpu().squeeze(-1).squeeze(0)  # [H, N]
    y_pred_real = y_pred_real.permute(1, 0).numpy()  # [N, H]

    # build output CSV
    cols = [f"H{i+1}" for i in range(args.horizon)]
    out_df = pd.DataFrame(y_pred_real, columns=cols)
    out_df.insert(0, "camera", cams_in_file)

    out_path = (
        Path(output_csv_path)
        if output_csv_path
        else Path(input_csv_path).with_name(Path(input_csv_path).stem + "_pred.csv")
    )
    out_df.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")
    return out_df, str(out_path)


# Optional convenience: run with hardcoded defaults matching user request
if __name__ == "__main__":
    run_inference(
        weights_path="/mnt/HDD/akashs/SCALE/TrendGCN/log/PEMS04/final-run-2/best_model.pth",
        input_csv_path="/mnt/HDD/akashs/SCALE/TrendGCN/inference_testing/sub-graph-1.csv",
        node_list_txt_path="inference_testing/PEMSD4.txt",
    )
