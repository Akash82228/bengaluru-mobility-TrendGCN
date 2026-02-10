"""Inference helper for TrendGCN (PEMS04), with optional subgraph support.

Key points vs training:
- The model learns a graph from node/time embeddings; the adj/dist file is unused for PEMS04.
- We therefore drop all adjacency loading and just pass a dummy identity to keep shapes.
- Checkpoint may include an argparse.Namespace; we load safely under torch>=2.6 weights_only default.

Usage
-----
# Full-graph (103 nodes)
python3 infer.py --checkpoint log/PEMS04/20260210235007/best_model.pth --data dataset/PEMS04/PEMS04.npz

# Subgraph by node indices (first 40 nodes)
python3 infer.py --checkpoint log/PEMS04/20260210235007/best_model.pth --data dataset/PEMS04/PEMS04.npz --subgraph_indices 0-39

# Subgraph by camera name list
python3 infer.py --checkpoint ... --data dataset/PEMS04/PEMS04.npz --node_id_txt dataset/PEMS04/PEMSD4.txt --subgraph_cameras my_40_cameras.txt
"""

import argparse
import configparser
import inspect
import os
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import torch
import pandas as pd

from dataloader import normalize_dataset, Add_Window_Horizon
from model.generator import DAAGCN as Generator
from utils.util import get_device


def parse_subgraph_indices(arg: str):
    pieces = [p.strip() for p in arg.split(',') if p.strip()]
    out = []
    for p in pieces:
        if '-' in p:
            a, b = p.split('-')
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(p))
    seen = set()
    uniq = []
    for i in out:
        if i not in seen:
            uniq.append(i)
            seen.add(i)
    return uniq


def cameras_to_indices(node_id_txt: str, camera_list_txt: str) -> List[int]:
    """Map camera names to node indices using the training node-id list."""
    with open(node_id_txt, 'r', encoding='utf-8') as f:
        master = [line.strip() for line in f if line.strip()]
    name_to_idx = {name: i for i, name in enumerate(master)}
    with open(camera_list_txt, 'r', encoding='utf-8') as f:
        wanted = [line.strip() for line in f if line.strip()]
    indices = []
    missing = []
    for cam in wanted:
        if cam in name_to_idx:
            indices.append(name_to_idx[cam])
        else:
            missing.append(cam)
    if missing:
        raise ValueError(f"Cameras not found in node list: {missing}")
    return indices


def build_args(cfg_path: str, dataset: str, gpu_id: int, num_nodes_override: Optional[int] = None):
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)
    num_nodes = num_nodes_override if num_nodes_override is not None else cfg.getint('data', 'num_nodes')
    return SimpleNamespace(
        dataset=dataset,
        gpu_id=gpu_id,
        num_nodes=num_nodes,
        lag=cfg.getint('data', 'lag'),
        horizon=cfg.getint('data', 'horizon'),
        normalizer=cfg.get('data', 'normalizer'),
        column_wise=cfg.getboolean('data', 'column_wise'),
        default_graph=cfg.getboolean('data', 'default_graph'),
        input_dim=cfg.getint('model', 'input_dim'),
        output_dim=cfg.getint('model', 'output_dim'),
        embed_dim=cfg.getint('model', 'embed_dim'),
        rnn_units=cfg.getint('model', 'rnn_units'),
        num_layers=cfg.getint('model', 'num_layers'),
        cheb_k=cfg.getint('model', 'cheb_order'),
        seed=cfg.getint('train', 'seed'),
    )


def load_checkpoint_bundle(path: str, device: torch.device):
    def _torch_load(p: str, *, map_location, weights_only: bool):
        try:
            if 'weights_only' in inspect.signature(torch.load).parameters:
                return torch.load(p, map_location=map_location, weights_only=weights_only)
        except Exception:
            pass
        return torch.load(p, map_location=map_location)

    try:
        if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'safe_globals'):
            with torch.serialization.safe_globals([argparse.Namespace]):
                ckpt = _torch_load(path, map_location=device, weights_only=True)
        else:
            ckpt = _torch_load(path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"[WARN] weights_only load failed ({e}); retrying with weights_only=False")
        ckpt = _torch_load(path, map_location=device, weights_only=False)

    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    cfg = ckpt.get('config') if isinstance(ckpt, dict) else None
    return state_dict, cfg


def load_csv_timeseries(csv_path: str, node_id_col: Optional[str] = None, orientation: str = "nodes_by_rows"):
    """
    Read a node x time CSV into ndarray (T, N, 1) and node name list.
    - orientation=nodes_by_rows: rows are nodes, columns are timesteps (default, matches your file).
    - orientation=nodes_by_cols: rows are timesteps, columns are nodes.
    """
    df = pd.read_csv(csv_path)

    # pick node id column: explicit or first non-numeric
    if node_id_col:
        if node_id_col.isdigit():
            col = df.columns[int(node_id_col)]
        else:
            col = node_id_col
        node_ids = df[col].astype(str).tolist()
        df = df.drop(columns=[col])
    else:
        node_ids = None
        for c in df.columns:
            if not pd.to_numeric(df[c], errors="coerce").notna().all():
                node_ids = df[c].astype(str).tolist()
                df = df.drop(columns=[c])
                break
        if node_ids is None:
            # no explicit id column; fabricate indices
            node_ids = [f"node_{i}" for i in range(len(df))]

    df_numeric = df.apply(pd.to_numeric, errors="coerce")
    if df_numeric.isnull().values.any():
        bad_cols = [c for c in df_numeric.columns if df_numeric[c].isnull().any()]
        raise ValueError(f"Non-numeric values in columns {bad_cols}")

    values = df_numeric.to_numpy(dtype=np.float32, copy=True)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D table, got {values.shape}")

    if orientation == "nodes_by_rows":
        data = values.T  # (T, N)
    else:
        data = values     # (T, N)
        node_ids = list(df.columns) if not node_id_col else node_ids

    data = data[..., None]  # (T, N, 1)
    return data, node_ids


def prepare_data(args, data_npz_path, data_csv_path, node_id_col, orientation, node_id_txt, sub_idx, use_csv):
    if use_csv:
        raw, names = load_csv_timeseries(data_csv_path, node_id_col=node_id_col, orientation=orientation)
        # map names to indices in master list to keep training order
        with open(node_id_txt, 'r', encoding='utf-8') as f:
            master = [line.strip() for line in f if line.strip()]
        name_to_idx = {n: i for i, n in enumerate(master)}
        idxs = []
        missing = []
        for n in names:
            if n in name_to_idx:
                idxs.append(name_to_idx[n])
            else:
                missing.append(n)
        if missing:
            raise ValueError(f"Camera names not found in master node list: {missing}")
        # reorder raw to master order of the selected nodes
        order = np.argsort(idxs)
        idxs = [idxs[i] for i in order]
        raw = raw[:, order, :]
        sub_idx = idxs
        print(f"CSV provided {len(names)} nodes; mapped to indices {sub_idx}")
    else:
        npz = np.load(data_npz_path)
        if 'data' not in npz:
            raise ValueError(f"{data_npz_path} missing key 'data'")
        raw = npz['data']
        if raw.ndim == 2:
            raw = raw[..., None]

    if sub_idx is not None and not use_csv:
        raw = raw[:, sub_idx, :]
        print(f"Subgraph slice -> {raw.shape} (nodes={len(sub_idx)})")

    normed, scaler = normalize_dataset(raw, args.normalizer, args.column_wise)
    X, Y = Add_Window_Horizon(normed, window=args.lag, horizon=args.horizon, single=False)
    print(f"Windows: X {X.shape}, Y {Y.shape}")
    return X, Y, scaler, sub_idx


def load_generator(args, state_dict, sub_idx, device):
    gen = Generator(args).to(device)
    sd = state_dict.copy()
    if sub_idx is not None:
        if 'node_embeddings' not in sd:
            raise KeyError('node_embeddings missing in checkpoint')
        sd['node_embeddings'] = sd['node_embeddings'][sub_idx]

        # LayerNorm over nodes has shape [num_nodes]; slice to subgraph
        ln_keys = [k for k in sd.keys() if 'layernorm_graph.weight' in k or 'layernorm_graph.bias' in k]
        for k in ln_keys:
            tensor = sd[k]
            if tensor.dim() == 1 and tensor.shape[0] >= max(sub_idx)+1:
                sd[k] = tensor[sub_idx]

    missing, unexpected = gen.load_state_dict(sd, strict=False if sub_idx else True)
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    return gen


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data', default='dataset/PEMS04/PEMS04.npz', help='NPZ file (fallback if CSV not given)')
    ap.add_argument('--data_csv', default=None, help='CSV with nodes as rows and timesteps as columns')
    ap.add_argument('--orientation', default='nodes_by_rows', choices=['nodes_by_rows', 'nodes_by_cols'], help='CSV layout')
    ap.add_argument('--node_id_col', default=None, help='CSV column name or index containing node ids (optional)')
    ap.add_argument('--dataset', default='PEMS04')
    ap.add_argument('--config', default='config/PEMS04.conf')
    ap.add_argument('--gpu_id', type=int, default=0)
    ap.add_argument('--subgraph_indices', type=str, default=None, help='e.g., 0-39 or 0,2,5')
    ap.add_argument('--node_id_txt', type=str, default='dataset/PEMS04/PEMSD4.txt', help='node id list used at training')
    ap.add_argument('--subgraph_cameras', type=str, default=None, help='file with one camera name per line')
    args_cli = ap.parse_args()

    use_csv = args_cli.data_csv is not None

    # resolve subgraph indices
    sub_idx = None
    if args_cli.subgraph_cameras:
        sub_idx = cameras_to_indices(args_cli.node_id_txt, args_cli.subgraph_cameras)
        print(f"Subgraph cameras -> indices: {sub_idx}")
    elif args_cli.subgraph_indices:
        sub_idx = parse_subgraph_indices(args_cli.subgraph_indices)

    # load checkpoint first
    state_dict, ckpt_cfg = load_checkpoint_bundle(args_cli.checkpoint, torch.device('cpu'))

    # build args (prefer checkpoint cfg if available)
    model_args = ckpt_cfg or build_args(args_cli.config, args_cli.dataset, args_cli.gpu_id)
    model_args.gpu_id = args_cli.gpu_id
    model_args.dataset = args_cli.dataset
    full_nodes = int(getattr(model_args, 'num_nodes', build_args(args_cli.config, args_cli.dataset, args_cli.gpu_id).num_nodes))
    if sub_idx is not None:
        model_args.num_nodes = len(sub_idx)

    device = get_device(model_args)

    # data
    X, Y, scaler, sub_idx = prepare_data(model_args,
                                         args_cli.data,
                                         args_cli.data_csv,
                                         args_cli.node_id_col,
                                         args_cli.orientation,
                                         args_cli.node_id_txt,
                                         sub_idx,
                                         use_csv)
    if sub_idx is not None:
        model_args.num_nodes = len(sub_idx)

    # norm_dis_matrix is unused by the model (dynamic graph); keep identity for shape
    norm_dis = torch.eye(model_args.num_nodes, device=device)

    # model
    gen = load_generator(model_args, state_dict, sub_idx, device)
    gen.eval()

    # inference
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        print(f"X_tensor: {X_tensor.shape}")
        out = []
        bs = 64
        for i in range(0, len(X_tensor), bs):
            xb = X_tensor[i:i+bs]
            yb = gen(xb, norm_dis)
            out.append(yb.cpu())
            print(f"batch {i//bs+1}: in {xb.shape} -> out {yb.shape}")
        y_pred = torch.cat(out, dim=0)
        print(f"y_pred: {y_pred.shape} (samples, horizon, nodes, 1)")

    y_pred_real = scaler.inverse_transform(y_pred)
    y_true_real = scaler.inverse_transform(Y)
    out_dir = os.path.join('log', args_cli.dataset, 'inference_outputs')
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'y_pred.npy'), y_pred_real.numpy())
    np.save(os.path.join(out_dir, 'y_true.npy'), y_true_real)
    print(f"Saved predictions to {out_dir}/y_pred.npy")


if __name__ == '__main__':
    run()
