#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pointwise NN training with Random Grids/Smiles — Local (no-CLI) version
------------------------------------------------------------------------
Simply run this file ("Run"). All options are set below
in the USER CONFIG block. No command line arguments required.

- Supported device: "auto" (cuda > mps > cpu), "cpu", "cuda", "mps".
- Local output in `OUTPUT_DIR` with subfolders datasets/ and models/.
- Maintains: Random Grids (train), Random Smiles (val), normalizations,
  PointwiseNetworkPricer, ReduceLROnPlateau, early stopping, checkpoints.
"""

from __future__ import annotations
import os
import sys
import json
import math
import gc
import time
import glob
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# =========================
# === USER CONFIG START ===
# =========================
PROCESS = "rough_bergomi"      # {"heston", "rough_heston", "rough_bergomi"}
DEVICE = "auto"                # {"auto", "cpu", "cuda", "mps"}
PROJECT_ROOT = None

# Dataset sizes
TRAIN_SURFACES = 10_000
TRAIN_MATURITIES = 11
TRAIN_STRIKES = 13
VAL_SMILES = 10_000
VAL_STRIKES = 13

# Monte Carlo + data
MC_PATHS = 50_000
SPOT = 1.0
SEED = 42

# Training
EPOCHS = 100
BATCH_SIZE = 4096
LEARNING_RATE = 1e-3
USE_LR_SCHEDULER = True
EARLY_STOPPING_PATIENCE = 20

# Resume / warm start
RESUME = False               # True to resume
RESUME_FROM = "latest"       # path to checkpoint or "latest"
WARM_START = False           # True for warm-start
WARM_START_PATH = None       # path to best_model.pt (if None tries models/best_model.pt)

# Network
HIDDEN_LAYERS = [30, 30, 30, 30]
ACTIVATION = "ELU"           # as used in your pricer

# Output
OUTPUT_DIR = f"runs/pointwise_random_grids/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# ---------------------------------------------------------------------
# Import project modules with a robust fallback
# ---------------------------------------------------------------------

def _import_project_modules(project_root: Path | None = None):
    try:
        from deepLearningVolatility.stochastic.stochastic_interface import ProcessFactory  # noqa: F401
        from deepLearningVolatility.nn.dataset_builder import DatasetBuilder  # noqa: F401
        from deepLearningVolatility.nn.pricer.pricer import PointwiseNetworkPricer  # noqa: F401
        return
    except Exception:
        pass

    if project_root is None:
        project_root = Path(__file__).resolve().parent

    candidates = [
        project_root,
        project_root / "deepLearningVolatility",
        project_root.parent,
    ]

    for cand in candidates:
        if not cand:
            continue
        if cand.exists() and cand.is_dir() and str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
            try:
                from deepLearningVolatility.stochastic.stochastic_interface import ProcessFactory  # noqa: F401
                from deepLearningVolatility.nn.dataset_builder import DatasetBuilder  # noqa: F401
                from deepLearningVolatility.nn.pricer.pricer import PointwiseNetworkPricer  # noqa: F401
                return
            except Exception:
                continue

    raise ImportError(
        "Could not import deepLearningVolatility modules. Install the package (pip install -e .)"
        "or set PROJECT_ROOT to your repo root."
    )


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------

def select_device(device_arg: str) -> torch.device:
    device_arg = device_arg.lower()
    if device_arg not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("DEVICE must be one of: auto, cpu, cuda, mps")

    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            print("[warn] CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")

    if device_arg == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print("[warn] MPS requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("mps")

    return torch.device("cpu")


def ensure_dirs(base: Path):
    (base / "models" / "checkpoints").mkdir(parents=True, exist_ok=True)
    (base / "datasets").mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2))


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(str(path), map_location=device)
    return ckpt


# ---------------------------------------------------------------------
# Dataset generation helpers
# ---------------------------------------------------------------------

def generate_datasets(CONFIG, process, device: torch.device):
    from deepLearningVolatility.nn.dataset_builder import DatasetBuilder

    print("" + "="*60)
    print("DATASET GENERATION WITH RANDOM GRIDS/SMILES")
    print("="*60)

    # TRAIN — Random Grids
    train_builder = DatasetBuilder(
        process=process,
        device=str(device),
        output_dir=CONFIG["output_dir"],
        dataset_type='train'
    )

    print(f">>> Generating Training Dataset with Random Grids")
    print(f"    Surfaces: {CONFIG['train_surfaces']}")
    print(f"    Maturities per surface: {CONFIG['train_maturities']}")
    print(f"    Strikes per maturity: {CONFIG['train_strikes']}")
    print(f"    Total points: {CONFIG['train_surfaces'] * CONFIG['train_maturities'] * CONFIG['train_strikes']:,}")

    t0 = time.time()
    theta_tr, T_tr, k_tr, iv_tr = train_builder.build_random_grids_dataset(
        n_surfaces=CONFIG["train_surfaces"],
        n_maturities=CONFIG["train_maturities"],
        n_strikes=CONFIG["train_strikes"],
        n_paths=CONFIG["mc_paths"],
        spot=CONFIG["spot"],
        normalize=True,
        compute_stats_from='train'
    )
    t1 = time.time()
    print(f"✔ Training dataset generated in {t1 - t0:.1f}s")

    # VAL — Random Smiles
    val_builder = DatasetBuilder(
        process=process,
        device=str(device),
        output_dir=CONFIG["output_dir"],
        dataset_type='val'
    )

    print(f">>> Generating Validation Dataset with Random Smiles")
    print(f"    Smiles: {CONFIG['val_smiles']}")
    print(f"    Strikes per smile: {CONFIG['val_strikes']}")
    print(f"    Total points: {CONFIG['val_smiles'] * CONFIG['val_strikes']:,}")

    t2 = time.time()
    theta_val, T_val, k_val, iv_val = val_builder.build_random_smiles_dataset(
        n_smiles=CONFIG["val_smiles"],
        n_strikes_per_smile=CONFIG["val_strikes"],
        n_paths=CONFIG["mc_paths"],
        spot=CONFIG["spot"],
        normalize=True,
        compute_stats_from='train',
        train_stats=train_builder.get_normalization_stats()
    )
    t3 = time.time()
    print(f"✔ Validation dataset generated in {t3 - t2:.1f}s")

    # Save tensors
    torch.save({'theta': theta_tr, 'T': T_tr, 'k': k_tr, 'iv': iv_tr}, str(Path(CONFIG['output_dir'])/"datasets"/"train_random_grids.pt"))
    torch.save({'theta': theta_val, 'T': T_val, 'k': k_val, 'iv': iv_val}, str(Path(CONFIG['output_dir'])/"datasets"/"val_random_smiles.pt"))

    return (train_builder, (theta_tr, T_tr, k_tr, iv_tr), (theta_val, T_val, k_val, iv_val))


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def _state_from_checkpoint_obj(ckpt_obj):
    from collections.abc import Mapping
    if isinstance(ckpt_obj, Mapping):
        for key in ("model_state_dict", "state_dict", "net_state_dict", "pricer_state_dict"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], Mapping):
                return ckpt_obj[key]
        return ckpt_obj
    return ckpt_obj


def train_pointwise_network(CONFIG, builder, train_data, val_data, device: torch.device):
    from deepLearningVolatility.nn.pricer.pricer import PointwiseNetworkPricer

    print("" + "="*60)
    print("POINTWISE NETWORK TRAINING")
    print("="*60)

    theta_tr, T_tr, k_tr, iv_tr = train_data
    theta_val, T_val, k_val, iv_val = val_data

    pricer = PointwiseNetworkPricer(
        process=builder.process,
        hidden_layers=CONFIG["hidden_layers"],
        activation=CONFIG["activation"],
        device=str(device),
        enable_smile_repair=False
    )

    pricer.set_normalization_stats(
        builder.theta_mean, builder.theta_std,
        builder.iv_mean, builder.iv_std
    )
    pricer.set_pointwise_normalization_stats(
        builder.T_mean, builder.T_std,
        builder.k_mean, builder.k_std
    )

    train_loader = DataLoader(
        TensorDataset(theta_tr, T_tr, k_tr, iv_tr),
        batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        TensorDataset(theta_val, T_val, k_val, iv_val),
        batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0, pin_memory=False
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(pricer.net.parameters(), lr=CONFIG["learning_rate"]) 
    scheduler = None
    if CONFIG.get("lr_scheduler", True):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

    start_epoch = 1

    ckpt_dir = Path(CONFIG['output_dir'])/"models"/"checkpoints"

    if CONFIG.get("resume_training", False):
        ckpt_path = CONFIG.get("resume_from")
        if ckpt_path in (None, "latest"):
            ckpts = glob.glob(str(ckpt_dir/"checkpoint_epoch_*.pt"))
            if ckpts:
                ckpt_path = max(
                    ckpts,
                    key=lambda p: int(Path(p).stem.split("_")[-1]) if Path(p).stem.split("_")[-1].isdigit() else -1
                )
        if ckpt_path and Path(ckpt_path).exists():
            print(f"Resuming from checkpoint: {ckpt_path}")
            ckpt = load_checkpoint(Path(ckpt_path), device)
            pricer.net.load_state_dict(_state_from_checkpoint_obj(ckpt))
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'epoch' in ckpt and isinstance(ckpt['epoch'], int):
                start_epoch = ckpt['epoch'] + 1

    elif CONFIG.get("warm_start", False):
        warm_path = CONFIG.get("warm_start_path")
        if warm_path is None:
            warm_path = Path(CONFIG['output_dir'])/"models"/"best_model.pt"
        warm_path = Path(warm_path)
        if warm_path.exists():
            print(f"Warm-starting from: {warm_path}")
            ckpt = load_checkpoint(warm_path, device)
            pricer.net.load_state_dict(_state_from_checkpoint_obj(ckpt.get('model_state_dict', ckpt)))

    best_val = float('inf')
    best_state = None
    best_epoch = start_epoch - 1
    patience = CONFIG.get("early_stopping_patience", 20)
    waited = 0

    history = {"train_loss": [], "val_loss": []}

    print(f">>> Starting training for {CONFIG['epochs']} epochs")
    print(f"    Batch size: {CONFIG['batch_size']}")
    print(f"    Learning rate: {CONFIG['learning_rate']}")

    for epoch in range(start_epoch, CONFIG['epochs'] + 1):
        pricer.net.train()
        run_loss = 0.0
        nb = 0
        for theta_b, T_b, k_b, iv_b in train_loader:
            pred = pricer.forward(theta_b, T_b, k_b, inputs_normalized=True, denormalize_output=False)
            loss = criterion(pred, iv_b)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            run_loss += loss.item()
            nb += 1

        train_loss = (run_loss / max(nb, 1))
        history["train_loss"].append(train_loss)

        pricer.net.eval()
        with torch.no_grad():
            v_loss = 0.0
            vb = 0
            for theta_b, T_b, k_b, iv_b in val_loader:
                pred = pricer.forward(theta_b, T_b, k_b, inputs_normalized=True, denormalize_output=False)
                l = criterion(pred, iv_b)
                v_loss += l.item()
                vb += 1
            val_loss = v_loss / max(vb, 1)
        history["val_loss"].append(val_loss)

        if scheduler is not None:
            scheduler.step(val_loss)

        print(f"Epoch {epoch:4d} | train {train_loss:.6f} | val {val_loss:.6f}")

        ckpt_payload = {
            'model_state_dict': pricer.net.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            'optimizer_state_dict': optimizer.state_dict(),
            'network_config': {
                'hidden_layers': CONFIG['hidden_layers'],
                'activation': CONFIG['activation']
            },
            'normalization_stats': {
                'theta_mean': builder.theta_mean,
                'theta_std': builder.theta_std,
                'T_mean': builder.T_mean,
                'T_std': builder.T_std,
                'k_mean': builder.k_mean,
                'k_std': builder.k_std,
                'iv_mean': builder.iv_mean,
                'iv_std': builder.iv_std,
            },
        }
        torch.save(ckpt_payload, str(ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pt"))
        torch.save(ckpt_payload, str(Path(CONFIG['output_dir'])/"models"/"latest_model.pt"))

        if val_loss < best_val:
            best_val = val_loss
            best_state = pricer.net.state_dict()
            best_epoch = epoch
            waited = 0
            torch.save({'model_state_dict': best_state, 'epoch': epoch, 'val_loss': best_val, 'config': CONFIG,
                        'normalization_stats': ckpt_payload['normalization_stats']},
                       str(Path(CONFIG['output_dir'])/"models"/"best_model.pt"))
        else:
            waited += 1

        if waited >= patience:
            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, val {best_val:.6f})")
            break

    if best_state is not None:
        pricer.net.load_state_dict(best_state)

    save_json(history, Path(CONFIG['output_dir'])/"training_history.json")

    return pricer, history


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

def build_config() -> dict:
    device = select_device(DEVICE)
    if device.type == "mps":
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    out_dir = Path(OUTPUT_DIR)
    ensure_dirs(out_dir)

    cfg = {
        "process": PROCESS,
        "device": str(device),
        "train_surfaces": TRAIN_SURFACES,
        "train_maturities": TRAIN_MATURITIES,
        "train_strikes": TRAIN_STRIKES,
        "val_smiles": VAL_SMILES,
        "val_strikes": VAL_STRIKES,
        "mc_paths": MC_PATHS,
        "spot": SPOT,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "lr_scheduler": USE_LR_SCHEDULER,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "resume_training": RESUME,
        "resume_from": RESUME_FROM,
        "warm_start": WARM_START,
        "warm_start_path": WARM_START_PATH,
        "hidden_layers": HIDDEN_LAYERS,
        "activation": ACTIVATION,
        "output_dir": str(out_dir),
    }
    return cfg


def main():
    # Project import
    _import_project_modules(Path(PROJECT_ROOT) if PROJECT_ROOT else None)
    from deepLearningVolatility.stochastic.stochastic_interface import ProcessFactory

    # Seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Config
    CONFIG = build_config()

    print("" + "="*60)
    print(" RANDOM GRIDS POINTWISE NETWORK TRAINING ")
    print("="*60)
    for k in ("process","device","train_surfaces","train_maturities","train_strikes",
              "val_smiles","val_strikes","mc_paths","batch_size","epochs","learning_rate"):
        print(f"  {k}: {CONFIG[k]}")

    # Save config
    save_json(CONFIG, Path(CONFIG['output_dir'])/"config.json")

    # Create process
    process = ProcessFactory.create(CONFIG["process"])
    print(f"✔ Created process: {process.__class__.__name__}")
    try:
        params = getattr(process, 'param_info', None)
        if params is not None and hasattr(params, 'names'):
            print(f"  Parameters: {params.names}")
    except Exception:
        pass
    if hasattr(process, 'supports_absorption'):
        print(f"  Supports absorption: {process.supports_absorption}")

    # Dataset
    builder, train_data, val_data = generate_datasets(CONFIG, process, torch.device(CONFIG["device"]))

    # Train
    pricer, history = train_pointwise_network(CONFIG, builder, train_data, val_data, torch.device(CONFIG["device"]))

    print("✓ COMPLETE EXECUTION FINISHED!")
    print(f"All results saved to: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
