#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Regime Grid Pricer Training for Local Execution
========================================================
Complete script for training multi-regime networks using separate grids
for short/mid/long term. Version adapted for local execution.
"""

import os
import sys
from pathlib import Path

# Add the project root to PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parents[1]  # Assuming the script is in a subfolder
sys.path.append(str(ROOT_DIR))

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import json
from datetime import datetime
import shutil
import pickle
import gc
from torch.utils.data import DataLoader, TensorDataset

from deepLearningVolatility.nn.pricer import MultiRegimeGridPricer
from deepLearningVolatility.nn.dataset_builder import MultiRegimeDatasetBuilder
from deepLearningVolatility.stochastic.stochastic_interface import ProcessFactory

import deepLearningVolatility.stochastic.wrappers.rough_bergomi_wrapper
import deepLearningVolatility.stochastic.wrappers.rough_heston_wrapper

print("GPU available?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))

# ===============================================================
# === MAIN CONFIGURATION ===
# ===============================================================

# Path to local project - MODIFY THIS PATH
PROJECT_DIR = "C:/Projects/NN/deepLearningVolatility"

CONFIG = {
    "process": "rough_heston",  # Options: "rough_heston", "rough_bergomi", ...
    "spot": 1.0,
    
    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # Dataset parameters
    "train_samples": {
        "short": 2000,
        "mid": 2000,
        "long": 2000
    },
    "val_samples": {
        "short": 400,
        "mid": 400,
        "long": 400
    },
    "mc_paths": 50000,
    "batch_size_generation": 100,
    
    # Training parameters
    "epochs_per_regime": 500,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "early_stopping_patience": 50,
    "scheduler_patience": 20,
    "scheduler_factor": 0.5,
    
    # Checkpoint settings
    "checkpoint_every": 100,
    "model_checkpoint_every": 100,
    "resume_from": None,  # 'latest' or path to checkpoint
    
    # Network architectures
    "short_hidden": [128, 64],
    "mid_hidden": [128, 64],
    "long_hidden": [128, 64],
    
    # Regime thresholds
    "short_term_threshold": 0.25,
    "mid_term_threshold": 1.0,
    
    # Output directory
    "output_dir": None,  # Will be set based on process
}

# Setup output directory based on process
process_key = CONFIG["process"]
CONFIG["output_dir"] = f"{PROJECT_DIR}/training_results/multi_regime_{process_key}"
output_dir = CONFIG["output_dir"]

# Create local directory structure
os.makedirs(f"{output_dir}/models", exist_ok=True)
os.makedirs(f"{output_dir}/models/checkpoints", exist_ok=True)
os.makedirs(f"{output_dir}/models/final", exist_ok=True)
os.makedirs(f"{output_dir}/logs", exist_ok=True)
os.makedirs(f"{output_dir}/visualizations", exist_ok=True)

print(f"\n✓ Directory create in: {output_dir}")

# ===============================================================
# === GRID DEFINITION FOR EACH REGIME ===
# ===============================================================

# Short term: 1 week - 1 month
SHORT_MATURITIES = torch.tensor([7/365, 14/365, 21/365, 30/365])
SHORT_LOGK = torch.linspace(-0.15, 0.15, 11)

# Mid term: 2 months - 9 months
MID_MATURITIES = torch.tensor([60/365, 90/365, 120/365, 180/365, 270/365])
MID_LOGK = torch.linspace(-0.3, 0.3, 11)

# Long term: 1 year - 5 years
LONG_MATURITIES = torch.tensor([1.0, 2.0, 3.0, 5.0])
LONG_LOGK = torch.linspace(-0.5, 0.5, 11)

# ===============================================================
# === VISUALIZATION FUNCTIONS ===
# ===============================================================

def visualize_surface_comparison(iv_true, iv_pred, regime_name, output_dir):
    """Displays IV surface comparison as heatmap."""
    n_examples = min(3, len(iv_true))
    if n_examples == 0:
        return

    fig, axes = plt.subplots(n_examples, 3, figsize=(18, 5 * n_examples), squeeze=False)
    fig.suptitle(f"Surface Comparison - {regime_name.upper()} Regime", fontsize=16)

    for i in range(n_examples):
        # Denormalized data
        true_surface = iv_true[i].cpu().detach().numpy()
        pred_surface = iv_pred[i].cpu().detach().numpy()
        error = np.abs(pred_surface - true_surface)

        # True surface
        im0 = axes[i, 0].imshow(true_surface, aspect='auto', cmap='viridis')
        axes[i, 0].set_title(f'Sample {i+1} - True IV Surface')
        axes[i, 0].set_xlabel('Strike Index')
        axes[i, 0].set_ylabel('Maturity Index')
        fig.colorbar(im0, ax=axes[i, 0])

        # Predicted surface
        im1 = axes[i, 1].imshow(pred_surface, aspect='auto', cmap='viridis')
        axes[i, 1].set_title(f'Sample {i+1} - Predicted IV Surface')
        axes[i, 1].set_xlabel('Strike Index')
        fig.colorbar(im1, ax=axes[i, 1])

        # Error surface
        im2 = axes[i, 2].imshow(error, aspect='auto', cmap='hot')
        axes[i, 2].set_title(f'Absolute Error (max: {error.max():.4f})')
        axes[i, 2].set_xlabel('Strike Index')
        fig.colorbar(im2, ax=axes[i, 2])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_dir}/{regime_name}_surface_comparison.png", dpi=150)
    plt.show()


def visualize_smile_comparison(theta_denorm, iv_true, iv_pred,
                              maturities, logK, regime_name, output_dir,
                              param_names=None):
    """Volatility smile comparison for each maturity."""
    
    def to_np(x):
        if isinstance(x, np.ndarray):
            return x
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    
    greek = {'eta': 'η', 'nu': 'ν', 'rho': 'ρ', 'kappa': 'κ', 
             'theta': 'θ', 'xi0': 'ξ₀', 'v0': 'v0', 'H': 'H'}
    
    def fmt_param_label(names, vals, per_line=3):
        parts = []
        for n, v in zip(names, vals):
            s = greek.get(n, n)
            parts.append(f"{s}={v:.2f}")
        if len(parts) <= per_line:
            return ", ".join(parts)
        return ", ".join(parts[:per_line]) + "\n" + ", ".join(parts[per_line:])
    
    theta_np = to_np(theta_denorm)
    ivp_np = to_np(iv_pred)
    ivt_np = None if iv_true is None else to_np(iv_true)
    maturities = to_np(maturities).reshape(-1)
    logK_vals = to_np(logK).reshape(-1)
    
    B, nT, nK = ivp_np.shape
    n_examples = int(min(3, B))
    if n_examples == 0:
        return
    
    if not param_names:
        param_names = [f"p{i+1}" for i in range(theta_np.shape[1])]
    
    fig, axes = plt.subplots(n_examples, nT, 
                             figsize=(5 * nT, 5 * n_examples), 
                             squeeze=False)
    fig.suptitle(f"Smile Comparison - {regime_name.upper()} Regime", fontsize=16)
    
    for i in range(n_examples):
        param_label = fmt_param_label(param_names, theta_np[i])
        for j, T in enumerate(maturities):
            ax = axes[i, j]
            iv_nn = ivp_np[i, j, :]
            
            if ivt_np is not None:
                iv_mc = ivt_np[i, j, :]
                ax.plot(logK_vals, iv_mc, 'b-o', linewidth=2, 
                       label='True (MC)', markersize=5)
            
            ax.plot(logK_vals, iv_nn, 'r--s', linewidth=2, 
                   label='Predicted (NN)', markersize=4)
            
            if i == 0:
                ax.set_title(f'T = {T:.3f} years', fontsize=12)
            if j == 0:
                ax.set_ylabel(f"Sample {i+1}\n{param_label}\n\nImplied Volatility", 
                             fontsize=10)
            
            ax.set_xlabel('Log-Moneyness (k)', fontsize=10)
            ax.grid(True, alpha=0.4)
            ax.legend(fontsize=9)
            
            if ivt_np is not None:
                mae = float(np.abs(iv_mc - iv_nn).mean())
                mae_txt = f"MAE: {mae:.4f}"
            else:
                mae_txt = "MAE: n/a"
            
            ax.text(0.03, 0.97, mae_txt, transform=ax.transAxes, va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                   fontsize=8)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{regime_name}_smile_comparison.png"), dpi=150)
    plt.show()


def analyze_dataset_quality(datasets, builder):
    """Analyzes the quality of the generated dataset"""
    print("\n" + "="*60)
    print("DATASET QUALITY ANALYSIS")
    print("="*60)

    for regime in ['short', 'mid', 'long']:
        theta = datasets[regime]['theta']
        iv = datasets[regime]['iv']

        # Denormalize for analysis
        theta_denorm = builder.denormalize_theta(theta)
        iv_denorm = builder.denormalize_iv_regime(iv, regime)

        print(f"\n{regime.upper()} TERM:")
        print(f"  Samples: {len(theta)}")
        print(f"  IV shape per sample: {iv.shape[1:]}")

        # IV statistics
        print(f"  IV range: [{iv_denorm.min():.4f}, {iv_denorm.max():.4f}]")
        print(f"  IV mean: {iv_denorm.mean():.4f} ± {iv_denorm.std():.4f}")

        # Check for anomalies
        anomalies = ((iv_denorm < 0.01) | (iv_denorm > 1.0)).sum()
        if anomalies > 0:
            print(f"  ! Anomalies: {anomalies} values outside [0.01, 1.0]")

        # Display distribution
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(iv_denorm.flatten().cpu(), bins=50, alpha=0.7)
        plt.xlabel('IV')
        plt.ylabel('Count')
        plt.title(f'{regime.upper()} - IV Distribution')

        plt.subplot(1, 2, 2)
        sample_iv = iv_denorm[0].cpu()
        plt.imshow(sample_iv, aspect='auto', cmap='viridis')
        plt.colorbar(label='IV')
        plt.xlabel('Strike')
        plt.ylabel('Maturity')
        plt.title(f'{regime.upper()} - Sample Surface')

        plt.tight_layout()
        plt.savefig(f"{CONFIG['output_dir']}/visualizations/{regime}_dataset_quality.png")
        plt.show()


# ===============================================================
# === TRAINING FUNCTIONS ===
# ===============================================================

def train_regime_direct(pricer, theta_train, iv_train, theta_val, iv_val,
                       epochs=30, batch_size=256, lr=1e-3, regime_name='',
                       output_dir=None, checkpoint_every=100):
    """Direct training of a single regime without additional normalization"""
    
    # Setup checkpoint directory
    if output_dir:
        checkpoint_dir = f"{output_dir}/models/checkpoints/{regime_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Flatten surfaces
    N_train = len(theta_train)
    iv_train_flat = iv_train.view(N_train, -1)

    train_loader = DataLoader(
        TensorDataset(theta_train, iv_train_flat),
        batch_size=batch_size, shuffle=True
    )

    if theta_val is not None and iv_val is not None:
        N_val = len(theta_val)
        iv_val_flat = iv_val.view(N_val, -1)
        val_loader = DataLoader(
            TensorDataset(theta_val, iv_val_flat),
            batch_size=batch_size, shuffle=False
        )
    else:
        val_loader = None

    optimizer = torch.optim.Adam(pricer.net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    best_weights = None
    best_epoch = 0
    train_losses = []
    val_losses = []

    # Check for existing checkpoint
    latest_checkpoint = None
    start_epoch = 0

    if output_dir and CONFIG.get("resume_from"):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        if checkpoints:
            # Get latest epoch
            epochs_found = []
            for cp in checkpoints:
                try:
                    epoch_num = int(cp.split('_epoch_')[1].split('.')[0])
                    epochs_found.append((epoch_num, cp))
                except:
                    continue

            if epochs_found:
                latest = max(epochs_found, key=lambda x: x[0])
                latest_checkpoint = os.path.join(checkpoint_dir, latest[1])

                # Load checkpoint
                checkpoint = torch.load(latest_checkpoint, map_location=pricer.device)
                pricer.net.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                best_epoch = checkpoint.get('best_epoch', 0)
                train_losses = checkpoint.get('train_losses', [])
                val_losses = checkpoint.get('val_losses', [])

                print(f"  Resuming {regime_name} from epoch {start_epoch}")

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=CONFIG["scheduler_factor"], 
        patience=CONFIG["scheduler_patience"], verbose=True
    )

    # Training loop
    patience_counter = 0
    
    for epoch in range(start_epoch, epochs + 1):
        # Training
        pricer.net.train()
        train_loss = 0.0
        for theta_batch, iv_batch in train_loader:
            pred = pricer.net(theta_batch)
            loss = loss_fn(pred, iv_batch)

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(pricer.net.parameters(), max_norm=1.0)
            
            optimizer.step()

            train_loss += loss.item() * theta_batch.size(0)

        train_loss /= N_train
        train_losses.append(train_loss)

        # Validation
        if val_loader is not None:
            pricer.net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for theta_val_batch, iv_val_batch in val_loader:
                    pred_val = pricer.net(theta_val_batch)
                    loss_val = loss_fn(pred_val, iv_val_batch)
                    val_loss += loss_val.item() * theta_val_batch.size(0)

            val_loss /= N_val
            val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = pricer.net.state_dict().copy()
                best_epoch = epoch
                patience_counter = 0

                # Save best model immediately
                if output_dir:
                    best_path = f"{output_dir}/models/{regime_name}_best_model.pt"
                    torch.save({
                        'model_state_dict': best_weights,
                        'epoch': best_epoch,
                        'val_loss': best_val_loss
                    }, best_path)
            else:
                patience_counter += 1

            if epoch % 5 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch:3d} | Train: {train_loss:.6f} | "
                      f"Val: {val_loss:.6f} | Best: {best_val_loss:.6f} @ {best_epoch} | "
                      f"LR: {current_lr:.1e}")
        else:
            if epoch % 5 == 0:
                print(f"  Epoch {epoch:3d} | Train Loss: {train_loss:.6f}")

        # Save checkpoint
        if output_dir and epoch % checkpoint_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': pricer.net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss if val_loader else None,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'regime': regime_name
            }

            checkpoint_path = f"{checkpoint_dir}/checkpoint_{regime_name}_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"    ✓ Checkpoint saved: epoch {epoch}")
        
        # Early stopping
        if patience_counter >= CONFIG["early_stopping_patience"]:
            print(f"    Early stopping triggered at epoch {epoch}")
            break

        # Clear cache
        if epoch % 50 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    if best_weights is not None:
        pricer.net.load_state_dict(best_weights)
        print(f"  ✓ {regime_name} - Best Val Loss: {best_val_loss:.6f} (epoch {best_epoch})")

    # Save training history
    if output_dir:
        history_path = f"{output_dir}/logs/{regime_name}_training_history.pkl"
        with open(history_path, 'wb') as f:
            pickle.dump({
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_epoch': best_epoch,
                'best_val_loss': best_val_loss
            }, f)

    return train_losses, val_losses


def train_multi_regime(multi_pricer, train_datasets, val_datasets, 
                      builder, val_builder):
    """Complete training of the multi-regime model"""
    
    print("\n" + "="*60)
    print("MULTI-REGIME TRAINING")
    print("="*60)
    
    # Set normalization statistics BEFORE training
    print("\nSetting normalization statistics...")
    
    for regime in ['short', 'mid', 'long']:
        pricer = getattr(multi_pricer, f"{regime}_term_pricer")
        pricer.set_normalization_stats(
            builder.theta_mean,
            builder.theta_std,
            builder.regime_stats[regime]['iv_mean'],
            builder.regime_stats[regime]['iv_std']
        )
    
    # Separate training for each regime
    training_results = {}
    
    for regime in ['short', 'mid', 'long']:
        print(f"\n{'='*50}")
        print(f"Training {regime.upper()} TERM regime...")
        print(f"{'='*50}")
        
        pricer = getattr(multi_pricer, f"{regime}_term_pricer")
        
        train_theta = train_datasets[regime]['theta']
        train_iv = train_datasets[regime]['iv']
        val_theta = val_datasets[regime]['theta'] if val_datasets else None
        val_iv = val_datasets[regime]['iv'] if val_datasets else None
        
        print(f"  Train shape: {train_iv.shape}")
        if val_theta is not None:
            print(f"  Val shape: {val_iv.shape}")
        
        # Training
        train_losses, val_losses = train_regime_direct(
            pricer,
            train_theta, train_iv,
            val_theta, val_iv,
            epochs=CONFIG['epochs_per_regime'],
            batch_size=CONFIG['batch_size'],
            lr=CONFIG['learning_rate'],
            regime_name=regime,
            output_dir=CONFIG['output_dir'],
            checkpoint_every=CONFIG['checkpoint_every']
        )
        
        training_results[regime] = {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    return training_results


# ===============================================================
# === SAVE & LOAD UTILITIES ===
# ===============================================================

def save_model(multi_pricer, output_dir, train_builder):
    """Salva il modello multi-regime"""
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_dir = os.path.join(output_dir, "models", "final")
    os.makedirs(final_dir, exist_ok=True)
    
    # Canonical process key
    process_key = ProcessFactory.key_for_instance(multi_pricer.process) or CONFIG["process"]
    
    # 1) Save weights of the three sub-models
    for regime in ['short', 'mid', 'long']:
        pricer = getattr(multi_pricer, f"{regime}_term_pricer")
        weights_path = os.path.join(final_dir, f"{process_key}_{regime}_weights_{ts}.pt")
        torch.save(pricer.net.state_dict(), weights_path)
        print(f"✓ {regime} network weights saved")
    
    # 2) Save normalization statistics
    stats_path = os.path.join(final_dir, f"{process_key}_norm_stats_{ts}.pt")
    try:
        torch.save({
            "theta_mean": train_builder.theta_mean.cpu(),
            "theta_std": train_builder.theta_std.cpu(),
            "regime_stats": train_builder.regime_stats,
        }, stats_path)
        print("✓ Normalization stats saved")
    except Exception as e:
        print(f"Warning: Could not save norm stats: {e}")
        stats_path = None
    
    # 3) Config to use in the loader
    cfg = {
        "process_key": process_key,
        "spot": getattr(multi_pricer.process, "spot", 1.0),
        "timestamp": ts,
        "weights": {
            "short": f"{process_key}_short_weights_{ts}.pt",
            "mid": f"{process_key}_mid_weights_{ts}.pt",
            "long": f"{process_key}_long_weights_{ts}.pt",
        },
        "thresholds": {
            "short_term_threshold": float(multi_pricer.short_term_threshold),
            "mid_term_threshold": float(multi_pricer.mid_term_threshold),
        },
        "grids": {
            "short": {
                "T": multi_pricer.short_term_pricer.Ts.cpu().tolist(),
                "k": multi_pricer.short_term_pricer.logKs.cpu().tolist(),
            },
            "mid": {
                "T": multi_pricer.mid_term_pricer.Ts.cpu().tolist(),
                "k": multi_pricer.mid_term_pricer.logKs.cpu().tolist(),
            },
            "long": {
                "T": multi_pricer.long_term_pricer.Ts.cpu().tolist(),
                "k": multi_pricer.long_term_pricer.logKs.cpu().tolist(),
            },
        },
        "norm_stats_path": os.path.basename(stats_path) if stats_path else None,
        "hidden": {
            "short": CONFIG["short_hidden"],
            "mid": CONFIG["mid_hidden"],
            "long": CONFIG["long_hidden"],
        },
    }
    
    # 4) Save config
    config_ts = os.path.join(final_dir, f"{process_key}_multi_regime_config_{ts}.json")
    config_latest = os.path.join(final_dir, f"{process_key}_multi_regime_config_latest.json")
    
    with open(config_ts, "w") as f:
        json.dump(cfg, f, indent=2)
    with open(config_latest, "w") as f:
        json.dump(cfg, f, indent=2)
    
    print(f"✓ Saved config: {config_ts}")
    print(f"✓ Saved latest: {config_latest}")
    
    return ts


# ===============================================================
# === MAIN FUNCTION ===
# ===============================================================

def run_complete_training():
    """Esegue il training completo del modello multi-regime"""
    
    print("\n" + "="*60)
    print(" MULTI-REGIME GRID PRICER TRAINING ")
    print("="*60)
    
    device = CONFIG["device"]
    
    # 1. Create the stochastic process
    print("\n" + "="*60)
    print("STEP 1: CREATING PROCESS")
    print("="*60)
    
    process = ProcessFactory.create(CONFIG["process"], spot=CONFIG["spot"])
    process_key = ProcessFactory.key_for_instance(process)
    print(f"✓ Created process: {process_key}")
    print(f"  Parameters: {process.param_info.names}")
    print(f"  Supports absorption: {process.supports_absorption}")
    
    # 2. Create the Multi-Regime Pricer
    print("\n" + "="*60)
    print("STEP 2: CREATING MULTI-REGIME PRICER")
    print("="*60)
    
    multi_pricer = MultiRegimeGridPricer(
        process=process,
        short_term_maturities=SHORT_MATURITIES.to(device),
        short_term_logK=SHORT_LOGK.to(device),
        mid_term_maturities=MID_MATURITIES.to(device),
        mid_term_logK=MID_LOGK.to(device),
        long_term_maturities=LONG_MATURITIES.to(device),
        long_term_logK=LONG_LOGK.to(device),
        short_term_threshold=CONFIG["short_term_threshold"],
        mid_term_threshold=CONFIG["mid_term_threshold"],
        short_term_hidden=CONFIG["short_hidden"],
        mid_term_hidden=CONFIG["mid_hidden"],
        long_term_hidden=CONFIG["long_hidden"],
        device=device
    )
    
    print("✓ Created MultiRegimeGridPricer")
    print(f"  Short term: T ≤ {CONFIG['short_term_threshold']}")
    print(f"  Mid term: {CONFIG['short_term_threshold']} < T ≤ {CONFIG['mid_term_threshold']}")
    print(f"  Long term: T > {CONFIG['mid_term_threshold']}")
    
    # 3. Generate training dataset
    print("\n" + "="*60)
    print("STEP 3: GENERATING TRAINING DATASET")
    print("="*60)
    
    train_builder = MultiRegimeDatasetBuilder(
        process=process,
        device=device,
        output_dir=CONFIG["output_dir"],
        dataset_type='train'
    )
    
    train_datasets = train_builder.build_multi_regime_dataset(
        multi_regime_pricer=multi_pricer,
        n_samples=CONFIG["train_samples"],
        n_paths=CONFIG["mc_paths"],
        batch_size=CONFIG["batch_size_generation"],
        normalize=True,
        compute_stats_from=None,
        sample_method='shared',
        force_regenerate=False,
        resume_from='latest' if CONFIG.get("resume_from") else None,
        checkpoint_every=1
    )
    
    print("✓ Training dataset generated")
    for regime in ['short', 'mid', 'long']:
        print(f"  {regime}: {len(train_datasets[regime]['theta'])} samples")
    
    # 4. Generate validation dataset
    print("\n" + "="*60)
    print("STEP 4: GENERATING VALIDATION DATASET")
    print("="*60)
    
    val_builder = MultiRegimeDatasetBuilder(
        process=process,
        device=device,
        output_dir=CONFIG["output_dir"],
        dataset_type='val'
    )
    
    val_datasets = val_builder.build_multi_regime_dataset(
        multi_regime_pricer=multi_pricer,
        n_samples=CONFIG["val_samples"],
        n_paths=CONFIG["mc_paths"],
        batch_size=CONFIG["batch_size_generation"],
        normalize=True,
        compute_stats_from=train_builder,  # USE TRAIN STATS!
        sample_method='shared',
        force_regenerate=False,
        checkpoint_every=1
    )
    
    print("✓ Validation dataset generated")
    for regime in ['short', 'mid', 'long']:
        print(f"  {regime}: {len(val_datasets[regime]['theta'])} samples")
    
    # 5. Analyze dataset quality
    print("\n" + "="*60)
    print("STEP 5: ANALYZING DATASET QUALITY")
    print("="*60)
    
    analyze_dataset_quality(train_datasets, train_builder)
    
    # 6. Training
    print("\n" + "="*60)
    print("STEP 6: TRAINING MULTI-REGIME MODEL")
    print("="*60)
    
    training_results = train_multi_regime(
        multi_pricer,
        train_datasets,
        val_datasets,
        train_builder,
        val_builder
    )
    
    # 7. Save the model
    print("\n" + "="*60)
    print("STEP 7: SAVING MODEL")
    print("="*60)
    
    timestamp = save_model(multi_pricer, CONFIG["output_dir"], train_builder)
    print(f"\n✓ Model saved with timestamp: {timestamp}")
    
    # 8. Test inference
    print("\n" + "="*60)
    print("STEP 8: TESTING INFERENCE")
    print("="*60)
    
    # Test with example parameters
    if CONFIG["process"] == "rough_heston":
        theta_values = {
            'H': 0.10,
            'nu': 0.35,
            'rho': -0.70,
            'kappa': 1.50,
            'theta_var': 0.04,
        }
    else:  # rough_bergomi
        theta_values = {
            'H': 0.10,
            'eta': 1.9,
            'rho': -0.70,
            'xi0': 0.09,
        }
    
    param_order = process.param_info.names
    test_theta = torch.tensor(
        [[theta_values.get(p, 0.1) for p in param_order]], 
        device=device, 
        dtype=torch.float32
    )
    
    print(f"Test theta (raw): {test_theta[0].cpu().numpy()}")
    
    # Price with denormalization
    with torch.no_grad():
        surfaces = multi_pricer.price_iv(test_theta, denormalize_output=True)
    
    for regime in ['short', 'mid', 'long']:
        surface = surfaces[regime]
        print(f"\n{regime.upper()} TERM:")
        print(f"  Shape: {surface.shape}")
        print(f"  Range: [{surface.min():.4f}, {surface.max():.4f}]")
        print(f"  Mean: {surface.mean():.4f} ± {surface.std():.4f}")
    
    # Test interpolation
    print("\nInterpolation test:")
    test_points = [
        (0.05, -0.1),   # Very short term
        (0.5, 0.0),     # Mid term ATM
        (3.0, 0.4),     # Long term OTM
    ]
    
    for T, k in test_points:
        iv = multi_pricer.interpolate_iv(T, k, test_theta[0])
        regime = multi_pricer._get_regime(T)
        print(f"  T={T:.2f}, k={k:.2f} ({regime}) -> IV={iv:.4f}")
    
    # 9. Visualize results on validation set
    print("\n" + "="*60)
    print("STEP 9: VISUALIZING RESULTS")
    print("="*60)
    
    visualize_results(multi_pricer, val_datasets, val_builder)
    
    # 10. Plot training curves
    print("\n" + "="*60)
    print("STEP 10: PLOTTING TRAINING CURVES")
    print("="*60)
    
    plot_all_training_curves(training_results)
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\nFinal Results:")
    print(f"  Process: {process_key}")
    print(f"  Training samples: {sum(CONFIG['train_samples'].values())} total")
    print(f"  Validation samples: {sum(CONFIG['val_samples'].values())} total")
    print(f"  Model saved to: {CONFIG['output_dir']}/models/final/")
    print(f"  Timestamp: {timestamp}")
    
    return multi_pricer, train_builder, val_builder


def visualize_results(multi_pricer, val_datasets, val_builder):
    """Visualizza i risultati sul validation set"""
    
    output_dir = f"{CONFIG['output_dir']}/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Take a small sample for visualization
    n_samples_to_show = 3
    
    for regime in ['short', 'mid', 'long']:
        print(f"\n--- Visualizing {regime.upper()} regime ---")
        pricer = getattr(multi_pricer, f"{regime}_term_pricer")
        
        # Data for the current regime
        theta_val_norm = val_datasets[regime]['theta']
        iv_val_norm = val_datasets[regime]['iv']
        
        # Select samples to visualize
        theta_sample = theta_val_norm[:n_samples_to_show]
        iv_true_sample = iv_val_norm[:n_samples_to_show]
        
        if len(theta_sample) == 0:
            print("No validation samples to visualize.")
            continue
        
        # Run prediction with the model
        pricer.net.eval()
        with torch.no_grad():
            iv_pred_sample_norm = pricer.net(theta_sample).view_as(iv_true_sample)
        
        # Denormalize all data for interpretable visualization
        theta_denorm = val_builder.denormalize_theta(theta_sample)
        iv_true_denorm = val_builder.denormalize_iv_regime(iv_true_sample, regime)
        iv_pred_denorm = val_builder.denormalize_iv_regime(iv_pred_sample_norm, regime)
        
        # Call plotting functions
        visualize_surface_comparison(iv_true_denorm, iv_pred_denorm, regime, output_dir)
        visualize_smile_comparison(
            theta_denorm,
            iv_true_denorm,
            iv_pred_denorm,
            pricer.Ts,
            pricer.logKs,
            regime,
            output_dir,
            param_names=multi_pricer.process.param_info.names
        )


def plot_all_training_curves(training_results):
    """Displays training curves for all regimes"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training Curves - All Regimes", fontsize=16)
    
    for idx, regime in enumerate(['short', 'mid', 'long']):
        ax = axes[idx]
        
        train_losses = training_results[regime]['train_losses']
        val_losses = training_results[regime]['val_losses']
        
        ax.plot(train_losses, label='Train Loss', linewidth=2)
        if val_losses:
            ax.plot(val_losses, label='Val Loss', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_title(f'{regime.upper()} Term')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{CONFIG['output_dir']}/visualizations/training_curves_all.png", dpi=150)
    plt.show()


# ===============================================================
# === MAIN EXECUTION ===
# ===============================================================

if __name__ == "__main__":
    # Check GPU availability
    if torch.cuda.is_available():
        print("GPU trovata! Si utilizzerà CUDA.")
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU non trovata. Si utilizzerà la CPU (potrebbe essere lento).")
    
    # Run complete training
    multi_pricer, train_builder, val_builder = run_complete_training()
    
    print("\n✓ COMPLETE EXECUTION FINISHED!")
    print(f"All results saved to: {CONFIG['output_dir']}")