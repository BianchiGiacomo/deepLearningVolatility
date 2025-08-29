#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correct script to load and test the Multi-Regime Grid Pricer
Compatible with the new JSON format that uses 'grids' instead of separate fields
"""

import sys
from pathlib import Path
# Add the project root to PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import glob
import os
from datetime import datetime
import time

from deepLearningVolatility.nn.pricer import MultiRegimeGridPricer
from deepLearningVolatility.stochastic.stochastic_interface import ProcessFactory

import deepLearningVolatility.stochastic.wrappers.rough_bergomi_wrapper
import deepLearningVolatility.stochastic.wrappers.rough_heston_wrapper


# ===============================================================
# === CONFIGURATION - MODIFY THESE PARAMETERS ===
# ===============================================================

# Path to the models directory (models/final)
MODELS_DIR = "C:/Projects/NN/deepLearningVolatility/models/Multiregime/RoughHeston"

# Process type ('rough_bergomi' or 'rough_heston')
PROCESS_TYPE = 'rough_heston'

# Device to use
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Directory where to save the plots
OUTPUT_DIR = "C:/Projects/NN/deepLearningVolatility/models/test_results"

# Test cases for Rough Heston: [H, nu, rho, kappa, theta_var]
CUSTOM_TEST_CASES = [
    {
        'name': 'Case 1',
        'theta': [0.35, 0.87, -0.1, 1.9, 0.09]
    },
    {
        'name': 'Case 2',
        'theta': [0.10, 0.94, -0.12, 0.26, 0.03]
    },
    {
        'name': 'Case 3',
        'theta': [0.44, 0.65, -0.77, 0.99, 0.02]
    },
    {
        'name': 'Case 4',
        'theta': [0.3, 0.45, -0.67, 0.91, 0.09]
    },
]


# ===============================================================
# === CORRECT LOADING FUNCTIONS ===
# ===============================================================

def load_multiregime_model_fixed(models_dir, config_filename=None, device='cpu'):
    """
    Fixed version that handles the new JSON format with 'grids'

    Args:
    models_dir: Directory containing the model files
    config_filename: Specific name of the config file (optional)
    device: Compute device

    Returns:
    Tuple (MultiRegimeGridPricer, config_dict)
    """
    print("\n" + "="*60)
    print("LOADING MULTI-REGIME MODEL")
    print("="*60)
    
    # 1. Find the configuration file
    if config_filename:
        config_path = os.path.join(models_dir, config_filename)
    else:
        config_pattern = f"{models_dir}/*_multi_regime_config_*.json"
        config_files = sorted(glob.glob(config_pattern))
        
        if not config_files:
            raise FileNotFoundError(f"No config file found in {models_dir}")
        
        config_path = config_files[-1] 
    
    print(f"Loading config from: {os.path.basename(config_path)}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 2. Determine the process
    process_key = config.get('process_key', 'rough_heston')
    spot = config.get('spot', 1.0)
    
    process = ProcessFactory.create(process_key, spot=spot)
    print(f"✓ Created process: {process.__class__.__name__}")
    print(f"  Parameters: {process.param_info.names}")
    
    # 3. Extract the grids from the new format
    grids = config.get('grids')
    if not grids:
        raise KeyError("Config file missing 'grids' field")
    
    # Extract maturities and strikes for each regime
    short_maturities = torch.tensor(grids['short']['T'], device=device)
    short_logK = torch.tensor(grids['short']['k'], device=device)
    
    mid_maturities = torch.tensor(grids['mid']['T'], device=device)
    mid_logK = torch.tensor(grids['mid']['k'], device=device)
    
    long_maturities = torch.tensor(grids['long']['T'], device=device)
    long_logK = torch.tensor(grids['long']['k'], device=device)
    
    print(f"✓ Loaded grids:")
    print(f"  Short: {len(short_maturities)} maturities x {len(short_logK)} strikes")
    print(f"  Mid: {len(mid_maturities)} maturities x {len(mid_logK)} strikes")
    print(f"  Long: {len(long_maturities)} maturities x {len(long_logK)} strikes")
    
    # 4. Extract thresholds
    thresholds = config.get('thresholds', {})
    short_threshold = thresholds.get('short_term_threshold', 0.25)
    mid_threshold = thresholds.get('mid_term_threshold', 1.0)
    
    # 5. Extract hidden layers architectures
    hidden = config.get('hidden', {})
    short_hidden = hidden.get('short', [128, 64])
    mid_hidden = hidden.get('mid', [128, 64])
    long_hidden = hidden.get('long', [128, 64])
    
    # 5.5 Extract dt configuration from metadata if available
    dt_config = config.get('regime_dt_config', {})
    short_term_dt = dt_config.get('short', 3e-5)
    mid_term_dt = dt_config.get('mid', 1e-4)
    long_term_dt = dt_config.get('long', 1/365)
    
    print(f"✓ DT configuration:")
    print(f"  Short: {short_term_dt:.2e}")
    print(f"  Mid: {mid_term_dt:.2e}")
    print(f"  Long: {long_term_dt:.6f}")
    
    # 6. Create the MultiRegimeGridPricer
    multi_pricer = MultiRegimeGridPricer(
        process=process,
        short_term_maturities=short_maturities,
        short_term_logK=short_logK,
        mid_term_maturities=mid_maturities,
        mid_term_logK=mid_logK,
        long_term_maturities=long_maturities,
        long_term_logK=long_logK,
        short_term_threshold=short_threshold,
        mid_term_threshold=mid_threshold,
        short_term_hidden=short_hidden,
        mid_term_hidden=mid_hidden,
        long_term_hidden=long_hidden,
        short_term_dt=short_term_dt,
        mid_term_dt=mid_term_dt,
        long_term_dt=long_term_dt,
        device=device
    )
    
    print("✓ Created MultiRegimeGridPricer")
    
    # 7. Load the network weights
    weights = config.get('weights', {})
    
    for regime in ['short', 'mid', 'long']:
        weights_filename = weights.get(regime)
        if not weights_filename:
            raise ValueError(f"No weights filename for {regime} regime")
        
        weights_path = os.path.join(models_dir, weights_filename)
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        # Select the correct pricer
        pricer = getattr(multi_pricer, f"{regime}_term_pricer")
        
        # Load the weights
        state_dict = torch.load(weights_path, map_location=device)
        pricer.net.load_state_dict(state_dict)
        
        print(f"✓ Loaded {regime} weights from: {weights_filename}")
    
    # 8. Load normalization statistics
    norm_stats_filename = config.get('norm_stats_filename')
    
    if norm_stats_filename:
        norm_stats_path = os.path.join(models_dir, norm_stats_filename)
        
        if os.path.exists(norm_stats_path):
            stats = torch.load(norm_stats_path, map_location=device)
            
            # Extract theta statistics
            theta_mean = stats['theta_mean'].to(device)
            theta_std = stats['theta_std'].to(device)
            
            # Extract statistics per regime
            regime_stats = stats.get('regime_stats', {})
            
            # Apply the statistics to each pricer
            for regime in ['short', 'mid', 'long']:
                pricer = getattr(multi_pricer, f"{regime}_term_pricer")
                
                if regime in regime_stats:
                    iv_mean = torch.tensor(regime_stats[regime]['iv_mean'], device=device)
                    iv_std = torch.tensor(regime_stats[regime]['iv_std'], device=device)
                else:
                    # Fallback if missing
                    print(f"  Warning: No stats for {regime}, using defaults")
                    iv_mean = torch.tensor(0.2, device=device)
                    iv_std = torch.tensor(0.1, device=device)
                
                pricer.set_normalization_stats(
                    theta_mean=theta_mean,
                    theta_std=theta_std,
                    iv_mean=iv_mean,
                    iv_std=iv_std
                )
            
            print(f"✓ Loaded normalization stats from: {norm_stats_filename}")
        else:
            print(f"! Warning: Normalization stats file not found: {norm_stats_path}")
    else:
        print("! Warning: No normalization stats filename in config")
    
    # 9. Set the model in evaluation mode
    multi_pricer.eval()
    
    print("✓ Model loaded successfully and ready for inference")
    
    return multi_pricer, config


# ===============================================================
# === MONTE CARLO FUNCTIONS ===
# ===============================================================

def compute_mc_surfaces(multi_pricer, theta, n_paths=50000, n_batches=10, confidence_level=0.95):
    """
    Computes the theoretical IV surfaces via Monte Carlo with confidence intervals
    """
    print("\n" + "="*60)
    print("COMPUTING MONTE CARLO SURFACES WITH CONFIDENCE INTERVALS")
    print("="*60)
    
    from scipy import stats as scipy_stats
    
    device = multi_pricer.device
    theta_tensor = torch.as_tensor(theta, device=device, dtype=torch.float32)
    
    if theta_tensor.dim() == 1:
        theta_tensor = theta_tensor.unsqueeze(0)
    
    results = {}
    
    regimes = {
        'short': multi_pricer.short_term_pricer,
        'mid': multi_pricer.mid_term_pricer,
        'long': multi_pricer.long_term_pricer,
    }
    
    batch_size = n_paths // n_batches
    z_score = scipy_stats.norm.ppf((1 + confidence_level) / 2)
    
    for name, pricer in regimes.items():
        print(f"\nComputing {name} regime with CI...")
        
        # Use the dt configured for this regime
        regime_dt = multi_pricer.regime_dt_config.get(name, 1/365)
        print(f"  Using dt={regime_dt:.2e} for {name} regime")
        
        # Temporarily set the fixed_regime_dt
        old_dt = pricer.fixed_regime_dt
        pricer.fixed_regime_dt = regime_dt
        
        batch_results = []
        
        # Run multiple batches for CI calculation
        for batch_idx in range(n_batches):
            try:
                iv_grid = pricer._mc_iv_grid(
                    theta=theta_tensor[0],
                    n_paths=batch_size,
                    spot=1.0,
                    use_antithetic=True,
                    adaptive_dt=False,
                    control_variate=True
                )
                
                # Handle invalid values
                iv_grid = torch.where(torch.isnan(iv_grid) | torch.isinf(iv_grid), 
                                     torch.tensor(0.2, device=device), iv_grid)
                iv_grid = torch.clamp(iv_grid, 0.01, 2.0)
                
                batch_results.append(iv_grid.unsqueeze(0))
                
            except Exception as e:
                print(f"  Error in batch {batch_idx}: {e}")
                continue
            
        # Restore the original dt
        pricer.fixed_regime_dt = old_dt
        
        if batch_results:
            # Stack all batches
            all_batches = torch.cat(batch_results, dim=0)  # [n_batches, nT, nK]
            
            # Calculate statistics
            mean_iv = all_batches.mean(dim=0, keepdim=True)  # [1, nT, nK]
            std_iv = all_batches.std(dim=0, keepdim=True)
            
            # Calculate confidence intervals
            ci_margin = z_score * std_iv / np.sqrt(n_batches)
            ci_lower = mean_iv - ci_margin
            ci_upper = mean_iv + ci_margin
            
            results[name] = {
                'mean': mean_iv,
                'std': std_iv,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
            
            print(f"  ✓ Computed: mean={mean_iv.mean():.4f}, "
                  f"CI width={(2*ci_margin).mean():.4f}")
        else:
            # Fallback
            nT = len(pricer.Ts)
            nK = len(pricer.logKs)
            results[name] = {
                'mean': torch.full((1, nT, nK), 0.2, device=device),
                'std': torch.full((1, nT, nK), 0.01, device=device),
                'ci_lower': torch.full((1, nT, nK), 0.18, device=device),
                'ci_upper': torch.full((1, nT, nK), 0.22, device=device)
            }
    
    return results


def visualize_smile_comparison(theta_denorm, iv_true, iv_pred, maturities, logK, 
                               regime_name, output_dir, param_names=None):
    """
    Displays smile comparison (IV vs logK) for each maturity with confidence intervals
    """
    import numpy as np
    
    # Convert everything to numpy
    def to_np(x):
        if isinstance(x, np.ndarray):
            return x
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    
    theta_np = to_np(theta_denorm)
    ivp_np = to_np(iv_pred)
    
    # Handle iv_true which can be None, a tensor, or a dictionary with CI
    if iv_true is not None:
        if isinstance(iv_true, dict):
            # New format with CI - convert each component
            ivt_dict = {
                'mean': to_np(iv_true['mean']),
                'std': to_np(iv_true['std']),
                'ci_lower': to_np(iv_true['ci_lower']),
                'ci_upper': to_np(iv_true['ci_upper'])
            }
        else:
            # Old format without CI
            ivt_dict = None
            ivt_np = to_np(iv_true)
    else:
        ivt_dict = None
        ivt_np = None
    
    maturities = to_np(maturities).reshape(-1)
    logK_vals = to_np(logK).reshape(-1)
    
    if theta_np.ndim == 1:
        theta_np = theta_np.reshape(1, -1)
    
    n_examples = 1
    n_maturities = len(maturities)
    
    # Limit the number of maturities to display
    max_maturities = 5
    if n_maturities > max_maturities:
        indices = np.linspace(0, n_maturities-1, max_maturities, dtype=int)
        maturities_to_plot = maturities[indices]
        plot_indices = indices
    else:
        maturities_to_plot = maturities
        plot_indices = range(n_maturities)
    
    fig, axes = plt.subplots(1, len(maturities_to_plot), 
                             figsize=(5 * len(maturities_to_plot), 5), 
                             squeeze=False)
    
    # Prepare parameter labels
    if param_names:
        param_str = ", ".join([f"{name}={val:.3f}" 
                              for name, val in zip(param_names, theta_np[0])])
    else:
        param_str = ", ".join([f"p{i+1}={val:.3f}" 
                              for i, val in enumerate(theta_np[0])])
    
    fig.suptitle(f"Smile Comparison - {regime_name.upper()} Regime\nθ = ({param_str})", 
                 fontsize=14)
    
    for plot_idx, (mat_idx, T) in enumerate(zip(plot_indices, maturities_to_plot)):
        ax = axes[0, plot_idx]
        
        # Extract NN prediction for this maturity
        iv_nn = ivp_np[0, mat_idx, :]
        
        # Plot NN prediction
        ax.plot(logK_vals, iv_nn, 'r--s', linewidth=2, 
                label='Predicted (NN)', markersize=4)
        
        # Plot MC if available
        if ivt_dict is not None:
            iv_mc_mean = ivt_dict['mean'][0, mat_idx, :]
            iv_mc_lower = ivt_dict['ci_lower'][0, mat_idx, :]
            iv_mc_upper = ivt_dict['ci_upper'][0, mat_idx, :]
            
            # Plot mean and CI
            ax.plot(logK_vals, iv_mc_mean, 'b-o', linewidth=2, 
                    label='True (MC)', markersize=5)
            ax.fill_between(logK_vals, iv_mc_lower, iv_mc_upper, 
                            alpha=0.3, color='blue', label='95% CI')
            
            # Check if NN is within CI
            within_ci = ((iv_nn >= iv_mc_lower) & (iv_nn <= iv_mc_upper)).mean()
            mae = np.abs(iv_mc_mean - iv_nn).mean()
            mae_txt = f'MAE: {mae:.4f}\nWithin CI: {within_ci:.1%}'
            
        elif ivt_np is not None:
            iv_mc = ivt_np[0, mat_idx, :]
            ax.plot(logK_vals, iv_mc, 'b-o', linewidth=2, 
                    label='True (MC)', markersize=5)
            mae = np.abs(iv_mc - iv_nn).mean()
            mae_txt = f'MAE: {mae:.4f}'
        else:
            mae_txt = 'MC not computed'
        
        # Formatting
        ax.set_title(f'T = {T:.3f} years', fontsize=12)
        ax.set_xlabel('Log-Moneyness (k)', fontsize=10)
        if plot_idx == 0:
            ax.set_ylabel('Implied Volatility', fontsize=10)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=9)
        
        # MAE annotation
        ax.text(0.05, 0.95, mae_txt, transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7), 
                fontsize=8)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/{regime_name}_smile_comparison.png"
        plt.savefig(filename, dpi=150)
        print(f"  Saved: {filename}")
    
    plt.show()


# ===============================================================
# === TEST FUNCTIONS ===
# ===============================================================

def test_model_with_mc_comparison(multi_pricer, test_cases, compute_mc=True, mc_paths=50000):
    """
    Tests the model on specified parameter sets, comparing NN predictions with MC surfaces
    """
    print("\n" + "="*60)
    print("TESTING MODEL WITH MC COMPARISON")
    print("="*60)
    
    device = multi_pricer.device
    
    for i, case in enumerate(test_cases, 1):
        name = case['name']
        theta_raw = case['theta']
        
        print(f"\n[{i}/{len(test_cases)}] {name}")
        print(f"  Parameters: {theta_raw}")
        
        # Convert to tensor
        theta = torch.tensor([theta_raw], device=device, dtype=torch.float32)
        
        # Calculate NN surfaces
        with torch.no_grad():
            nn_surfaces = multi_pricer.price_iv(theta, denormalize_output=True)
        
        # Compute MC surfaces if requested
        mc_surfaces = None
        if compute_mc:
            mc_surfaces = compute_mc_surfaces(multi_pricer, theta_raw, n_paths=mc_paths)
        
        # Create output directory for this test
        if OUTPUT_DIR:
            test_dir = os.path.join(OUTPUT_DIR, f"test_{i:02d}_{name.replace(' ', '_')}")
            os.makedirs(test_dir, exist_ok=True)
        else:
            test_dir = None
        
        # Display smile comparison for each regime
        regimes = {
            'short': multi_pricer.short_term_pricer,
            'mid': multi_pricer.mid_term_pricer,
            'long': multi_pricer.long_term_pricer,
        }
        
        param_names = multi_pricer.process.param_info.names if hasattr(multi_pricer.process, 'param_info') else None
        
        for regime_name, pricer in regimes.items():
            maturities = pricer.Ts
            logK = pricer.logKs
            
            # Predicted and MC surfaces
            iv_pred = nn_surfaces[regime_name]
            iv_true = mc_surfaces[regime_name] if mc_surfaces else None
            
            # Display smile comparison
            visualize_smile_comparison(
                theta_denorm=theta,
                iv_true=iv_true,
                iv_pred=iv_pred,
                maturities=maturities,
                logK=logK,
                regime_name=regime_name,
                output_dir=test_dir,
                param_names=param_names
            )
        
        # Report statistics
        print(f"\n  Surface Statistics:")
        for regime in ['short', 'mid', 'long']:
            nn_surf = nn_surfaces[regime]
            print(f"    {regime} NN: mean={nn_surf.mean():.4f}, std={nn_surf.std():.4f}, "
                  f"range=[{nn_surf.min():.4f}, {nn_surf.max():.4f}]")
            
            if mc_surfaces:
                mc_data = mc_surfaces[regime]
                mc_mean_surf = mc_data['mean']
                mc_ci_lower = mc_data['ci_lower']
                mc_ci_upper = mc_data['ci_upper']
                
                # Calculate MAE between NN and MC mean
                mae = (nn_surf - mc_mean_surf).abs().mean().item()
                
                # Calculate percentage of NN points within CI
                within_ci = ((nn_surf >= mc_ci_lower) & (nn_surf <= mc_ci_upper)).float().mean().item()
                
                # Calculate average CI width
                ci_width = (mc_ci_upper - mc_ci_lower).mean().item()
                
                print(f"    {regime} MC: mean={mc_mean_surf.mean():.4f}, "
                      f"MAE vs NN={mae:.5f}, "
                      f"CI width={ci_width:.4f}")
                print(f"    {regime} Coverage: {within_ci:.1%} of NN predictions within 95% CI")


def visualize_surface_heatmap_comparison(multi_pricer, theta, compute_mc=True, mc_paths=50000):
    """
    Displays surface comparison as heatmap (NN vs MC vs Error)
    """
    device = multi_pricer.device
    theta_tensor = torch.tensor([theta], device=device, dtype=torch.float32)
    
    # Compute surfaces
    with torch.no_grad():
        nn_surfaces = multi_pricer.price_iv(theta_tensor, denormalize_output=True)
    
    mc_surfaces = None
    if compute_mc:
        mc_surfaces = compute_mc_surfaces(multi_pricer, theta, n_paths=mc_paths)
    
    # Create figure for each regime
    regimes = ['short', 'mid', 'long']
    
    for regime in regimes:
        n_cols = 4 if (compute_mc and mc_surfaces) else 1
        fig, axes = plt.subplots(1, n_cols, figsize=(6*n_cols, 5))
        
        if not compute_mc:
            axes = [axes]
        elif not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # NN Surface
        nn_surf = nn_surfaces[regime][0].cpu().numpy()
        
        im0 = axes[0].imshow(nn_surf, aspect='auto', cmap='viridis', origin='lower')
        axes[0].set_title(f'{regime.upper()} - NN Predicted Surface')
        axes[0].set_xlabel('Strike Index')
        axes[0].set_ylabel('Maturity Index')
        plt.colorbar(im0, ax=axes[0])
        
        if compute_mc and mc_surfaces:
            # MC Surface (mean)
            mc_surf_mean = mc_surfaces[regime]['mean'][0].cpu().numpy()
            
            im1 = axes[1].imshow(mc_surf_mean, aspect='auto', cmap='viridis', origin='lower')
            axes[1].set_title(f'{regime.upper()} - MC Mean Surface')
            axes[1].set_xlabel('Strike Index')
            plt.colorbar(im1, ax=axes[1])
            
            # CI Width Surface
            ci_lower = mc_surfaces[regime]['ci_lower'][0].cpu().numpy()
            ci_upper = mc_surfaces[regime]['ci_upper'][0].cpu().numpy()
            ci_width = ci_upper - ci_lower
            
            im2 = axes[2].imshow(ci_width, aspect='auto', cmap='plasma', origin='lower')
            axes[2].set_title(f'{regime.upper()} - CI Width\n(mean={ci_width.mean():.4f})')
            axes[2].set_xlabel('Strike Index')
            plt.colorbar(im2, ax=axes[2])
            
            # Error Surface
            error = np.abs(nn_surf - mc_surf_mean)
            
            # Calculate if NN is within CI
            within_ci = ((nn_surf >= ci_lower) & (nn_surf <= ci_upper))
            pct_within = within_ci.mean() * 100
            
            im3 = axes[3].imshow(error, aspect='auto', cmap='hot', origin='lower')
            axes[3].set_title(f'{regime.upper()} - Absolute Error\n(max={error.max():.4f}, mean={error.mean():.4f})\n{pct_within:.1f}% within CI')
            axes[3].set_xlabel('Strike Index')
            plt.colorbar(im3, ax=axes[3])
        
        # Add parameters as title
        param_str = ", ".join([f"{p:.3f}" for p in theta])
        fig.suptitle(f'Surface Comparison - θ = ({param_str})', fontsize=14)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        
        if OUTPUT_DIR:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            filename = f"{OUTPUT_DIR}/{regime}_surface_heatmap_with_ci.png"
            plt.savefig(filename, dpi=150)
            print(f"  Saved: {filename}")
        
        plt.show()


def visualize_surfaces(multi_pricer, theta, title="", output_dir=None):
    """
    Displays the surfaces for all regimes
    """
    device = multi_pricer.device
    theta_tensor = torch.tensor([theta], device=device, dtype=torch.float32)
    
    with torch.no_grad():
        surfaces = multi_pricer.price_iv(theta_tensor, denormalize_output=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    param_str = ", ".join([f"{p:.3f}" for p in theta])
    fig.suptitle(f"{title}\nθ = ({param_str})", fontsize=14)
    
    regime_info = [
        ('short', multi_pricer.short_term_pricer, 'Short-Term'),
        ('mid', multi_pricer.mid_term_pricer, 'Mid-Term'),
        ('long', multi_pricer.long_term_pricer, 'Long-Term')
    ]
    
    for idx, (regime, pricer, label) in enumerate(regime_info):
        ax = axes[idx]
        
        # Extract surface
        iv_surface = surfaces[regime][0].cpu().numpy()
        
        # Plot heatmap
        im = ax.imshow(iv_surface, aspect='auto', cmap='viridis', origin='lower')
        ax.set_title(f'{label} Regime')
        ax.set_xlabel('Strike Index')
        ax.set_ylabel('Maturity Index')
        plt.colorbar(im, ax=ax)
        
        # Add statistics
        stats_text = (f"Mean: {iv_surface.mean():.3f}\n"
                     f"Std: {iv_surface.std():.3f}\n"
                     f"Range: [{iv_surface.min():.3f}, {iv_surface.max():.3f}]")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/surfaces_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=150)
        print(f"  Saved plot to: {filename}")
    
    plt.show()


# ===============================================================
# === MAIN FUNCTION ===
# ===============================================================

def main():
    """
    Main function with smile visualization and MC comparison
    """
    print("\n" + "="*60)
    print(" MULTI-REGIME GRID PRICER TEST (WITH MC COMPARISON) ")
    print("="*60)
    
    # Setup
    device = torch.device(DEVICE)
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Models directory: {MODELS_DIR}")
    print(f"  Process type: {PROCESS_TYPE}")
    print(f"  Compute MC: True")
    print(f"  MC paths: 50000")
    print(f"  Output directory: {OUTPUT_DIR if OUTPUT_DIR else 'None'}")
    
    # Crea output directory se necessario
    if OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create output directory if needed
    try:
        config_filename = "rough_heston_multi_regime_config_20250826_090959.json"
        
        multi_pricer, config = load_multiregime_model_fixed(
            MODELS_DIR,
            config_filename=config_filename,
            device=device
        )
    except Exception as e:
        print(f"\nERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nTesting with MC comparison and smile visualization...")
    test_model_with_mc_comparison(
        multi_pricer, 
        CUSTOM_TEST_CASES[:4],
        compute_mc=True,
        mc_paths=50000
    )
    
    # print("\n" + "="*60)
    # print("SURFACE HEATMAP COMPARISON")
    # print("="*60)
    
    # visualize_surface_heatmap_comparison(
    #     multi_pricer,
    #     CUSTOM_TEST_CASES[0]['theta'],
    #     compute_mc=True,
    #     mc_paths=50000
    # )
    
    print("\n" + "="*60)
    print("INTERPOLATION TEST")
    print("="*60)
    
    test_theta = CUSTOM_TEST_CASES[0]['theta']
    theta_tensor = torch.tensor([test_theta], device=device, dtype=torch.float32)
    
    test_points = [
        (0.02, -0.05),   # Very short term
        (0.1, 0.0),      # Short term ATM
        (0.5, 0.0),      # Mid term ATM
        (2.0, 0.3),      # Long term OTM
        (5.0, -0.4),     # Very long term ITM
    ]
    
    print(f"\nUsing theta: {test_theta}")
    print("\nInterpolated values:")
    for T, k in test_points:
        iv = multi_pricer.interpolate_iv(T, k, theta_tensor[0])
        regime = multi_pricer._get_regime(T)
        print(f"  T={T:.3f}, k={k:+.2f} ({regime:5s}) → IV={iv:.4f}")
    
    print("\n" + "="*60)
    print("TESTING COMPLETED SUCCESSFULLY")
    print("="*60)
    
    if OUTPUT_DIR:
        print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()