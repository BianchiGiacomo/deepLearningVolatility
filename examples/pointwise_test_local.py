#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified script to test PointwiseNetworkPricer with NN vs MC smile comparison
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

from deepLearningVolatility.nn.pricer.pricer import PointwiseNetworkPricer, GridNetworkPricer
from deepLearningVolatility.stochastic.stochastic_interface import ProcessFactory

import deepLearningVolatility.stochastic.wrappers.rough_bergomi_wrapper
import deepLearningVolatility.stochastic.wrappers.rough_heston_wrapper

# ===============================================================
# CONFIGURATION
# ===============================================================

MODEL_PATH = "C:/Projects/NN/deepLearningVolatility/models/Pointwise/RoughBergomi/best_model.pt"
CONFIG_PATH = "C:/Projects/NN/deepLearningVolatility/models/Pointwise/RoughBergomi/roughbergomiprocess_pointwise_random_grids_latest_config.json"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = "C:/Projects/NN/deepLearningVolatility/models/Pointwise/RoughBergomi/test_results_smile"

# Test cases for Rough Bergomi: [H, eta, rho, xi0]
TEST_CASES = [
    {'name': 'Case 1', 'theta': [0.15, 1.0, -0.2, 0.11]},
    {'name': 'Case 2', 'theta': [0.25, 2.0, -0.8, 0.15]},
    {'name': 'Case 3', 'theta': [0.3, 1.5, -0.5, 0.08]},
]

# Test grid configuration
TEST_MATURITIES = [0.1, 0.25, 0.5, 1.0]  # Maturities to display
N_STRIKES = 31  # Number of strikes

# ===============================================================
# LOADING FUNCTIONS
# ===============================================================

def load_pointwise_model(model_path, config_path=None, device='cpu'):
    """Load the PointwiseNetworkPricer model"""
    print("\nLoading Pointwise Network Model...")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load config
    config = None
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Determine process
    if 'process_key' in checkpoint:
        process_key = checkpoint['process_key']
    elif config and 'process_key' in config:
        process_key = config['process_key']
    else:
        process_key = 'rough_bergomi'  # Default
    
    process = ProcessFactory.create(process_key)
    print(f"  Process: {process.__class__.__name__}")
    
    # Extract architecture
    if 'network_config' in checkpoint:
        hidden_layers = checkpoint['network_config']['hidden_layers']
        activation = checkpoint['network_config']['activation']
    elif config and 'training_config' in config:
        hidden_layers = config['training_config']['hidden_layers']
        activation = config['training_config']['activation']
    else:
        hidden_layers = [30, 30, 30, 30]
        activation = 'ELU'
    
    # Create pricer
    pricer = PointwiseNetworkPricer(
        process=process,
        hidden_layers=hidden_layers,
        activation=activation,
        device=device,
        enable_smile_repair=False
    )
    
    # Load weights
    model_state = checkpoint.get('model_state_dict', checkpoint)
    pricer.net.load_state_dict(model_state)
    
    # Load normalization stats
    if 'normalization_stats' in checkpoint:
        norm_stats = checkpoint['normalization_stats']
        theta_mean = norm_stats['theta_mean'].to(device)
        theta_std = norm_stats['theta_std'].to(device)
        iv_mean = norm_stats['iv_mean'].to(device)
        iv_std = norm_stats['iv_std'].to(device)
        pricer.set_normalization_stats(theta_mean, theta_std, iv_mean, iv_std)
        
        if 'T_mean' in norm_stats:
            T_mean = norm_stats['T_mean'].to(device)
            T_std = norm_stats['T_std'].to(device)
            k_mean = norm_stats['k_mean'].to(device)
            k_std = norm_stats['k_std'].to(device)
            pricer.set_pointwise_normalization_stats(T_mean, T_std, k_mean, k_std)
    
    pricer.eval()
    print("  Model loaded successfully!")
    
    return pricer, checkpoint, config

def get_maturity_dependent_k_range(T, l=0.55, u=0.30, spot=1.0, n_points=31):
    """
    Calculates the log-moneyness range dependent on maturity
    following the formula K_min = S(1 - l*sqrt(T)), K_max = S(1 + u*sqrt(T))
    """
    import math
    sqrtT = math.sqrt(T)
    K_min = max(spot * (1.0 - l * sqrtT), 0.05 * spot)
    K_max = min(spot * (1.0 + u * sqrtT), 3.00 * spot)
    k_min = math.log(K_min / spot) + 1e-6
    k_max = math.log(K_max / spot) - 1e-6
    return np.linspace(k_min, k_max, n_points)

# ===============================================================
# MONTE CARLO FUNCTIONS
# ===============================================================

def compute_mc_with_ci(pricer, theta, T_values, k_grid_per_T, n_paths=50000, n_batches=10):
    """
    Computes MC with confidence intervals
    k_grid_per_T: dict with key T_idx and value array of k for that T
    """
    from scipy import stats as scipy_stats
    
    print(f"  Computing MC ({n_paths} paths)...")
    
    device = pricer.device
    confidence_level = 0.95
    z_score = scipy_stats.norm.ppf((1 + confidence_level) / 2)
    
    theta_tensor = torch.tensor(theta, device=device, dtype=torch.float32)
    
    # Results for each maturity
    mc_results_per_T = {}
    
    for T_idx, T in enumerate(T_values):
        k_values = k_grid_per_T[T_idx]
        
        # Create temporary grid pricer for this specific T
        T_tensor = torch.tensor([T], device=device, dtype=torch.float32)
        k_tensor = torch.tensor(k_values, device=device, dtype=torch.float32)
        
        temp_pricer = GridNetworkPricer(
            maturities=T_tensor,
            logK=k_tensor,
            process=pricer.process,
            device=device,
            enable_smile_repair=False
        )
        
        # Copy normalization stats if available
        if hasattr(pricer, 'theta_mean') and pricer.theta_mean is not None:
            temp_pricer.set_normalization_stats(
                pricer.theta_mean, pricer.theta_std,
                pricer.iv_mean, pricer.iv_std
            )
        
        batch_size = n_paths // n_batches
        batch_results = []
        
        for batch_idx in range(n_batches):
            try:
                iv_grid = temp_pricer._mc_iv_grid(
                    theta=theta_tensor,
                    n_paths=batch_size,
                    spot=1.0,
                    use_antithetic=True,
                    adaptive_dt=True,
                    control_variate=True
                )
                
                iv_grid = torch.where(torch.isnan(iv_grid) | torch.isinf(iv_grid), 
                                     torch.tensor(0.2, device=device), iv_grid)
                iv_grid = torch.clamp(iv_grid, 0.01, 2.0)
                batch_results.append(iv_grid.squeeze(0))  # Remove T dimension
                
            except Exception as e:
                continue
        
        if batch_results:
            # Calculate statistics
            all_batches = torch.stack(batch_results)
            mean_iv = all_batches.mean(dim=0)
            std_iv = all_batches.std(dim=0)
            ci_margin = z_score * std_iv / np.sqrt(n_batches)
            # Confidence intervals
            mc_results_per_T[T_idx] = {
                'mean': mean_iv.cpu().numpy(),
                'ci_lower': (mean_iv - ci_margin).cpu().numpy(),
                'ci_upper': (mean_iv + ci_margin).cpu().numpy()
            }
    
    return mc_results_per_T


# ===============================================================
# VISUALIZATION FUNCTIONS
# ===============================================================

def compute_nn_surface(pricer, theta, T_values, k_grid_per_T):
    """
    Calculates IV surface with neural network
    k_grid_per_T: dict with key T_idx and value array of k for that T
    """
    device = pricer.device
    theta_tensor = torch.tensor(theta, device=device, dtype=torch.float32)
    
    nn_results_per_T = {}
    
    for T_idx, T in enumerate(T_values):
        k_values = k_grid_per_T[T_idx]
        
        T_tensor = torch.tensor([T], device=device, dtype=torch.float32)
        k_tensor = torch.tensor(k_values, device=device, dtype=torch.float32)
        
        n_points = len(k_values)
        theta_expanded = theta_tensor.unsqueeze(0).repeat(n_points, 1)
        T_expanded = T_tensor.repeat(n_points)
        
        with torch.no_grad():
            iv_pred = pricer.price_iv(
                theta_expanded, 
                T_expanded, 
                k_tensor,
                denormalize_output=True,
                inputs_normalized=False
            )
        
        nn_results_per_T[T_idx] = iv_pred.cpu().numpy()
    
    return nn_results_per_T


def visualize_smile_comparison(pricer, theta, test_name="Test", 
                              maturities=None, 
                              compute_mc=True, mc_paths=50000,
                              output_dir=None):
    """
    Displays NN vs MC smile comparison with CI for different maturities
    with k-range dependent on maturity
    """
    if maturities is None:
        maturities = TEST_MATURITIES
    
    n_maturities = len(maturities)
    
    # Calculate k-grid for each maturity
    k_grid_per_T = {}
    for T_idx, T in enumerate(maturities):
        k_grid_per_T[T_idx] = get_maturity_dependent_k_range(T, n_points=N_STRIKES)
    
    # Calculate NN surface
    print(f"\nComputing NN surface for: {test_name}")
    nn_results = compute_nn_surface(pricer, theta, maturities, k_grid_per_T)
    
    # Calculate MC if requested
    mc_results = None
    if compute_mc:
        mc_results = compute_mc_with_ci(pricer, theta, maturities, k_grid_per_T, n_paths=mc_paths)
    
    # Create plot
    fig, axes = plt.subplots(1, n_maturities, figsize=(5*n_maturities, 5))
    if n_maturities == 1:
        axes = [axes]
    
    # Title with parameters
    param_names = pricer.process.param_info.names if hasattr(pricer.process, 'param_info') else None
    if param_names:
        param_str = ", ".join([f"{name}={val:.3f}" for name, val in zip(param_names, theta)])
    else:
        param_str = ", ".join([f"{val:.3f}" for val in theta])
    
    fig.suptitle(f'MC vs NN Smile Comparison\n{test_name}: θ = ({param_str})', fontsize=14)
    
    summary_mae = []
    summary_coverage = []
    
    for i, (ax, T) in enumerate(zip(axes, maturities)):
        k_values = k_grid_per_T[i]
        
        # NN smile
        iv_nn = nn_results[i]
        ax.plot(k_values, iv_nn, 'r--s', label='Predicted (NN)', linewidth=2, 
                markersize=4, markevery=3)
        
        # MC smile with CI if available
        if mc_results and i in mc_results:
            ax.plot(k_values, mc_results[i]['mean'], 'b-o', 
                   label='True (MC)', linewidth=2, markersize=5, markevery=3)
            ax.fill_between(k_values, 
                           mc_results[i]['ci_lower'],
                           mc_results[i]['ci_upper'],
                           alpha=0.3, color='blue')
            
            mae = np.abs(iv_nn - mc_results[i]['mean']).mean()
            within_ci = ((iv_nn >= mc_results[i]['ci_lower']) & 
                        (iv_nn <= mc_results[i]['ci_upper'])).mean()
            
            metrics_text = f'MAE: {mae:.4f}'
            summary_mae.append(mae)
            summary_coverage.append(within_ci)
        else:
            metrics_text = 'MC not computed'
        
        ax.set_title(f'T = {T:.3f} years', fontsize=12)
        ax.set_xlabel('Log-Moneyness (k)', fontsize=10)
        if i == 0:
            ax.set_ylabel('Implied Volatility', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        ax.text(0.03, 0.97, metrics_text, transform=ax.transAxes, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7), 
               fontsize=9)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/smile_{test_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=150)
        print(f"  Saved: {filename}")
    
    plt.show()
    
    # Print global metrics
    if summary_mae:
        print(f"  Overall MAE: {np.mean(summary_mae):.5f}")
        if summary_coverage:
            print(f"  Overall Coverage: {np.mean(summary_coverage):.1%}")
    
    return nn_results, mc_results


# ===============================================================
# MAIN FUNCTION
# ===============================================================

def main():
    """Main testing function"""
    print("\n" + "="*60)
    print(" POINTWISE MODEL - SMILE COMPARISON TEST ")
    print("="*60)
    
    # Setup
    device = torch.device(DEVICE)
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Model: {os.path.basename(MODEL_PATH)}")
    print(f"  Maturities: {TEST_MATURITIES}")
    print(f"  Strikes: {N_STRIKES} points per maturity (range depends on T)")
    print(f"  MC paths: 50000")
    print(f"  Output: {OUTPUT_DIR if OUTPUT_DIR else 'Display only'}")
    
    # Load model
    try:
        pricer, checkpoint, config = load_pointwise_model(MODEL_PATH, CONFIG_PATH, device)
    except Exception as e:
        print(f"\nERROR: {e}")
        return
    
    print("\n" + "="*60)
    print("TESTING PARAMETER SETS")
    print("="*60)
    
    # k_values = np.linspace(STRIKE_RANGE[0], STRIKE_RANGE[1], N_STRIKES)
    
    summary_results = []
    
    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] {case['name']}")
        
        nn_surface, mc_results = visualize_smile_comparison(
            pricer=pricer,
            theta=case['theta'],
            test_name=case['name'],
            maturities=TEST_MATURITIES,
            # k_values=k_values,
            compute_mc=True,
            mc_paths=50000,
            output_dir=OUTPUT_DIR
        )

        # Global statistics
        if mc_results:
            all_mae = []
            all_coverage = []
            
            for T_idx in nn_surface.keys():
                if T_idx in mc_results:
                    mae_T = np.abs(nn_surface[T_idx] - mc_results[T_idx]['mean']).mean()
                    coverage_T = ((nn_surface[T_idx] >= mc_results[T_idx]['ci_lower']) & 
                                 (nn_surface[T_idx] <= mc_results[T_idx]['ci_upper'])).mean()
                    all_mae.append(mae_T)
                    all_coverage.append(coverage_T)
            
            if all_mae:
                mae_global = np.mean(all_mae)
                within_ci_global = np.mean(all_coverage)
                
                summary_results.append({
                    'name': case['name'],
                    'mae': mae_global,
                    'coverage': within_ci_global
                })
                
                print(f"  Overall MAE: {mae_global:.5f}")
                print(f"  Overall Coverage: {within_ci_global:.1%}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if summary_results:
        print("\nModel Performance vs Monte Carlo:")
        for res in summary_results:
            print(f"  {res['name']:20s} - MAE: {res['mae']:.5f}, Coverage: {res['coverage']:.1%}")
        
        avg_mae = np.mean([r['mae'] for r in summary_results])
        avg_coverage = np.mean([r['coverage'] for r in summary_results])
        print(f"\n  Average MAE: {avg_mae:.5f}")
        print(f"  Average Coverage: {avg_coverage:.1%}")
    
    print("\n" + "="*60)
    print("TESTING COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()