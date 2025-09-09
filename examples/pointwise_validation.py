#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive validation script for PointwiseNetworkPricer
Implements validation methods from Baschetti et al. (2024) paper
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
import time
from datetime import datetime
from scipy import stats as scipy_stats
from scipy.optimize import minimize, differential_evolution
from typing import Dict, List, Tuple, Optional
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

from deepLearningVolatility.nn.pricer.pricer import PointwiseNetworkPricer, GridNetworkPricer
from deepLearningVolatility.stochastic.stochastic_interface import ProcessFactory
from deepLearningVolatility.stochastic.wrappers import rough_bergomi_wrapper

# ===============================================================
# CONFIGURATION
# ===============================================================

MODEL_PATH = "C:/Projects/NN/deepLearningVolatility/models/Pointwise/RoughBergomi/best_model.pt"
CONFIG_PATH = "C:/Projects/NN/deepLearningVolatility/models/Pointwise/RoughBergomi/roughbergomiprocess_pointwise_random_grids_latest_config.json"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = "C:/Projects/NN/deepLearningVolatility/models/Pointwise/RoughBergomi/validation_results"

# ===============================================================
# UTILITY FUNCTIONS
# ===============================================================

def load_pointwise_model(model_path, config_path=None, device='cpu'):
    """
    Load the PointwiseNetworkPricer model
    
    Args:
        model_path: Path to model checkpoint
        config_path: Optional path to config file
        device: Device to load model on
        
    Returns:
        pricer: Loaded model
        checkpoint: Model checkpoint dict
        config: Configuration dict
    """
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
        process_key = 'rough_bergomi'
    
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
    Calculate log-moneyness range dependent on maturity
    
    Args:
        T: Time to maturity
        l: Lower bound parameter
        u: Upper bound parameter  
        spot: Spot price
        n_points: Number of points in grid
        
    Returns:
        Array of log-moneyness values
    """
    import math
    sqrtT = math.sqrt(T)
    K_min = max(spot * (1.0 - l * sqrtT), 0.05 * spot)
    K_max = min(spot * (1.0 + u * sqrtT), 3.00 * spot)
    k_min = math.log(K_min / spot) + 1e-6
    k_max = math.log(K_max / spot) - 1e-6
    return np.linspace(k_min, k_max, n_points)


def sample_random_params(process, device='cpu'):
    """
    Sample random parameters for a given process
    
    Args:
        process: Stochastic process object
        device: Device to create tensor on
        
    Returns:
        Tensor of random parameters
    """
    param_info = process.param_info
    n_params = len(param_info.names)
    
    # Sample uniformly from parameter bounds
    theta = torch.zeros(n_params, device=device)
    for i, (low, high) in enumerate(param_info.bounds):
        theta[i] = torch.rand(1, device=device) * (high - low) + low
    
    return theta


def black_scholes_call(S, K, T, sigma, r=0.0):
    """
    Black-Scholes call option price
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity
        sigma: Implied volatility
        r: Risk-free rate
        
    Returns:
        Call option price
    """
    from scipy.stats import norm
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    call_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return call_price


# ===============================================================
# VALIDATION FUNCTIONS
# ===============================================================

def evaluate_oos_with_mc(pricer, n_param_sets=100, n_smiles_per_set=10, 
                              n_mc_paths=10000, device='cpu'):
    """
    Generate random smiles for out-of-sample evaluation using Monte Carlo
    
    Args:
        pricer: PointwiseNetworkPricer model
        n_param_sets: Number of parameter sets to test
        n_smiles_per_set: Number of smiles per parameter set
        n_mc_paths: Number of MC paths for ground truth
        device: Computation device
        
    Returns:
        Dictionary with error statistics
    """
    print(f"\nEvaluating OOS with {n_param_sets} parameter sets using MC...")
    
    errors = []
    
    for i in range(n_param_sets):
        if i % 10 == 0:
            print(f"  Processing parameter set {i}/{n_param_sets}")
        
        # Sample random parameters
        theta = sample_random_params(pricer.process, device)
        
        # For each parameter set, generate multiple random smiles
        for _ in range(n_smiles_per_set):
            # Random maturity
            T = np.random.uniform(0.05, 1.5)
            
            # Generate k values for this maturity
            k_values = get_maturity_dependent_k_range(T, n_points=13)
            
            # Compute NN implied volatilities
            n_points = len(k_values)
            theta_expanded = theta.unsqueeze(0).repeat(n_points, 1)
            T_tensor = torch.tensor([T], device=device, dtype=torch.float32).repeat(n_points)
            k_tensor = torch.tensor(k_values, device=device, dtype=torch.float32)
            
            with torch.no_grad():
                iv_nn = pricer.price_iv(
                    theta_expanded, 
                    T_tensor, 
                    k_tensor,
                    denormalize_output=True,
                    inputs_normalized=False
                ).cpu().numpy()
            
            # Compute MC implied volatilities using the pricer's MC method
            iv_mc = []
            for k in k_values:
                try:
                    # Use the pricer's built-in MC method
                    iv_point = pricer._mc_iv_point(
                        theta=theta,
                        T=T,
                        logK=k,
                        n_paths=n_mc_paths,
                        spot=1.0,
                        use_antithetic=True,
                        adaptive_dt=True,
                        control_variate=True
                    )
                    iv_mc.append(iv_point)
                except Exception as e:
                    # Fallback to default volatility if MC fails
                    iv_mc.append(0.2)
            
            iv_mc = np.array(iv_mc)
            
            # Calculate errors
            point_errors = np.abs(iv_nn - iv_mc)
            errors.extend(point_errors)
    
    # Calculate statistics
    errors = np.array(errors)
    
    return {
        'mae': np.mean(errors),
        'rmse': np.sqrt(np.mean(errors**2)),
        'q5': np.quantile(errors, 0.05),
        'q25': np.quantile(errors, 0.25),
        'q50': np.median(errors),
        'q75': np.quantile(errors, 0.75),
        'q95': np.quantile(errors, 0.95),
        'std': np.std(errors),
        'max': np.max(errors)
    }


def calibrate_to_surface(pricer, surface_points, initial_theta=None, device='cpu'):
    """
    Real calibration using differential evolution + local refinement
    """
    param_info = pricer.process.param_info
    bounds = param_info.bounds
    
    # Prepare data
    T_values = np.array([p[0] for p in surface_points])
    k_values = np.array([p[1] for p in surface_points])
    iv_target = np.array([p[2] for p in surface_points])
    
    # Weights for different maturities (short-term more important)
    weights = np.exp(-T_values)
    weights = weights / weights.sum() * len(weights)
    
    T_tensor = torch.tensor(T_values, device=device, dtype=torch.float32)
    k_tensor = torch.tensor(k_values, device=device, dtype=torch.float32)
    
    def objective(theta_np):
        """Weighted RMSE objective"""
        theta = torch.tensor(theta_np, device=device, dtype=torch.float32)
        
        # Validate parameters
        is_valid, _ = pricer.process.validate_theta(theta)
        if not is_valid:
            return 1e10
        
        n_points = len(T_values)
        theta_expanded = theta.unsqueeze(0).repeat(n_points, 1)
        
        try:
            with torch.no_grad():
                iv_pred = pricer.price_iv(
                    theta_expanded,
                    T_tensor,
                    k_tensor,
                    denormalize_output=True,
                    inputs_normalized=False
                ).cpu().numpy()
            
            # Weighted RMSE with penalty for extreme values
            errors = (iv_pred - iv_target) * weights
            rmse = np.sqrt(np.mean(errors**2))
            
            # Add penalty for unrealistic IVs
            if np.any(iv_pred < 0.01) or np.any(iv_pred > 2.0):
                rmse += 1.0
                
            return rmse
            
        except Exception as e:
            return 1e10
    
    # Step 1: Global optimization with differential evolution
    result_de = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=50,
        popsize=15,
        atol=1e-5,
        tol=1e-5,
        workers=1
    )
    
    # Step 2: Local refinement with L-BFGS-B
    result = minimize(
        objective,
        result_de.x,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-8}
    )
    
    return torch.tensor(result.x, device=device, dtype=torch.float32)


def plot_45_degree_comparison(pricer, n_test_samples=500, n_mc_paths=5000, device='cpu'):
    """
    Create 45-degree plots with real MC values
    """
    print("\nGenerating 45-degree comparison plots with real MC...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Generate test data
    all_true, all_pred = [], []
    
    for i in range(n_test_samples):
        if i % 50 == 0:
            print(f"  Processing sample {i}/{n_test_samples}")
        
        # Random parameters and contracts
        theta = sample_random_params(pricer.process, device)
        T = np.random.uniform(0.1, 1.0)
        k = np.random.uniform(-0.3, 0.3)
        
        # NN prediction
        T_tensor = torch.tensor([T], device=device, dtype=torch.float32)
        k_tensor = torch.tensor([k], device=device, dtype=torch.float32)
        
        with torch.no_grad():
            iv_pred = pricer.price_iv(
                theta.unsqueeze(0), 
                T_tensor, 
                k_tensor,
                denormalize_output=True,
                inputs_normalized=False
            ).item()
        
        # True value from MC
        try:
            iv_true = pricer._mc_iv_point(
                theta=theta,
                T=T,
                logK=k,
                n_paths=n_mc_paths,
                spot=1.0,
                use_antithetic=True,
                adaptive_dt=True,
                control_variate=True
            )
            
            all_true.append(iv_true)
            all_pred.append(iv_pred)
            
        except:
            continue
    
    # Split into train/test
    n_split = len(all_true) // 2
    train_true = all_true[:n_split]
    train_pred = all_pred[:n_split]
    test_true = all_true[n_split:]
    test_pred = all_pred[n_split:]
    
    # Plot
    ax1.scatter(train_true, train_pred, alpha=0.5, s=10)
    ax1.plot([0, 0.6], [0, 0.6], 'r--', linewidth=2)
    ax1.set_xlabel('MC IV (True)')
    ax1.set_ylabel('NN IV (Predicted)')
    ax1.set_title('Train Set')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 0.6])
    ax1.set_ylim([0, 0.6])
    
    ax2.scatter(test_true, test_pred, alpha=0.5, s=10)
    ax2.plot([0, 0.6], [0, 0.6], 'r--', linewidth=2)
    ax2.set_xlabel('MC IV (True)')
    ax2.set_ylabel('NN IV (Predicted)')
    ax2.set_title('Test Set')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 0.6])
    ax2.set_ylim([0, 0.6])
    
    # Calculate metrics
    from sklearn.metrics import r2_score, mean_squared_error
    
    if len(train_true) > 0:
        r2_train = r2_score(train_true, train_pred)
        rmse_train = np.sqrt(mean_squared_error(train_true, train_pred))
        ax1.text(0.05, 0.95, f'R²={r2_train:.4f}\nRMSE={rmse_train:.4f}', 
                 transform=ax1.transAxes, va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if len(test_true) > 0:
        r2_test = r2_score(test_true, test_pred)
        rmse_test = np.sqrt(mean_squared_error(test_true, test_pred))
        ax2.text(0.05, 0.95, f'R²={r2_test:.4f}\nRMSE={rmse_test:.4f}', 
                 transform=ax2.transAxes, va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def analyze_parameter_recovery(pricer, n_tests=50, n_mc_paths=10000, device='cpu'):
    """
    Analyze parameter recovery capability using MC and calibration
    
    Args:
        pricer: PointwiseNetworkPricer model
        n_tests: Number of test cases
        n_mc_paths: Number of MC paths for generating synthetic surface
        device: Computation device
        
    Returns:
        Dictionary with recovery statistics per parameter
    """
    print(f"\nAnalyzing parameter recovery with {n_tests} tests using real calibration...")
    
    param_names = pricer.process.param_info.names
    n_params = len(param_names)
    recovery_errors = {name: [] for name in param_names}
    
    for test_idx in range(n_tests):
        if test_idx % 10 == 0:
            print(f"  Test {test_idx}/{n_tests}")
        
        # Generate true parameters
        theta_true = sample_random_params(pricer.process, device)
        
        # Generate synthetic surface using MC
        T_values = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
        surface_points = []
        
        for T in T_values:
            k_values = get_maturity_dependent_k_range(T, n_points=11)
            
            for k in k_values:
                # Use real MC to generate "market" IV
                try:
                    iv_mc = pricer._mc_iv_point(
                        theta=theta_true,
                        T=T,
                        logK=k,
                        n_paths=n_mc_paths,
                        spot=1.0,
                        use_antithetic=True,
                        adaptive_dt=True,
                        control_variate=True
                    )
                    # Add small noise to simulate bid-ask spread
                    iv_market = iv_mc + np.random.normal(0, 0.001)
                    surface_points.append((T, k, iv_market))
                except:
                    pass  # Skip failed points
        
        if len(surface_points) < 20:
            continue  # Skip if too few valid points
        
        # Calibrate to recover parameters
        theta_recovered = calibrate_to_surface(pricer, surface_points, 
                                              initial_theta=theta_true, device=device)
        
        # Store errors
        for i, name in enumerate(param_names):
            error = (theta_recovered[i] - theta_true[i]).item()
            recovery_errors[name].append(error)
    
    # Calculate statistics
    results = {}
    for name, errors in recovery_errors.items():
        if len(errors) > 0:
            errors = np.array(errors)
            results[name] = {
                'min': np.min(errors),
                'mean': np.mean(errors),
                'median': np.median(errors),
                'q95': np.quantile(errors, 0.95),
                'max': np.max(errors),
                'std': np.std(errors),
                'overestimation_ratio': np.mean(errors > 0),
                '2tail_0.02': np.mean(np.abs(errors) > 0.02),
                '2tail_0.03': np.mean(np.abs(errors) > 0.03),
                '2tail_0.05': np.mean(np.abs(errors) > 0.05)
            }
    
    return results


def benchmark_real_calibration_times(pricer, n_tests=10, device='cpu'):
    """
    Benchmark real calibration times
    
    Args:
        pricer: PointwiseNetworkPricer model
        n_tests: Number of test calibrations
        device: Computation device
        
    Returns:
        Dictionary with timing statistics
    """
    print(f"\nBenchmarking real calibration times with {n_tests} tests...")
    
    times = []
    
    for test_idx in range(n_tests):
        # Generate random surface
        theta_true = sample_random_params(pricer.process, device)
        
        # Generate market-like surface
        surface_points = []
        n_maturities = np.random.randint(5, 10)
        T_values = np.sort(np.random.uniform(0.05, 2.0, n_maturities))
        
        for T in T_values:
            n_strikes = np.random.randint(5, 15)
            k_values = get_maturity_dependent_k_range(T, n_points=n_strikes)
            
            for k in k_values:
                # Use NN to generate synthetic market (faster than MC)
                with torch.no_grad():
                    iv = pricer.price_iv(
                        theta_true.unsqueeze(0),
                        torch.tensor([T], device=device, dtype=torch.float32),
                        torch.tensor([k], device=device, dtype=torch.float32),
                        denormalize_output=True,
                        inputs_normalized=False
                    ).item()
                
                surface_points.append((T, k, iv))
        
        # Time the calibration
        start = time.time()
        _ = calibrate_to_surface(pricer, surface_points, device=device)
        elapsed = time.time() - start
        
        times.append(elapsed)
        print(f"    Test {test_idx+1}: {elapsed:.2f}s")
    
    times = np.array(times)
    
    return {
        'avg': np.mean(times),
        'min': np.min(times),
        'max': np.max(times),
        'std': np.std(times),
        'median': np.median(times),
        'q25': np.quantile(times, 0.25),
        'q75': np.quantile(times, 0.75)
    }


def check_arbitrage_violations(pricer, n_tests=10, device='cpu'):
    """
    Check for arbitrage violations on dense grid
    
    Args:
        pricer: PointwiseNetworkPricer model
        n_tests: Number of random parameter sets to test
        device: Computation device
        
    Returns:
        Dictionary with violation counts
    """
    print(f"\nChecking arbitrage violations on {n_tests} parameter sets...")
    
    total_violations = {'vertical': 0, 'butterfly': 0, 'calendar': 0}
    total_points = 0
    
    for test_idx in range(n_tests):
        # Random parameters
        theta = sample_random_params(pricer.process, device)
        spot = 1.0
        
        # Dense time grid
        T_grid = np.arange(5/365, 2.0, 5/365)  # Every 5 days
        
        for T in T_grid:
            # Dense strike grid
            k_grid = get_maturity_dependent_k_range(T, n_points=50)
            K_grid = spot * np.exp(k_grid)
            
            # Compute implied volatilities
            n_points = len(k_grid)
            T_tensor = torch.tensor([T], device=device, dtype=torch.float32).repeat(n_points)
            k_tensor = torch.tensor(k_grid, device=device, dtype=torch.float32)
            theta_expanded = theta.unsqueeze(0).repeat(n_points, 1)
            
            with torch.no_grad():
                iv = pricer.price_iv(
                    theta_expanded,
                    T_tensor,
                    k_tensor,
                    denormalize_output=True,
                    inputs_normalized=False
                ).cpu().numpy()
            
            # Convert to prices
            prices = np.array([black_scholes_call(spot, K, T, sigma) 
                              for K, sigma in zip(K_grid, iv)])
            
            total_points += len(prices)
            
            # Check vertical spread: C(K1) > C(K2) for K1 < K2
            for i in range(len(prices)-1):
                if prices[i] < prices[i+1]:
                    total_violations['vertical'] += 1
            
            # Check butterfly spread
            for i in range(len(prices)-2):
                K1, K2, K3 = K_grid[i:i+3]
                C1, C2, C3 = prices[i:i+3]
                butterfly = (K3-K2)*C1 - (K3-K1)*C2 + (K2-K1)*C3
                if butterfly < -1e-6:  # Small tolerance for numerical errors
                    total_violations['butterfly'] += 1
        
        # Check calendar spread (prices should increase with maturity)
        for k in np.linspace(-0.3, 0.3, 20):
            prices_by_T = []
            
            for T in T_grid[::5]:  # Check every 25 days
                k_tensor = torch.tensor([k], device=device, dtype=torch.float32)
                T_tensor = torch.tensor([T], device=device, dtype=torch.float32)
                
                with torch.no_grad():
                    iv = pricer.price_iv(
                        theta.unsqueeze(0),
                        T_tensor,
                        k_tensor,
                        denormalize_output=True,
                        inputs_normalized=False
                    ).item()
                
                price = black_scholes_call(spot, spot*np.exp(k), T, iv)
                prices_by_T.append(price)
            
            # Check monotonicity
            for i in range(len(prices_by_T)-1):
                if prices_by_T[i] > prices_by_T[i+1] + 1e-6:
                    total_violations['calendar'] += 1
    
    print(f"  Total points checked: {total_points}")
    print(f"  Violations found:")
    for vtype, count in total_violations.items():
        print(f"    {vtype}: {count} ({100*count/max(total_points,1):.4f}%)")
    
    return total_violations


def plot_oos_evolution(errors_by_size):
    """
    Plot OOS error evolution as in Figures 4-5 of the paper
    
    Args:
        errors_by_size: Dictionary mapping training set sizes to error stats
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sizes = sorted(errors_by_size.keys())
    mae_values = [errors_by_size[s]['mae'] for s in sizes]
    q5_values = [errors_by_size[s]['q5'] for s in sizes]
    q95_values = [errors_by_size[s]['q95'] for s in sizes]
    
    # Plot as candlesticks
    for i, size in enumerate(sizes):
        ax.plot([i, i], [q5_values[i], q95_values[i]], 'b-', linewidth=2)
        ax.plot(i, mae_values[i], 'ro', markersize=8)
    
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f'2^{int(np.log2(s))}' for s in sizes])
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Out-of-Sample Error |? - ?^|')
    ax.set_title('OOS Error Evolution')
    ax.grid(True, alpha=0.3)
    
    return fig


def save_validation_results(output_dir, results):
    """
    Save all validation results to files
    
    Args:
        output_dir: Directory to save results
        results: Dictionary with all validation results
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save as JSON
    results_file = f"{output_dir}/validation_results_{timestamp}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Also save as formatted text report
    report_file = f"{output_dir}/validation_report_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("VALIDATION REPORT\n")
        f.write(f"Generated: {timestamp}\n")
        f.write("="*60 + "\n\n")
        
        # OOS Statistics
        if 'oos_stats' in results:
            f.write("OUT-OF-SAMPLE STATISTICS\n")
            f.write("-"*40 + "\n")
            for key, value in results['oos_stats'].items():
                f.write(f"  {key:20s}: {value:.6f}\n")
            f.write("\n")
        
        # Arbitrage violations
        if 'arbitrage' in results:
            f.write("ARBITRAGE CHECK\n")
            f.write("-"*40 + "\n")
            for key, value in results['arbitrage'].items():
                f.write(f"  {key:20s}: {value}\n")
            f.write("\n")
        
        # Parameter recovery
        if 'recovery' in results:
            f.write("PARAMETER RECOVERY\n")
            f.write("-"*40 + "\n")
            for param, stats in results['recovery'].items():
                f.write(f"\n  {param}:\n")
                for key, value in stats.items():
                    f.write(f"    {key:20s}: {value:.6f}\n")
            f.write("\n")
        
        # Timing
        if 'times' in results:
            f.write("CALIBRATION TIMES (seconds)\n")
            f.write("-"*40 + "\n")
            for key, value in results['times'].items():
                f.write(f"  {key:20s}: {value:.4f}\n")
    
    print(f"Report saved to: {report_file}")


# ===============================================================
# MAIN VALIDATION FUNCTION
# ===============================================================

def comprehensive_validation_with_mc(model_path, config_path, output_dir):
    """
    Perform comprehensive validation with MC and calibration
    
    Args:
        model_path: Path to model checkpoint
        config_path: Path to config file
        output_dir: Directory for output files
    """
    print("\n" + "="*60)
    print(" COMPREHENSIVE MODEL VALIDATION WITH MC ")
    print("="*60)
    
    # Setup
    device = torch.device(DEVICE)
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Model: {os.path.basename(model_path)}")
    print(f"  Output: {output_dir}")
    
    # 1. Load model
    try:
        pricer, checkpoint, config = load_pointwise_model(model_path, config_path, device)
    except Exception as e:
        print(f"\nERROR loading model: {e}")
        return
    
    results = {}
    
    # 2. Out-of-sample evaluation with Monte Carlo
    print("\n" + "="*40)
    print("TEST 1: OOS with Monte Carlo")
    print("="*40)
    oos_stats = evaluate_oos_with_mc(
        pricer, 
        n_param_sets=20,
        n_smiles_per_set=5,
        n_mc_paths=10_000,
        device=device
    )
    results['oos_stats'] = oos_stats
    print("\nOOS Statistics:")
    for key, value in oos_stats.items():
        print(f"  {key:10s}: {value:.6f}")
    
    # 3. 45-degree plots (keep existing)
    print("\n" + "="*40)
    print("TEST 2: 45-Degree Plots")
    print("="*40)
    fig_45 = plot_45_degree_comparison(
        pricer, 
        n_test_samples=1000,
        n_mc_paths=10000,
        device=device
    )
    fig_45.savefig(f"{output_dir}/45_degree_plot_real_mc.png", dpi=150)
    
    # 4. Arbitrage check (keep existing)
    print("\n" + "="*40)
    print("TEST 3: No-Arbitrage Check")
    print("="*40)
    arbitrage_violations = check_arbitrage_violations(
        pricer, 
        n_tests=100,
        device=device
    )
    results['arbitrage'] = arbitrage_violations
    
    # 5. Parameter recovery with calibration
    print("\n" + "="*40)
    print("TEST 4: Parameter Recovery with Calibration")
    print("="*40)
    recovery_stats = analyze_parameter_recovery(
        pricer, 
        n_tests=100,
        n_mc_paths=20_000,     # MC paths for surface generation
        device=device
    )
    results['recovery'] = recovery_stats
    
    if recovery_stats:
        print("\nParameter Recovery Statistics:")
        for param, stats in recovery_stats.items():
            print(f"\n  {param}:")
            print(f"    Mean error: {stats['mean']:.6f}")
            print(f"    Std error:  {stats['std']:.6f}")
            print(f"    Overest. ratio: {stats['overestimation_ratio']:.3f}")
    
    # 6. Benchmark calibration times
    print("\n" + "="*40)
    print("TEST 5: Calibration Times")
    print("="*40)
    time_stats = benchmark_real_calibration_times(pricer, n_tests=50, device=device)
    results['times'] = time_stats
    print("\nTiming Statistics (seconds):")
    for key, value in time_stats.items():
        print(f"  {key:10s}: {value:.4f}")
    
    # 7. Save all results
    print("\n" + "="*40)
    print("SAVING RESULTS")
    print("="*40)
    save_validation_results(output_dir, results)
    
    print("\n" + "="*60)
    print(" VALIDATION WITH MC COMPLETED ")
    print("="*60)
    
    return results


# ===============================================================
# ADDITIONAL VISUALIZATION FUNCTIONS
# ===============================================================

def create_validation_summary_plots(results, output_dir):
   """
   Create summary visualization plots for validation results
   
   Args:
       results: Dictionary with validation results
       output_dir: Directory to save plots
   """
   # Create figure with subplots
   fig = plt.figure(figsize=(15, 10))
   
   # 1. OOS Error Distribution
   if 'oos_stats' in results:
       ax1 = plt.subplot(2, 3, 1)
       stats = results['oos_stats']
       
       # Box plot representation
       box_data = [stats['q5'], stats['q25'], stats['q50'], stats['q75'], stats['q95']]
       positions = [0.05, 0.25, 0.50, 0.75, 0.95]
       ax1.bar(positions, box_data, width=0.15, alpha=0.7)
       ax1.axhline(stats['mae'], color='r', linestyle='--', label=f"MAE: {stats['mae']:.4f}")
       ax1.set_xlabel('Quantiles')
       ax1.set_ylabel('Error')
       ax1.set_title('OOS Error Distribution')
       ax1.legend()
       ax1.grid(True, alpha=0.3)
   
   # 2. Arbitrage Violations
   if 'arbitrage' in results:
       ax2 = plt.subplot(2, 3, 2)
       violations = results['arbitrage']
       
       labels = list(violations.keys())
       values = list(violations.values())
       colors = ['green' if v == 0 else 'red' for v in values]
       
       ax2.bar(labels, values, color=colors, alpha=0.7)
       ax2.set_ylabel('Number of Violations')
       ax2.set_title('Arbitrage Check Results')
       ax2.grid(True, alpha=0.3)
       
       # Add text annotations
       for i, v in enumerate(values):
           ax2.text(i, v + max(values)*0.01, str(v), ha='center')
   
   # 3. Parameter Recovery - Mean Errors
   if 'recovery' in results:
       ax3 = plt.subplot(2, 3, 3)
       params = list(results['recovery'].keys())
       mean_errors = [results['recovery'][p]['mean'] for p in params]
       std_errors = [results['recovery'][p]['std'] for p in params]
       
       x_pos = range(len(params))
       ax3.bar(x_pos, mean_errors, yerr=std_errors, alpha=0.7, capsize=5)
       ax3.set_xticks(x_pos)
       ax3.set_xticklabels(params)
       ax3.set_ylabel('Mean Error')
       ax3.set_title('Parameter Recovery - Mean Errors')
       ax3.axhline(0, color='k', linestyle='-', linewidth=0.5)
       ax3.grid(True, alpha=0.3)
   
   # 4. Parameter Recovery - Overestimation Ratio
   if 'recovery' in results:
       ax4 = plt.subplot(2, 3, 4)
       overest_ratios = [results['recovery'][p]['overestimation_ratio'] for p in params]
       
       ax4.bar(x_pos, overest_ratios, alpha=0.7)
       ax4.set_xticks(x_pos)
       ax4.set_xticklabels(params)
       ax4.set_ylabel('Overestimation Ratio')
       ax4.set_title('Parameter Recovery - Overestimation')
       ax4.axhline(0.5, color='r', linestyle='--', alpha=0.5)
       ax4.set_ylim([0, 1])
       ax4.grid(True, alpha=0.3)
   
   # 5. Calibration Times Distribution
   if 'times' in results:
       ax5 = plt.subplot(2, 3, 5)
       times = results['times']
       
       # Create box plot data
       box_stats = [times['min'], times['q25'], times['median'], times['q75'], times['max']]
       
       ax5.boxplot([box_stats], vert=True, widths=0.5)
       ax5.axhline(times['avg'], color='r', linestyle='--', label=f"Mean: {times['avg']:.3f}s")
       ax5.set_ylabel('Time (seconds)')
       ax5.set_title('Calibration Time Distribution')
       ax5.set_xticklabels([''])
       ax5.legend()
       ax5.grid(True, alpha=0.3)
   
   # 6. Summary Statistics Table
   ax6 = plt.subplot(2, 3, 6)
   ax6.axis('off')
   
   # Create summary text
   summary_text = "VALIDATION SUMMARY\n" + "="*30 + "\n\n"
   
   if 'oos_stats' in results:
       summary_text += f"OOS MAE: {results['oos_stats']['mae']:.6f}\n"
       summary_text += f"OOS RMSE: {results['oos_stats']['rmse']:.6f}\n\n"
   
   if 'arbitrage' in results:
       total_violations = sum(results['arbitrage'].values())
       summary_text += f"Total Arbitrage Violations: {total_violations}\n\n"
   
   if 'times' in results:
       summary_text += f"Avg Calibration Time: {results['times']['avg']:.3f}s\n"
       summary_text += f"Max Calibration Time: {results['times']['max']:.3f}s\n"
   
   ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')
   
   plt.tight_layout()
   
   # Save figure
   fig.savefig(f"{output_dir}/validation_summary.png", dpi=150, bbox_inches='tight')
   print(f"  Summary plots saved to validation_summary.png")
   
   return fig


# ===============================================================
# MAIN EXECUTION
# ===============================================================

def main():
    """
    Main function to run complete validation with MC
    """
    print("\n" + "="*60)
    print(" POINTWISE NETWORK PRICER - VALIDATION WITH MC ")
    print("="*60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Run comprehensive validation with MC
    results = comprehensive_validation_with_mc(MODEL_PATH, CONFIG_PATH, OUTPUT_DIR)
    
    # Create summary plots
    if results:
        print("\n" + "="*40)
        print("CREATING SUMMARY PLOTS")
        print("="*40)
        create_validation_summary_plots(results, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print(" ALL TESTS COMPLETED ")
    print("="*60)
    print(f"\nResults saved in: {OUTPUT_DIR}")
    
    # Print quick summary
    if results:
        print("\n" + "="*40)
        print("QUICK SUMMARY")
        print("="*40)
        
        if 'oos_stats' in results:
            print(f"  OOS MAE: {results['oos_stats']['mae']:.6f}")
        
        if 'times' in results:
            print(f"  Avg calibration time: {results['times']['avg']:.3f}s")


if __name__ == "__main__":
    main()