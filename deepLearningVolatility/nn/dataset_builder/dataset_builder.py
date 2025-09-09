# -*- coding: utf-8 -*-

"""Generation of synthetic training and validation datasets for neural volatility models.

This module provides the tools to create large-scale, high-quality datasets
required to train the neural network pricers. It automates the computationally
intensive process of generating data by orchestrating parameter sampling,
Monte Carlo simulations, and data normalization.

The core components are designed to be generic, relying on the `StochasticProcess`
interface to adapt the data generation to any underlying financial model.
The module supports various data generation strategies, from fixed grids to the
"Random Grids" approach proposed by Baschetti et al.
Key classes include:

- `DatasetBuilder`: The main factory class, configured for a specific process.
- `MultiRegimeDatasetBuilder`: A specialization for creating datasets tailored
  to the three maturity regimes (short, mid, long).
- `CheckpointManager`: A utility for managing persistence and resumption of
  long-running data generation tasks.
"""

import torch
import numpy as np
from scipy.stats import qmc
from typing import Dict, Tuple, Optional, Union
from tqdm import tqdm
import time
import os, pickle, re
import gc
import math
from datetime import datetime
import matplotlib.pyplot as plt
from deepLearningVolatility.stochastic.stochastic_interface import StochasticProcess, ProcessFactory
from deepLearningVolatility.nn.pricer.pricer import GridNetworkPricer, PointwiseNetworkPricer, MultiRegimeGridPricer


class DatasetBuilder:
    """
    Dataset builder that supports any StochasticProcess.
    """
    
    def __init__(self, process, device='cpu', output_dir=None, dataset_type: str = 'train', **kwargs):
        """
        Args:
            process: Name of the process or StochasticProcess instance
            device: Device torch
            output_dir: Output directory
        """
        self.device = torch.device(device)
        self.output_dir = output_dir
        dt = (dataset_type or 'train').lower()
        assert dt in ('train', 'val'), "dataset_type must be 'train' or 'val'"
        self.dataset_type = dt

        # Setup directories if output_dir is specified
        if self.output_dir:
            self.setup_directories()
        
        # Create process if passed as a string
        if isinstance(process, str):
            self.process = ProcessFactory.create(process)
        else:
            self.process = process
        self.process_name = self.process.__class__.__name__.lower()
        
        # Update process-based bounds
        self._update_param_bounds()
        
        # Initialize statistics
        self.theta_mean = None
        self.theta_std = None
        self.iv_mean = None
        self.iv_std = None
        self.T_mean = None
        self.T_std = None
        self.k_mean = None
        self.k_std = None
    
    def _update_param_bounds(self):
        """Update process-based parameter bounds."""
        param_info = self.process.param_info
        self.param_bounds = {}
        
        for i, (name, bounds) in enumerate(zip(param_info.names, param_info.bounds)):
            self.param_bounds[name] = bounds
        
        # Also maintain a per-index mapping
        self.param_names = param_info.names
        self.param_defaults = param_info.defaults
    
    def compute_normalization_stats(self, thetas, ivs, Ts=None, ks=None):
        """
        Calculate mean and standard deviation for normalization.

        Args:
            thetas: model parameters [N, 4]
            ivs: implied volatilities (grid or points)
            Ts: maturity (pointwise only)
            ks: log-moneyness (pointwise only)
        """
        # Statistics for theta
        self.theta_mean = thetas.mean(dim=0)
        self.theta_std = thetas.std(dim=0)
        # Avoid division by zero
        self.theta_std = torch.where(self.theta_std > 1e-6, self.theta_std, torch.ones_like(self.theta_std))
        
        # Statistics for IV
        self.iv_mean = ivs.mean()
        self.iv_std = ivs.std()
        if self.iv_std < 1e-6:
            self.iv_std = torch.tensor(1.0, device=self.device)
        
        # Statistics for T and k (pointwise only)
        if Ts is not None:
            self.T_mean = Ts.mean()
            self.T_std = Ts.std()
            if self.T_std < 1e-6:
                self.T_std = torch.tensor(1.0, device=self.device)
                
        if ks is not None:
            self.k_mean = ks.mean()
            self.k_std = ks.std()
            if self.k_std < 1e-6:
                self.k_std = torch.tensor(1.0, device=self.device)
    
    def normalize_theta(self, theta):
        """Normalize theta parameters using z-score normalization"""
        if self.theta_mean is None or self.theta_std is None:
            raise ValueError("Normalization stats not computed. Call compute_normalization_stats first.")
        return (theta - self.theta_mean) / self.theta_std
    
    def denormalize_theta(self, theta_norm):
        """Denormalize theta parameters"""
        if self.theta_mean is None or self.theta_std is None:
            raise ValueError("Normalization stats not computed.")
        return theta_norm * self.theta_std + self.theta_mean
    
    def normalize_iv(self, iv):
        """Normalize implied volatilities"""
        if self.iv_mean is None or self.iv_std is None:
            raise ValueError("Normalization stats not computed. Call compute_normalization_stats first.")
        return (iv - self.iv_mean) / self.iv_std
    
    def denormalize_iv(self, iv_norm):
        """Denormalizes implied volatilities"""
        if self.iv_mean is None or self.iv_std is None:
            raise ValueError("Normalization stats not computed.")
        return iv_norm * self.iv_std + self.iv_mean
    
    def normalize_T(self, T):
        """Normalize the maturity"""
        if self.T_mean is None or self.T_std is None:
            raise ValueError("Normalization stats not computed.")
        return (T - self.T_mean) / self.T_std
    
    def denormalize_T(self, T_norm):
        """Denormalize the maturity"""
        if self.T_mean is None or self.T_std is None:
            raise ValueError("Normalization stats not computed.")
        return T_norm * self.T_std + self.T_mean
    
    def normalize_k(self, k):
        """Normalize the log-moneyness"""
        if self.k_mean is None or self.k_std is None:
            raise ValueError("Normalization stats not computed.")
        return (k - self.k_mean) / self.k_std
    
    def denormalize_k(self, k_norm):
        """Denormalize the log-moneyness"""
        if self.k_mean is None or self.k_std is None:
            raise ValueError("Normalization stats not computed.")
        return k_norm * self.k_std + self.k_mean
    
    def save_normalization_stats(self, path=None):
        """Save normalization statistics"""
        if path is None and self.output_dir:
            path = f"{self.dirs['stats']}/{self.process_name}_normalization_stats.pt"
        
        stats = {
            'theta_mean': self.theta_mean,
            'theta_std': self.theta_std,
            'iv_mean': self.iv_mean,
            'iv_std': self.iv_std,
            'T_mean': self.T_mean,
            'T_std': self.T_std,
            'k_mean': self.k_mean,
            'k_std': self.k_std,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(stats, path)
        print(f"✓ Normalization stats saved: {path}")
    
    def load_normalization_stats(self, path=None):
        """Load normalization statistics"""
        if path is None and self.output_dir:
            path = f"{self.dirs['stats']}/{self.process_name}_normalization_stats.pt"
            
        stats = torch.load(path, map_location=self.device)
        self.theta_mean = stats['theta_mean']
        self.theta_std = stats['theta_std']
        self.iv_mean = stats['iv_mean']
        self.iv_std = stats['iv_std']
        self.T_mean = stats.get('T_mean')
        self.T_std = stats.get('T_std')
        self.k_mean = stats.get('k_mean')
        self.k_std = stats.get('k_std')
        print(f"✓ Normalization stats loaded from: {path}")
        
    def setup_directories(self):
        """Create directory structure for Google Colab"""
        self.dirs = {
            'datasets': f"{self.output_dir}/datasets",
            'checkpoints': f"{self.output_dir}/checkpoints",
            'models': f"{self.output_dir}/models",
            'logs': f"{self.output_dir}/logs",
            'stats': f"{self.output_dir}/stats"
        }
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            
    def sample_theta_lhs(self, n_samples, seed=None):
        """
        LHS sampling for the specific process.
        """
        # Extract bounds as array
        bounds = np.array([self.param_bounds[name] for name in self.param_names])
        
        # Create LHS sampler
        sampler = qmc.LatinHypercube(d=self.process.num_params, seed=seed)
        
        # Generate samples in the unit hypercube
        unit_samples = sampler.random(n=n_samples)
        
        # Scale to real bounds
        scaled_samples = qmc.scale(unit_samples, bounds[:, 0], bounds[:, 1])
        
        return torch.tensor(scaled_samples, dtype=torch.float32, device=self.device)
    
    def sample_theta_lhs_restricted(self, n_samples, seed=None, restriction_factor=0.5):
        """
        LHS with restricted range for testing.

        Args:
            restriction_factor: Fraction of the range to use (0.5 = use half the range)
        """
        bounds = np.array([self.param_bounds[name] for name in self.param_names])
        
        # Reduce the range
        centers = (bounds[:, 0] + bounds[:, 1]) / 2
        ranges = bounds[:, 1] - bounds[:, 0]
        restricted_ranges = ranges * restriction_factor
        
        restricted_bounds = np.column_stack([
            centers - restricted_ranges / 2,
            centers + restricted_ranges / 2
        ])
        
        # Ensures that restricted bounds are valid
        restricted_bounds[:, 0] = np.maximum(restricted_bounds[:, 0], bounds[:, 0])
        restricted_bounds[:, 1] = np.minimum(restricted_bounds[:, 1], bounds[:, 1])
        
        sampler = qmc.LatinHypercube(d=self.process.num_params, seed=seed)
        unit_samples = sampler.random(n=n_samples)
        scaled_samples = qmc.scale(unit_samples, restricted_bounds[:, 0], restricted_bounds[:, 1])
        
        return torch.tensor(scaled_samples, dtype=torch.float32, device=self.device)
    
    def sample_theta_uniform(self, n_samples):
        """Standard uniform sampling for comparison"""
        bounds = np.array([self.param_bounds[name] for name in self.param_names])
        samples = []
        for _ in range(n_samples):
            sample = []
            for i, name in enumerate(self.param_names):
                value = np.random.uniform(bounds[i, 0], bounds[i, 1])
                sample.append(value)
            samples.append(sample)
        
        return torch.tensor(samples, dtype=torch.float32, device=self.device)
    
    def create_pricer(self, maturities, logK, **pricer_kwargs):
        """
        Create a GridNetworkPricer for the process.
        """
        return GridNetworkPricer(
            maturities=maturities,
            logK=logK,
            process=self.process,
            device=self.device,
            **pricer_kwargs
        )
    
    def get_process_specific_mc_params(self, base_n_paths: int = 30000) -> Dict:
        """
        Gets MC parameters optimized for the specific process.
        """
        process_name = self.process.__class__.__name__.lower()
        
        if 'rough' in process_name:
            # Rough models need more paths
            return {
                'n_paths': int(base_n_paths),
                'use_antithetic': True,
                'adaptive_paths': False,
                'adaptive_dt': True,
                'control_variate': True
            }
        elif 'jump' in process_name:
            # Jump models
            return {
                'n_paths': int(base_n_paths),
                'use_antithetic': True,
                'adaptive_paths': False,
                'adaptive_dt': False,
                'control_variate': True
            }
        elif 'heston' in process_name:
            # Heston
            return {
                'n_paths': base_n_paths,
                'use_antithetic': True,
                'adaptive_paths': False,
                'adaptive_dt': True,
                'control_variate': True
            }
        else:
            # Default
            return {
                'n_paths': base_n_paths,
                'use_antithetic': True,
                'adaptive_paths': False,
                'adaptive_dt': True,
                'control_variate': True
            }
       
    def build_grid_dataset(self, pricer: GridNetworkPricer, n_samples, n_paths=30000, 
                          restricted=False, show_progress=True, normalize=True,
                          compute_stats_from=None):
        """
        Builds datasets for GridNetworkPricer with optional normalization.
        Updated version for generic processes.
        """
        # Sample theta
        if restricted:
            thetas = self.sample_theta_lhs_restricted(n_samples)
        else:
            thetas = self.sample_theta_lhs(n_samples)
        
        # Get process-optimized MC parameters
        mc_params = self.get_process_specific_mc_params()
        
        # Generate IV surfaces
        ivs = []
        invalid_count = 0
        
        iterator = tqdm(thetas, desc=f"Building {self.process.__class__.__name__} grid dataset") if show_progress else thetas
        
        for i, theta in enumerate(iterator):
            try:
                # Use optimized MC parameters
                iv = pricer._mc_iv_grid(
                    theta, 
                    n_paths=mc_params['n_paths'],
                    use_antithetic=mc_params['use_antithetic'],
                    adaptive_paths=mc_params['adaptive_paths'],
                    adaptive_dt=mc_params['adaptive_dt'],
                    control_variate=mc_params['control_variate']
                )
                
                # Check validity
                if torch.isnan(iv).any() or torch.isinf(iv).any():
                    print(f"\nWarning: Invalid IV for theta {i}: {theta}")
                    invalid_count += 1
                    # Use process-based default volatility
                    default_vol = self._get_process_default_volatility(theta)
                    iv = torch.full_like(iv, default_vol)
                elif (iv < 0.01).any() or (iv > 2.0).any():
                    print(f"\nWarning: Extreme IV values for theta {i}: min={iv.min():.4f}, max={iv.max():.4f}")
                
                ivs.append(iv.cpu())
                
            except Exception as e:
                print(f"\nError processing theta {i}: {e}")
                invalid_count += 1
                # Fallback with default volatility
                default_vol = self._get_process_default_volatility(theta)
                iv = torch.full(
                    (len(pricer.Ts), len(pricer.logKs)), 
                    default_vol, 
                    device=self.device
                )
                ivs.append(iv.cpu())
        
        iv_tensor = torch.stack(ivs).to(self.device)
        
        if show_progress:
            print(f"\nDataset statistics (before normalization):")
            print(f"  Process: {self.process.__class__.__name__}")
            print(f"  Shape: {iv_tensor.shape}")
            print(f"  IV range: [{iv_tensor.min():.4f}, {iv_tensor.max():.4f}]")
            print(f"  IV mean: {iv_tensor.mean():.4f}, std: {iv_tensor.std():.4f}")
            print(f"  Invalid samples: {invalid_count}")
        
        # Normalization
        if normalize:
            if compute_stats_from is None:
                # Calculate statistics from this dataset
                self.compute_normalization_stats(thetas, iv_tensor)
                
                if show_progress:
                    print(f"\nComputed normalization statistics:")
                    print(f"  Theta mean: {self.theta_mean}")
                    print(f"  Theta std: {self.theta_std}")
                    print(f"  IV mean: {self.iv_mean:.4f}, std: {self.iv_std:.4f}")
            else:
                # Use stats from another builder
                self.theta_mean = compute_stats_from.theta_mean
                self.theta_std = compute_stats_from.theta_std
                self.iv_mean = compute_stats_from.iv_mean
                self.iv_std = compute_stats_from.iv_std
                
                if show_progress:
                    print(f"\nUsing normalization statistics from training set")
            
            thetas_norm = self.normalize_theta(thetas)
            iv_tensor_norm = self.normalize_iv(iv_tensor)
            
            return thetas_norm, iv_tensor_norm
        else:
            return thetas, iv_tensor
    
    def build_grid_dataset_colab(self, pricer: GridNetworkPricer, n_samples, n_paths=30000, 
                                    batch_size=50, checkpoint_every=5,
                                    normalize=True, compute_stats_from=None,
                                    resume_from=None, mixed_precision=True, chunk_size: int = None,
                                    split: str = 'train'):
        """
        Optimized version for Google Colab with checkpoints and memory management.
        """
        # --- split dirs (train/val) ---
        split = split.lower()
        assert split in ('train', 'val'), "split must be 'train' or 'val'"
        ds_dir = (os.path.join(self.dirs['datasets'], split) 
                  if self.output_dir else os.path.join('./datasets', split))
        ckpt_dir = (os.path.join(self.dirs['checkpoints'], split) 
                    if self.output_dir else os.path.join('./checkpoints', split))
        os.makedirs(ds_dir, exist_ok=True); os.makedirs(ckpt_dir, exist_ok=True)
        # Check if the final dataset already exists
        final_dataset_path = f"{ds_dir}/{self.process_name}_grid_dataset_final.pkl"
        if os.path.exists(final_dataset_path):
            print(f"✓ Final dataset already exists for {self.process.__class__.__name__} ({split})!")
            return self._load_final_dataset(final_dataset_path, normalize, compute_stats_from)
        
        # Initialize checkpoint manager with process-specific prefix
        checkpoint_manager = CheckpointManager(
            ckpt_dir,
            prefix=f'{self.process_name}_grid_dataset'
        )
        
        # Resume from checkpoint if available
        start_idx = 0
        all_theta = []
        all_iv = []
        
        if resume_from and os.path.exists(resume_from):
            print(f"Resuming from checkpoint: {resume_from}")
            checkpoint = self._load_checkpoint(resume_from)
            all_theta = checkpoint['theta_list']
            all_iv = checkpoint['iv_list']
            start_idx = len(all_theta)
            print(f"Resuming from sample {start_idx}/{n_samples}")
            
            if start_idx >= n_samples:
                print("Already complete dataset!")
                return self._process_completed_dataset(
                    all_theta, all_iv, normalize, compute_stats_from, split=split
                )
        elif resume_from is None:
            pref = f"{self.process_name}_grid_dataset_checkpoint_"
            cand = sorted([f for f in os.listdir(ckpt_dir)
                        if f.startswith(pref) and f.endswith(".pkl")])
            if cand:
                resume_from = os.path.join(ckpt_dir, cand[-1])
                print(f"Resuming from latest checkpoint [{split}]: {os.path.basename(resume_from)}")
                checkpoint = self._load_checkpoint(resume_from)
                all_theta = checkpoint['theta_list']; all_iv = checkpoint['iv_list']
                start_idx = len(all_theta)
                print(f"Resuming from sample {start_idx}/{n_samples}")

        # Generate only the missing thetas
        remaining_samples = n_samples - start_idx
        if remaining_samples <= 0:
            return self._process_completed_dataset(
                    all_theta, all_iv, normalize, compute_stats_from, split=split
                )
        
        print(f"\nGenerating {remaining_samples} theta samples for {self.process.__class__.__name__}...")
        thetas = self.sample_theta_lhs(remaining_samples)
        
        # Get optimized MC parameters
        mc_params = self.get_process_specific_mc_params(base_n_paths=n_paths)
        
        # Process in batches
        print(f"Processing in batches of {batch_size}...")
        
        for batch_idx in range(0, len(thetas), batch_size):
            batch_end = min(batch_idx + batch_size, len(thetas))
            batch_thetas = thetas[batch_idx:batch_end]
            
            current_total = start_idx + batch_end
            print(f"\n[Batch {(batch_idx//batch_size)+1}] Samples {start_idx + batch_idx + 1}-{current_total}/{n_samples}")
            
            # Timer per batch
            batch_start_time = time.time()
            
            # Batch process with mixed precision
            iv_batch = []
            with torch.cuda.amp.autocast(enabled=mixed_precision and self.device.type == 'cuda'):
                for theta in tqdm(batch_thetas, desc=f"Computing {self.process_name} IV grids"):
                    try:
                        iv_grid = pricer._mc_iv_grid(
                            theta, 
                            n_paths=mc_params['n_paths'],
                            use_antithetic=mc_params['use_antithetic'],
                            adaptive_paths=mc_params['adaptive_paths'],
                            adaptive_dt=mc_params['adaptive_dt'],
                            control_variate=mc_params['control_variate'],
                            chunk_size=chunk_size
                        )
                        
                        # Check validity
                        if torch.isnan(iv_grid).any() or torch.isinf(iv_grid).any():
                            print(f"!  Invalid IV detected, using fallback")
                            default_vol = self._get_process_default_volatility(theta)
                            iv_grid = torch.full_like(iv_grid, default_vol)
                        elif (iv_grid < 0.01).any() or (iv_grid > 2.0).any():
                            print(f"!  Extreme IV values: [{iv_grid.min():.4f}, {iv_grid.max():.4f}]")
                            iv_grid = torch.clamp(iv_grid, 0.01, 2.0)
                        
                        iv_batch.append(iv_grid.cpu())
                        
                    except Exception as e:
                        print(f"X Error processing theta: {e}")
                        default_vol = self._get_process_default_volatility(theta)
                        iv_grid = torch.full(
                            (len(pricer.Ts), len(pricer.logKs)),
                            default_vol,
                            device=self.device
                        )
                        iv_batch.append(iv_grid.cpu())
                    
                    # Clear cache periodically
                    if len(iv_batch) % 10 == 0 and self.device.type == 'cuda':
                        torch.cuda.empty_cache()
            
            # Add to complete lists
            all_theta.extend([t.cpu() for t in batch_thetas])
            all_iv.extend(iv_batch)
            
            batch_time = time.time() - batch_start_time
            print(f"✓ Batch completed in {batch_time:.1f}s")
            
            # Periodic checkpoints
            if ((batch_idx // batch_size + 1) % checkpoint_every == 0 or 
                batch_end == len(thetas)):
                
                checkpoint_data = {
                    'theta_list': all_theta,
                    'iv_list': all_iv,
                    'n_samples_done': start_idx + batch_end,
                    'n_samples_total': n_samples,
                    'n_paths': n_paths,
                    'process': self.process.__class__.__name__,
                    'timestamp': datetime.now().isoformat()
                }
                
                checkpoint_name = f"{checkpoint_manager.prefix}_checkpoint_{start_idx + batch_end}.pkl"
                checkpoint_manager.save_checkpoint(checkpoint_data, checkpoint_name)
                
                # Clear GPU memory
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()
        
        # Process complete dataset
        return self._process_completed_dataset(
                    all_theta, all_iv, normalize, compute_stats_from, split=split
                )
    
    def _get_process_default_volatility(self, theta: torch.Tensor) -> float:
        """
        Gets a default volatility based on the process and parameters.
        """
        if hasattr(self.process, 'param_info'):
            param_info = self.process.param_info
            
            # Search for volatility-related parameters in priority order
            for vol_name in ['xi0', 'sigma', 'theta', 'vol']:
                if vol_name in param_info.names:
                    idx = param_info.names.index(vol_name)
                    value = theta[idx].item()
                    
                    # For variance parameters, take sqrt
                    if vol_name in ['xi0', 'theta']:
                        return math.sqrt(value)
                    else:
                        return value
            
            # If it's a jump model, use sigma if present
            if 'jump' in self.process.__class__.__name__.lower():
                if 'sigma' in param_info.names:
                    idx = param_info.names.index('sigma')
                    return theta[idx].item()
        
        # General default
        return 0.2
    
        
    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint with error handling"""
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
        
    def _load_final_dataset(self, path, normalize, compute_stats_from):
        """Load final dataset and apply normalization if necessary"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        theta_tensor = data['theta'].to(self.device)
        iv_tensor = data['iv'].to(self.device)
        
        if normalize:
            if compute_stats_from is None:
                # Calculate statistics
                self.compute_normalization_stats(theta_tensor, iv_tensor)
                self.save_normalization_stats()
            else:
                # Use existing statistics
                self._copy_stats_from(compute_stats_from)
            
            # Normalize
            theta_norm = self.normalize_theta(theta_tensor)
            iv_norm = self.normalize_iv(iv_tensor)
            return theta_norm, iv_norm
        
        return theta_tensor, iv_tensor
    
    def _process_completed_dataset(self, theta_list, iv_list, normalize, 
                                   compute_stats_from, cleanup_checkpoints=False,
                                   split: str = 'train'):
        """Process complete dataset: conversion, normalization, saving"""
        # Split folders
        split = split.lower()
        assert split in ('train', 'val')
        ds_dir = (os.path.join(self.dirs['datasets'], split)
                  if self.output_dir else os.path.join('./datasets', split))
        ckpt_dir = (os.path.join(self.dirs['checkpoints'], split)
                    if self.output_dir else os.path.join('./checkpoints', split))
        os.makedirs(ds_dir, exist_ok=True); os.makedirs(ckpt_dir, exist_ok=True)
        # Convert to tensors
        print("\nConverting to tensors...")
        theta_tensor = torch.stack(theta_list).to(self.device)
        iv_tensor = torch.stack(iv_list).to(self.device)
        
        print(f"Dataset shape - Theta: {theta_tensor.shape}, IV: {iv_tensor.shape}")
        
        # Save raw dataset if we have output_dir
        final_data = {
            'theta': theta_tensor.cpu(),
            'iv': iv_tensor.cpu(),
            'process': self.process.__class__.__name__,
            'timestamp': datetime.now().isoformat()
        }
        final_path = f"{ds_dir}/{self.process_name}_grid_dataset_final.pkl"
        with open(final_path, 'wb') as f:
            pickle.dump(final_data, f)
        print(f"✓ Raw dataset saved: {final_path}")
        
        # Cleanup checkpoints
        if cleanup_checkpoints:
            prefix = f"{self.process_name}_grid_dataset_checkpoint_"
            removed = 0
            for cp in os.listdir(ckpt_dir):
                if cp.startswith(prefix) and cp.endswith('.pkl'):
                    os.remove(os.path.join(ckpt_dir, cp)); removed += 1
            print(f"✓ Removed {removed} checkpoint files from {ckpt_dir}")
        
        # Normalization
        if normalize:
            if compute_stats_from is None:
                # Calculate new statistics (training set)
                self.compute_normalization_stats(theta_tensor, iv_tensor)
                if self.output_dir:
                    self.save_normalization_stats()
                
                print(f"\nNormalization statistics computed:")
                print(f"  Theta mean: {self.theta_mean}")
                print(f"  Theta std: {self.theta_std}")
                print(f"  IV mean: {self.iv_mean:.4f}, std: {self.iv_std:.4f}")
            else:
                # Use existing statistics (validation set)
                self._copy_stats_from(compute_stats_from)
                print(f"\nUsing normalization statistics from training set")
            
            # Apply normalization
            theta_norm = self.normalize_theta(theta_tensor)
            iv_norm = self.normalize_iv(iv_tensor)
            
            norm_data = {
                'theta_norm': theta_norm.cpu(),
                'iv_norm': iv_norm.cpu(),
                'theta_mean': self.theta_mean.cpu(),
                'theta_std': self.theta_std.cpu(),
                'iv_mean': self.iv_mean.cpu(),
                'iv_std': self.iv_std.cpu(),
                'timestamp': datetime.now().isoformat()
            }
            norm_path = f"{ds_dir}/{self.process_name}_grid_dataset_normalized.pkl"
            with open(norm_path, 'wb') as f:
                pickle.dump(norm_data, f)
            print(f"✓ Normalized dataset saved: {norm_path}")
            
            return theta_norm, iv_norm
        
        return theta_tensor, iv_tensor
    
    def _copy_stats_from(self, other_builder):
        """Copy normalization statistics from another builder"""
        self.theta_mean = other_builder.theta_mean
        self.theta_std = other_builder.theta_std
        self.iv_mean = other_builder.iv_mean
        self.iv_std = other_builder.iv_std
        self.T_mean = other_builder.T_mean
        self.T_std = other_builder.T_std
        self.k_mean = other_builder.k_mean
        self.k_std = other_builder.k_std
        
    def build_pointwise_dataset(self, pricer, 
                           thetas, maturities, logK, 
                           n_paths=30000, show_progress=True, normalize=True,
                           compute_stats_from=None):
        """
        Builds pointwise datasets with optional normalization.
        """
        n_theta = len(thetas)
        n_T = len(maturities)
        n_K = len(logK)
        total_points = n_theta * n_T * n_K
        
        if show_progress:
            print(f"Building pointwise dataset for {self.process.__class__.__name__}: {total_points} points")
        
        # Pre-allocate
        theta_pw = torch.empty(total_points, self.process.num_params, device=self.device)
        T_pw = torch.empty(total_points, device=self.device)
        k_pw = torch.empty(total_points, device=self.device)
        iv_pw = torch.empty(total_points, device=self.device)
        
        # Get optimized MC parameters
        mc_params = self.get_process_specific_mc_params(base_n_paths=n_paths)
        
        idx = 0
        iterator = tqdm(thetas, desc=f"Building {self.process.__class__.__name__} pointwise") if show_progress else thetas
        
        for i, theta in enumerate(iterator):
            try:
                # Calculate grid for this theta
                iv_grid = pricer._mc_iv_grid(
                    theta, maturities, logK, 
                    n_paths=mc_params['n_paths'],
                    use_antithetic=mc_params['use_antithetic'],
                    adaptive_dt=mc_params['adaptive_dt'],
                    control_variate=mc_params['control_variate']
                )
                
                # Check validity
                if torch.isnan(iv_grid).any() or torch.isinf(iv_grid).any():
                    print(f"\nWarning: Invalid IV for theta {i}")
                    default_vol = self._get_process_default_volatility(theta)
                    iv_grid = torch.full_like(iv_grid, default_vol)
                
                # Flatten
                for j, T in enumerate(maturities):
                    for k_idx, k in enumerate(logK):
                        theta_pw[idx] = theta
                        T_pw[idx] = T
                        k_pw[idx] = k
                        iv_pw[idx] = iv_grid[j, k_idx]
                        idx += 1
                        
            except Exception as e:
                print(f"\nError processing theta {i}: {e}")
                default_vol = self._get_process_default_volatility(theta)
                
                # Fill con default
                for j, T in enumerate(maturities):
                    for k_idx, k in enumerate(logK):
                        theta_pw[idx] = theta
                        T_pw[idx] = T
                        k_pw[idx] = k
                        iv_pw[idx] = default_vol
                        idx += 1
        
        if normalize:
            if compute_stats_from is None:
                # Calculate statistics
                self.compute_normalization_stats(theta_pw, iv_pw, T_pw, k_pw)
                
                if show_progress:
                    print(f"\nComputed normalization statistics for {self.process.__class__.__name__}:")
                    print(f"  Theta mean: {self.theta_mean}")
                    print(f"  Theta std: {self.theta_std}")
                    print(f"  T mean: {self.T_mean:.4f}, std: {self.T_std:.4f}")
                    print(f"  k mean: {self.k_mean:.4f}, std: {self.k_std:.4f}")
                    print(f"  IV mean: {self.iv_mean:.4f}, std: {self.iv_std:.4f}")
                    
                if self.output_dir:
                    self.save_normalization_stats()
            else:
                # Use existing statistics
                self._copy_stats_from(compute_stats_from)
                
                if show_progress:
                    print(f"\nUsing normalization statistics from training set")
            
            # Normalize
            theta_pw_norm = self.normalize_theta(theta_pw)
            T_pw_norm = self.normalize_T(T_pw)
            k_pw_norm = self.normalize_k(k_pw)
            iv_pw_norm = self.normalize_iv(iv_pw)
            
            return theta_pw_norm, T_pw_norm, k_pw_norm, iv_pw_norm
        else:
            return theta_pw, T_pw, k_pw, iv_pw
    
    def build_pointwise_dataset_colab(self, pricer: PointwiseNetworkPricer, 
                                     thetas, maturities, logK,
                                     n_paths=30000, batch_size=50,
                                     checkpoint_every=5, show_progress=True,
                                     normalize=True, compute_stats_from=None,
                                     resume_from=None, mixed_precision=True,
                                     chunk_size: int = None,
                                     split: str = 'train'):
        """
        Colab-optimized version of build_pointwise_dataset.
        """
        # --- split dirs (train/val) ---
        split = split.lower()
        assert split in ('train', 'val'), "split must be 'train' or 'val'"
        ds_dir = (os.path.join(self.dirs['datasets'], split) 
                  if self.output_dir else os.path.join('./datasets', split))
        ckpt_dir = (os.path.join(self.dirs['checkpoints'], split) 
                    if self.output_dir else os.path.join('./checkpoints', split))
        os.makedirs(ds_dir, exist_ok=True); os.makedirs(ckpt_dir, exist_ok=True)

        n_theta = len(thetas)
        n_T = len(maturities)
        n_K = len(logK)
        total_points = n_theta * n_T * n_K
        
        # Check if the final dataset already exists
        if self.output_dir:
            final_dataset_path = f"{ds_dir}/{self.process_name}_pointwise_dataset_final.pkl"
            if os.path.exists(final_dataset_path) and not resume_from:
                print(f"✓ Dataset finale già esistente per {self.process.__class__.__name__}!")
                return self._load_final_pointwise_dataset(
                    final_dataset_path, normalize, compute_stats_from
                )
        
        # Initialize checkpoint manager
        checkpoint_manager = CheckpointManager(ckpt_dir, prefix=f'{self.process_name}_pointwise_dataset')
        
        # Resume from checkpoint if available
        start_idx = 0
        all_data = {'theta': [], 'T': [], 'k': [], 'iv': []}
        
        if resume_from and os.path.exists(resume_from):
            print(f"Resuming from checkpoint: {resume_from}")
            checkpoint = self._load_checkpoint(resume_from)
            all_data = checkpoint['data']
            start_idx = checkpoint['n_theta_done']
            print(f"Resuming from theta {start_idx}/{n_theta}")
            
            if start_idx >= n_theta:
                print("Dataset già completo!")
                return self._process_completed_pointwise_dataset(
                    all_data, normalize, compute_stats_from, split=split
                )
        elif resume_from is None:
            # Auto-resume: last pointwise checkpoint in the current split
            pref = f"{self.process_name}_pointwise_dataset_checkpoint_"
            cand = sorted([f for f in os.listdir(ckpt_dir)
                           if f.startswith(pref) and f.endswith(".pkl")])
            if cand:
                resume_from = os.path.join(ckpt_dir, cand[-1])
                print(f"Resuming from latest checkpoint [{split}]: {os.path.basename(resume_from)}")
                checkpoint = self._load_checkpoint(resume_from)
                all_data = checkpoint['data']
                start_idx = checkpoint.get('n_theta_done', 0)
                print(f"Resuming from theta {start_idx}/{n_theta}")

        # Process remaining thetas
        remaining_thetas = thetas[start_idx:]
        if show_progress:
            print(f"\nGenerating pointwise dataset for {self.process.__class__.__name__}: {total_points} total points")
            print(f"  Remaining thetas to process: {len(remaining_thetas)}")
        
        # Get optimized MC parameters
        mc_params = self.get_process_specific_mc_params(base_n_paths=n_paths)
        
        # Process in batches
        for batch_idx in range(0, len(remaining_thetas), batch_size):
            batch_end = min(batch_idx + batch_size, len(remaining_thetas))
            batch_thetas = remaining_thetas[batch_idx:batch_end]
            current_idx = start_idx + batch_idx
            print(f"\n[Batch {(batch_idx//batch_size)+1}] Processing theta {current_idx+1}-{current_idx+len(batch_thetas)}/{n_theta}")
            batch_start_time = time.time()
            
            with torch.cuda.amp.autocast(enabled=mixed_precision and self.device.type == 'cuda'):
                iterator = tqdm(batch_thetas, desc=f"Computing {self.process_name} IV grids") if show_progress else batch_thetas
                for theta in iterator:
                    try:
                        iv_grid = pricer._mc_iv_grid(
                            theta, maturities, logK, 
                            n_paths=mc_params['n_paths'],
                            use_antithetic=mc_params['use_antithetic'],
                            adaptive_paths=mc_params['adaptive_paths'],
                            adaptive_dt=mc_params['adaptive_dt'],
                            control_variate=mc_params['control_variate'],
                            chunk_size=chunk_size
                        )
                        
                        if torch.isnan(iv_grid).any() or torch.isinf(iv_grid).any():
                            print(f"Invalid IV detected, using fallback")
                            default_vol = self._get_process_default_volatility(theta)
                            iv_grid = torch.full_like(iv_grid, default_vol)
                        elif (iv_grid < 0.01).any() or (iv_grid > 2.0).any():
                            print(f"Extreme IV values, clamping")
                            iv_grid = torch.clamp(iv_grid, 0.01, 2.0)
                        
                        # Check repair stats if available
                        if hasattr(pricer, 'last_repair_stats') and pricer.last_repair_stats:
                            repair_ratio = pricer.last_repair_stats['repair_ratio']
                            if repair_ratio > 0.2:
                                print(f"High repair ratio: {repair_ratio:.1%}")
                        
                        for i, T in enumerate(maturities):
                            for j, k in enumerate(logK):
                                all_data['theta'].append(theta.cpu())
                                all_data['T'].append(T.cpu())
                                all_data['k'].append(k.cpu())
                                all_data['iv'].append(iv_grid[i, j].cpu())
                                
                    except Exception as e:
                        print(f"Error processing theta: {e}")
                        default_vol = self._get_process_default_volatility(theta)
                        for i, T in enumerate(maturities):
                            for j, k in enumerate(logK):
                                all_data['theta'].append(theta.cpu())
                                all_data['T'].append(T.cpu())
                                all_data['k'].append(k.cpu())
                                all_data['iv'].append(torch.tensor(default_vol))
                    
                    if self.device.type == 'cuda' and len(batch_thetas) % 10 == 0:
                        torch.cuda.empty_cache()
            
            batch_time = time.time() - batch_start_time
            points_generated = len(batch_thetas) * n_T * n_K
            print(f"✓ Batch completed in {batch_time:.1f}s ({points_generated} points)")
            
            if ((batch_idx // batch_size + 1) % checkpoint_every == 0 or batch_end == len(remaining_thetas)):
                checkpoint_data = {
                    'data': all_data,
                    'n_theta_done': start_idx + batch_end,
                    'n_theta_total': n_theta,
                    'n_paths': n_paths,
                    'maturities': maturities.cpu(),
                    'logK': logK.cpu(),
                    'process': self.process.__class__.__name__,
                    'timestamp': datetime.now().isoformat()
                }
                checkpoint_name = f"{checkpoint_manager.prefix}_checkpoint_{start_idx + batch_end}.pkl"
                checkpoint_manager.save_checkpoint(checkpoint_data, checkpoint_name)
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()
        
        return self._process_completed_pointwise_dataset(
            all_data, normalize, compute_stats_from, cleanup_checkpoints=True, split=split
        )

    def _process_completed_pointwise_dataset(self, all_data, normalize,
                                           compute_stats_from, cleanup_checkpoints=False,
                                           split: str = 'train'):
        """Process complete dataset: conversion, normalization, saving"""
        print("\nConverting to tensors...")
        theta_tensor = torch.stack(all_data['theta']).to(self.device)
        T_tensor = torch.stack(all_data['T']).to(self.device)
        k_tensor = torch.stack(all_data['k']).to(self.device)
        iv_tensor = torch.stack(all_data['iv']).to(self.device)
        
        print(f"Dataset shape: {theta_tensor.shape[0]} points")
        print(f"Unique thetas: {len(torch.unique(theta_tensor, dim=0))}")
    
        # split folders
        split = split.lower()
        assert split in ('train','val')
        ds_dir = (os.path.join(self.dirs['datasets'], split)
                  if self.output_dir else os.path.join('./datasets', split))
        ckpt_dir = (os.path.join(self.dirs['checkpoints'], split)
                    if self.output_dir else os.path.join('./checkpoints', split))
        os.makedirs(ds_dir, exist_ok=True); os.makedirs(ckpt_dir, exist_ok=True)

        if self.output_dir:
            final_data = {
                'theta': theta_tensor.cpu(),
                'T': T_tensor.cpu(),
                'k': k_tensor.cpu(),
                'iv': iv_tensor.cpu(),
                'timestamp': datetime.now().isoformat()
            }
            final_path = f"{ds_dir}/{self.process_name}_pointwise_dataset_final.pkl"
            with open(final_path, 'wb') as f:
                pickle.dump(final_data, f)
            print(f"✓ Raw dataset saved: {final_path}")
            
            if cleanup_checkpoints:
                prefix = f"{self.process_name}_pointwise_dataset_checkpoint_"
                removed = 0
                for cp in os.listdir(ckpt_dir):
                    if cp.startswith(prefix) and cp.endswith('.pkl'):
                        os.remove(os.path.join(ckpt_dir, cp)); removed += 1
                if removed:
                    print(f"✓ Removed {removed} checkpoint files from {ckpt_dir}")
    
        if normalize:
            if compute_stats_from is None:
                self.compute_normalization_stats(theta_tensor, iv_tensor, T_tensor, k_tensor)
                if self.output_dir:
                    self.save_normalization_stats()
                print(f"\nNormalization statistics computed and saved.")
            else:
                self._copy_stats_from(compute_stats_from)
                print(f"\nUsing normalization statistics from training set.")
    
            theta_norm = self.normalize_theta(theta_tensor)
            T_norm = self.normalize_T(T_tensor)
            k_norm = self.normalize_k(k_tensor)
            iv_norm = self.normalize_iv(iv_tensor)
    
            if self.output_dir:
                norm_data = {
                    'theta_norm': theta_norm.cpu(),
                    'T_norm': T_norm.cpu(),
                    'k_norm': k_norm.cpu(),
                    'iv_norm': iv_norm.cpu(),
                    'theta_mean': self.theta_mean.cpu(),
                    'theta_std': self.theta_std.cpu(),
                    'T_mean': self.T_mean.cpu(),
                    'T_std': self.T_std.cpu(),
                    'k_mean': self.k_mean.cpu(),
                    'k_std': self.k_std.cpu(),
                    'iv_mean': self.iv_mean.cpu(),
                    'iv_std': self.iv_std.cpu(),
                    'timestamp': datetime.now().isoformat()
                }
                norm_path = f"{ds_dir}/{self.process_name}_pointwise_dataset_normalized.pkl"
                with open(norm_path, 'wb') as f:
                    pickle.dump(norm_data, f)
                print(f"✓ Normalized dataset saved: {norm_path}")
            
            return theta_norm, T_norm, k_norm, iv_norm
        
        return theta_tensor, T_tensor, k_tensor, iv_tensor
    
    def _load_final_pointwise_dataset(self, path, normalize, compute_stats_from):
        """Load final dataset and apply normalization if necessary"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        theta_tensor = data['theta'].to(self.device)
        T_tensor = data['T'].to(self.device)
        k_tensor = data['k'].to(self.device)
        iv_tensor = data['iv'].to(self.device)
    
        if normalize:
            if compute_stats_from is None:
                self.compute_normalization_stats(theta_tensor, iv_tensor, T_tensor, k_tensor)
                self.save_normalization_stats()
            else:
                self._copy_stats_from(compute_stats_from)
            
            theta_norm = self.normalize_theta(theta_tensor)
            T_norm = self.normalize_T(T_tensor)
            k_norm = self.normalize_k(k_tensor)
            iv_norm = self.normalize_iv(iv_tensor)
            
            return theta_norm, T_norm, k_norm, iv_norm
        
        return theta_tensor, T_tensor, k_tensor, iv_tensor
    

    # ---------------- Random Grid / Random Smiles (Baschetti et al.) ----------------
    # Temporal buckets (Eq. 2) and strikes parameters (Eq. 3)
    _maturity_buckets = [
        (0.003, 0.030), (0.030, 0.090), (0.090, 0.150), (0.150, 0.300),
        (0.300, 0.500), (0.500, 0.750), (0.750, 1.000), (1.000, 1.250),
        (1.250, 1.500), (1.500, 2.000), (2.000, 2.500)
    ]
    _strike_params = dict(l=0.55, u=0.30, n_left_tail=4, n_center=7, n_right_tail=2, center_width=0.20)

    def _sample_random_maturities(self, n_maturities: int = 11, buckets=None, seed: int = None):
        """Sample random maturities from the buckets and sort them"""
        if buckets is None:
            buckets = self._maturity_buckets[:n_maturities]
        rng = np.random.default_rng(seed)
        mats = []
        if n_maturities >= len(buckets):
            for lo, hi in buckets:
                mats.append(rng.uniform(lo, hi))
            for _ in range(n_maturities - len(buckets)):
                lo, hi = buckets[rng.integers(len(buckets))]
                mats.append(rng.uniform(lo, hi))
        else:
            idx = rng.choice(len(buckets), n_maturities, replace=False)
            for i in idx:
                lo, hi = buckets[i]
                mats.append(rng.uniform(lo, hi))
        mats.sort()
        return torch.tensor(mats, dtype=torch.float32, device=self.device)

    def _sample_random_strikes(self, T: float, spot: float = 1.0, n_strikes: int = 13):
        """Sample random strikes respecting the typical granularity of the market"""
        p = self._strike_params
        sqrt_T = float(np.sqrt(T))
        K_min = spot * (1 - p['l'] * sqrt_T)
        K_max = spot * (1 + p['u'] * sqrt_T)
        # Guardrail for numerically problematic extremes
        K_min = max(K_min, 0.05 * spot)
        K_max = min(K_max, 3.00 * spot)
        center_lower = spot * (1 - p['center_width'] * sqrt_T)
        center_upper = spot * (1 + p['center_width'] * sqrt_T)

        n_left, n_center, n_right = p['n_left_tail'], p['n_center'], p['n_right_tail']
        tot_default = n_left + n_center + n_right
        if n_strikes != tot_default:
            # redistribute proportionally, ensuring at least 1 in each zone
            n_left = max(1, int(round(p['n_left_tail'] * n_strikes / tot_default)))
            n_center = max(1, int(round(p['n_center'] * n_strikes / tot_default)))
            n_right = max(1, n_strikes - n_left - n_center)

        rng = np.random.default_rng()
        strikes = []
        if n_left > 0 and K_min < center_lower:
            strikes.extend(rng.uniform(K_min, center_lower, n_left))
        if n_center > 0:
            strikes.extend(rng.uniform(center_lower, center_upper, n_center))
        if n_right > 0 and center_upper < K_max:
            strikes.extend(rng.uniform(center_upper, K_max, n_right))
        strikes = np.clip(np.sort(strikes), K_min, K_max)
        return torch.tensor(strikes, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def _generate_random_grid(self, theta: torch.Tensor, n_maturities: int = 11,
                              n_strikes_per_maturity: int = 13, spot: float = 1.0,
                              mc_params: dict = None):
        """Generates a random IV(T,K) grid for a set of theta parameters"""
        if mc_params is None:
            mc_params = self.get_process_specific_mc_params()
        mats = self._sample_random_maturities(n_maturities)
        all_strikes, iv_grid = [], []
        for T in mats:
            strikes = self._sample_random_strikes(float(T.item()), spot, n_strikes_per_maturity)
            logK = torch.log(strikes / spot)
            pricer = GridNetworkPricer(
                maturities=T.unsqueeze(0),
                logK=logK,
                process=self.process,
                device=self.device,
                r=0.0,
                enable_smile_repair=True,
                smile_repair_method='pchip'
            )
            iv_smile = pricer._mc_iv_grid(
                theta=theta,
                n_paths=mc_params.get('n_paths', 30000),
                spot=spot,
                use_antithetic=mc_params.get('use_antithetic', True),
                adaptive_paths=mc_params.get('adaptive_paths', False),
                adaptive_dt=mc_params.get('adaptive_dt', True),
                control_variate=mc_params.get('control_variate', True)
            ).squeeze(0)
            all_strikes.append(strikes)
            iv_grid.append(iv_smile)
        return {'maturities': mats, 'strikes': all_strikes, 'iv_grid': torch.stack(iv_grid)}

    def build_random_grids_dataset(self, n_surfaces: int = 10000, n_maturities: int = 11,
                                         n_strikes: int = 13, n_paths: int = 30000, 
                                         spot: float = 1.0, normalize: bool = True, 
                                         compute_stats_from=None, show_progress: bool = True,
                                         batch_size: int = 50, checkpoint_every: int = 100,
                                         resume_from: str = None, save_all_thetas: bool = True,
                                         base_seed: int = 42, split: str | None = None):
        """
        Build pointwise datasets with Random Grids approach using optimized regime-based simulation.
        This version groups maturities by dt regime and reuses simulations.
        """        
        # Setup checkpoint directory
        split = (split or getattr(self, 'dataset_type', 'train')).lower()
        assert split in ('train','val'), "split must be 'train' or 'val'"
        base_ckpt = (self.dirs['checkpoints'] if getattr(self, 'dirs', None) and 'checkpoints' in self.dirs
                     else (os.path.join(self.output_dir, 'checkpoints') if self.output_dir else './checkpoints'))
        checkpoint_dir = os.path.join(base_ckpt, 'random_grids', split)

        os.makedirs(checkpoint_dir, exist_ok=True)
        proc_slug = getattr(self.process.__class__, "__name__", "process").lower()
        # Check for existing checkpoint
        start_idx = 0
        all_theta, all_T, all_k, all_iv = [], [], [], []
        thetas_total = None
        
        if resume_from == 'latest':
            # Find the last checkpoint in the train/val subfolder (numeric sort order)
            pattern = rf'^{proc_slug}_pointwise_checkpoint_(\d+)(?:\.pkl)?$'
            cands = [f for f in os.listdir(checkpoint_dir) if re.match(pattern, f)]
            # Fallback for backward compatibility: also search the parent (without split) if you don't find anything
            if not cands:
                parent_dir = os.path.dirname(checkpoint_dir)  # .../random_grids
                if os.path.isdir(parent_dir):
                    cands = [f for f in os.listdir(parent_dir) if re.match(pattern, f)]
                    if cands:
                        cands.sort(key=lambda n: int(re.match(pattern, n).group(1)))
                        resume_from = os.path.join(parent_dir, cands[-1])
            if cands:
                cands.sort(key=lambda n: int(re.match(pattern, n).group(1)))
                resume_from = os.path.join(checkpoint_dir, cands[-1])
        
            print(f"\nBuilding Random Grids Pointwise Dataset: {n_surfaces} × {n_maturities} × {n_strikes}")
            print(f"Starting from surface {start_idx} [{split}]")

        if resume_from and os.path.exists(resume_from):
            print(f"Resuming from checkpoint: {resume_from}")
            with open(resume_from, 'rb') as f:
                checkpoint = pickle.load(f)
            # Tensors on CPU for portability
            all_theta = [t.cpu() for t in checkpoint['theta']]
            all_T     = [t.cpu() for t in checkpoint['T']]
            all_k     = [t.cpu() for t in checkpoint['k']]
            all_iv    = [t.cpu() for t in checkpoint['iv']]
            if save_all_thetas and ('thetas_total' in checkpoint) and isinstance(checkpoint['thetas_total'], torch.Tensor):
                thetas_total = checkpoint['thetas_total'].cpu()
            start_idx = checkpoint['n_surfaces_done']
            print(f"Resuming from surface {start_idx}/{n_surfaces}")
            
            if start_idx >= n_surfaces:
                print("Dataset already complete!")
                return self._finalize_random_grids_dataset(
                    all_theta, all_T, all_k, all_iv, normalize, compute_stats_from
                )
        
        print(f"\nBuilding Random Grids Pointwise Dataset: {n_surfaces} × {n_maturities} × {n_strikes}")
        print(f"Starting from surface {start_idx}")
        
        mc_params = self.get_process_specific_mc_params(base_n_paths=n_paths)
        
        # Build the tail of the full theta sequence robustly (resume-safe)
        def _ensure_full_thetas_total(t_total, n, seed):
            # If missing or too short, rebuild deterministically
            if (t_total is None) or (isinstance(t_total, torch.Tensor) and t_total.size(0) < n):
                full = self.sample_theta_lhs(n, seed=base_seed).cpu()
                if isinstance(t_total, torch.Tensor) and t_total.numel() > 0:
                    keep = min(t_total.size(0), full.size(0))
                    full[:keep] = t_total[:keep]
                return full
            return t_total

        thetas_total = _ensure_full_thetas_total(thetas_total, n_surfaces, base_seed)
        tail = thetas_total[start_idx:n_surfaces]
        if tail.numel() == 0 and start_idx < n_surfaces:
            # Extreme fallback: generate only the missing queue
            tail = self.sample_theta_lhs(n_surfaces - start_idx).cpu()
        thetas = tail.to(self.device)
        
        iterator = range(start_idx, n_surfaces)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Generating random grids", initial=start_idx, total=n_surfaces)
            except Exception:
                pass
        
        for i in iterator:
            if i < start_idx:
                continue  # Skip already processed
                
            theta = thetas[i - start_idx]
            grid = self._generate_random_grid(theta, n_maturities, n_strikes, spot, mc_params)
            
            for j, T in enumerate(grid['maturities']):
                strikes = grid['strikes'][j]
                logK = torch.log(strikes / spot)
                iv_smile = grid['iv_grid'][j]
                # Store to CPU to keep checkpoints light/safe
                all_theta.extend([theta.detach().cpu()] * len(strikes))
                all_T.extend([T.detach().cpu()] * len(strikes))
                all_k.extend([lk.detach().cpu() for lk in logK])
                all_iv.extend([iv.detach().cpu() for iv in iv_smile])
            
            # Save checkpoint periodically
            if (i + 1) % checkpoint_every == 0 or (i + 1) == n_surfaces:
                checkpoint_data = {
                    'theta': all_theta,
                    'T': all_T,
                    'k': all_k,
                    'iv': all_iv,
                    'n_surfaces_done': i + 1,
                    'n_surfaces_total': n_surfaces,
                    'timestamp': datetime.now().isoformat()
                }
                checkpoint_path = f"{checkpoint_dir}/random_grids_checkpoint_{i+1}.pkl"
                if save_all_thetas:
                    checkpoint_data['thetas_total'] = thetas_total
                # File name: <processclass>_pointwise_checkpoint_<N>.pkl
                checkpoint_path = f"{checkpoint_dir}/{proc_slug}_pointwise_checkpoint_{i+1}.pkl"
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                print(f"\n✓ Checkpoint saved: {i+1}/{n_surfaces} surfaces")
            
            if (i + 1) % 10 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return self._finalize_random_grids_dataset(
            all_theta, all_T, all_k, all_iv, normalize, compute_stats_from
        )

    def _finalize_random_grids_dataset(self, all_theta, all_T, all_k, all_iv, 
                                    normalize, compute_stats_from):
        """Finalize the dataset after generation."""
        # Stack CPU lists and realign to builder device
        theta_pw = torch.stack(all_theta).to(self.device, non_blocking=True)
        T_pw     = torch.stack(all_T).to(self.device, non_blocking=True)
        k_pw     = torch.stack(all_k).to(self.device, non_blocking=True)
        iv_pw    = torch.stack(all_iv).to(self.device, non_blocking=True)
        
        if normalize:
            if compute_stats_from is None:
                self.compute_normalization_stats(theta_pw, iv_pw, T_pw, k_pw)
                if self.output_dir:
                    self.save_normalization_stats()
            else:
                self._copy_stats_from(compute_stats_from)
            
            return (self.normalize_theta(theta_pw), self.normalize_T(T_pw),
                    self.normalize_k(k_pw), self.normalize_iv(iv_pw))
        
        return theta_pw, T_pw, k_pw, iv_pw

    def build_random_smiles_dataset(self,
                                    n_smiles: int = 50000,
                                    n_strikes_per_smile: int = 13,
                                    n_paths: int = 30000,
                                    spot: float = 1.0,
                                    normalize: bool = True,
                                    compute_stats_from=None,
                                    show_progress: bool = True,
                                    checkpoint_every: int = 100,
                                    resume_from: str | None = None,
                                    base_seed: int = 42,
                                    split: str | None = None,
                                    save_all_thetas: bool = True):
        """
        Light variant: a random T for each theta (now with checkpoint + resume).
        Checkpoint path: <output_dir>/checkpoints/random_smiles/<train|val>/<proc>_smiles_checkpoint_<N>.pkl
        """
        # ----------------------------
        # Setup checkpoint directory
        # ----------------------------
        split = (split or getattr(self, 'dataset_type', 'val')).lower()
        assert split in ('train', 'val'), "split must be 'train' or 'val'"
        base_ckpt = (self.dirs['checkpoints'] if getattr(self, 'dirs', None) and 'checkpoints' in self.dirs
                     else (os.path.join(self.output_dir, 'checkpoints') if self.output_dir else './checkpoints'))
        checkpoint_dir = os.path.join(base_ckpt, 'random_smiles', split)
        os.makedirs(checkpoint_dir, exist_ok=True)
        proc_slug = getattr(self.process.__class__, "__name__", "process").lower()

        # State
        start_idx = 0
        all_theta, all_T, all_k, all_iv = [], [], [], []
        thetas_total = None

        # ----------------------------
        # Resolve 'latest' resume
        # ----------------------------
        if resume_from == 'latest':
            pattern = rf'^{proc_slug}_smiles_checkpoint_(\d+)(?:\.pkl)?$'
            cands = [f for f in os.listdir(checkpoint_dir) if re.match(pattern, f)]
            # fallback: also search in the parent (without split) if empty
            if not cands:
                parent_dir = os.path.dirname(checkpoint_dir)  # .../random_smiles
                if os.path.isdir(parent_dir):
                    cands = [f for f in os.listdir(parent_dir) if re.match(pattern, f)]
                    if cands:
                        cands.sort(key=lambda n: int(re.match(pattern, n).group(1)))
                        resume_from = os.path.join(parent_dir, cands[-1])
            if cands:
                cands.sort(key=lambda n: int(re.match(pattern, n).group(1)))
                resume_from = os.path.join(checkpoint_dir, cands[-1])

        # ----------------------------
        # Load checkpoint if provided
        # ----------------------------
        if resume_from and os.path.exists(resume_from):
            print(f"\nResuming Random Smiles from checkpoint: {resume_from}")
            with open(resume_from, 'rb') as f:
                checkpoint = pickle.load(f)
            all_theta = [t.cpu() for t in checkpoint.get('theta', [])]
            all_T     = [t.cpu() for t in checkpoint.get('T', [])]
            all_k     = [t.cpu() for t in checkpoint.get('k', [])]
            all_iv    = [t.cpu() for t in checkpoint.get('iv', [])]
            if save_all_thetas and ('thetas_total' in checkpoint) and isinstance(checkpoint['thetas_total'], torch.Tensor):
                thetas_total = checkpoint['thetas_total'].cpu()
            start_idx = int(checkpoint.get('n_smiles_done', 0))
            if start_idx >= n_smiles:
                print("Random Smiles dataset already complete!")
                return self._finalize_random_grids_dataset(all_theta, all_T, all_k, all_iv,
                                                           normalize, compute_stats_from)

        print(f"\nBuilding Random Smiles Dataset: {n_smiles} × {n_strikes_per_smile}")
        print(f"Starting from smile {start_idx} [{split}]")
        mc_params = self.get_process_specific_mc_params(base_n_paths=n_paths)

        # Pre-sample deterministic LHS for reproducibility (and resume)
        # Make resume robust to missing/short thetas_total in old checkpoints
        def _ensure_full_thetas_total(t_total, n, seed):
            # Rebuild full sequence if missing or shorter than requested
            if (t_total is None) or (isinstance(t_total, torch.Tensor) and t_total.size(0) < n):
                full = self.sample_theta_lhs(n, seed=base_seed).cpu()
                if isinstance(t_total, torch.Tensor) and t_total.numel() > 0:
                    keep = min(t_total.size(0), full.size(0))
                    full[:keep] = t_total[:keep]
                return full
            return t_total

        thetas_total = _ensure_full_thetas_total(thetas_total, n_smiles, base_seed)
        tail = thetas_total[start_idx:n_smiles]
        # Extreme fallback if tail is still empty (old/truncated checkpoint)
        if tail.numel() == 0 and start_idx < n_smiles:
            tail = self.sample_theta_lhs(n_smiles - start_idx, seed=base_seed + 1).cpu()
        thetas = tail.to(self.device)

        iterator = range(start_idx, n_smiles)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Generating random smiles",
                                initial=start_idx, total=n_smiles)
            except Exception:
                pass

        for i in iterator:
            lo, hi = self._maturity_buckets[np.random.randint(len(self._maturity_buckets))]
            T = float(np.random.uniform(lo, hi))
            T_t = torch.tensor(T, dtype=torch.float32, device=self.device)
            strikes = self._sample_random_strikes(T, spot, n_strikes_per_smile)
            logK = torch.log(strikes / spot)
            # current theta
            theta = thetas[i - start_idx]
            pricer = GridNetworkPricer(
                maturities=T_t.unsqueeze(0),
                logK=logK,
                process=self.process,
                device=self.device,
                r=0.0,
                enable_smile_repair=True,
                smile_repair_method='pchip'
            )
            iv_smile = pricer._mc_iv_grid(
                theta=theta,
                n_paths=mc_params.get('n_paths', 30000),
                spot=spot,
                use_antithetic=mc_params.get('use_antithetic', True),
                adaptive_paths=mc_params.get('adaptive_paths', False),
                adaptive_dt=mc_params.get('adaptive_dt', True),
                control_variate=mc_params.get('control_variate', True)
            ).squeeze(0)

            # Accumulate on CPU for portable checkpoints
            theta_cpu  = theta.detach().cpu()
            T_cpu      = T_t.detach().cpu()
            logK_cpu   = logK.detach().cpu()
            iv_cpu     = iv_smile.detach().cpu()
            all_theta.extend([theta_cpu] * len(strikes))
            all_T.extend([T_cpu] * len(strikes))
            all_k.extend(list(logK_cpu))
            all_iv.extend(list(iv_cpu))

            # ----------------------------
            # Periodic checkpoint
            # ----------------------------
            smiles_done = (i + 1)
            if checkpoint_every and (smiles_done % checkpoint_every == 0):
                ckpt_path = os.path.join(
                    checkpoint_dir, f"{proc_slug}_smiles_checkpoint_{smiles_done}.pkl"
                )
                tmp = {
                    'theta': all_theta,
                    'T': all_T,
                    'k': all_k,
                    'iv': all_iv,
                    'n_smiles_done': smiles_done,
                }
                if save_all_thetas and isinstance(thetas_total, torch.Tensor):
                    tmp['thetas_total'] = thetas_total
                with open(ckpt_path, 'wb') as f:
                    pickle.dump(tmp, f)
                print(f"[checkpoint] Saved: {ckpt_path}")

            if (i + 1) % 100 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()

        theta_pw = torch.stack(all_theta); T_pw = torch.stack(all_T)
        k_pw = torch.stack(all_k); iv_pw = torch.stack(all_iv)
        return self._finalize_random_grids_dataset(all_theta, all_T, all_k, all_iv,
                                                   normalize, compute_stats_from)
    
class MultiRegimeDatasetBuilder(DatasetBuilder):
    """
    Extends DatasetBuilder to handle multi-regime datasets with generic processes.
    """
    
    def __init__(self, process: Union[str, StochasticProcess], 
                 device='cpu', output_dir=None, dataset_type=None,
                 enable_smile_repair: bool = True,
                 smile_repair_method: str = 'pchip'):
        
        super().__init__(process, device, output_dir)
        
        # Separate statistics by regime
        self.regime_stats = {
            'short': {'iv_mean': None, 'iv_std': None},
            'mid': {'iv_mean': None, 'iv_std': None},
            'long': {'iv_mean': None, 'iv_std': None}
        }
        
        # Additional directories for multi-regime
        if self.output_dir:
            process_name = self.process.__class__.__name__.lower()
            self.regime_dirs = {
                'short': f"{self.output_dir}/datasets/{process_name}_short_term",
                'mid':   f"{self.output_dir}/datasets/{process_name}_mid_term",
                'long':  f"{self.output_dir}/datasets/{process_name}_long_term",
                'unified': f"{self.output_dir}/datasets/{process_name}_unified"
            }
            for dir_path in self.regime_dirs.values():
                os.makedirs(dir_path, exist_ok=True)
                
        self.enable_smile_repair = enable_smile_repair
        self.smile_repair_method = smile_repair_method

        # Normalize dataset_type → "train" | "val"
        self.dataset_type = (dataset_type or "train").lower()
        if self.dataset_type in ("validation", "valid"):
            self.dataset_type = "val"
        assert self.dataset_type in ("train", "val"), "dataset_type must be 'train' or 'val'"

        # Add subfolders per phase (train/val) to all output directories
        if self.output_dir:
            # checkpoints/<phase>
            self.dirs['checkpoints'] = os.path.join(self.dirs['checkpoints'], self.dataset_type)
            os.makedirs(self.dirs['checkpoints'], exist_ok=True)
            # regime dirs /<phase>
            for k in ['short','mid','long','unified']:
                self.regime_dirs[k] = os.path.join(self.regime_dirs[k], self.dataset_type)
                os.makedirs(self.regime_dirs[k], exist_ok=True)
        else:
            # local fallback
            self.dirs['checkpoints'] = os.path.join("./checkpoints", self.dataset_type)
            os.makedirs(self.dirs['checkpoints'], exist_ok=True)
            for k in ['short','mid','long','unified']:
                base = f"./{k}" if k != 'unified' else "./unified"
                self.regime_dirs[k] = os.path.join(base, self.dataset_type)
                os.makedirs(self.regime_dirs[k], exist_ok=True)
    
    def compute_regime_normalization_stats(self, thetas, iv_short, iv_mid, iv_long):
        """
        Calculate normalization statistics for each regime.

        Args:
        thetas: parameters common to all regimes
        iv_short: IV of the short-term regime
        iv_mid: IV of the mid-term regime
        iv_long: IV of the long-term regime
        """
        # Common theta statistics
        self.theta_mean = thetas.mean(dim=0)
        self.theta_std = thetas.std(dim=0)
        self.theta_std = torch.where(self.theta_std > 1e-6, self.theta_std, torch.ones_like(self.theta_std))
        
        # IV Statistics by Regime
        for regime, iv_data in [('short', iv_short), ('mid', iv_mid), ('long', iv_long)]:
            self.regime_stats[regime]['iv_mean'] = iv_data.mean()
            self.regime_stats[regime]['iv_std'] = iv_data.std()
            if self.regime_stats[regime]['iv_std'] < 1e-6:
                self.regime_stats[regime]['iv_std'] = torch.tensor(1.0, device=self.device)
    
    def normalize_iv_regime(self, iv, regime):
        """Normalize IV using regime-specific statistics"""
        stats = self.regime_stats[regime]
        if stats['iv_mean'] is None or stats['iv_std'] is None:
            raise ValueError(f"Normalization stats for regime '{regime}' not computed.")
        return (iv - stats['iv_mean']) / stats['iv_std']
    
    def denormalize_iv_regime(self, iv_norm, regime):
        """Denormalize IV using regime-specific statistics"""
        stats = self.regime_stats[regime]
        if stats['iv_mean'] is None or stats['iv_std'] is None:
            raise ValueError(f"Normalization stats for regime '{regime}' not computed.")
        return iv_norm * stats['iv_std'] + stats['iv_mean']
    
    def save_regime_normalization_stats(self, path=None):
        """Save all normalization statistics including regimes"""
        if path is None and self.output_dir:
            path = f"{self.dirs['stats']}/regime_normalization_stats.pt"
        
        stats = {
            'theta_mean': self.theta_mean,
            'theta_std': self.theta_std,
            'regime_stats': self.regime_stats,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(stats, path)
        print(f"✓ Regime normalization stats saved: {path}")
    
    def load_regime_normalization_stats(self, path=None):
        """Load multi-regime normalization statistics"""
        if path is None and self.output_dir:
            path = f"{self.dirs['stats']}/regime_normalization_stats.pt"
        
        stats = torch.load(path, map_location=self.device)
        self.theta_mean = stats['theta_mean']
        self.theta_std = stats['theta_std']
        self.regime_stats = stats['regime_stats']
        print(f"✓ Regime normalization stats loaded from: {path}")
    
    def build_multi_regime_dataset(self, multi_regime_pricer, n_samples,
                                n_paths=30000, batch_size=50,
                                checkpoint_every=5, normalize=True,
                                compute_stats_from=None, resume_from=None,
                                mixed_precision=True, chunk_size=None,
                                sample_method='shared', force_regenerate=False):
        """
        Builds datasets for MultiRegimeGridPricer (with per-regime resumes and separate per-phase checkpoints).
        """
        import os, gc, pickle, time
        from datetime import datetime
        import torch
        from tqdm.auto import tqdm

        process_name = self.process.__class__.__name__.lower()
        phase = getattr(self, "dataset_type", "train")

        # If the final dataset already exists (for phase), load and return
        if self.output_dir:
            final_path = os.path.join(self.regime_dirs['unified'], "multi_regime_dataset_final.pkl")
            if os.path.exists(final_path) and not resume_from and not force_regenerate:
                print(f"✓ Multi-regime dataset {phase} already exists for {self.process.__class__.__name__}!")
                return self._load_final_multi_regime_dataset(final_path, normalize, compute_stats_from)

        # Separate checkpoint manager per phase
        checkpoint_manager = CheckpointManager(
            self.dirs['checkpoints'] if self.output_dir else './checkpoints',
            prefix=f'{self.process_name}_multi_regime',
            keep_last_n=9
        )

        # Initial aggregate state
        all_data = {
            'short': {'theta': [], 'iv': []},
            'mid':   {'theta': [], 'iv': []},
            'long':  {'theta': [], 'iv': []}
        }

        # Resume per-regime from the current phase
        if not force_regenerate:
            latest_per_regime = checkpoint_manager.find_latest_per_regime()
        else:
            latest_per_regime = {}

        if latest_per_regime:
            print("Found per-regime checkpoints:",
                ", ".join(os.path.basename(p) for p in latest_per_regime.values()))

            # Take, for each regime, the most complete dataset
            best = { 'short': (0, None), 'mid': (0, None), 'long': (0, None) }
            for _, path in latest_per_regime.items():
                with open(path, 'rb') as f:
                    ck = pickle.load(f)
                data = ck['data']
                for r in ['short','mid','long']:
                    n = len(data[r]['theta'])
                    if n > best[r][0]:
                        best[r] = (n, data)
            for r in ['short','mid','long']:
                if best[r][1] is not None:
                    all_data[r]['theta'] = best[r][1][r]['theta']
                    all_data[r]['iv']    = best[r][1][r]['iv']

        # Target per-regime: int (same for all) | dict | tuples/list
        if isinstance(n_samples, dict):
            targets = {
                'short': int(n_samples.get('short', 0)),
                'mid':   int(n_samples.get('mid',   0)),
                'long':  int(n_samples.get('long',  0)),
            }
        elif isinstance(n_samples, (tuple, list)) and len(n_samples) == 3:
            targets = {'short': int(n_samples[0]), 'mid': int(n_samples[1]), 'long': int(n_samples[2])}
        else:
            targets = {'short': int(n_samples), 'mid': int(n_samples), 'long': int(n_samples)}

        # Calculate progress and remaining per-regime
        done = {reg: len(all_data[reg]['theta']) for reg in ['short','mid','long']}
        remaining = {reg: max(0, targets[reg] - done[reg]) for reg in ['short','mid','long']}
        
        all_complete = all(done[reg] >= targets[reg] for reg in ['short', 'mid', 'long'])
        
        if all_complete:
            print("Dataset already complete for all regimes!")
            return self._process_completed_multi_regime_dataset(
                all_data, normalize, compute_stats_from
            )
        
        print("\nDataset status:")
        for reg in ['short', 'mid', 'long']:
            status = "✓ COMPLETE" if done[reg] >= targets[reg] else f"INCOMPLETE ({done[reg]}/{targets[reg]})"
            print(f"  {reg}: {status}")
        
        if any(done[reg] > 0 and done[reg] < targets[reg] for reg in ['short', 'mid', 'long']):
            print("\n! WARNING: Dataset partially complete. Continuing generation...")

        if all(v == 0 for v in remaining.values()):
            print("Already complete dataset!")
            return self._process_completed_multi_regime_dataset(
                all_data, normalize, compute_stats_from
            )

        # Remaining thetas generation for each regime
        theta_dict = {}
        if sample_method == 'shared':
            max_rem = max(remaining.values())
            if max_rem > 0:
                shared_thetas = self.sample_theta_lhs(max_rem)
            else:
                shared_thetas = torch.empty((0, len(self.param_names)), dtype=torch.float32, device=self.device)
            for reg in ['short','mid','long']:
                theta_dict[reg] = shared_thetas[:remaining[reg]]
        else:
            seeds = {'short': 42, 'mid': 43, 'long': 44}
            for reg in ['short','mid','long']:
                r = remaining[reg]
                theta_dict[reg] = self.sample_theta_lhs(r, seed=seeds[reg]) if r > 0 else \
                                torch.empty((0, len(self.param_names)), dtype=torch.float32, device=self.device)

        # Process in batches for incomplete regimes only (short → mid → long)
        for regime in ['short','mid','long']:
            rem = remaining[regime]
            if rem == 0:
                print(f"\n{regime.upper()} TERM regime già completo ({done[regime]}/{n_samples})")
                continue

            print(f"\n{'='*50}")
            print(f"Processing {regime.upper()} TERM regime "
                  f"[target={targets[regime]}, done={done[regime]}, remaining={remaining[regime]}]")
            print(f"{'='*50}")

            regime_thetas = theta_dict[regime]
            pricer = getattr(multi_regime_pricer, f"{regime}_term_pricer")

            # local progress to the regime
            start_idx_reg = done[regime]

            for batch_idx in range(0, len(regime_thetas), batch_size):
                batch_end = min(batch_idx + batch_size, len(regime_thetas))
                batch_thetas = regime_thetas[batch_idx:batch_end]

                current_total_reg = start_idx_reg + batch_end
                print(f"\n[{regime} - Batch {(batch_idx//batch_size)+1}] "
                    f"Samples {start_idx_reg + batch_idx + 1}-{current_total_reg}/{n_samples}")

                batch_start_time = time.time()

                # Compute IV
                iv_batch = []
                # use the MultiRegimeGridPricer config, if present
                regime_dt_config = getattr(multi_regime_pricer, 'regime_dt_config',
                                        {'short': 3e-5, 'mid': 1/1460, 'long': 1/365})
                # do not overwrite if already set
                if getattr(pricer, 'fixed_regime_dt', None) is None:
                    pricer.fixed_regime_dt = regime_dt_config[regime]

                with torch.cuda.amp.autocast(enabled=mixed_precision and self.device.type == 'cuda'):
                    for theta in tqdm(batch_thetas, desc=f"Computing {regime} IV grids"):
                        try:
                            iv_grid = pricer._mc_iv_grid(
                                theta, n_paths=n_paths,
                                use_antithetic=True,
                                adaptive_paths=False,
                                adaptive_dt=False,
                                control_variate=True,
                                chunk_size=chunk_size
                            )
                            if torch.isnan(iv_grid).any() or torch.isinf(iv_grid).any():
                                print("!  Invalid IV detected, using fallback")
                                iv_grid = torch.full_like(iv_grid, 0.2)
                            iv_batch.append(iv_grid.cpu())
                        except Exception as e:
                            print(f"X Error processing theta: {e}")
                            iv_grid = torch.full(
                                (len(pricer.Ts), len(pricer.logKs)), 0.2, device=self.device
                            )
                            iv_batch.append(iv_grid.cpu())

                        if len(iv_batch) % 10 == 0 and self.device.type == 'cuda':
                            torch.cuda.empty_cache()

                # Accumulate
                all_data[regime]['theta'].extend([t.cpu() for t in batch_thetas])
                all_data[regime]['iv'].extend(iv_batch)

                batch_time = time.time() - batch_start_time
                print(f"✓ {regime} batch completed in {batch_time:.1f}s")

                # Consistent and per-regime checkpoint (prefix includes phase)
                if ((batch_idx // batch_size + 1) % checkpoint_every == 0):
                    n_done_reg = start_idx_reg + batch_end
                    checkpoint_data = {
                        'data': all_data,
                        'n_samples_total': n_samples,
                        'current_regime': regime,
                        'timestamp': datetime.now().isoformat()
                    }
                    checkpoint_name = f"{checkpoint_manager.prefix}_checkpoint_{regime}_{n_done_reg}.pkl"
                    checkpoint_manager.save_checkpoint(checkpoint_data, checkpoint_name)

                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()

            # End-of-regime checkpoint with the same prefix
            n_done_reg = len(all_data[regime]['theta'])
            checkpoint_data = {
                'data': all_data,
                'n_samples_total': n_samples,
                'n_paths': n_paths,
                'current_regime': regime,
                'timestamp': datetime.now().isoformat()
            }
            checkpoint_name = f"{checkpoint_manager.prefix}_checkpoint_{regime}_{n_done_reg}.pkl"
            checkpoint_manager.save_checkpoint(checkpoint_data, checkpoint_name)

        # All regimens should be complete now
        return self._process_completed_multi_regime_dataset(
            all_data, normalize, compute_stats_from, cleanup_checkpoints=True
        )

    
    def _process_completed_multi_regime_dataset(self, all_data, normalize,
                                                compute_stats_from, cleanup_checkpoints=False):
        """Process and save complete multi-regime datasets (per phase), with checkpoint cleanup"""
        import os, pickle
        from datetime import datetime
        import torch
        import re

        phase = getattr(self, "dataset_type", "train")

        # Convert to tensors
        print("\nConverting to tensors...")
        datasets = {}
        for regime in ['short', 'mid', 'long']:
            theta_tensor = torch.stack(all_data[regime]['theta']).to(self.device)
            iv_tensor = torch.stack(all_data[regime]['iv']).to(self.device)

            datasets[regime] = {
                'theta': theta_tensor,
                'iv': iv_tensor
            }
            print(f"{regime}: Theta shape {theta_tensor.shape}, IV shape {iv_tensor.shape}")

        # Save raw dataset (per-regime and unified in the per-phase folder)
        if self.output_dir:
            for regime in ['short', 'mid', 'long']:
                regime_data = {
                    'theta': datasets[regime]['theta'].cpu(),
                    'iv': datasets[regime]['iv'].cpu(),
                    'timestamp': datetime.now().isoformat()
                }
                regime_path = os.path.join(self.regime_dirs[regime], "dataset_final.pkl")
                with open(regime_path, 'wb') as f:
                    pickle.dump(regime_data, f)
                print(f"✓ {regime} dataset saved: {regime_path}")

            # Unified (in the unified dir of the phase)
            unified_data = {
                'short': datasets['short'],
                'mid': datasets['mid'],
                'long': datasets['long'],
                'timestamp': datetime.now().isoformat()
            }
            unified_path = os.path.join(self.regime_dirs['unified'], "multi_regime_dataset_final.pkl")
            with open(unified_path, 'wb') as f:
                pickle.dump(unified_data, f)
            print(f"✓ Unified dataset saved: {unified_path}")

            # Cleanup checkpoint files: keep the last one for each regime in the current phase
            if cleanup_checkpoints:
                checkpoint_dir = self.dirs['checkpoints']
                files = [f for f in os.listdir(checkpoint_dir)
                        if f.startswith(f"{self.process_name}_multi_regime_checkpoint_") and f.endswith('.pkl')]

                rx = re.compile(
                    rf"{re.escape(self.process_name)}_multi_regime_checkpoint_(short|mid|long)_(\d+)\.pkl$"
                )
                latest = {}
                for f in files:
                    m = rx.match(f)
                    if not m:
                        continue
                    reg, n = m.group(1), int(m.group(2))
                    if (reg not in latest) or (n > latest[reg][0]):
                        latest[reg] = (n, f)

                keep = {v[1] for v in latest.values()}
                removed = 0
                for f in files:
                    if f not in keep:
                        os.remove(os.path.join(checkpoint_dir, f)); removed += 1
                print(f"✓ [{phase}] Removed {removed} checkpoints; kept: {', '.join(sorted(keep))}")

        # Normalization (optional; beware of upstream data leakage)
        if normalize:
            if compute_stats_from is None:
                # Calculate new statistics: use ALL concatenated thetas (robust across sizes)
                theta_all = torch.cat([datasets[r]['theta'] for r in ['short','mid','long']], dim=0)
                self.compute_regime_normalization_stats(
                    theta_all,
                    datasets['short']['iv'],
                    datasets['mid']['iv'],
                    datasets['long']['iv']
                )
                if self.output_dir:
                    self.save_regime_normalization_stats()

                print(f"\nRegime normalization statistics computed:")
                print(f"  Theta mean: {self.theta_mean}")
                print(f"  Theta std: {self.theta_std}")
                for regime in ['short', 'mid', 'long']:
                    stats = self.regime_stats[regime]
                    print(f"  {regime} IV: mean={stats['iv_mean']:.4f}, std={stats['iv_std']:.4f}")
            else:
                # Use existing statistics (e.g. from TRAIN)
                self._copy_regime_stats_from(compute_stats_from)
                print(f"\nUsing normalization statistics from training set")

            # Apply normalization
            normalized_datasets = {}
            for regime in ['short', 'mid', 'long']:
                theta_norm = self.normalize_theta(datasets[regime]['theta'])
                iv_norm = self.normalize_iv_regime(datasets[regime]['iv'], regime)
                normalized_datasets[regime] = {'theta': theta_norm, 'iv': iv_norm}

            # Save normalized datasets (optional)
            if self.output_dir:
                norm_data = {
                    'datasets': normalized_datasets,
                    'theta_mean': self.theta_mean.cpu(),
                    'theta_std': self.theta_std.cpu(),
                    'regime_stats': {k: {kk: vv.cpu() if torch.is_tensor(vv) else vv
                                        for kk, vv in v.items()}
                                    for k, v in self.regime_stats.items()},
                    'timestamp': datetime.now().isoformat()
                }
                norm_path = os.path.join(self.regime_dirs['unified'], "multi_regime_dataset_normalized.pkl")
                with open(norm_path, 'wb') as f:
                    pickle.dump(norm_data, f)
                print(f"✓ Normalized dataset saved: {norm_path}")

            return normalized_datasets

        return datasets

    
    def _copy_regime_stats_from(self, other_builder):
        """Copy statistics from another MultiRegimeDatasetBuilder"""
        self.theta_mean = other_builder.theta_mean
        self.theta_std = other_builder.theta_std
        self.regime_stats = other_builder.regime_stats.copy()
    
    def _load_final_multi_regime_dataset(self, path, normalize, compute_stats_from):
        """Load final multi-regime dataset"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        datasets = {}
        for regime in ['short', 'mid', 'long']:
            datasets[regime] = {
                'theta': data[regime]['theta'].to(self.device),
                'iv': data[regime]['iv'].to(self.device)
            }
        
        if normalize:
            if compute_stats_from is None:
                # Calculate statistics: use ALL concatenated thetas
                theta_all = torch.cat([datasets[r]['theta'] for r in ['short','mid','long']], dim=0)
                self.compute_regime_normalization_stats(
                    theta_all,
                    datasets['short']['iv'],
                    datasets['mid']['iv'],
                    datasets['long']['iv']
                )
                self.save_regime_normalization_stats()
            else:
                self._copy_regime_stats_from(compute_stats_from)
            
            # Normalize
            normalized_datasets = {}
            for regime in ['short', 'mid', 'long']:
                normalized_datasets[regime] = {
                    'theta': self.normalize_theta(datasets[regime]['theta']),
                    'iv': self.normalize_iv_regime(datasets[regime]['iv'], regime)
                }
            
            return normalized_datasets
        
        return datasets
    
    def split_multi_regime_dataset(self, datasets, train_ratio=0.8, seed=42):
        """
        Splits multi-regime datasets into train/validation datasets while maintaining consistency.

        Args:
        datasets: dict with datasets for each regime
        train_ratio: percentage for training
        seed: random seed

        Returns:
        train_datasets, val_datasets (both dicts with the same structure)
        """
        torch.manual_seed(seed)
        
        # If the sizes match, we maintain a "consistent" split between regimes; otherwise, we split per regime.
        sizes = {r: len(datasets[r]['theta']) for r in ['short','mid','long']}
        same_size = len(set(sizes.values())) == 1

        train_datasets, val_datasets = {}, {}
        if same_size:
            n = next(iter(sizes.values()))
            n_train = int(n * train_ratio)
            idx = torch.randperm(n)
            tr, va = idx[:n_train], idx[n_train:]
            for r in ['short','mid','long']:
                train_datasets[r] = {'theta': datasets[r]['theta'][tr], 'iv': datasets[r]['iv'][tr]}
                val_datasets[r]   = {'theta': datasets[r]['theta'][va], 'iv': datasets[r]['iv'][va]}
            print(f"Split dataset (coupled): {n_train} train, {n - n_train} validation per regime")
        else:
            print("Warning: regime sizes differ; performing per-regime split.")
            for r in ['short','mid','long']:
                n = sizes[r]
                n_train = int(n * train_ratio)
                idx = torch.randperm(n)
                tr, va = idx[:n_train], idx[n_train:]
                train_datasets[r] = {'theta': datasets[r]['theta'][tr], 'iv': datasets[r]['iv'][tr]}
                val_datasets[r]   = {'theta': datasets[r]['theta'][va], 'iv': datasets[r]['iv'][va]}
            print("Split dataset: per-regime counts:",
                  {r: (len(train_datasets[r]['theta']), len(val_datasets[r]['theta'])) for r in ['short','mid','long']})
        
        return train_datasets, val_datasets
    
class ExternalPointwiseDatasetBuilder(DatasetBuilder):
    """
    Dataset builder that uses external pricing backend instead of Monte Carlo simulation.
    Inherits from DatasetBuilder to reuse sampling and normalization logic.
    """
    
    def __init__(self, process, backend: 'PricingBackend', **kwargs):
        """
        Args:
            process: StochasticProcess instance
            backend: PricingBackend implementation for external pricing
            **kwargs: Additional arguments passed to DatasetBuilder
        """
        super().__init__(process=process, **kwargs)
        self.backend = backend
    
    def _theta_tensor_to_dict(self, theta: torch.Tensor) -> Dict[str, float]:
        """Convert theta tensor to parameter dictionary for backend."""
        return {name: float(theta[i].item()) 
                for i, name in enumerate(self.process.param_info.names)}
    
    @torch.no_grad()
    def _generate_random_grid(self, theta: torch.Tensor,
                              n_maturities: int = 11,
                              n_strikes_per_maturity: int = 13,
                              spot: float = 1.0,
                              mc_params: dict = None):  # Ignored for external backend
        """
        Generates a random IV(T,K) grid using external backend instead of MC simulation.
        
        Args:
            theta: Model parameters tensor
            n_maturities: Number of random maturities
            n_strikes_per_maturity: Number of strikes per maturity
            spot: Spot price
            mc_params: Ignored (kept for interface compatibility)
            
        Returns:
            Dictionary with maturities, strikes, and IV grid
        """
        # Sample random maturities using base class method
        mats = self._sample_random_maturities(n_maturities)
        
        all_strikes, iv_grid = [], []
        theta_dict = self._theta_tensor_to_dict(theta)
        
        for T in mats:
            T_val = float(T.item())
            
            # Sample random strikes using base class method
            strikes = self._sample_random_strikes(T_val, spot, n_strikes_per_maturity)
            
            # Calculate IVs using backend
            ivs = []
            for K in strikes:
                K_val = float(K.item())
                try:
                    # Price via backend
                    price = self.backend.price(
                        theta_dict, T_val, K_val, 'C', spot, 1.0
                    )
                    
                    # Calculate IV via backend
                    if price > 1e-10:
                        iv = self.backend.implied_vol(
                            T_val, spot, K_val, price, 'C', 1.0
                        )
                    else:
                        iv = self._get_process_default_volatility(theta)
                        
                except Exception as e:
                    # Fallback to default volatility
                    iv = self._get_process_default_volatility(theta)
                    if hasattr(self, 'verbose') and self.verbose:
                        print(f"Backend pricing failed: {e}")
                
                ivs.append(iv)
            
            all_strikes.append(strikes)
            iv_grid.append(torch.tensor(ivs, dtype=torch.float32, device=self.device))
        
        return {
            'maturities': mats,
            'strikes': all_strikes,
            'iv_grid': torch.stack(iv_grid)
        }
    
    def build_external_random_grids_dataset(self,
                                           n_surfaces: int = 10000,
                                           n_maturities: int = 11,
                                           n_strikes: int = 13,
                                           spot: float = 1.0,
                                           normalize: bool = True,
                                           compute_stats_from=None,
                                           show_progress: bool = True,
                                           batch_size: int = 50,
                                           checkpoint_every: int = 100,
                                           resume_from: str = None,
                                           base_seed: int = 42,
                                           split: str = None):
        """
        Build pointwise dataset using external backend with random grids approach.
        This is a wrapper around build_random_grids_dataset that uses the external backend.
        
        Args:
            n_surfaces: Number of parameter sets
            n_maturities: Number of maturities per surface
            n_strikes: Number of strikes per maturity
            spot: Spot price
            normalize: Whether to normalize the data
            compute_stats_from: DatasetBuilder to copy normalization stats from
            show_progress: Show progress bar
            batch_size: Batch size for processing
            checkpoint_every: Checkpoint frequency
            resume_from: Path to resume from checkpoint
            base_seed: Random seed
            split: 'train' or 'val'
            
        Returns:
            Tuple of (theta, T, k, iv) tensors
        """
        # Override the internal _generate_random_grid to use external backend
        # This is already done above, so we can just call the base method
        return self.build_random_grids_dataset(
            n_surfaces=n_surfaces,
            n_maturities=n_maturities,
            n_strikes=n_strikes,
            n_paths=0,  # Not used with external backend
            spot=spot,
            normalize=normalize,
            compute_stats_from=compute_stats_from,
            show_progress=show_progress,
            batch_size=batch_size,
            checkpoint_every=checkpoint_every,
            resume_from=resume_from,
            base_seed=base_seed,
            split=split
        )

class CheckpointManager:
    """Manages checkpoints for dataset generation"""
    
    def __init__(self, checkpoint_dir, prefix='checkpoint', keep_last_n=3):
        self.checkpoint_dir = checkpoint_dir
        self.prefix = prefix
        self.keep_last_n = keep_last_n
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, data, name):
        """Saves checkpoints and manages cleanup of old checkpoints"""
        path = f"{self.checkpoint_dir}/{name}"
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Checkpoint saved: {name}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
    # Keeps only the last N for mtime, more robust than alphabetical order
        paths = [
            os.path.join(self.checkpoint_dir, f)
            for f in os.listdir(self.checkpoint_dir)
            if f.startswith(self.prefix) and f.endswith('.pkl')
        ]
        paths.sort(key=lambda p: os.path.getmtime(p))
        if len(paths) > self.keep_last_n:
            for old_p in paths[:-self.keep_last_n]:
                os.remove(old_p)

    # List only checkpoints that match the full prefix and pattern "_checkpoint_"
    def _list_checkpoints(self):
        return [
            os.path.join(self.checkpoint_dir, f)
            for f in os.listdir(self.checkpoint_dir)
            if f.startswith(self.prefix + "_checkpoint_") and f.endswith(".pkl")
        ]
    
    def find_latest(self):
        """Find the last available checkpoint"""
        checkpoints = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.endswith('.pkl') and self.prefix in f
        ]
        
        if not checkpoints:
            return None
        
        # Extract timestamps or numbers from checkpoint names
        checkpoint_info = []
        for cp in checkpoints:
            try:
                # Try to draw a number after the last underscore first.
                parts = cp.replace('.pkl', '').split('_')
                if parts[-1].isdigit():
                    num = int(parts[-1])
                    checkpoint_info.append((num, cp))
                else:
                    # If there is no number, use the file modification time
                    mtime = os.path.getmtime(os.path.join(self.checkpoint_dir, cp))
                    checkpoint_info.append((mtime, cp))
            except:
                continue
        
        if checkpoint_info:
            # Find the checkpoint with the highest value (number or timestamp)
            latest = max(checkpoint_info, key=lambda x: x[0])
            return os.path.join(self.checkpoint_dir, latest[1])
        
        return None
    
    def _list_multi_regime_checkpoints(self):
        return [
            os.path.join(self.checkpoint_dir, f)
            for f in os.listdir(self.checkpoint_dir)
            if f.startswith(self.prefix + "_checkpoint_") and f.endswith(".pkl")
        ]

    def find_latest_per_regime(self):
        """Return last checkpoint per regime using current prefix (e.g. multi_regime_train/val)."""
        import re, os
        rx = re.compile(rf"{re.escape(self.prefix)}_checkpoint_(short|mid|long)_(\d+)\.pkl$")
        latest = {}
        for p in self._list_multi_regime_checkpoints():
            m = rx.search(os.path.basename(p))
            if not m:
                continue
            reg, n = m.group(1), int(m.group(2))
            if (reg not in latest) or (n > latest[reg][0]):
                latest[reg] = (n, p)
        return {reg: path for reg, (n, path) in latest.items()}

    