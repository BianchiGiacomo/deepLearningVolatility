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
import os
import pickle
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
    
    def __init__(self, process: Union[str, StochasticProcess], device='cpu', output_dir=None):
        """
        Args:
            process: Name of the process or StochasticProcess instance
            device: Device torch
            output_dir: Output directory
        """
        self.device = torch.device(device)
        self.output_dir = output_dir
        
        # Setup directories if output_dir is specified
        if self.output_dir:
            self.setup_directories()
        
        # Create process if passed as a string
        if isinstance(process, str):
            self.process = ProcessFactory.create(process)
        else:
            self.process = process
        
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
            path = f"{self.dirs['stats']}/normalization_stats.pt"
        
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
            path = f"{self.dirs['stats']}/normalization_stats.pt"
            
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
    
    def visualize_sampling_with_labels(self, n_samples=100):
        """
        View sampling with process-specific labels.
        """        
        # Generate samples
        uniform_samples = self.sample_theta_uniform(n_samples).cpu().numpy()
        lhs_samples = self.sample_theta_lhs(n_samples).cpu().numpy()
        
        param_names = self.param_names
        n_params = len(param_names)
        
        # Create subplots for each pair of parameters
        n_pairs = n_params * (n_params - 1) // 2
        n_cols = min(3, n_pairs)
        n_rows = (n_pairs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(5 * n_cols, 5 * n_rows * 2))
        if n_pairs == 1:
            axes = axes.reshape(2, 1)
        
        pair_idx = 0
        for i in range(n_params):
            for j in range(i + 1, n_params):
                row_uniform = (pair_idx // n_cols) * 2
                row_lhs = row_uniform + 1
                col = pair_idx % n_cols
                
                # Uniform sampling
                axes[row_uniform, col].scatter(uniform_samples[:, i], uniform_samples[:, j], 
                                             alpha=0.6, s=30)
                axes[row_uniform, col].set_xlabel(param_names[i])
                axes[row_uniform, col].set_ylabel(param_names[j])
                axes[row_uniform, col].set_title(f'Uniform: {param_names[i]} vs {param_names[j]}')
                axes[row_uniform, col].grid(True, alpha=0.3)
                
                # LHS sampling
                axes[row_lhs, col].scatter(lhs_samples[:, i], lhs_samples[:, j], 
                                         alpha=0.6, s=30, color='red')
                axes[row_lhs, col].set_xlabel(param_names[i])
                axes[row_lhs, col].set_ylabel(param_names[j])
                axes[row_lhs, col].set_title(f'LHS: {param_names[i]} vs {param_names[j]}')
                axes[row_lhs, col].grid(True, alpha=0.3)
                
                pair_idx += 1
        
        # Hide empty subplots
        for idx in range(pair_idx, n_rows * n_cols * 2):
            row = idx // n_cols
            col = idx % n_cols
            if row < axes.shape[0] and col < axes.shape[1]:
                axes[row, col].axis('off')
        
        plt.suptitle(f'{self.process.__class__.__name__} Parameter Sampling', fontsize=16)
        plt.tight_layout()
        plt.show()
    
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
                                resume_from=None, mixed_precision=True, chunk_size: int = None):
        """
        Optimized version for Google Colab with checkpoints and memory management.
        """
        # Check if the final dataset already exists
        if self.output_dir:
            process_name = self.process.__class__.__name__.lower()
            final_dataset_path = f"{self.dirs['datasets']}/{process_name}_grid_dataset_final.pkl"
            if os.path.exists(final_dataset_path):
                print(f"✓ Dataset finale già esistente per {self.process.__class__.__name__}!")
                return self._load_final_dataset(final_dataset_path, normalize, compute_stats_from)
        
        # Initialize checkpoint manager with process-specific prefix
        checkpoint_manager = CheckpointManager(
            self.dirs['checkpoints'] if self.output_dir else './checkpoints',
            prefix=f'{process_name}_grid_dataset'
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
                print("Dataset già completo!")
                return self._process_completed_dataset(
                    all_theta, all_iv, normalize, compute_stats_from
                )
        
        # Generate only the missing thetas
        remaining_samples = n_samples - start_idx
        if remaining_samples <= 0:
            return self._process_completed_dataset(
                all_theta, all_iv, normalize, compute_stats_from
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
                for theta in tqdm(batch_thetas, desc=f"Computing {process_name} IV grids"):
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
                    
                    # Clear cache periodicamente
                    if len(iv_batch) % 10 == 0 and self.device.type == 'cuda':
                        torch.cuda.empty_cache()
            
            # Aggiungi a liste complete
            all_theta.extend([t.cpu() for t in batch_thetas])
            all_iv.extend(iv_batch)
            
            batch_time = time.time() - batch_start_time
            print(f"✓ Batch completed in {batch_time:.1f}s")
            
            # Checkpoint periodici
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
                
                checkpoint_name = f"{process_name}_checkpoint_{start_idx + batch_end}.pkl"
                checkpoint_manager.save_checkpoint(checkpoint_data, checkpoint_name)
                
                # Clear GPU memory
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()
        
        # Processa dataset completo
        return self._process_completed_dataset(
            all_theta, all_iv, normalize, compute_stats_from,
            cleanup_checkpoints=True
        )
    
    def _get_process_default_volatility(self, theta: torch.Tensor) -> float:
        """
        Ottiene una volatilità di default basata sul processo e i parametri.
        """
        if hasattr(self.process, 'param_info'):
            param_info = self.process.param_info
            
            # Cerca parametri volatilità-related nell'ordine di priorità
            for vol_name in ['xi0', 'sigma', 'theta', 'vol']:
                if vol_name in param_info.names:
                    idx = param_info.names.index(vol_name)
                    value = theta[idx].item()
                    
                    # Per parametri di varianza, prendi sqrt
                    if vol_name in ['xi0', 'theta']:
                        return math.sqrt(value)
                    else:
                        return value
            
            # Se è un modello jump, usa sigma se presente
            if 'jump' in self.process.__class__.__name__.lower():
                if 'sigma' in param_info.names:
                    idx = param_info.names.index('sigma')
                    return theta[idx].item()
        
        # Default generale
        return 0.2
    
        
    def _load_checkpoint(self, checkpoint_path):
        """Carica checkpoint con gestione errori"""
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
        
    def _load_final_dataset(self, path, normalize, compute_stats_from):
        """Carica dataset finale e applica normalizzazione se necessario"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        theta_tensor = data['theta'].to(self.device)
        iv_tensor = data['iv'].to(self.device)
        
        if normalize:
            if compute_stats_from is None:
                # Calcola statistiche
                self.compute_normalization_stats(theta_tensor, iv_tensor)
                self.save_normalization_stats()
            else:
                # Usa statistiche esistenti
                self._copy_stats_from(compute_stats_from)
            
            # Normalizza
            theta_norm = self.normalize_theta(theta_tensor)
            iv_norm = self.normalize_iv(iv_tensor)
            return theta_norm, iv_norm
        
        return theta_tensor, iv_tensor
    
    def _process_completed_dataset(self, theta_list, iv_list, normalize, 
                                 compute_stats_from, cleanup_checkpoints=False):
        """Processa dataset completo: conversione, normalizzazione, salvataggio"""
        # Converti a tensori
        print("\nConverting to tensors...")
        theta_tensor = torch.stack(theta_list).to(self.device)
        iv_tensor = torch.stack(iv_list).to(self.device)
        
        print(f"Dataset shape - Theta: {theta_tensor.shape}, IV: {iv_tensor.shape}")
        
        # Salva dataset raw se abbiamo output_dir
        if self.output_dir:
            process_name = self.process.__class__.__name__.lower()
            final_data = {
                'theta': theta_tensor.cpu(),
                'iv': iv_tensor.cpu(),
                'process': self.process.__class__.__name__,
                'timestamp': datetime.now().isoformat()
            }
            final_path = f"{self.dirs['datasets']}/{process_name}_grid_dataset_final.pkl"
            with open(final_path, 'wb') as f:
                pickle.dump(final_data, f)
            print(f"✓ Raw dataset saved: {final_path}")
            
            # Cleanup checkpoints
            if cleanup_checkpoints:
                checkpoint_dir = self.dirs['checkpoints']
                checkpoints = [f for f in os.listdir(checkpoint_dir) 
                             if f.startswith('checkpoint_') and f.endswith('.pkl')]
                for cp in checkpoints:
                    os.remove(f"{checkpoint_dir}/{cp}")
                print(f"✓ Removed {len(checkpoints)} checkpoint files")
        
        # Normalizzazione
        if normalize:
            if compute_stats_from is None:
                # Calcola nuove statistiche (training set)
                self.compute_normalization_stats(theta_tensor, iv_tensor)
                if self.output_dir:
                    self.save_normalization_stats()
                
                print(f"\nNormalization statistics computed:")
                print(f"  Theta mean: {self.theta_mean}")
                print(f"  Theta std: {self.theta_std}")
                print(f"  IV mean: {self.iv_mean:.4f}, std: {self.iv_std:.4f}")
            else:
                # Usa statistiche esistenti (validation set)
                self._copy_stats_from(compute_stats_from)
                print(f"\nUsing normalization statistics from training set")
            
            # Applica normalizzazione
            theta_norm = self.normalize_theta(theta_tensor)
            iv_norm = self.normalize_iv(iv_tensor)
            
            # Salva dataset normalizzato
            if self.output_dir:
                norm_data = {
                    'theta_norm': theta_norm.cpu(),
                    'iv_norm': iv_norm.cpu(),
                    'theta_mean': self.theta_mean.cpu(),
                    'theta_std': self.theta_std.cpu(),
                    'iv_mean': self.iv_mean.cpu(),
                    'iv_std': self.iv_std.cpu(),
                    'timestamp': datetime.now().isoformat()
                }
                norm_path = f"{self.dirs['datasets']}/grid_dataset_normalized.pkl"
                with open(norm_path, 'wb') as f:
                    pickle.dump(norm_data, f)
                print(f"✓ Normalized dataset saved: {norm_path}")
            
            return theta_norm, iv_norm
        
        return theta_tensor, iv_tensor
    
    def _copy_stats_from(self, other_builder):
        """Copia statistiche di normalizzazione da un altro builder"""
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
        Costruisce dataset pointwise con normalizzazione opzionale.
        Versione aggiornata per processi generici.
        """
        n_theta = len(thetas)
        n_T = len(maturities)
        n_K = len(logK)
        total_points = n_theta * n_T * n_K
        
        if show_progress:
            print(f"Building pointwise dataset for {self.process.__class__.__name__}: {total_points} points")
        
        # Pre-alloca
        theta_pw = torch.empty(total_points, self.process.num_params, device=self.device)
        T_pw = torch.empty(total_points, device=self.device)
        k_pw = torch.empty(total_points, device=self.device)
        iv_pw = torch.empty(total_points, device=self.device)
        
        # Ottieni parametri MC ottimizzati
        mc_params = self.get_process_specific_mc_params(base_n_paths=n_paths)
        
        idx = 0
        iterator = tqdm(thetas, desc=f"Building {self.process.__class__.__name__} pointwise") if show_progress else thetas
        
        for i, theta in enumerate(iterator):
            try:
                # Calcola griglia per questo theta
                iv_grid = pricer._mc_iv_grid(
                    theta, maturities, logK, 
                    n_paths=mc_params['n_paths'],
                    use_antithetic=mc_params['use_antithetic'],
                    adaptive_dt=mc_params['adaptive_dt'],
                    control_variate=mc_params['control_variate']
                )
                
                # Verifica validità
                if torch.isnan(iv_grid).any() or torch.isinf(iv_grid).any():
                    print(f"\nWarning: Invalid IV for theta {i}")
                    default_vol = self._get_process_default_volatility(theta)
                    iv_grid = torch.full_like(iv_grid, default_vol)
                
                # Appiattisci
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
                # Calcola statistiche
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
                # Usa statistiche esistenti
                self._copy_stats_from(compute_stats_from)
                
                if show_progress:
                    print(f"\nUsing normalization statistics from training set")
            
            # Normalizza
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
                                     chunk_size: int = None):
        """
        Versione ottimizzata per Colab del build_pointwise_dataset.
        Aggiornata per processi generici.
        """
        n_theta = len(thetas)
        n_T = len(maturities)
        n_K = len(logK)
        total_points = n_theta * n_T * n_K
        
        process_name = self.process.__class__.__name__.lower()
        
        # Check se esiste già il dataset finale
        if self.output_dir:
            final_dataset_path = f"{self.dirs['datasets']}/{process_name}_pointwise_dataset_final.pkl"
            if os.path.exists(final_dataset_path) and not resume_from:
                print(f"✓ Dataset finale già esistente per {self.process.__class__.__name__}!")
                return self._load_final_pointwise_dataset(
                    final_dataset_path, normalize, compute_stats_from
                )
        
        # Inizializza checkpoint manager
        checkpoint_manager = CheckpointManager(
            self.dirs['checkpoints'] if self.output_dir else './checkpoints',
            prefix=f'{process_name}_pointwise_dataset'
        )
        
        # Resume da checkpoint se disponibile
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
                    all_data, normalize, compute_stats_from
                )
        
        # Process remaining thetas
        remaining_thetas = thetas[start_idx:]
        if show_progress:
            print(f"\nGenerating pointwise dataset for {self.process.__class__.__name__}: {total_points} total points")
            print(f"  Remaining thetas to process: {len(remaining_thetas)}")
        
        # Ottieni parametri MC ottimizzati
        mc_params = self.get_process_specific_mc_params(base_n_paths=n_paths)
        
        # Process in batches
        for batch_idx in range(0, len(remaining_thetas), batch_size):
            batch_end = min(batch_idx + batch_size, len(remaining_thetas))
            batch_thetas = remaining_thetas[batch_idx:batch_end]
            current_idx = start_idx + batch_idx
            print(f"\n[Batch {(batch_idx//batch_size)+1}] Processing theta {current_idx+1}-{current_idx+len(batch_thetas)}/{n_theta}")
            batch_start_time = time.time()
            
            with torch.cuda.amp.autocast(enabled=mixed_precision and self.device.type == 'cuda'):
                iterator = tqdm(batch_thetas, desc=f"Computing {process_name} IV grids") if show_progress else batch_thetas
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
                        
                        # Check repair stats se disponibili
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
                checkpoint_name = f"{process_name}_checkpoint_{start_idx + batch_end}.pkl"
                checkpoint_manager.save_checkpoint(checkpoint_data, checkpoint_name)
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()
        
        return self._process_completed_pointwise_dataset(
            all_data, normalize, compute_stats_from, cleanup_checkpoints=True
        )

    def _process_completed_pointwise_dataset(self, all_data, normalize,
                                           compute_stats_from, cleanup_checkpoints=False):
        """Processa dataset completo: conversione, normalizzazione, salvataggio"""
        print("\nConverting to tensors...")
        theta_tensor = torch.stack(all_data['theta']).to(self.device)
        T_tensor = torch.stack(all_data['T']).to(self.device)
        k_tensor = torch.stack(all_data['k']).to(self.device)
        iv_tensor = torch.stack(all_data['iv']).to(self.device)
        
        print(f"Dataset shape: {theta_tensor.shape[0]} points")
        print(f"Unique thetas: {len(torch.unique(theta_tensor, dim=0))}")
    
        if self.output_dir:
            final_data = {
                'theta': theta_tensor.cpu(),
                'T': T_tensor.cpu(),
                'k': k_tensor.cpu(),
                'iv': iv_tensor.cpu(),
                'timestamp': datetime.now().isoformat()
            }
            final_path = f"{self.dirs['datasets']}/pointwise_dataset_final.pkl"
            with open(final_path, 'wb') as f:
                pickle.dump(final_data, f)
            print(f"✓ Raw dataset saved: {final_path}")
            
            if cleanup_checkpoints:
                checkpoint_dir = self.dirs['checkpoints']
                # FIX: Cerca i checkpoint con il pattern corretto
                checkpoints = [f for f in os.listdir(checkpoint_dir) 
                              if f.startswith('checkpoint_') and f.endswith('.pkl')]
                for cp in checkpoints:
                    os.remove(f"{checkpoint_dir}/{cp}")
                if checkpoints:
                    print(f"✓ Removed {len(checkpoints)} checkpoint files")
    
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
                norm_path = f"{self.dirs['datasets']}/pointwise_dataset_normalized.pkl"
                with open(norm_path, 'wb') as f:
                    pickle.dump(norm_data, f)
                print(f"✓ Normalized dataset saved: {norm_path}")
            
            return theta_norm, T_norm, k_norm, iv_norm
        
        return theta_tensor, T_tensor, k_tensor, iv_tensor
    
    def _load_final_pointwise_dataset(self, path, normalize, compute_stats_from):
        """Carica dataset finale e applica normalizzazione se necessario"""
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
    # Buckets temporali (Eq. 2) e parametri strikes (Eq. 3)
    _maturity_buckets = [
        (0.003, 0.030), (0.030, 0.090), (0.090, 0.150), (0.150, 0.300),
        (0.300, 0.500), (0.500, 0.750), (0.750, 1.000), (1.000, 1.250),
        (1.250, 1.500), (1.500, 2.000), (2.000, 2.500)
    ]
    _strike_params = dict(l=0.55, u=0.30, n_left_tail=4, n_center=7, n_right_tail=2, center_width=0.20)

    def _sample_random_maturities(self, n_maturities: int = 11, buckets=None, seed: int = None):
        """Campiona maturities casuali dai buckets e le ordina."""
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
        """Campiona strikes random rispettando la granularità tipica del mercato."""
        p = self._strike_params
        sqrt_T = float(np.sqrt(T))
        K_min = spot * (1 - p['l'] * sqrt_T)
        K_max = spot * (1 + p['u'] * sqrt_T)
        # Guardrail per estremi numericamente problematici
        K_min = max(K_min, 0.05 * spot)
        K_max = min(K_max, 3.00 * spot)
        center_lower = spot * (1 - p['center_width'] * sqrt_T)
        center_upper = spot * (1 + p['center_width'] * sqrt_T)

        n_left, n_center, n_right = p['n_left_tail'], p['n_center'], p['n_right_tail']
        tot_default = n_left + n_center + n_right
        if n_strikes != tot_default:
            # ridistribuisci in proporzione, garantendo almeno 1 in ogni zona
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
        """Genera una griglia IV(T,K) random per un set di parametri theta."""
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
                                   n_strikes: int = 13, n_paths: int = 30000, spot: float = 1.0,
                                   normalize: bool = True, compute_stats_from=None,
                                   show_progress: bool = True, batch_size: int = 50):
        """Costruisce dataset pointwise con approccio Random Grids."""
        print(f"\nBuilding Random Grids Pointwise Dataset: {n_surfaces} × {n_maturities} × {n_strikes}")
        all_theta, all_T, all_k, all_iv = [], [], [], []
        mc_params = self.get_process_specific_mc_params(base_n_paths=n_paths)

        thetas = self.sample_theta_lhs(n_surfaces)
        iterator = range(n_surfaces)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Generating random grids")
            except Exception:
                pass

        for i in iterator:
            theta = thetas[i]
            grid = self._generate_random_grid(theta, n_maturities, n_strikes, spot, mc_params)
            for j, T in enumerate(grid['maturities']):
                strikes = grid['strikes'][j]
                logK = torch.log(strikes / spot)
                iv_smile = grid['iv_grid'][j]
                all_theta.extend([theta] * len(strikes))
                all_T.extend([T] * len(strikes))
                all_k.extend(list(logK))
                all_iv.extend(list(iv_smile))
            if (i + 1) % 10 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()

        theta_pw = torch.stack(all_theta); T_pw = torch.stack(all_T)
        k_pw = torch.stack(all_k); iv_pw = torch.stack(all_iv)

        if normalize:
            if compute_stats_from is None:
                self.compute_normalization_stats(theta_pw, iv_pw, T_pw, k_pw)
                self.save_normalization_stats()
            else:
                self._copy_stats_from(compute_stats_from)
            return (self.normalize_theta(theta_pw), self.normalize_T(T_pw),
                    self.normalize_k(k_pw), self.normalize_iv(iv_pw))

        return theta_pw, T_pw, k_pw, iv_pw

    def build_random_smiles_dataset(self, n_smiles: int = 50000, n_strikes_per_smile: int = 13,
                                    n_paths: int = 30000, spot: float = 1.0, normalize: bool = True,
                                    compute_stats_from=None, show_progress: bool = True):
        """Variante leggera: una T random per ogni theta."""
        print(f"\nBuilding Random Smiles Dataset: {n_smiles} × {n_strikes_per_smile}")
        mc_params = self.get_process_specific_mc_params(base_n_paths=n_paths)
        thetas = self.sample_theta_lhs(n_smiles)
        all_theta, all_T, all_k, all_iv = [], [], [], []

        iterator = range(n_smiles)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Generating random smiles")
            except Exception:
                pass

        for i in iterator:
            lo, hi = self._maturity_buckets[np.random.randint(len(self._maturity_buckets))]
            T = float(np.random.uniform(lo, hi))
            T_t = torch.tensor(T, dtype=torch.float32, device=self.device)
            strikes = self._sample_random_strikes(T, spot, n_strikes_per_smile)
            logK = torch.log(strikes / spot)
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
                theta=thetas[i],
                n_paths=mc_params.get('n_paths', 30000),
                spot=spot,
                use_antithetic=mc_params.get('use_antithetic', True),
                adaptive_paths=mc_params.get('adaptive_paths', False),
                adaptive_dt=mc_params.get('adaptive_dt', True),
                control_variate=mc_params.get('control_variate', True)
            ).squeeze(0)

            all_theta.extend([thetas[i]] * len(strikes))
            all_T.extend([T_t] * len(strikes))
            all_k.extend(list(logK))
            all_iv.extend(list(iv_smile))

            if (i + 1) % 100 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()

        theta_pw = torch.stack(all_theta); T_pw = torch.stack(all_T)
        k_pw = torch.stack(all_k); iv_pw = torch.stack(all_iv)
        if normalize:
            if compute_stats_from is None:
                self.compute_normalization_stats(theta_pw, iv_pw, T_pw, k_pw)
                self.save_normalization_stats()
            else:
                self._copy_stats_from(compute_stats_from)
            return (self.normalize_theta(theta_pw), self.normalize_T(T_pw),
                    self.normalize_k(k_pw), self.normalize_iv(iv_pw))
        return theta_pw, T_pw, k_pw, iv_pw
class MultiRegimeDatasetBuilder(DatasetBuilder):
    """
    Estende DatasetBuilder per gestire dataset multi-regime con processi generici.
    """
    
    def __init__(self, process: Union[str, StochasticProcess], 
                 device='cpu', output_dir=None, dataset_type=None,
                 enable_smile_repair: bool = True,
                 smile_repair_method: str = 'pchip'):
        
        super().__init__(process, device, output_dir)
        
        # Statistiche separate per regime
        self.regime_stats = {
            'short': {'iv_mean': None, 'iv_std': None},
            'mid': {'iv_mean': None, 'iv_std': None},
            'long': {'iv_mean': None, 'iv_std': None}
        }
        
        # Directory aggiuntive per multi-regime
        if self.output_dir:
            process_name = self.process.__class__.__name__.lower()
            self.regime_dirs = {
                'short': f"{self.output_dir}/datasets/{process_name}_short_term",
                'mid': f"{self.output_dir}/datasets/{process_name}_mid_term",
                'long': f"{self.output_dir}/datasets/{process_name}_long_term",
                'unified': f"{self.output_dir}/datasets/{process_name}_unified"
            }
            for dir_path in self.regime_dirs.values():
                os.makedirs(dir_path, exist_ok=True)
                
        self.enable_smile_repair = enable_smile_repair
        self.smile_repair_method = smile_repair_method

        # Normalizza dataset_type → "train" | "val"
        self.dataset_type = (dataset_type or "train").lower()
        if self.dataset_type in ("validation", "valid"):
            self.dataset_type = "val"
        assert self.dataset_type in ("train", "val"), "dataset_type deve essere 'train' o 'val'"

        # Aggiungi sottocartelle per fase (train/val)
        # Assumiamo che self.dirs e self.regime_dirs siano già creati prima
        if self.output_dir:
            # checkpoints/<phase>
            self.dirs['checkpoints'] = os.path.join(self.dirs['checkpoints'], self.dataset_type)
            os.makedirs(self.dirs['checkpoints'], exist_ok=True)
            # unified/<phase>
            self.regime_dirs['unified'] = os.path.join(self.regime_dirs['unified'], self.dataset_type)
            os.makedirs(self.regime_dirs['unified'], exist_ok=True)
        else:
            # fallback locale
            self.dirs['checkpoints'] = os.path.join("./checkpoints", self.dataset_type)
            os.makedirs(self.dirs['checkpoints'], exist_ok=True)
            self.regime_dirs['unified'] = os.path.join("./unified", self.dataset_type)
            os.makedirs(self.regime_dirs['unified'], exist_ok=True)
    
    def compute_regime_normalization_stats(self, thetas, iv_short, iv_mid, iv_long):
        """
        Calcola statistiche di normalizzazione per ogni regime.
        
        Args:
            thetas: parametri comuni a tutti i regimi
            iv_short: IV del regime short term
            iv_mid: IV del regime mid term
            iv_long: IV del regime long term
        """
        # Statistiche theta comuni
        self.theta_mean = thetas.mean(dim=0)
        self.theta_std = thetas.std(dim=0)
        self.theta_std = torch.where(self.theta_std > 1e-6, self.theta_std, torch.ones_like(self.theta_std))
        
        # Statistiche IV per regime
        for regime, iv_data in [('short', iv_short), ('mid', iv_mid), ('long', iv_long)]:
            self.regime_stats[regime]['iv_mean'] = iv_data.mean()
            self.regime_stats[regime]['iv_std'] = iv_data.std()
            if self.regime_stats[regime]['iv_std'] < 1e-6:
                self.regime_stats[regime]['iv_std'] = torch.tensor(1.0, device=self.device)
    
    def normalize_iv_regime(self, iv, regime):
        """Normalizza IV usando le statistiche del regime specifico"""
        stats = self.regime_stats[regime]
        if stats['iv_mean'] is None or stats['iv_std'] is None:
            raise ValueError(f"Normalization stats for regime '{regime}' not computed.")
        return (iv - stats['iv_mean']) / stats['iv_std']
    
    def denormalize_iv_regime(self, iv_norm, regime):
        """Denormalizza IV usando le statistiche del regime specifico"""
        stats = self.regime_stats[regime]
        if stats['iv_mean'] is None or stats['iv_std'] is None:
            raise ValueError(f"Normalization stats for regime '{regime}' not computed.")
        return iv_norm * stats['iv_std'] + stats['iv_mean']
    
    def save_regime_normalization_stats(self, path=None):
        """Salva tutte le statistiche di normalizzazione inclusi i regimi"""
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
        """Carica le statistiche di normalizzazione multi-regime"""
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
        Costruisce dataset per MultiRegimeGridPricer (con resume per-regime e checkpoint separati per phase).
        """
        import os, gc, pickle, time
        from datetime import datetime
        import torch
        from tqdm.auto import tqdm

        process_name = self.process.__class__.__name__.lower()
        phase = getattr(self, "dataset_type", "train")

        # Se esiste già il dataset finale (per phase), carica e ritorna
        if self.output_dir:
            final_path = os.path.join(self.regime_dirs['unified'], "multi_regime_dataset_final.pkl")
            if os.path.exists(final_path) and not resume_from and not force_regenerate:
                print(f"✓ Multi-regime dataset {phase} già esistente per {self.process.__class__.__name__}!")
                return self._load_final_multi_regime_dataset(final_path, normalize, compute_stats_from)

        # Checkpoint manager separato per phase
        checkpoint_manager = CheckpointManager(
            self.dirs['checkpoints'] if self.output_dir else './checkpoints',
            prefix=f'multi_regime_{phase}',
            keep_last_n=9
        )

        # Stato aggregato iniziale
        all_data = {
            'short': {'theta': [], 'iv': []},
            'mid':   {'theta': [], 'iv': []},
            'long':  {'theta': [], 'iv': []}
        }

        # Resume per‑regime dalla phase corrente
        if not force_regenerate:
            latest_per_regime = checkpoint_manager.find_latest_per_regime()
        else:
            latest_per_regime = {}

        if latest_per_regime:
            print("Found per-regime checkpoints:",
                ", ".join(os.path.basename(p) for p in latest_per_regime.values()))

            # Prendi, per ciascun regime, il dataset più completo
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

        # Calcola progresso e rimanenti per-regime
        done = {reg: len(all_data[reg]['theta']) for reg in ['short','mid','long']}
        remaining = {reg: max(0, n_samples - done[reg]) for reg in ['short','mid','long']}

        if all(v == 0 for v in remaining.values()):
            print("Dataset già completo!")
            return self._process_completed_multi_regime_dataset(
                all_data, normalize, compute_stats_from
            )

        # Generazione thetas rimanenti per ciascun regime
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

        # Process in batches per i soli regimi incompleti (short → mid → long)
        for regime in ['short','mid','long']:
            rem = remaining[regime]
            if rem == 0:
                print(f"\n{regime.upper()} TERM regime già completo ({done[regime]}/{n_samples})")
                continue

            print(f"\n{'='*50}")
            print(f"Processing {regime.upper()} TERM regime")
            print(f"{'='*50}")

            regime_thetas = theta_dict[regime]
            pricer = getattr(multi_regime_pricer, f"{regime}_term_pricer")

            # progress locale al regime
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
                with torch.cuda.amp.autocast(enabled=mixed_precision and self.device.type == 'cuda'):
                    for theta in tqdm(batch_thetas, desc=f"Computing {regime} IV grids"):
                        try:
                            iv_grid = pricer._mc_iv_grid(
                                theta, n_paths=n_paths,
                                use_antithetic=True,
                                adaptive_paths=False,
                                adaptive_dt=True,
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

                # Accumula
                all_data[regime]['theta'].extend([t.cpu() for t in batch_thetas])
                all_data[regime]['iv'].extend(iv_batch)

                batch_time = time.time() - batch_start_time
                print(f"✓ {regime} batch completed in {batch_time:.1f}s")

                # Checkpoint coerente e per-regime (prefisso include phase)
                if ((batch_idx // batch_size + 1) % checkpoint_every == 0):
                    n_done_reg = start_idx_reg + batch_end
                    checkpoint_data = {
                        'data': all_data,
                        'n_samples_total': n_samples,
                        'current_regime': regime,
                        'timestamp': datetime.now().isoformat()
                    }
                    checkpoint_name = f"multi_regime_{phase}_checkpoint_{regime}_{n_done_reg}.pkl"
                    checkpoint_manager.save_checkpoint(checkpoint_data, checkpoint_name)

                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()

            # Checkpoint di fine regime con stesso prefisso
            n_done_reg = len(all_data[regime]['theta'])
            checkpoint_data = {
                'data': all_data,
                'n_samples_total': n_samples,
                'n_paths': n_paths,
                'current_regime': regime,
                'timestamp': datetime.now().isoformat()
            }
            checkpoint_name = f"multi_regime_{phase}_checkpoint_{regime}_{n_done_reg}.pkl"
            checkpoint_manager.save_checkpoint(checkpoint_data, checkpoint_name)

        # Tutti i regimi dovrebbero essere completi ora
        return self._process_completed_multi_regime_dataset(
            all_data, normalize, compute_stats_from, cleanup_checkpoints=True
        )

    
    def _process_completed_multi_regime_dataset(self, all_data, normalize,
                                                compute_stats_from, cleanup_checkpoints=False):
        """Processa e salva dataset multi-regime completo (per phase), con cleanup dei checkpoint."""
        import os, pickle
        from datetime import datetime
        import torch
        import re

        phase = getattr(self, "dataset_type", "train")

        # Converti a tensori
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

        # Salva dataset raw (per-regime e unificato nella cartella per phase)
        if self.output_dir:
            # Per regime
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

            # Unificato (nella dir unified della phase)
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

            # Cleanup checkpoint files: tieni l'ultimo per ogni regime nella phase corrente
            if cleanup_checkpoints:
                checkpoint_dir = self.dirs['checkpoints']  # già include /train o /val se hai settato in __init__
                files = [f for f in os.listdir(checkpoint_dir)
                        if f.startswith(f"multi_regime_{phase}_checkpoint_") and f.endswith('.pkl')]

                rx = re.compile(rf"multi_regime_{phase}_checkpoint_(short|mid|long)_(\d+)\.pkl$")
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

        # Normalizzazione (opzionale; attenzione al data leakage a monte)
        if normalize:
            if compute_stats_from is None:
                # Calcola nuove statistiche (theta condivisi: prendi 'short' per coerenza)
                theta_all = datasets['short']['theta']
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
                # Usa statistiche esistenti (es. dal TRAIN)
                self._copy_regime_stats_from(compute_stats_from)
                print(f"\nUsing normalization statistics from training set")

            # Applica normalizzazione
            normalized_datasets = {}
            for regime in ['short', 'mid', 'long']:
                theta_norm = self.normalize_theta(datasets[regime]['theta'])
                iv_norm = self.normalize_iv_regime(datasets[regime]['iv'], regime)
                normalized_datasets[regime] = {'theta': theta_norm, 'iv': iv_norm}

            # Salva dataset normalizzati (opzionale)
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
        """Copia statistiche da un altro MultiRegimeDatasetBuilder"""
        self.theta_mean = other_builder.theta_mean
        self.theta_std = other_builder.theta_std
        self.regime_stats = other_builder.regime_stats.copy()
    
    def _load_final_multi_regime_dataset(self, path, normalize, compute_stats_from):
        """Carica dataset finale multi-regime"""
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
                # Calcola statistiche
                theta_all = datasets['short']['theta']
                self.compute_regime_normalization_stats(
                    theta_all,
                    datasets['short']['iv'],
                    datasets['mid']['iv'],
                    datasets['long']['iv']
                )
                self.save_regime_normalization_stats()
            else:
                self._copy_regime_stats_from(compute_stats_from)
            
            # Normalizza
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
        Divide dataset multi-regime in train/validation mantenendo consistency.
        
        Args:
            datasets: dict con dataset per ogni regime
            train_ratio: percentuale per training
            seed: random seed
        
        Returns:
            train_datasets, val_datasets (entrambi dict con stessa struttura)
        """
        torch.manual_seed(seed)
        
        # Assumiamo che tutti i regimi abbiano lo stesso numero di samples
        n_samples = len(datasets['short']['theta'])
        n_train = int(n_samples * train_ratio)
        
        # Genera indici shuffled
        indices = torch.randperm(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # Split datasets
        train_datasets = {}
        val_datasets = {}
        
        for regime in ['short', 'mid', 'long']:
            train_datasets[regime] = {
                'theta': datasets[regime]['theta'][train_indices],
                'iv': datasets[regime]['iv'][train_indices]
            }
            val_datasets[regime] = {
                'theta': datasets[regime]['theta'][val_indices],
                'iv': datasets[regime]['iv'][val_indices]
            }
        
        print(f"Split dataset: {n_train} train, {n_samples - n_train} validation")
        
        return train_datasets, val_datasets
    
class CheckpointManager:
    """Gestisce checkpoint per dataset generation"""
    
    def __init__(self, checkpoint_dir, prefix='checkpoint', keep_last_n=3):
        self.checkpoint_dir = checkpoint_dir
        self.prefix = prefix
        self.keep_last_n = keep_last_n
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, data, name):
        """Salva checkpoint e gestisce pulizia vecchi checkpoint"""
        path = f"{self.checkpoint_dir}/{name}"
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Checkpoint saved: {name}")
        
        # Cleanup vecchi checkpoint
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
    # Mantiene solo gli ultimi N per mtime, più robusto dell'ordine alfabetico
        paths = [
            os.path.join(self.checkpoint_dir, f)
            for f in os.listdir(self.checkpoint_dir)
            if f.startswith(self.prefix) and f.endswith('.pkl')
        ]
        paths.sort(key=lambda p: os.path.getmtime(p))
        if len(paths) > self.keep_last_n:
            for old_p in paths[:-self.keep_last_n]:
                os.remove(old_p)

    # Elenco solo i checkpoint che rispettano il prefisso completo e il pattern "_checkpoint_"
    def _list_checkpoints(self):
        return [
            os.path.join(self.checkpoint_dir, f)
            for f in os.listdir(self.checkpoint_dir)
            if f.startswith(self.prefix + "_checkpoint_") and f.endswith(".pkl")
        ]

    # Trova l'ultimo checkpoint per ciascun regime (short/mid/long) filtrando per prefix (quindi per phase)
    def find_latest_per_regime(self):
        import re, os
        rx = re.compile(rf"{re.escape(self.prefix)}_checkpoint_(short|mid|long)_(\d+)\.pkl$")
        latest = {}
        for p in self._list_checkpoints():
            m = rx.search(os.path.basename(p))
            if not m:
                continue
            reg, n = m.group(1), int(m.group(2))
            if (reg not in latest) or (n > latest[reg][0]):
                latest[reg] = (n, p)
        return {reg: path for reg, (n, path) in latest.items()}
    
    def find_latest(self):
        """Trova l'ultimo checkpoint disponibile"""
        checkpoints = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.endswith('.pkl') and self.prefix in f
        ]
        
        if not checkpoints:
            return None
        
        # Estrai timestamp o numeri dai nomi dei checkpoint
        checkpoint_info = []
        for cp in checkpoints:
            try:
                # Prova prima a estrarre un numero dopo l'ultimo underscore
                parts = cp.replace('.pkl', '').split('_')
                if parts[-1].isdigit():
                    num = int(parts[-1])
                    checkpoint_info.append((num, cp))
                else:
                    # Se non c'è numero, usa il tempo di modifica del file
                    mtime = os.path.getmtime(os.path.join(self.checkpoint_dir, cp))
                    checkpoint_info.append((mtime, cp))
            except:
                continue
        
        if checkpoint_info:
            # Trova il checkpoint con il valore più alto (numero o timestamp)
            latest = max(checkpoint_info, key=lambda x: x[0])
            return os.path.join(self.checkpoint_dir, latest[1])
        
        return None
    
    def _list_multi_regime_checkpoints(self):
        base = self.checkpoint_dir
        return [
            os.path.join(base, f) for f in os.listdir(base)
            if f.startswith('multi_regime_checkpoint_') and f.endswith('.pkl')
        ]

    def find_latest_per_regime(self):
        """
        Restituisce il path dell'ultimo checkpoint per ciascun regime
        in base al numero alla fine del nome (…_{N}.pkl).
        """
        import re
        latest = {}  # { 'short': (N, path), 'mid': (N, path), 'long': (N, path) }
        rx = re.compile(r"multi_regime_checkpoint_(short|mid|long)_(\d+)\.pkl$")
        for p in self._list_multi_regime_checkpoints():
            m = rx.search(os.path.basename(p))
            if not m:
                continue
            reg, n = m.group(1), int(m.group(2))
            if (reg not in latest) or (n > latest[reg][0]):
                latest[reg] = (n, p)
        # ritorna solo i path
        return {reg: path for reg, (n, path) in latest.items()}

    