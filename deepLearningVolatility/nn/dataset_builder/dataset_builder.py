# -*- coding: utf-8 -*-
"""

Questo modulo contiene le classi per la costruzione e la gestione di dataset
utilizzati per l'addestramento dei modelli di pricing.

"""

__author__ = "Giacomo Bianchi"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "giacomo.bianchi.97.bs@gmail.com"
__creation_date__ = "01/08/2025"

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
    Dataset builder che supporta qualsiasi StochasticProcess.
    """
    
    def __init__(self, process: Union[str, StochasticProcess], device='cpu', output_dir=None):
        """
        Args:
            process: Nome del processo o istanza di StochasticProcess
            device: Device torch
            output_dir: Directory per output
        """
        self.device = torch.device(device)
        self.output_dir = output_dir
        
        # Setup directories se output_dir è specificato
        if self.output_dir:
            self.setup_directories()
        
        # Crea processo se passato come stringa
        if isinstance(process, str):
            self.process = ProcessFactory.create(process)
        else:
            self.process = process
        
        # Aggiorna bounds basati sul processo
        self._update_param_bounds()
        
        # Inizializza statistiche (come nell'originale)
        self.theta_mean = None
        self.theta_std = None
        self.iv_mean = None
        self.iv_std = None
        self.T_mean = None
        self.T_std = None
        self.k_mean = None
        self.k_std = None
    
    def _update_param_bounds(self):
        """Aggiorna i bounds dei parametri basati sul processo."""
        param_info = self.process.param_info
        self.param_bounds = {}
        
        for i, (name, bounds) in enumerate(zip(param_info.names, param_info.bounds)):
            self.param_bounds[name] = bounds
        
        # Mantieni anche un mapping per indice
        self.param_names = param_info.names
        self.param_defaults = param_info.defaults
    
    def compute_normalization_stats(self, thetas, ivs, Ts=None, ks=None):
        """
        Calcola media e deviazione standard per la normalizzazione.
        
        Args:
            thetas: parametri del modello [N, 4]
            ivs: volatilità implicite (griglia o punti)
            Ts: maturità (solo per pointwise)
            ks: log-moneyness (solo per pointwise)
        """
        # Statistiche per theta
        self.theta_mean = thetas.mean(dim=0)
        self.theta_std = thetas.std(dim=0)
        # Evita divisione per zero
        self.theta_std = torch.where(self.theta_std > 1e-6, self.theta_std, torch.ones_like(self.theta_std))
        
        # Statistiche per IV
        self.iv_mean = ivs.mean()
        self.iv_std = ivs.std()
        if self.iv_std < 1e-6:
            self.iv_std = torch.tensor(1.0, device=self.device)
        
        # Statistiche per T e k (solo per pointwise)
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
        """Normalizza i parametri theta usando z-score normalization"""
        if self.theta_mean is None or self.theta_std is None:
            raise ValueError("Normalization stats not computed. Call compute_normalization_stats first.")
        return (theta - self.theta_mean) / self.theta_std
    
    def denormalize_theta(self, theta_norm):
        """Denormalizza i parametri theta"""
        if self.theta_mean is None or self.theta_std is None:
            raise ValueError("Normalization stats not computed.")
        return theta_norm * self.theta_std + self.theta_mean
    
    def normalize_iv(self, iv):
        """Normalizza le volatilità implicite"""
        if self.iv_mean is None or self.iv_std is None:
            raise ValueError("Normalization stats not computed. Call compute_normalization_stats first.")
        return (iv - self.iv_mean) / self.iv_std
    
    def denormalize_iv(self, iv_norm):
        """Denormalizza le volatilità implicite"""
        if self.iv_mean is None or self.iv_std is None:
            raise ValueError("Normalization stats not computed.")
        return iv_norm * self.iv_std + self.iv_mean
    
    def normalize_T(self, T):
        """Normalizza le maturità"""
        if self.T_mean is None or self.T_std is None:
            raise ValueError("Normalization stats not computed.")
        return (T - self.T_mean) / self.T_std
    
    def denormalize_T(self, T_norm):
        """Denormalizza le maturità"""
        if self.T_mean is None or self.T_std is None:
            raise ValueError("Normalization stats not computed.")
        return T_norm * self.T_std + self.T_mean
    
    def normalize_k(self, k):
        """Normalizza il log-moneyness"""
        if self.k_mean is None or self.k_std is None:
            raise ValueError("Normalization stats not computed.")
        return (k - self.k_mean) / self.k_std
    
    def denormalize_k(self, k_norm):
        """Denormalizza il log-moneyness"""
        if self.k_mean is None or self.k_std is None:
            raise ValueError("Normalization stats not computed.")
        return k_norm * self.k_std + self.k_mean
    
    def save_normalization_stats(self, path=None):
        """Salva le statistiche di normalizzazione"""
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
        """Carica le statistiche di normalizzazione"""
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
        """Crea struttura directory per Google Colab"""
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
        LHS sampling per il processo specifico.
        """
        # Estrai bounds come array
        bounds = np.array([self.param_bounds[name] for name in self.param_names])
        
        # Crea sampler LHS
        sampler = qmc.LatinHypercube(d=self.process.num_params, seed=seed)
        
        # Genera campioni nell'ipercubo unitario
        unit_samples = sampler.random(n=n_samples)
        
        # Scala ai bounds reali
        scaled_samples = qmc.scale(unit_samples, bounds[:, 0], bounds[:, 1])
        
        return torch.tensor(scaled_samples, dtype=torch.float32, device=self.device)
    
    def sample_theta_lhs_restricted(self, n_samples, seed=None, restriction_factor=0.5):
        """
        LHS con range ristretto per test.
        
        Args:
            restriction_factor: Frazione del range da usare (0.5 = usa metà del range)
        """
        bounds = np.array([self.param_bounds[name] for name in self.param_names])
        
        # Riduci il range
        centers = (bounds[:, 0] + bounds[:, 1]) / 2
        ranges = bounds[:, 1] - bounds[:, 0]
        restricted_ranges = ranges * restriction_factor
        
        restricted_bounds = np.column_stack([
            centers - restricted_ranges / 2,
            centers + restricted_ranges / 2
        ])
        
        # Assicura che i bounds ristretti siano validi
        restricted_bounds[:, 0] = np.maximum(restricted_bounds[:, 0], bounds[:, 0])
        restricted_bounds[:, 1] = np.minimum(restricted_bounds[:, 1], bounds[:, 1])
        
        sampler = qmc.LatinHypercube(d=self.process.num_params, seed=seed)
        unit_samples = sampler.random(n=n_samples)
        scaled_samples = qmc.scale(unit_samples, restricted_bounds[:, 0], restricted_bounds[:, 1])
        
        return torch.tensor(scaled_samples, dtype=torch.float32, device=self.device)
    
    def sample_theta_uniform(self, n_samples):
        """Campionamento uniforme standard per confronto"""
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
        Crea un GridNetworkPricer per il processo.
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
        Visualizza il sampling con labels specifici del processo.
        """
        import matplotlib.pyplot as plt
        
        # Genera campioni
        uniform_samples = self.sample_theta_uniform(n_samples).cpu().numpy()
        lhs_samples = self.sample_theta_lhs(n_samples).cpu().numpy()
        
        param_names = self.param_names
        n_params = len(param_names)
        
        # Crea subplot per ogni coppia di parametri
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
        
        # Nascondi subplot vuoti
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
        Ottiene parametri MC ottimizzati per il processo specifico.
        """
        process_name = self.process.__class__.__name__.lower()
        
        if 'rough' in process_name:
            # Rough models need more paths
            return {
                'n_paths': int(base_n_paths * 1.5),
                'use_antithetic': True,
                'adaptive_paths': True,
                'adaptive_dt': True,
                'control_variate': True
            }
        elif 'jump' in process_name:
            # Jump models
            return {
                'n_paths': int(base_n_paths * 1.2),
                'use_antithetic': True,
                'adaptive_paths': False,  # Jumps don't benefit much
                'adaptive_dt': False,
                'control_variate': True
            }
        elif 'heston' in process_name:
            # Heston
            return {
                'n_paths': base_n_paths,
                'use_antithetic': True,
                'adaptive_paths': True,
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
        Costruisce dataset per GridNetworkPricer con normalizzazione opzionale.
        Versione aggiornata per processi generici.
        """
        # Campiona theta
        if restricted:
            thetas = self.sample_theta_lhs_restricted(n_samples)
        else:
            thetas = self.sample_theta_lhs(n_samples)
        
        # Ottieni parametri MC ottimizzati per il processo
        mc_params = self.get_process_specific_mc_params()
        
        # Genera IV surfaces
        ivs = []
        invalid_count = 0
        
        iterator = tqdm(thetas, desc=f"Building {self.process.__class__.__name__} grid dataset") if show_progress else thetas
        
        for i, theta in enumerate(iterator):
            try:
                # Usa parametri MC ottimizzati
                iv = pricer._mc_iv_grid(
                    theta, 
                    n_paths=mc_params['n_paths'],
                    use_antithetic=mc_params['use_antithetic'],
                    adaptive_paths=mc_params['adaptive_paths'],
                    adaptive_dt=mc_params['adaptive_dt'],
                    control_variate=mc_params['control_variate']
                )
                
                # Verifica validità
                if torch.isnan(iv).any() or torch.isinf(iv).any():
                    print(f"\nWarning: Invalid IV for theta {i}: {theta}")
                    invalid_count += 1
                    # Usa volatilità di default basata sul processo
                    default_vol = self._get_process_default_volatility(theta)
                    iv = torch.full_like(iv, default_vol)
                elif (iv < 0.01).any() or (iv > 2.0).any():
                    print(f"\nWarning: Extreme IV values for theta {i}: min={iv.min():.4f}, max={iv.max():.4f}")
                
                ivs.append(iv.cpu())
                
            except Exception as e:
                print(f"\nError processing theta {i}: {e}")
                invalid_count += 1
                # Fallback con volatilità di default
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
        
        # Normalizzazione
        if normalize:
            if compute_stats_from is None:
                # Calcola statistiche da questo dataset
                self.compute_normalization_stats(thetas, iv_tensor)
                
                if show_progress:
                    print(f"\nComputed normalization statistics:")
                    print(f"  Theta mean: {self.theta_mean}")
                    print(f"  Theta std: {self.theta_std}")
                    print(f"  IV mean: {self.iv_mean:.4f}, std: {self.iv_std:.4f}")
            else:
                # Usa statistiche da un altro builder
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
        Versione ottimizzata per Google Colab con checkpoint e gestione memoria.
        Aggiornata per processi generici.
        """
        # Check se esiste già il dataset finale
        if self.output_dir:
            process_name = self.process.__class__.__name__.lower()
            final_dataset_path = f"{self.dirs['datasets']}/{process_name}_grid_dataset_final.pkl"
            if os.path.exists(final_dataset_path):
                print(f"✓ Dataset finale già esistente per {self.process.__class__.__name__}!")
                return self._load_final_dataset(final_dataset_path, normalize, compute_stats_from)
        
        # Inizializza checkpoint manager con prefix specifico per processo
        checkpoint_manager = CheckpointManager(
            self.dirs['checkpoints'] if self.output_dir else './checkpoints',
            prefix=f'{process_name}_grid_dataset'
        )
        
        # Resume da checkpoint se disponibile
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
        
        # Genera solo i theta mancanti
        remaining_samples = n_samples - start_idx
        if remaining_samples <= 0:
            return self._process_completed_dataset(
                all_theta, all_iv, normalize, compute_stats_from
            )
        
        print(f"\nGenerating {remaining_samples} theta samples for {self.process.__class__.__name__}...")
        thetas = self.sample_theta_lhs(remaining_samples)
        
        # Ottieni parametri MC ottimizzati
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
            
            # Process batch con mixed precision
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
                        
                        # Verifica validità
                        if torch.isnan(iv_grid).any() or torch.isinf(iv_grid).any():
                            print(f"⚠️  Invalid IV detected, using fallback")
                            default_vol = self._get_process_default_volatility(theta)
                            iv_grid = torch.full_like(iv_grid, default_vol)
                        elif (iv_grid < 0.01).any() or (iv_grid > 2.0).any():
                            print(f"⚠️  Extreme IV values: [{iv_grid.min():.4f}, {iv_grid.max():.4f}]")
                            iv_grid = torch.clamp(iv_grid, 0.01, 2.0)
                        
                        iv_batch.append(iv_grid.cpu())
                        
                    except Exception as e:
                        print(f"❌ Error processing theta: {e}")
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
    

class MultiRegimeDatasetBuilder(DatasetBuilder):
    """
    Estende DatasetBuilder per gestire dataset multi-regime con processi generici.
    """
    
    def __init__(self, process: Union[str, StochasticProcess], 
                 device='cpu', output_dir=None, dataset_type='train',
                 enable_smile_repair: bool = True,
                 smile_repair_method: str = 'pchip'):
        
        super().__init__(process, device, output_dir)
        self.dataset_type = dataset_type
        
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
        Costruisce dataset per MultiRegimeGridPricer.
        
        Args:
            multi_regime_pricer: istanza di MultiRegimeGridPricer
            n_samples: numero di campioni theta
            n_paths: paths per Monte Carlo
            batch_size: campioni per batch
            checkpoint_every: salva checkpoint ogni N batch
            normalize: se normalizzare i dati
            compute_stats_from: MultiRegimeDatasetBuilder da cui prendere stats
            resume_from: checkpoint da cui riprendere
            mixed_precision: usa FP16 su GPU
            chunk_size: dimensione chunk per MC
            sample_method: 'shared' (stessi theta per tutti i regimi) o 
                          'independent' (theta diversi per regime)
        
        Returns:
            dict con keys 'short', 'mid', 'long', ognuno contenente (theta, iv)
        """
        
        process_name = self.process.__class__.__name__.lower()
        
        # Check se esiste già il dataset finale
        if self.output_dir:
            final_path = f"{self.regime_dirs['unified']}/{process_name}_multi_regime_dataset_final.pkl"
            if os.path.exists(final_path) and not resume_from and not force_regenerate:
                print(f"✓ Multi-regime dataset finale già esistente per {self.process.__class__.__name__}!")
                return self._load_final_multi_regime_dataset(
                    final_path, normalize, compute_stats_from
                )
        
        # Inizializza checkpoint manager
        checkpoint_manager = CheckpointManager(
            self.dirs['checkpoints'] if self.output_dir else './checkpoints',
            prefix='multi_regime'
        )
        
        # Check for existing checkpoints
        latest_checkpoint = checkpoint_manager.find_latest()
        
        # Resume da checkpoint se disponibile
        start_idx = 0
        all_data = {
            'short': {'theta': [], 'iv': []},
            'mid': {'theta': [], 'iv': []},
            'long': {'theta': [], 'iv': []}
        }
        
        if latest_checkpoint and not force_regenerate:
            print(f"Found checkpoint: {os.path.basename(latest_checkpoint)}")
            with open(latest_checkpoint, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            all_data = checkpoint_data['data']
            start_idx = checkpoint_data['n_samples_done']
            current_regime = checkpoint_data.get('current_regime', 'short')
            
            if current_regime == 'short' and start_idx >= n_samples:
                # Short completato, inizia dal mid
                regimes = ['mid', 'long']
            elif current_regime == 'mid' and start_idx >= n_samples:
                # Mid completato, inizia da long
                regimes = ['long']
            else:
                # Riprendi dal regime corrente
                regimes = [current_regime, 'mid', 'long'] if current_regime == 'short' else \
                          [current_regime, 'long'] if current_regime == 'mid' else \
                          ['long']
                            
            print(f"Resuming from sample {start_idx}/{n_samples}")
            
            if start_idx >= n_samples:
                print("Dataset già completo!")
                return self._process_completed_multi_regime_dataset(
                    all_data, normalize, compute_stats_from
                )
            
        # Genera theta samples
        remaining_samples = n_samples - start_idx
        if remaining_samples <= 0:
            return self._process_completed_multi_regime_dataset(
                all_data, normalize, compute_stats_from
            )
        
        print(f"\nGenerating {remaining_samples} theta samples...")
        
        if sample_method == 'shared':
            # Stessi theta per tutti i regimi
            thetas = self.sample_theta_lhs(remaining_samples)
            theta_dict = {
                'short': thetas,
                'mid': thetas,
                'long': thetas
            }
        else:
            # Theta indipendenti per regime (potenzialmente con bounds diversi)
            theta_dict = {
                'short': self.sample_theta_lhs(remaining_samples, seed=42),
                'mid': self.sample_theta_lhs(remaining_samples, seed=43),
                'long': self.sample_theta_lhs(remaining_samples, seed=44)
            }
        
        # Process in batches per ogni regime
        regimes = ['short', 'mid', 'long']
        
        for regime in regimes:
            print(f"\n{'='*50}")
            print(f"Processing {regime.upper()} TERM regime")
            print(f"{'='*50}")
            
            regime_thetas = theta_dict[regime]
            pricer = getattr(multi_regime_pricer, f"{regime}_term_pricer")
            
            for batch_idx in range(0, len(regime_thetas), batch_size):
                batch_end = min(batch_idx + batch_size, len(regime_thetas))
                batch_thetas = regime_thetas[batch_idx:batch_end]
                
                current_total = start_idx + batch_end
                print(f"\n[{regime} - Batch {(batch_idx//batch_size)+1}] Samples {start_idx + batch_idx + 1}-{current_total}/{n_samples}")
                
                batch_start_time = time.time()
                
                # Process batch
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
                            
                            # Verifica validità
                            if torch.isnan(iv_grid).any() or torch.isinf(iv_grid).any():
                                print(f"⚠️  Invalid IV detected, using fallback")
                                iv_grid = torch.full_like(iv_grid, 0.2)
                            
                            iv_batch.append(iv_grid.cpu())
                            
                        except Exception as e:
                            print(f"❌ Error processing theta: {e}")
                            iv_grid = torch.full(
                                (len(pricer.Ts), len(pricer.logKs)),
                                0.2,
                                device=self.device
                            )
                            iv_batch.append(iv_grid.cpu())
                        
                        # Clear cache periodicamente
                        if len(iv_batch) % 10 == 0 and self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                
                # Aggiungi ai dati
                all_data[regime]['theta'].extend([t.cpu() for t in batch_thetas])
                all_data[regime]['iv'].extend(iv_batch)
                
                batch_time = time.time() - batch_start_time
                print(f"✓ {regime} batch completed in {batch_time:.1f}s")
                
                # Checkpoint dopo ogni N batch
                if ((batch_idx // batch_size + 1) % checkpoint_every == 0):
                    checkpoint_data = {
                        'data': all_data,
                        'n_samples_done': start_idx + batch_end,
                        'n_samples_total': n_samples,
                        'current_regime': regime,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    checkpoint_name = f"multi_regime_checkpoint_{regime}_{start_idx + batch_end}.pkl"
                    checkpoint_manager.save_checkpoint(checkpoint_data, checkpoint_name)
                
                # Clear GPU memory
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Checkpoint dopo ogni regime
            checkpoint_data = {
                'data': all_data,
                'n_samples_done': start_idx + len(regime_thetas),
                'n_samples_total': n_samples,
                'n_paths': n_paths,
                'current_regime': regime,
                'timestamp': datetime.now().isoformat()
            }
            
            checkpoint_name = f"checkpoint_{regime}_{start_idx + len(regime_thetas)}.pkl"
            checkpoint_manager.save_checkpoint(checkpoint_data, checkpoint_name)
        
        # Processa dataset completo
        return self._process_completed_multi_regime_dataset(
            all_data, normalize, compute_stats_from, cleanup_checkpoints=True
        )
    
    def _process_completed_multi_regime_dataset(self, all_data, normalize,
                                              compute_stats_from, cleanup_checkpoints=False):
        """Processa e salva dataset multi-regime completo"""
        
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
        
        # Salva dataset raw
        if self.output_dir:
            # Salva per regime
            for regime in ['short', 'mid', 'long']:
                regime_data = {
                    'theta': datasets[regime]['theta'].cpu(),
                    'iv': datasets[regime]['iv'].cpu(),
                    'timestamp': datetime.now().isoformat()
                }
                regime_path = f"{self.regime_dirs[regime]}/dataset_final.pkl"
                with open(regime_path, 'wb') as f:
                    pickle.dump(regime_data, f)
                print(f"✓ {regime} dataset saved: {regime_path}")
            
            # Salva dataset unificato
            unified_data = {
                'short': datasets['short'],
                'mid': datasets['mid'],
                'long': datasets['long'],
                'timestamp': datetime.now().isoformat()
            }
            unified_path = f"{self.regime_dirs['unified']}/multi_regime_dataset_final.pkl"
            with open(unified_path, 'wb') as f:
                pickle.dump(unified_data, f)
            print(f"✓ Unified dataset saved: {unified_path}")
            
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
                # Calcola nuove statistiche
                # Assumiamo theta comuni (o prendiamo la media)
                theta_all = datasets['short']['theta']  # Se shared
                
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
                # Usa statistiche esistenti
                self._copy_regime_stats_from(compute_stats_from)
                print(f"\nUsing normalization statistics from training set")
            
            # Applica normalizzazione
            normalized_datasets = {}
            
            for regime in ['short', 'mid', 'long']:
                theta_norm = self.normalize_theta(datasets[regime]['theta'])
                iv_norm = self.normalize_iv_regime(datasets[regime]['iv'], regime)
                
                normalized_datasets[regime] = {
                    'theta': theta_norm,
                    'iv': iv_norm
                }
            
            # Salva dataset normalizzati
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
                norm_path = f"{self.regime_dirs['unified']}/multi_regime_dataset_normalized.pkl"
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
        """Mantiene solo gli ultimi N checkpoint"""
        checkpoints = sorted([
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith(self.prefix) and f.endswith('.pkl')
        ])
        
        if len(checkpoints) > self.keep_last_n:
            for old_cp in checkpoints[:-self.keep_last_n]:
                os.remove(f"{self.checkpoint_dir}/{old_cp}")
    
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
    