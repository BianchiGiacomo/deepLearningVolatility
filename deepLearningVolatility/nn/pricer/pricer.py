# -*- coding: utf-8 -*-

"""Approximation of implied volatility surfaces using neural networks.

This module provides a framework for learning and generating implied volatility
surfaces for various stochastic models. It includes implementations for
grid-based, point-wise, and multi-regime architectures that map stochastic
process parameters to their corresponding IV surfaces.

The core components are designed to be generic, relying on a common
`StochasticProcess` interface to generate training data and a `NeuralSurfacePricer`
base class to define the model architectures. Once a network is trained, it can
produce entire IV surfaces, which can then be used in standard pricing formulas.
"""

__author__ = "Giacomo Bianchi"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "giacomo.bianchi.97.bs@gmail.com"
__creation_date__ = "01/08/2025"

import math
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Union
from scipy.interpolate import RBFInterpolator
from scipy.interpolate import interp1d, PchipInterpolator

from deepLearningVolatility.nn.modules import BlackScholes
from deepLearningVolatility.instruments import EuropeanOption, BrownianStock
from deepLearningVolatility.stochastic.stochastic_interface import StochasticProcess

import warnings
warnings.filterwarnings('ignore')

# Activation Function Map
ACTIVATION_MAP = {
    "Threshold": nn.Threshold,
    "ReLU": nn.ReLU,
    "RReLU": nn.RReLU,
    "Hardtanh": nn.Hardtanh,
    "ReLU6": nn.ReLU6,
    "Sigmoid": nn.Sigmoid,
    "Hardsigmoid": nn.Hardsigmoid,
    "Tanh": nn.Tanh,
    "SiLU": nn.SiLU,
    "Mish": nn.Mish,
    "Hardswish": nn.Hardswish,
    "ELU": nn.ELU,
    "CELU": nn.CELU,
    "SELU": nn.SELU,
    "GLU": nn.GLU,
    "GELU": nn.GELU,
    "Hardshrink": nn.Hardshrink,
    "LeakyReLU": nn.LeakyReLU,
    "LogSigmoid": nn.LogSigmoid,
    "Softplus": nn.Softplus,
    "Softshrink": nn.Softshrink,
    "PReLU": nn.PReLU,
    "Softsign": nn.Softsign,
    "Tanhshrink": nn.Tanhshrink,
}

def build_network(input_dim, output_dim, hidden_layers, activation='ReLU'):
    """
    Builds a flexible neural network architecture.
    
    Args:
        input_dim: size of the input layer
        output_dim: size of the output layer
        hidden_layers: list of neuron counts for each hidden layer
                       e.g. [128, 64, 32] for three hidden layers
        activation: name of activation function or nn.Module class
    """

    layers = []
    
    # Handle the case where activation is a string
    if isinstance(activation, str):
        if activation not in ACTIVATION_MAP:
            raise ValueError(f"Activation {activation} not supported. Choose from {list(ACTIVATION_MAP.keys())}")
        activation_fn = ACTIVATION_MAP[activation]
    else:
        activation_fn = activation
    
    # Input layer
    if len(hidden_layers) > 0:
        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        # Some activations require specific parameters
        if activation_fn in [nn.Threshold]:
            layers.append(activation_fn(0.0, 0.0))
        elif activation_fn in [nn.RReLU]:
            layers.append(activation_fn())
        elif activation_fn in [nn.Hardtanh]:
            layers.append(activation_fn())
        else:
            layers.append(activation_fn())
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            if activation_fn in [nn.Threshold]:
                layers.append(activation_fn(0.0, 0.0))
            elif activation_fn in [nn.RReLU]:
                layers.append(activation_fn())
            elif activation_fn in [nn.Hardtanh]:
                layers.append(activation_fn())
            else:
                layers.append(activation_fn())
        
        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
    else:
        # Case without hidden layers
        layers.append(nn.Linear(input_dim, output_dim))
    
    return nn.Sequential(*layers)

class VolatilityInterpolator:
    """
    Volatility surface interpolator using Radial Basis Functions (RBF).
    RBF is well suited for volatility surfaces because it:
    - Ensures smoothness
    - Handles sparse data effectively
    - Respects typical smile shapes
    """

    def __init__(self, method='thin_plate_spline', extrapolation='flat'):
        self.method = method  # 'thin_plate_spline', 'multiquadric', 'inverse_multiquadric', 'gaussian', 'linear', 'cubic'
        self.extrapolation = extrapolation  # 'flat', 'linear'
        self.interpolator = None
        self.T_min = None
        self.T_max = None
        self.k_min = None
        self.k_max = None
        self.boundary_values = {}
        
    def fit(self, T_grid, k_grid, iv_surface):
        """
        Fits the interpolator on a volatility grid.
        
        Args:
            T_grid: array of maturities
            k_grid: array of log-moneyness values
            iv_surface: matrix [len(T_grid), len(k_grid)] of implied volatilities
        """

        # Create meshgrid
        T_mesh, k_mesh = np.meshgrid(T_grid, k_grid, indexing='ij')
        
        # Flatten for RBF
        points = np.column_stack([T_mesh.ravel(), k_mesh.ravel()])
        values = iv_surface.ravel()
        
        # Save bounds for extrapolation
        self.T_min, self.T_max = T_grid.min(), T_grid.max()
        self.k_min, self.k_max = k_grid.min(), k_grid.max()
        
        # Save the values at the edges for flat extrapolation
        if self.extrapolation == 'flat':
            # Boundaies for T
            self.boundary_values['T_min'] = iv_surface[0, :]  # First row
            self.boundary_values['T_max'] = iv_surface[-1, :]  # Last row
            # Boundaies for k
            self.boundary_values['k_min'] = iv_surface[:, 0]  # First column
            self.boundary_values['k_max'] = iv_surface[:, -1]  # Last column
            # Angles
            self.boundary_values['corner_TminKmin'] = iv_surface[0, 0]
            self.boundary_values['corner_TminKmax'] = iv_surface[0, -1]
            self.boundary_values['corner_TmaxKmin'] = iv_surface[-1, 0]
            self.boundary_values['corner_TmaxKmax'] = iv_surface[-1, -1]
        
        # Create RBF interpolator
        self.interpolator = RBFInterpolator(points, values, kernel=self.method)
        
        # Also save original grid for linear interpolation if needed
        self.T_grid = T_grid
        self.k_grid = k_grid
        self.iv_surface = iv_surface
        
    def __call__(self, T, k):
        """
        Interpolates or extrapolates implied volatility values.
        
        Args:
            T: maturity or array of maturities
            k: log-moneyness or array of log-moneyness values
        
        Returns:
            Array of interpolated implied volatilities
        """

        if self.interpolator is None:
            raise ValueError("Interpolator not fitted. Call fit() first.")
        
        # Convert to numpy arrays
        T = np.asarray(T)
        k = np.asarray(k)
        
        # Gestisci input scalari
        scalar_input = False
        if T.ndim == 0:
            T = T.reshape(1)
            k = k.reshape(1)
            scalar_input = True
        
        # Crea punti per interpolazione
        points = np.column_stack([T, k])
        
        if self.extrapolation == 'flat':
            # Applica estrapolazione flat
            result = np.zeros(len(points))
            
            for i, (t, k_val) in enumerate(points):
                # Controlla se il punto è dentro i bounds
                if self.T_min <= t <= self.T_max and self.k_min <= k_val <= self.k_max:
                    # Interpola normalmente
                    result[i] = self.interpolator(points[i:i+1])[0]
                else:
                    # Estrapolazione flat
                    if t < self.T_min and k_val < self.k_min:
                        result[i] = self.boundary_values['corner_TminKmin']
                    elif t < self.T_min and k_val > self.k_max:
                        result[i] = self.boundary_values['corner_TminKmax']
                    elif t > self.T_max and k_val < self.k_min:
                        result[i] = self.boundary_values['corner_TmaxKmin']
                    elif t > self.T_max and k_val > self.k_max:
                        result[i] = self.boundary_values['corner_TmaxKmax']
                    elif t < self.T_min:
                        # Interpola lungo k al T minimo
                        k_idx = np.searchsorted(self.k_grid, k_val)
                        if k_idx == 0:
                            result[i] = self.boundary_values['T_min'][0]
                        elif k_idx >= len(self.k_grid):
                            result[i] = self.boundary_values['T_min'][-1]
                        else:
                            # Interpolazione lineare
                            w = (k_val - self.k_grid[k_idx-1]) / (self.k_grid[k_idx] - self.k_grid[k_idx-1])
                            result[i] = (1-w) * self.boundary_values['T_min'][k_idx-1] + w * self.boundary_values['T_min'][k_idx]
                    elif t > self.T_max:
                        # Interpola lungo k al T massimo
                        k_idx = np.searchsorted(self.k_grid, k_val)
                        if k_idx == 0:
                            result[i] = self.boundary_values['T_max'][0]
                        elif k_idx >= len(self.k_grid):
                            result[i] = self.boundary_values['T_max'][-1]
                        else:
                            w = (k_val - self.k_grid[k_idx-1]) / (self.k_grid[k_idx] - self.k_grid[k_idx-1])
                            result[i] = (1-w) * self.boundary_values['T_max'][k_idx-1] + w * self.boundary_values['T_max'][k_idx]
                    elif k_val < self.k_min:
                        # Interpola lungo T al k minimo
                        T_idx = np.searchsorted(self.T_grid, t)
                        if T_idx == 0:
                            result[i] = self.boundary_values['k_min'][0]
                        elif T_idx >= len(self.T_grid):
                            result[i] = self.boundary_values['k_min'][-1]
                        else:
                            w = (t - self.T_grid[T_idx-1]) / (self.T_grid[T_idx] - self.T_grid[T_idx-1])
                            result[i] = (1-w) * self.boundary_values['k_min'][T_idx-1] + w * self.boundary_values['k_min'][T_idx]
                    else:  # k_val > self.k_max
                        # Interpola lungo T al k massimo
                        T_idx = np.searchsorted(self.T_grid, t)
                        if T_idx == 0:
                            result[i] = self.boundary_values['k_max'][0]
                        elif T_idx >= len(self.T_grid):
                            result[i] = self.boundary_values['k_max'][-1]
                        else:
                            w = (t - self.T_grid[T_idx-1]) / (self.T_grid[T_idx] - self.T_grid[T_idx-1])
                            result[i] = (1-w) * self.boundary_values['k_max'][T_idx-1] + w * self.boundary_values['k_max'][T_idx]
        else:
            # Interpolazione standard (RBF gestisce l'estrapolazione)
            result = self.interpolator(points)
        
        if scalar_input:
            return result[0]
        return result
    
    
class SmileRepair:
    """
    Utility to repair volatility smiles with zeros or extreme values.
    """

    @staticmethod
    def repair_smile_simple(iv_smile, logK, min_valid_points=3, 
                           fallback_vol=0.2, method='pchip'):
        """
        Repairs a 1D volatility smile by interpolating over valid points.
        
        Args:
            iv_smile: 1D array of implied volatilities at various strikes
            logK: 1D array of log-moneyness values
            min_valid_points: minimum number of valid points required for interpolation
            fallback_vol: fallback vol if too few valid points
            method: interpolation method ('linear', 'cubic', 'pchip')
        
        Returns:
            iv_smile_repaired: repaired volatility smile
            n_repaired: number of points repaired
        """

        iv_smile = np.array(iv_smile)
        logK = np.array(logK)
        
        # Identifica punti validi (non zero e ragionevoli)
        valid_mask = (iv_smile > 0.01) & (iv_smile < 2.0) & ~np.isnan(iv_smile)
        n_valid = valid_mask.sum()
        
        if n_valid < min_valid_points:
            # Troppo pochi punti validi, usa fallback
            if n_valid > 0:
                # Usa la media dei punti validi
                fallback_vol = iv_smile[valid_mask].mean()
            return np.full_like(iv_smile, fallback_vol), len(iv_smile) - n_valid
        
        if n_valid == len(iv_smile):
            # Tutti i punti sono validi
            return iv_smile, 0
        
        # Interpola usando solo punti validi
        logK_valid = logK[valid_mask]
        iv_valid = iv_smile[valid_mask]
        
        try:
            if method == 'pchip':
                # PCHIP preserva la forma e evita oscillazioni
                interpolator = PchipInterpolator(logK_valid, iv_valid, extrapolate=False)
            else:
                # Linear o cubic
                interpolator = interp1d(logK_valid, iv_valid, kind=method, 
                                      bounds_error=False, fill_value='extrapolate')
            
            # Ripara punti non validi
            iv_repaired = iv_smile.copy()
            invalid_mask = ~valid_mask
            iv_repaired[invalid_mask] = interpolator(logK[invalid_mask])
            
            # Gestisci extrapolazione con flat extension
            # Per punti a sinistra del range valido
            left_invalid = invalid_mask & (logK < logK_valid.min())
            if left_invalid.any():
                iv_repaired[left_invalid] = iv_valid[0]
            
            # Per punti a destra del range valido
            right_invalid = invalid_mask & (logK > logK_valid.max())
            if right_invalid.any():
                iv_repaired[right_invalid] = iv_valid[-1]
            
            # Assicura valori ragionevoli
            iv_repaired = np.clip(iv_repaired, 0.01, 2.0)
            
            # Se ancora ci sono NaN (non dovrebbe succedere), usa fallback
            nan_mask = np.isnan(iv_repaired)
            if nan_mask.any():
                iv_repaired[nan_mask] = fallback_vol
            
            return iv_repaired, invalid_mask.sum()
            
        except Exception as e:
            # Fallback in caso di errore
            print(f"Interpolation failed: {e}, using fallback")
            return np.full_like(iv_smile, fallback_vol), len(iv_smile) - n_valid
    
    @staticmethod
    def repair_surface(iv_surface, logK, min_valid_per_smile=3, 
                      fallback_vol=0.2, method='pchip'):
        """
        Repairs an entire IV surface smile by smile.
        
        Args:
            iv_surface: 2D array or tensor [n_maturities, n_strikes]
            logK: 1D array of log-moneyness values
        
        Returns:
            iv_surface_repaired: repaired IV surface
            repair_stats: dict with repair statistics
        """

        if torch.is_tensor(iv_surface):
            device = iv_surface.device
            iv_surface_np = iv_surface.cpu().numpy()
            logK_np = logK.cpu().numpy() if torch.is_tensor(logK) else logK
        else:
            device = None
            iv_surface_np = iv_surface
            logK_np = logK
        
        n_maturities, n_strikes = iv_surface_np.shape
        iv_repaired = np.zeros_like(iv_surface_np)
        total_repaired = 0
        problematic_maturities = []
        
        for i in range(n_maturities):
            smile = iv_surface_np[i, :]
            smile_repaired, n_rep = SmileRepair.repair_smile_simple(
                smile, logK_np, min_valid_per_smile, fallback_vol, method
            )
            iv_repaired[i, :] = smile_repaired
            total_repaired += n_rep
            
            if n_rep > n_strikes * 0.5:  # Più del 50% riparato
                problematic_maturities.append(i)
        
        repair_stats = {
            'total_repaired': total_repaired,
            'total_points': n_maturities * n_strikes,
            'repair_ratio': total_repaired / (n_maturities * n_strikes),
            'problematic_maturities': problematic_maturities
        }
        
        if device is not None:
            iv_repaired = torch.tensor(iv_repaired, device=device, dtype=torch.float32)
        
        return iv_repaired, repair_stats

class NeuralSurfacePricer(nn.Module):
    """
    Generic interface for neural network–based volatility surface pricers.
    """

    def __init__(self, device='cpu', r=0.0):
        super().__init__()
        self.device = torch.device(device)
        self.r = r  # Tasso risk-free
        
        # Statistiche di normalizzazione
        self.register_buffer('theta_mean', None)
        self.register_buffer('theta_std', None)
        self.register_buffer('iv_mean', None)
        self.register_buffer('iv_std', None)
        
    def set_normalization_stats(self, theta_mean, theta_std, iv_mean, iv_std):
        """Imposta le statistiche di normalizzazione"""
        self.register_buffer('theta_mean', theta_mean.to(self.device))
        self.register_buffer('theta_std', theta_std.to(self.device))
        self.register_buffer('iv_mean', torch.tensor(iv_mean, device=self.device))
        self.register_buffer('iv_std', torch.tensor(iv_std, device=self.device))
    
    def normalize_theta(self, theta):
        """Normalize model parameters theta"""
        if self.theta_mean is None or self.theta_std is None:
            return theta  # Se non ci sono stats, restituisci non normalizzato
        return (theta - self.theta_mean) / self.theta_std
    
    def denormalize_theta(self, theta_norm):
        """Denormalize model parameters theta"""
        if self.theta_mean is None or self.theta_std is None:
            return theta_norm
        return theta_norm * self.theta_std + self.theta_mean
    
    def normalize_iv(self, iv):
        """Normalize implied volatilities"""
        if self.iv_mean is None or self.iv_std is None:
            return iv
        return (iv - self.iv_mean) / self.iv_std
    
    def denormalize_iv(self, iv_norm):
        """Denormalize implied volatilities"""
        if self.iv_mean is None or self.iv_std is None:
            return iv_norm
        return iv_norm * self.iv_std + self.iv_mean
        
    def fit(self, *args, **kwargs):
        raise NotImplementedError
        
    def price_iv(self, theta):
        raise NotImplementedError
        
    def to(self, device):
        """Override per gestire correttamente il device"""
        self.device = device
        return super().to(device)
    
    def save_model_only(self, path: str):
        """Save only model weights, excluding normalization stats."""
        # Crea una copia dello state dict
        state = self.state_dict()
        
        # Rimuovi le statistiche di normalizzazione
        keys_to_remove = []
        for key in state.keys():
            if key in ['theta_mean', 'theta_std', 'iv_mean', 'iv_std', 'T_mean', 'T_std', 'k_mean', 'k_std']:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            state.pop(key)
        
        # Salva solo i pesi del modello
        torch.save(state, path)
    
    def save_full(self, path: str):
        """Save the full model including normalization stats."""
        torch.save(self.state_dict(), path)

#--------------------------------------------------------------
class ZeroAbsorptionHandler:
    """
    Handles zero-absorption paths in stochastic processes.
    Provides utilities for absorbed-path option pricing.
    """

    
    @staticmethod
    def find_absorption_times(S_paths, dt):
        """
        DEPRECATED: Use process.handle_absorption() instead.
        Kept for backward compatibility.
        """
        # Delega a una implementazione comune
        zero_mask = S_paths <= 1e-10
        cumsum = zero_mask.cumsum(dim=1)
        first_zero_mask = (cumsum == 1) & zero_mask
        
        padded_mask = torch.cat([first_zero_mask, 
                                torch.ones(S_paths.shape[0], 1, dtype=torch.bool, device=S_paths.device)], 
                               dim=1)
        
        absorption_indices = padded_mask.to(torch.float32).argmax(dim=1)
        n_steps = S_paths.shape[1]
        absorbed_mask = absorption_indices < n_steps
        absorption_times = absorption_indices.float() * dt
        
        return absorption_times, absorbed_mask
    
    @staticmethod
    def compute_barrier_adjusted_price_vectorized(S_paths, K, T, r, dt, 
                                                  process=None):
        """
        Computes option price with zero-barrier adjustment.
        
        Args:
            S_paths: underlying price paths [n_paths, n_steps]
            K: strike price (scalar or tensor)
            T: maturity
            r: risk-free rate
            dt: time step
            process: optional custom stochastic process for absorption handling
        
        Returns:
            price: barrier-adjusted option price
            effective_paths: number of paths surviving to T
        """

        n_paths = S_paths.shape[0]
        
        # Usa il processo se fornito, altrimenti fallback all'implementazione locale
        if process and hasattr(process, 'handle_absorption'):
            absorption_times, absorbed_mask = process.handle_absorption(S_paths, dt)
        else:
            absorption_times, absorbed_mask = ZeroAbsorptionHandler.find_absorption_times(S_paths, dt)
        
        # Paths sopravvissuti
        survived_mask = ~absorbed_mask
        n_survived = survived_mask.sum().item()
        
        if n_survived == 0:
            return 0.0, 0
        
        # Calcola payoff per TUTTI i paths (0 per quelli assorbiti)
        ST = S_paths[:, -1]
        payoff = (ST - K).clamp(min=0.0)
        
        # Azzera il payoff per i paths assorbiti
        payoff = payoff * survived_mask.float()
        
        # Prezzo medio su tutti i paths
        price = payoff.mean() * torch.exp(torch.tensor(-r * T))
        
        return price.item(), n_survived
    
    @staticmethod
    def compute_implied_vol_with_absorption_batch(S_paths_list, K_list, T_list, r, dt, 
                                                  spot=1.0, process=None):
        """
        Computes implied volatilities for batches of options, handling absorption.
        
        Args:
        S_paths_list: List of S_paths tensors for different options
        K_list: List of strikes
        T_list: List of maturities
        r: Risk-free rate
        dt: Time step
        spot: Spot price
        process: (optional) Stochastic process for custom handle_absorption
        
        Returns:
        iv_list: List of implied volatilities
        absorbed_ratios: List of absorption ratios
        """
        from deepLearningVolatility.nn import BlackScholes
        from deepLearningVolatility.instruments import EuropeanOption, BrownianStock
        
        iv_list = []
        absorbed_ratios = []
        
        for S_paths, K, T in zip(S_paths_list, K_list, T_list):
            n_paths = S_paths.shape[0]
            
            # Calcola prezzo con gestione absorption
            price, n_survived = ZeroAbsorptionHandler.compute_barrier_adjusted_price_vectorized(
                S_paths, K, T, r, dt, process
            )
            
            absorbed_ratio = 1.0 - (n_survived / n_paths)
            absorbed_ratios.append(absorbed_ratio)
            
            # Calcola IV
            if price > 1e-10 and absorbed_ratio < 0.95:
                try:
                    bs = BlackScholes(
                        EuropeanOption(
                            BrownianStock(),
                            strike=K,
                            maturity=T
                        )
                    )
                    iv = bs.implied_volatility(
                        log_moneyness=torch.log(torch.tensor(spot/K)),
                        time_to_maturity=T,
                        price=price
                    )
                    iv_list.append(float(iv))
                except:
                    # Stima volatilità dai returns dei path sopravvissuti
                    if n_survived > 10:
                        survived_mask = S_paths[:, -1] > 0
                        S_survived = S_paths[survived_mask]
                        if S_survived.shape[0] > 0:
                            returns = torch.log(S_survived[:, 1:] / S_survived[:, :-1] + 1e-10)
                            vol_estimate = returns.std() * torch.sqrt(1/dt)
                            iv_list.append(vol_estimate.item())
                        else:
                            iv_list.append(0.2)
                    else:
                        iv_list.append(0.2)
            else:
                iv_list.append(0.0 if absorbed_ratio > 0.95 else 0.2)
        
        return iv_list, absorbed_ratios

class GridNetworkPricer(NeuralSurfacePricer):
    """Models the entire implied volatility surface for a given set of parameters.

    This class implements a neural pricer that learns the mapping between the
    parameters of a stochastic process (theta) and its corresponding
    implied volatility (IV) surface.

    The architecture, which generates the entire grid (maturities vs.
    log-moneyness) in a single pass, is inspired by the approach described
    in the paper "Deep Learning Volatility" by Horvath, Muguruza, and
    Tomas (2019). This method is efficient as a single forward pass
    produces the entire surface.

    Attributes:
        process (StochasticProcess): The underlying stochastic model, used to
            generate training data.
        net (nn.Module): The neural network that maps model parameters to the
            volatility surface.
        Ts (torch.Tensor): A tensor containing the maturities of the output grid.
        logKs (torch.Tensor): A tensor containing the log-moneyness values
            of the output grid.
    """
    def __init__(self,
                 maturities: torch.Tensor,
                 logK: torch.Tensor,
                 process: StochasticProcess,
                 hidden_layers: list = [128],
                 activation: str = 'ReLU',
                 dt: float = 1/365,
                 device: str = 'cpu',
                 r: float = 0.0,
                 interpolation_method: str = 'thin_plate_spline',
                 extrapolation: str = 'flat',
                 enable_smile_repair: bool = True,
                 smile_repair_method: str = 'pchip'):
        
        super().__init__(device=device, r=r)
        
        self.register_buffer('Ts', maturities.to(device))
        self.register_buffer('logKs', logK.to(device))
        self.process = process
        self.dt = dt
        self.out_dim = len(self.Ts) * len(self.logKs)
        
        # Rete neurale con architettura dinamica basata su num_params
        self.net = build_network(
            process.num_params,
            self.out_dim, 
            hidden_layers, 
            activation
        ).to(device)
        
        # Interpolatore
        self.interpolator = VolatilityInterpolator(
            method=interpolation_method,
            extrapolation=extrapolation
        )
        self._interpolator_fitted = False
        self.enable_smile_repair = enable_smile_repair
        self.smile_repair_method = smile_repair_method
        self.last_repair_stats = None
    
    @torch.no_grad()
    def _mc_iv_grid(self,
                    theta: torch.Tensor,
                    n_paths: int,
                    spot: float = 1.0,
                    price_floor: float = 1e-8,
                    use_antithetic: bool = True,
                    adaptive_paths: bool = False,
                    adaptive_dt: bool = True,
                    control_variate: bool = True,
                    chunk_size: int = None,
                    handle_absorption: bool = True) -> torch.Tensor:
        """
        Calcola IV su griglia via Monte Carlo.
        Versione aggiornata che usa l'interfaccia process.
        """
        # Valida parametri
        is_valid, error_msg = self.process.validate_theta(theta)
        if not is_valid:
            raise ValueError(f"Invalid parameters: {error_msg}")
        
        # Determina dt adattivo
        T_min = self.Ts.min().item()
        if adaptive_dt:
            if T_min <= 0.1:
                dt_base = 1/1460
            elif T_min < 1.0:
                dt_base = 1/730
            else:
                dt_base = 1/365
        else:
            dt_base = self.dt
        
        # Chunk size default
        if chunk_size is None:
            chunk_size = 50000 if self.device.type == 'cuda' else 20000
        
        # Inizializza griglia risultati
        iv = torch.zeros(len(self.Ts), len(self.logKs), device=self.device)
        
        if handle_absorption and self.process.supports_absorption:
            absorption_stats = torch.zeros(len(self.Ts), len(self.logKs), device=self.device)
        
        # Processa ogni scadenza
        for iT, T in enumerate(self.Ts):
            T_val = T.item()
            disc = math.exp(-self.r * T_val)
            
            # Step necessari
            n_steps_T = int(round(T_val / dt_base)) + 1
            
            # Numero di paths adattivo
            n_paths_T = n_paths
            if adaptive_paths:
                if T_val <= 0.05:
                    n_paths_T = int(n_paths * 5)
                elif T_val <= 0.1:
                    n_paths_T = int(n_paths * 3)
                elif T_val <= 0.25:
                    n_paths_T = int(n_paths * 1.5)
            
            # Pre-calcola tutti gli strikes per questa maturità
            K_values = spot * torch.exp(self.logKs)
            
            # Inizializza accumulatori per statistics
            payoff_sums = torch.zeros(len(self.logKs), device=self.device)
            n_valid_per_strike = torch.zeros(len(self.logKs), device=self.device)
            
            if control_variate:
                dS_sums = torch.zeros(len(self.logKs), device=self.device)
                payoff_dS_sums = torch.zeros(len(self.logKs), device=self.device)
                dS_sq_sums = torch.zeros(len(self.logKs), device=self.device)
            
            total_paths_processed = 0
            
            # Processa in chunks
            n_chunks = (n_paths_T + chunk_size - 1) // chunk_size
            
            for chunk_idx in range(n_chunks):
                current_chunk_size = min(chunk_size, n_paths_T - chunk_idx * chunk_size)
                
                # Simula usando il processo
                sim_result = self.process.simulate(
                    theta=theta,
                    n_paths=current_chunk_size,
                    n_steps=n_steps_T,
                    dt=dt_base,
                    init_state=(spot,) if not self.process.requires_variance_state else None,
                    device=self.device,
                    antithetic=use_antithetic
                )
                
                S_chunk = sim_result.spot
                
                # Gestisci absorption se supportato
                if handle_absorption and self.process.supports_absorption:
                    # Il metodo handle_absorption esiste sempre in BaseStochasticProcess
                    _, absorbed_mask = self.process.handle_absorption(S_chunk, dt_base)
                    survived_mask = ~absorbed_mask
                    n_survived_chunk = survived_mask.sum()
                else:
                    # Tutti i paths sopravvivono
                    survived_mask = torch.ones(current_chunk_size, dtype=torch.bool, device=self.device)
                    n_survived_chunk = current_chunk_size
                
                if n_survived_chunk > 0:
                    # Estrai ST solo per i path sopravvissuti
                    ST_chunk = S_chunk[:, -1]
                    ST_survived = ST_chunk[survived_mask]
                    
                    # Calcola payoff per tutti gli strikes
                    payoffs = (ST_survived.unsqueeze(1) - K_values.unsqueeze(0)).clamp(min=0.0)
                    
                    # Accumula statistiche
                    payoff_sums += payoffs.sum(dim=0)
                    n_valid_per_strike += n_survived_chunk
                    
                    if control_variate:
                        dS = ST_survived - spot
                        dS_sums += dS.sum() * torch.ones(len(self.logKs), device=self.device)
                        
                        for jK in range(len(self.logKs)):
                            payoff_dS_sums[jK] += (payoffs[:, jK] * dS).sum()
                        
                        dS_sq_sums += (dS ** 2).sum() * torch.ones(len(self.logKs), device=self.device)
                
                total_paths_processed += current_chunk_size
                
                # Libera memoria
                del S_chunk
                if n_survived_chunk > 0:
                    del ST_chunk, ST_survived, payoffs
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Calcola IV per tutti gli strikes
            for jK, k in enumerate(self.logKs):
                K = K_values[jK]
                n_valid = n_valid_per_strike[jK].item()
                
                # Fallback se nessun path è sopravvissuto
                if n_valid == 0:
                    # Usa volatilità di default basata su parametri
                    default_vol = self._get_default_volatility(theta)
                    iv[iT, jK] = default_vol
                    if handle_absorption and self.process.supports_absorption:
                        absorption_stats[iT, jK] = 1.0
                    continue
                
                # Media normalizzata sul numero totale di paths
                payoff_mean = payoff_sums[jK] / total_paths_processed
                
                # Control variate
                if control_variate and n_valid > 1:
                    dS_mean = dS_sums[jK] / total_paths_processed
                    payoff_dS_mean = payoff_dS_sums[jK] / total_paths_processed
                    dS_var = (dS_sq_sums[jK] / total_paths_processed) - dS_mean**2
                    
                    if dS_var > 1e-10:
                        cov = payoff_dS_mean - payoff_mean * dS_mean
                        beta = cov / dS_var
                        price_call = (payoff_mean - beta * dS_mean) * disc
                    else:
                        price_call = payoff_mean * disc
                else:
                    price_call = payoff_mean * disc
                
                # Assicura prezzo sopra intrinseco
                intrinsic_value = max(0, spot - K.item() * disc)
                price_call = max(price_call, intrinsic_value + price_floor)
                
                # Absorption ratio
                if handle_absorption and self.process.supports_absorption:
                    absorbed_ratio = 1.0 - (n_valid / total_paths_processed)
                    absorption_stats[iT, jK] = absorbed_ratio
                
                # Calcola IV
                if price_call > 1e-10:
                    try:
                        bs = BlackScholes(
                            EuropeanOption(
                                BrownianStock(),
                                strike=K.item(),
                                maturity=float(T)
                            )
                        )
                        iv[iT, jK] = bs.implied_volatility(
                            log_moneyness=-float(k),
                            time_to_maturity=float(T),
                            price=price_call
                        )
                    except:
                        iv[iT, jK] = self._get_default_volatility(theta)
                else:
                    iv[iT, jK] = self._get_default_volatility(theta)
        
        # Post-processing con smile repair
        if self.enable_smile_repair:
            fallback_vol = self._get_default_volatility(theta)
            iv_repaired, repair_stats = SmileRepair.repair_surface(
                iv, 
                self.logKs,
                min_valid_per_smile=3,
                fallback_vol=fallback_vol,
                method=self.smile_repair_method
            )
            
            self.last_repair_stats = repair_stats
            
            if repair_stats['repair_ratio'] > 0.1:
                print(f"  Smile repair: fixed {repair_stats['repair_ratio']:.1%} of points")
            
            iv = iv_repaired
        
        # Salva statistiche absorption
        if handle_absorption and self.process.supports_absorption:
            self.last_absorption_stats = absorption_stats
        
        return iv
    
    def _get_default_volatility(self, theta: torch.Tensor) -> float:
        """
        Ottiene una volatilità di default basata sui parametri del modello.
        """
        # Per Rough Bergomi, usa sqrt(xi0)
        # Per altri modelli, questa logica dovrà essere adattata
        if hasattr(self.process, 'param_info'):
            param_info = self.process.param_info
            if 'xi0' in param_info.names:
                xi0_idx = param_info.names.index('xi0')
                return math.sqrt(theta[xi0_idx].item())
            elif 'sigma' in param_info.names:
                sigma_idx = param_info.names.index('sigma')
                return theta[sigma_idx].item()
        
        # Default generale
        return 0.2
    
    # Tutti gli altri metodi rimangono invariati...
    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """Forward pass della rete."""
        batch_size = theta.shape[0]
        out = self.net(theta)
        return out.view(batch_size, len(self.Ts), len(self.logKs))
    
    def fit(self,
            theta_train: torch.Tensor,
            iv_train: torch.Tensor,
            theta_val: torch.Tensor = None,
            iv_val: torch.Tensor = None,
            n_paths: int = 4096,
            epochs: int = 30,
            batch_size: int = 512,
            lr: float = 1e-3):
        """
        Fit della rete su grid-based datasets.
        """
        # Appiattisco la superficie
        N_train = len(theta_train)
        iv_train_flat = iv_train.view(N_train, self.out_dim)
        
        train_loader = DataLoader(
            TensorDataset(theta_train, iv_train_flat),
            batch_size=batch_size, shuffle=True
        )
        
        if theta_val is not None and iv_val is not None:
            N_val = len(theta_val)
            iv_val_flat = iv_val.view(N_val, self.out_dim)
            val_loader = DataLoader(
                TensorDataset(theta_val, iv_val_flat),
                batch_size=batch_size, shuffle=False
            )
        else:
            val_loader = None
        
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        
        best_val_loss = float('inf')
        best_weights = None
        
        for epoch in range(1, epochs + 1):
            # Training
            self.net.train()
            train_loss = 0.0
            for theta_batch, iv_batch in train_loader:
                pred = self.net(theta_batch)
                loss = loss_fn(pred, iv_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * theta_batch.size(0)
            
            train_loss /= N_train
            
            # Validation
            if val_loader is not None:
                self.net.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for theta_val_batch, iv_val_batch in val_loader:
                        pred_val = self.net(theta_val_batch)
                        loss_val = loss_fn(pred_val, iv_val_batch)
                        val_loss += loss_val.item() * theta_val_batch.size(0)
                
                val_loss /= N_val
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = self.net.state_dict().copy()
                
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            else:
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f}")
        
        if best_weights is not None:
            self.net.load_state_dict(best_weights)
            self.save("best_grid_weights.pt") 
            print(f"Loaded best weights (Val Loss = {best_val_loss:.6f})")
    
    def price_iv(self, theta: torch.Tensor, denormalize_output: bool = True) -> torch.Tensor:
        """
        Calcola la superficie IV per i parametri theta dati.
        
        Args:
            theta: parametri del modello (NON normalizzati)
            denormalize_output: se True, denormalizza l'output
        """
        self.net.eval()
        
        # Normalizza l'input se abbiamo le statistiche
        theta_norm = self.normalize_theta(theta)
        
        # Forward pass con input normalizzato
        iv_surface_norm = self.forward(theta_norm)
        
        # Denormalizza l'output se richiesto
        if denormalize_output:
            iv_surface = self.denormalize_iv(iv_surface_norm)
        else:
            iv_surface = iv_surface_norm
        
        # Se richiesto un singolo theta, fitta l'interpolatore
        if theta.shape[0] == 1 and not self._interpolator_fitted:
            self._fit_interpolator(iv_surface[0].detach())
        
        return iv_surface
    
    def _fit_interpolator(self, iv_surface: torch.Tensor):
        """Fitta l'interpolatore su una superficie IV."""
        T_np = self.Ts.cpu().numpy()
        k_np = self.logKs.cpu().numpy()
        iv_np = iv_surface.cpu().numpy()
        
        self.interpolator.fit(T_np, k_np, iv_np)
        self._interpolator_fitted = True
    
    def interpolate_iv(self, T, k, theta=None):
        """
        Interpola IV per punti arbitrari (T, k).
        
        Args:
            T: maturità (scalare o array)
            k: log-moneyness (scalare o array)
            theta: parametri del modello (opzionale se già fittato)
            
        Returns:
            IV interpolata
        """
        if theta is not None:
            # Calcola la superficie per questo theta
            iv_surface = self.price_iv(theta.unsqueeze(0) if theta.dim() == 1 else theta)
            self._fit_interpolator(iv_surface[0])
        
        if not self._interpolator_fitted:
            raise ValueError("Interpolator not fitted. Provide theta or call price_iv first.")
        
        return self.interpolator(T, k)
        return self.interpolator(T, k)
    
    # Aggiungi questo metodo alla classe GridNetworkPricer
    def get_repair_statistics(self):
        """
        Ritorna le statistiche dell'ultima riparazione smile
        """
        if self.last_repair_stats is None:
            return "No repair statistics available. Run _mc_iv_grid first."
        
        stats = self.last_repair_stats
        return {
            'total_points_repaired': stats['total_repaired'],
            'total_points': stats['total_points'],
            'repair_percentage': f"{stats['repair_ratio']*100:.1f}%",
            'problematic_maturities': [self.Ts[i].item() for i in stats['problematic_maturities']]
        }
    
    def save(self, path: str, include_norm_stats: bool = False):
        """
        Salva i pesi del modello.
        
        Args:
            path: percorso del file
            include_norm_stats: se True, include le statistiche di normalizzazione
        """
        if include_norm_stats:
            self.save_full(path)
        else:
            self.save_model_only(path)
    
    @classmethod
    def load(cls, path: str, maturities: torch.Tensor, logK: torch.Tensor, 
             load_norm_stats: bool = False, **kwargs):
        """
        Carica un modello salvato.
        
        Args:
            path: percorso del file
            maturities: array delle maturità
            logK: array dei log-moneyness
            load_norm_stats: se True, prova a caricare anche le statistiche di normalizzazione
            **kwargs: altri parametri per il costruttore
        """
        device = kwargs.get('device', maturities.device)
        obj = cls(maturities, logK, **kwargs)
        
        # Carica lo state dict
        sd = torch.load(path, map_location=device, weights_only=True)
        
        if load_norm_stats:
            # Carica tutto incluse le statistiche se presenti
            obj.load_state_dict(sd, strict=False)
        else:
            # Carica solo i pesi del modello
            model_state = {k: v for k, v in sd.items() 
                          if k not in ['theta_mean', 'theta_std', 'iv_mean', 'iv_std']}
            obj.load_state_dict(model_state, strict=False)
        
        obj.eval()
        return obj
    
    
class PointwiseNetworkPricer(NeuralSurfacePricer):
    """
    Versione aggiornata di PointwiseNetworkPricer che supporta qualsiasi StochasticProcess.
    Calcola IV per singoli punti (theta, T, k).
    """
    def __init__(self,
                 process: StochasticProcess,  # <-- CAMBIATO
                 hidden_layers: list = [128],
                 activation: str = 'ReLU',
                 dt: float = 1/365,
                 device: str = 'cpu',
                 r: float = 0.0,
                 enable_smile_repair: bool = True,
                 smile_repair_method: str = 'pchip'):
        super().__init__(device=device, r=r)
        
        self.process = process  # <-- CAMBIATO
        self.dt = dt
        
        # Rete: num_params + 2 (T, k) -> 1 output
        self.net = build_network(
            process.num_params + 2,  # <-- CAMBIATO
            1, 
            hidden_layers, 
            activation
        ).to(device)
        
        self.enable_smile_repair = enable_smile_repair
        self.smile_repair_method = smile_repair_method
        self.last_repair_stats = None
    
    @torch.no_grad()
    def _mc_iv_point(self,
                     theta: torch.Tensor,
                     T: float,
                     logK: float,
                     n_paths: int = 4096,
                     spot: float = 1.0,
                     price_floor: float = 1e-8,
                     use_antithetic: bool = True,
                     adaptive_dt: bool = True,
                     control_variate: bool = True,
                     chunk_size: int = None) -> float:
        """
        Calcola IV per un singolo punto via Monte Carlo con variance reduction.
        
        Args:
            theta: parametri del modello (H, eta, rho, xi0)
            T: maturità
            logK: log-moneyness
            n_paths: numero di paths
            spot: prezzo spot iniziale
            price_floor: prezzo minimo per stabilità numerica
            use_antithetic: usa variabili antitetiche
            adaptive_dt: usa dt più piccolo per scadenze brevi
            control_variate: usa control variate basato su Black-Scholes
            chunk_size: processa il MC in chunck per gestire la memoria limitata
        """
        # Valida parametri
        is_valid, error_msg = self.process.validate_theta(theta)
        if not is_valid:
            raise ValueError(f"Invalid parameters: {error_msg}")
        
        # Determina dt
        if adaptive_dt:
            if T <= 0.1:
                dt_use = T / 100
            elif T <= 0.25:
                dt_use = 1/365 / 2
            else:
                dt_use = 1/365
        else:
            dt_use = self.dt
        
        n_steps = max(2, int(math.ceil(T / dt_use)))
        
        # Determina chunk_size
        if chunk_size is None:
            chunk_size = 10000 if self.device.type == 'cuda' else 20000
        
        # Accumula statistiche
        payoff_sum = 0.0
        dS_sum = 0.0
        payoff_dS_sum = 0.0
        payoff_sq_sum = 0.0
        dS_sq_sum = 0.0
        n_processed = 0
        n_valid_total = 0
        
        K = spot * math.exp(logK)
        disc = math.exp(-self.r * T)
        
        # Processa in chunks
        n_chunks = (n_paths + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(n_chunks):
            current_chunk_size = min(chunk_size, n_paths - chunk_idx * chunk_size)
            
            # Simula usando il processo
            sim_result = self.process.simulate(
                theta=theta,
                n_paths=current_chunk_size,
                n_steps=n_steps,
                dt=dt_use,
                init_state=(spot,) if not self.process.requires_variance_state else None,
                device=self.device,
                antithetic=use_antithetic
            )
            
            S_chunk = sim_result.spot
            ST_chunk = S_chunk[:, -1]
            
            # Gestisci absorption se supportato
            if self.process.supports_absorption:
                valid_mask = ST_chunk > 0
                n_valid = valid_mask.sum().item()
                if n_valid > 0:
                    ST_valid = ST_chunk[valid_mask]
                else:
                    continue
            else:
                ST_valid = ST_chunk
                n_valid = current_chunk_size
            
            # Calcola payoff e accumula statistiche
            payoff = (ST_valid - K).clamp(min=0.0)
            dS = ST_valid - spot
            
            payoff_sum += payoff.sum().item()
            dS_sum += dS.sum().item()
            
            if control_variate:
                payoff_dS_sum += (payoff * dS).sum().item()
                payoff_sq_sum += (payoff ** 2).sum().item()
                dS_sq_sum += (dS ** 2).sum().item()
            
            n_processed += current_chunk_size
            n_valid_total += n_valid
            
            # Libera memoria
            del S_chunk, ST_chunk
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Se nessun path valido, ritorna volatilità di default
        if n_valid_total == 0:
            return self._get_default_volatility(theta)
        
        # Calcola prezzo finale normalizzando sul numero totale di paths processati
        payoff_mean = payoff_sum / n_processed
        
        if control_variate and n_processed > 1:
            dS_mean = dS_sum / n_processed
            payoff_dS_mean = payoff_dS_sum / n_processed
            dS_var = (dS_sq_sum / n_processed) - dS_mean**2
            
            if dS_var > 1e-10:
                cov = payoff_dS_mean - payoff_mean * dS_mean
                beta = cov / dS_var
                price_call = (payoff_mean - beta * dS_mean) * disc
            else:
                price_call = payoff_mean * disc
        else:
            price_call = payoff_mean * disc
        
        # Assicura prezzo sopra intrinseco
        intrinsic_value = max(0, spot - K * disc)
        price_call = max(price_call, intrinsic_value + price_floor)
        
        # Calcola IV
        if price_call > 1e-10:
            try:
                bs = BlackScholes(
                    EuropeanOption(
                        BrownianStock(),
                        strike=K,
                        maturity=T
                    )
                )
                return float(bs.implied_volatility(
                    log_moneyness=-logK,
                    time_to_maturity=T,
                    price=price_call
                ))
            except:
                return self._get_default_volatility(theta)
        else:
            return self._get_default_volatility(theta)
    
    def _get_default_volatility(self, theta: torch.Tensor) -> float:
        """
        Ottiene una volatilità di default basata sui parametri del modello.
        """
        if hasattr(self.process, 'param_info'):
            param_info = self.process.param_info
            # Cerca parametri volatilità-related
            for vol_name in ['xi0', 'sigma', 'theta']:
                if vol_name in param_info.names:
                    idx = param_info.names.index(vol_name)
                    value = theta[idx].item()
                    # Per xi0 e theta (varianza), prendi sqrt
                    if vol_name in ['xi0', 'theta']:
                        return math.sqrt(value)
                    else:
                        return value
        
        # Default generale
        return 0.2
    
    @torch.no_grad()
    def _mc_iv_grid(self,
                    theta: torch.Tensor,
                    maturities: torch.Tensor,
                    logK: torch.Tensor,
                    n_paths: int = 4096,
                    spot: float = 1.0,
                    price_floor: float = 1e-8,
                    use_antithetic: bool = True,
                    adaptive_dt: bool = True,
                    control_variate: bool = True,
                    chunk_size: int = None) -> torch.Tensor:
        """
        Calcola IV su griglia via Monte Carlo.
        Versione aggiornata per supportare processi generici.
        """
        # Valida parametri
        is_valid, error_msg = self.process.validate_theta(theta)
        if not is_valid:
            raise ValueError(f"Invalid parameters: {error_msg}")
        
        # Determina chunk_size se non specificato
        if chunk_size is None:
            chunk_size = 10000 if self.device.type == 'cuda' else 20000
        
        # Inizializza griglia risultati
        iv = torch.zeros(len(maturities), len(logK), device=self.device)
        
        # Processa ogni scadenza separatamente
        for iT, T in enumerate(maturities):
            T_val = T.item()
            disc = math.exp(-self.r * T_val)
            
            # Determina dt per questa scadenza
            if adaptive_dt:
                if T_val <= 0.1:
                    dt_use = T_val / 100
                elif T_val <= 0.25:
                    dt_use = 1/365 / 2
                else:
                    dt_use = 1/365
            else:
                dt_use = self.dt
            
            n_steps = max(2, int(math.ceil(T_val / dt_use)))
            
            # Accumula statistiche per ogni strike
            strike_stats = {jK: {'payoff_sum': 0.0, 'dS_sum': 0.0, 'payoff_dS_sum': 0.0,
                                'payoff_sq_sum': 0.0, 'dS_sq_sum': 0.0, 
                                'n_processed': 0, 'n_valid': 0}
                           for jK in range(len(logK))}
            
            # Processa in chunks
            n_chunks = (n_paths + chunk_size - 1) // chunk_size
            
            for chunk_idx in range(n_chunks):
                current_chunk_size = min(chunk_size, n_paths - chunk_idx * chunk_size)
                
                # Simula usando il processo
                sim_result = self.process.simulate(
                    theta=theta,
                    n_paths=current_chunk_size,
                    n_steps=n_steps,
                    dt=dt_use,
                    init_state=(spot,) if not self.process.requires_variance_state else None,
                    device=self.device,
                    antithetic=use_antithetic
                )
                
                S_chunk = sim_result.spot
                ST_chunk = S_chunk[:, -1]
                
                # Gestisci absorption
                if self.process.supports_absorption:
                    valid_mask = ST_chunk > 0
                else:
                    valid_mask = torch.ones_like(ST_chunk, dtype=torch.bool)
                
                n_valid_chunk = valid_mask.sum().item()
                
                # Accumula statistiche per ogni strike
                for jK, k in enumerate(logK):
                    K = spot * math.exp(k.item())
                    
                    if n_valid_chunk > 0:
                        ST_valid = ST_chunk[valid_mask]
                        payoff_valid = (ST_valid - K).clamp(min=0.0)
                        dS_valid = ST_valid - spot
                        
                        stats = strike_stats[jK]
                        stats['payoff_sum'] += payoff_valid.sum().item()
                        stats['dS_sum'] += dS_valid.sum().item()
                        
                        if control_variate:
                            stats['payoff_dS_sum'] += (payoff_valid * dS_valid).sum().item()
                            stats['payoff_sq_sum'] += (payoff_valid ** 2).sum().item()
                            stats['dS_sq_sum'] += (dS_valid ** 2).sum().item()
                        
                        stats['n_valid'] += n_valid_chunk
                    
                    stats['n_processed'] += current_chunk_size
                
                # Libera memoria
                del S_chunk, ST_chunk
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Calcola IV per ogni strike
            for jK, k in enumerate(logK):
                stats = strike_stats[jK]
                n_total = stats['n_processed']
                n_valid = stats['n_valid']
                
                if n_total == 0 or n_valid == 0:
                    iv[iT, jK] = self._get_default_volatility(theta)
                    continue
                
                payoff_mean = stats['payoff_sum'] / n_total
                
                # Control variate
                if control_variate and n_total > 1:
                    dS_mean = stats['dS_sum'] / n_total
                    payoff_dS_mean = stats['payoff_dS_sum'] / n_total
                    dS_var = (stats['dS_sq_sum'] / n_total) - dS_mean**2
                    
                    if dS_var > 1e-10:
                        cov = payoff_dS_mean - payoff_mean * dS_mean
                        beta = cov / dS_var
                        price_call = (payoff_mean - beta * dS_mean) * disc
                    else:
                        price_call = payoff_mean * disc
                else:
                    price_call = payoff_mean * disc
                
                # Assicura prezzo sopra intrinseco
                K = spot * math.exp(k.item())
                intrinsic_value = max(0, spot - K * disc)
                price_call = max(price_call, intrinsic_value + price_floor)
                
                # Warning se molti paths assorbiti
                if self.process.supports_absorption:
                    absorption_ratio = 1.0 - (n_valid / n_total)
                    if absorption_ratio > 0.2:
                        print(f"Warning: {absorption_ratio:.1%} paths absorbed at T={T_val:.3f}, K={K:.3f}")
                
                # Calcola IV
                if price_call > 1e-10:
                    try:
                        bs = BlackScholes(
                            EuropeanOption(
                                BrownianStock(),
                                strike=K,
                                maturity=float(T)
                            )
                        )
                        iv[iT, jK] = bs.implied_volatility(
                            log_moneyness=float(-k),
                            time_to_maturity=float(T),
                            price=price_call
                        )
                    except:
                        iv[iT, jK] = self._get_default_volatility(theta)
                else:
                    iv[iT, jK] = self._get_default_volatility(theta)
        
        # Smile repair
        if self.enable_smile_repair:
            fallback_vol = self._get_default_volatility(theta)
            
            iv_repaired, repair_stats = SmileRepair.repair_surface(
                iv, 
                logK,
                min_valid_per_smile=3,
                fallback_vol=fallback_vol,
                method=self.smile_repair_method
            )
            
            self.last_repair_stats = repair_stats
            
            if repair_stats['repair_ratio'] > 0.1:
                print(f"  Smile repair: fixed {repair_stats['repair_ratio']:.1%} of points")
                if repair_stats['problematic_maturities']:
                    prob_T = [maturities[i].item() for i in repair_stats['problematic_maturities']]
                    print(f"  Problematic maturities: {prob_T}")
            
            iv = iv_repaired
        
        return iv

    def forward(self,
                theta: torch.Tensor,
                T: torch.Tensor,
                logK: torch.Tensor):
        """Forward pass."""
        if T.dim() == 1:
            T = T.unsqueeze(-1)
        if logK.dim() == 1:
            logK = logK.unsqueeze(-1)
        
        x = torch.cat([theta, T, logK], dim=1)
        iv = self.net(x)
        return iv.squeeze(-1)
    
    def fit(self,
            theta_train: torch.Tensor,
            iv_train: torch.Tensor,
            T_train: torch.Tensor,
            k_train: torch.Tensor,
            theta_val: torch.Tensor = None,
            iv_val: torch.Tensor = None,
            T_val: torch.Tensor = None,
            k_val: torch.Tensor = None,
            epochs: int = 30,
            batch_size: int = 4096,
            lr: float = 1e-3):
        """Train su dataset point-wise."""
        
        train_loader = DataLoader(
            TensorDataset(theta_train, T_train, k_train, iv_train),
            batch_size=batch_size, shuffle=True
        )
        
        if theta_val is not None:
            val_loader = DataLoader(
                TensorDataset(theta_val, T_val, k_val, iv_val),
                batch_size=batch_size, shuffle=False
            )
        else:
            val_loader = None
        
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        
        best_val_loss = float('inf')
        best_weights = None
        
        for epoch in range(1, epochs + 1):
            # Training
            self.net.train()
            train_loss = 0.0
            n_train = 0
            
            for theta, T, k, iv in train_loader:
                pred = self.forward(theta, T, k)
                loss = loss_fn(pred, iv)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(theta)
                n_train += len(theta)
            
            train_loss /= n_train
            
            # Validation
            if val_loader is not None:
                self.net.eval()
                val_loss = 0.0
                n_val = 0
                
                with torch.no_grad():
                    for theta, T, k, iv in val_loader:
                        pred = self.forward(theta, T, k)
                        loss = loss_fn(pred, iv)
                        val_loss += loss.item() * len(theta)
                        n_val += len(theta)
                
                val_loss /= n_val
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = self.net.state_dict().copy()
                
                print(f"Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
            else:
                print(f"Epoch {epoch:3d} | Train: {train_loss:.6f}")
        
        if best_weights is not None:
            self.net.load_state_dict(best_weights)
            self.save("best_pw_weights.pt") 
            print(f"Loaded best weights (Val Loss = {best_val_loss:.6f})")
    
    def set_pointwise_normalization_stats(self, T_mean, T_std, k_mean, k_std):
        """Imposta le statistiche di normalizzazione"""
        self.register_buffer('T_mean', T_mean.to(self.device))
        self.register_buffer('T_std', T_std.to(self.device))
        self.register_buffer('k_mean', torch.tensor(k_mean, device=self.device))
        self.register_buffer('k_std', torch.tensor(k_std, device=self.device))
    
    def normalize_T(self, T):
        """Normalizza le maturità"""
        if self.T_mean is None or self.T_std is None:
            return T
        return (T - self.T_mean) / self.T_std
    
    def normalize_k(self, k):
        """Normalizza il log-moneyness"""
        if self.k_mean is None or self.k_std is None:
            return k
        return (k - self.k_mean) / self.k_std

    def price_iv_grid(self, theta: torch.Tensor, maturities: torch.Tensor, 
                  logK: torch.Tensor, denormalize_output: bool = True) -> torch.Tensor:
        """
        Calcola griglia completa di IV.
        
        Args:
            theta: parametri del modello (NON normalizzati)
            maturities: array di maturità (NON normalizzate)
            logK: array di log-moneyness (NON normalizzati)
            denormalize_output: se True, denormalizza l'output
        """
        self.net.eval()
        
        B = theta.size(0)
        T_len = maturities.size(0)
        K_len = logK.size(0)
        
        # Crea mesh
        theta_exp = theta.view(B, 1, 1, 4).expand(B, T_len, K_len, 4)
        mat_mesh, k_mesh = torch.meshgrid(maturities, logK, indexing='ij')
        mat_exp = mat_mesh.unsqueeze(0).expand(B, T_len, K_len)
        k_exp = k_mesh.unsqueeze(0).expand(B, T_len, K_len)
        
        # Flatten
        theta_flat = theta_exp.reshape(-1, 4)
        T_flat = mat_exp.reshape(-1)
        k_flat = k_exp.reshape(-1)
        
        # Forward con normalizzazione/denormalizzazione gestita da price_iv
        with torch.no_grad():
            iv_flat = self.price_iv(theta_flat, T_flat, k_flat, denormalize_output)
        
        return iv_flat.view(B, T_len, K_len)
    
    
    def price_iv(self, theta: torch.Tensor, T: torch.Tensor, k: torch.Tensor, 
                 denormalize_output: bool = True) -> torch.Tensor:
        """
        Calcola IV per punti specifici (theta, T, k).
        
        Args:
            theta: parametri del modello (NON normalizzati)
            T: maturità (NON normalizzata)
            k: log-moneyness (NON normalizzato)
            denormalize_output: se True, denormalizza l'output
        """
        self.net.eval()
        
        # Normalizza tutti gli input
        theta_norm = self.normalize_theta(theta)
        T_norm = self.normalize_T(T)
        k_norm = self.normalize_k(k)
        
        # Forward pass con input normalizzati
        iv_norm = self.forward(theta_norm, T_norm, k_norm)
        
        # Denormalizza l'output se richiesto
        if denormalize_output:
            return self.denormalize_iv(iv_norm)
        else:
            return iv_norm
        
    
    def get_repair_statistics(self):
        """
        Ritorna le statistiche dell'ultima riparazione smile
        """
        if self.last_repair_stats is None:
            return "No repair statistics available. Run _mc_iv_grid first."
        
        stats = self.last_repair_stats
        return {
            'total_points_repaired': stats['total_repaired'],
            'total_points': stats['total_points'],
            'repair_percentage': f"{stats['repair_ratio']*100:.1f}%",
            'problematic_maturities': stats['problematic_maturities']  # Indici, non valori
        }

    def save(self, path: str, include_norm_stats: bool = False):
        """
        Salva i pesi del modello.
        
        Args:
            path: percorso del file
            include_norm_stats: se True, include le statistiche di normalizzazione
        """
        if include_norm_stats:
            self.save_full(path)
        else:
            self.save_model_only(path)
    
    @classmethod
    def load(cls, path: str, process: Union[str, StochasticProcess], 
             load_norm_stats: bool = False, **kwargs):
        """
        Carica un modello salvato.
        
        Args:
            path: percorso del file
            process: nome del processo o istanza StochasticProcess
            load_norm_stats: se True, prova a caricare anche le statistiche
            **kwargs: parametri per il costruttore
        """
        device = kwargs.get('device', 'cpu')
        
        # Crea processo se passato come stringa
        if isinstance(process, str):
            from .stochastic_interface import ProcessFactory
            process_obj = ProcessFactory.create(process)
        else:
            process_obj = process
        
        obj = cls(process=process_obj, **kwargs)
        
        # Carica lo state dict
        sd = torch.load(path, map_location=device, weights_only=True)
        
        if load_norm_stats:
            # Carica tutto incluse le statistiche se presenti
            obj.load_state_dict(sd, strict=False)
        else:
            # Carica solo i pesi del modello
            model_state = {k: v for k, v in sd.items() 
                          if k not in ['theta_mean', 'theta_std', 'iv_mean', 'iv_std', 
                                       'T_mean', 'T_std', 'k_mean', 'k_std']}
            obj.load_state_dict(model_state, strict=False)
        
        obj.eval()
        return obj
    
    
class MultiRegimeGridPricer(NeuralSurfacePricer):
    """
    Versione aggiornata di MultiRegimeGridPricer che supporta qualsiasi StochasticProcess.
    Gestisce 3 regimi separati (short, mid, long term).
    """
    
    def __init__(self,
                 # Processo stocastico
                 process: StochasticProcess,  # <-- CAMBIATO
                 # Parametri per i 3 regimi
                 short_term_maturities: torch.Tensor,
                 short_term_logK: torch.Tensor,
                 mid_term_maturities: torch.Tensor,
                 mid_term_logK: torch.Tensor,
                 long_term_maturities: torch.Tensor,
                 long_term_logK: torch.Tensor,
                 # Soglie per dividere i regimi (in anni)
                 short_term_threshold: float = 0.25,
                 mid_term_threshold: float = 1.0,
                 # Parametri comuni
                 dt: float = 1/365,
                 device: str = 'cpu',
                 r: float = 0.0,
                 # Parametri architettura per regime
                 short_term_hidden: list = [128, 64],
                 mid_term_hidden: list = [128, 64],
                 long_term_hidden: list = [128, 64],
                 short_term_activation: str = 'ReLU',
                 mid_term_activation: str = 'ReLU',
                 long_term_activation: str = 'ReLU',
                 # Parametri interpolazione
                 interpolation_method: str = 'thin_plate_spline',
                 extrapolation: str = 'flat'):
        
        super().__init__(device=device, r=r)
        
        # Salva processo e soglie
        self.process = process  # <-- CAMBIATO
        self.short_term_threshold = short_term_threshold
        self.mid_term_threshold = mid_term_threshold
        
        # Crea i 3 pricer separati usando GridNetworkPricerV2
        self.short_term_pricer = GridNetworkPricer(
            maturities=short_term_maturities,
            logK=short_term_logK,
            process=process,
            hidden_layers=short_term_hidden,
            activation=short_term_activation,
            dt=dt,
            device=device,
            r=r,
            interpolation_method=interpolation_method,
            extrapolation=extrapolation,
            enable_smile_repair=True,
            smile_repair_method='pchip'
        )
        
        self.mid_term_pricer = GridNetworkPricer(
            maturities=mid_term_maturities,
            logK=mid_term_logK,
            process=process,
            hidden_layers=mid_term_hidden,
            activation=mid_term_activation,
            dt=dt,
            device=device,
            r=r,
            interpolation_method=interpolation_method,
            extrapolation=extrapolation,
            enable_smile_repair=True,
            smile_repair_method='pchip'
        )
        
        self.long_term_pricer = GridNetworkPricer(
            maturities=long_term_maturities,
            logK=long_term_logK,
            process=process,
            hidden_layers=long_term_hidden,
            activation=long_term_activation,
            dt=dt,
            device=device,
            r=r,
            interpolation_method=interpolation_method,
            extrapolation=extrapolation,
            enable_smile_repair=True,
            smile_repair_method='pchip'
        )
        
        # Salva tutte le maturità e strikes per reference
        self.all_maturities = torch.cat([
            short_term_maturities,
            mid_term_maturities,
            long_term_maturities
        ]).unique().sort()[0]
        
        # Per l'interpolazione finale
        self.global_interpolator = VolatilityInterpolator(
            method=interpolation_method,
            extrapolation=extrapolation
        )
        self._global_interpolator_fitted = False
    
    def _get_regime(self, T: float) -> str:
        """Determina il regime basato sulla maturità."""
        if T <= self.short_term_threshold:
            return 'short'
        elif T <= self.mid_term_threshold:
            return 'mid'
        else:
            return 'long'
    
    def _get_pricer_for_maturity(self, T: float) -> GridNetworkPricer:
        """Restituisce il pricer appropriato per la maturità data."""
        regime = self._get_regime(T)
        if regime == 'short':
            return self.short_term_pricer
        elif regime == 'mid':
            return self.mid_term_pricer
        else:
            return self.long_term_pricer
    
    def set_normalization_stats(self, theta_mean, theta_std, 
                               short_iv_mean, short_iv_std,
                               mid_iv_mean, mid_iv_std,
                               long_iv_mean, long_iv_std):
        """
        Imposta le statistiche di normalizzazione per tutti i regimi.
        Nota: theta_mean e theta_std sono comuni a tutti i regimi.
        """
        # Imposta statistiche theta per tutti i pricer
        self.short_term_pricer.set_normalization_stats(
            theta_mean, theta_std, short_iv_mean, short_iv_std
        )
        self.mid_term_pricer.set_normalization_stats(
            theta_mean, theta_std, mid_iv_mean, mid_iv_std
        )
        self.long_term_pricer.set_normalization_stats(
            theta_mean, theta_std, long_iv_mean, long_iv_std
        )
        
        # Salva anche nella classe parent per consistency
        super().set_normalization_stats(
            theta_mean, theta_std, 
            (short_iv_mean + mid_iv_mean + long_iv_mean) / 3,  # Media approssimativa
            (short_iv_std + mid_iv_std + long_iv_std) / 3
        )
    
    def fit(self,
            # Dati short term
            theta_train_short: torch.Tensor,
            iv_train_short: torch.Tensor,
            theta_val_short: torch.Tensor = None,
            iv_val_short: torch.Tensor = None,
            # Dati mid term
            theta_train_mid: torch.Tensor = None,
            iv_train_mid: torch.Tensor = None,
            theta_val_mid: torch.Tensor = None,
            iv_val_mid: torch.Tensor = None,
            # Dati long term
            theta_train_long: torch.Tensor = None,
            iv_train_long: torch.Tensor = None,
            theta_val_long: torch.Tensor = None,
            iv_val_long: torch.Tensor = None,
            # Parametri training
            n_paths: int = 4096,
            epochs: int = 30,
            batch_size: int = 512,
            lr: float = 1e-3,
            lr_decay: float = 0.95,  # Decay del learning rate tra regimi
            verbose: bool = True):
        """
        Fitta i 3 pricer sequenzialmente o in parallelo.
        Se i dati mid/long non sono forniti, usa gli stessi theta dello short.
        """
        
        # Se non forniti, usa gli stessi theta per tutti i regimi
        if theta_train_mid is None:
            theta_train_mid = theta_train_short
        if theta_train_long is None:
            theta_train_long = theta_train_short
            
        if theta_val_short is not None:
            if theta_val_mid is None:
                theta_val_mid = theta_val_short
            if theta_val_long is None:
                theta_val_long = theta_val_short
        
        # Fit short term
        if verbose:
            print("=" * 50)
            print("Fitting SHORT TERM regime...")
            print("=" * 50)
        
        self.short_term_pricer.fit(
            theta_train_short, iv_train_short,
            theta_val_short, iv_val_short,
            n_paths=n_paths,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr
        )
        
        # Fit mid term con learning rate ridotto
        if iv_train_mid is not None:
            if verbose:
                print("\n" + "=" * 50)
                print("Fitting MID TERM regime...")
                print("=" * 50)
            
            self.mid_term_pricer.fit(
                theta_train_mid, iv_train_mid,
                theta_val_mid, iv_val_mid,
                n_paths=n_paths,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr * lr_decay
            )
        
        # Fit long term con learning rate ulteriormente ridotto
        if iv_train_long is not None:
            if verbose:
                print("\n" + "=" * 50)
                print("Fitting LONG TERM regime...")
                print("=" * 50)
            
            self.long_term_pricer.fit(
                theta_train_long, iv_train_long,
                theta_val_long, iv_val_long,
                n_paths=n_paths,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr * (lr_decay ** 2)
            )
    
    def price_iv(self, theta: torch.Tensor, denormalize_output: bool = True) -> dict:
        """
        Calcola le superfici IV per tutti i regimi.
        
        Returns:
            dict con keys 'short', 'mid', 'long' contenenti le rispettive superfici
        """
        results = {}
        
        # Short term
        results['short'] = self.short_term_pricer.price_iv(theta, denormalize_output)
        
        # Mid term
        results['mid'] = self.mid_term_pricer.price_iv(theta, denormalize_output)
        
        # Long term
        results['long'] = self.long_term_pricer.price_iv(theta, denormalize_output)
        
        return results
    
    def price_iv_unified(self, theta: torch.Tensor, 
                        maturities: torch.Tensor = None,
                        logK: torch.Tensor = None,
                        denormalize_output: bool = True) -> torch.Tensor:
        """
        Calcola una superficie IV unificata su una griglia comune.
        
        Args:
            theta: parametri del modello
            maturities: maturità desiderate (se None, usa tutte)
            logK: log-moneyness desiderati (se None, usa unione di tutti)
            denormalize_output: se denormalizzare l'output
            
        Returns:
            Superficie unificata [batch_size, len(maturities), len(logK)]
        """
        if maturities is None:
            maturities = self.all_maturities
        
        if logK is None:
            # Unione di tutti i logK dei 3 regimi
            all_logK = torch.cat([
                self.short_term_pricer.logKs,
                self.mid_term_pricer.logKs,
                self.long_term_pricer.logKs
            ]).unique().sort()[0]
            logK = all_logK
        
        batch_size = theta.shape[0]
        n_T = len(maturities)
        n_K = len(logK)
        
        # Inizializza output
        unified_surface = torch.zeros(batch_size, n_T, n_K, device=self.device)
        
        # Per ogni batch
        for b in range(batch_size):
            theta_b = theta[b:b+1]
            
            # Ottieni superfici dai 3 regimi
            surfaces = self.price_iv(theta_b, denormalize_output)
            
            # Fitta interpolatori per ogni regime
            self.short_term_pricer._fit_interpolator(surfaces['short'][0].detach())
            self.mid_term_pricer._fit_interpolator(surfaces['mid'][0].detach())
            self.long_term_pricer._fit_interpolator(surfaces['long'][0].detach())
            
            # Interpola per ogni punto della griglia unificata
            for i, T in enumerate(maturities):
                T_val = T.item()
                pricer = self._get_pricer_for_maturity(T_val)
                
                for j, k in enumerate(logK):
                    k_val = k.item()
                    
                    # Usa l'interpolatore del regime appropriato
                    unified_surface[b, i, j] = pricer.interpolate_iv(T_val, k_val)
        
        return unified_surface
    
    def interpolate_iv(self, T, k, theta=None):
        """
        Interpola IV per punti arbitrari (T, k) usando il regime appropriato.
        
        Args:
            T: maturità (scalare o array)
            k: log-moneyness (scalare o array)
            theta: parametri del modello (opzionale se già fittato)
            
        Returns:
            IV interpolata
        """
        # Converti in numpy se necessario
        T_np = np.asarray(T)
        k_np = np.asarray(k)
        scalar_input = T_np.ndim == 0
        
        if scalar_input:
            T_np = T_np.reshape(1)
            k_np = k_np.reshape(1)
        
        # Se fornito theta, calcola le superfici
        if theta is not None:
            surfaces = self.price_iv(theta.unsqueeze(0) if theta.dim() == 1 else theta)
            
            # Fitta interpolatori
            self.short_term_pricer._fit_interpolator(surfaces['short'][0].detach())
            self.mid_term_pricer._fit_interpolator(surfaces['mid'][0].detach())
            self.long_term_pricer._fit_interpolator(surfaces['long'][0].detach())
        
        # Interpola usando il regime appropriato per ogni punto
        results = np.zeros_like(T_np, dtype=float)
        
        for i, (t, k_val) in enumerate(zip(T_np, k_np)):
            pricer = self._get_pricer_for_maturity(t)
            results[i] = pricer.interpolate_iv(t, k_val)
        
        if scalar_input:
            return results[0]
        return results
    
    def save(self, path_prefix: str, include_norm_stats: bool = False):
        """
        Salva tutti e 3 i modelli.
        
        Args:
            path_prefix: prefisso per i file (es. 'model' -> 'model_short.pt', etc.)
            include_norm_stats: se includere le statistiche di normalizzazione
        """
        self.short_term_pricer.save(f"{path_prefix}_short.pt", include_norm_stats)
        self.mid_term_pricer.save(f"{path_prefix}_mid.pt", include_norm_stats)
        self.long_term_pricer.save(f"{path_prefix}_long.pt", include_norm_stats)
        
        # Salva anche metadati sui regimi
        metadata = {
            'short_term_threshold': self.short_term_threshold,
            'mid_term_threshold': self.mid_term_threshold,
            'short_term_maturities': self.short_term_pricer.Ts.cpu(),
            'short_term_logK': self.short_term_pricer.logKs.cpu(),
            'mid_term_maturities': self.mid_term_pricer.Ts.cpu(),
            'mid_term_logK': self.mid_term_pricer.logKs.cpu(),
            'long_term_maturities': self.long_term_pricer.Ts.cpu(),
            'long_term_logK': self.long_term_pricer.logKs.cpu(),
        }
        torch.save(metadata, f"{path_prefix}_metadata.pt")
    
    @classmethod
    def load(cls, path_prefix: str, process: Union[str, StochasticProcess],
             load_norm_stats: bool = False, **kwargs):
        """
        Carica un modello multi-regime salvato.
        
        Args:
            path_prefix: prefisso dei file
            process: nome del processo o istanza StochasticProcess
            load_norm_stats: se caricare le statistiche
            **kwargs: parametri aggiuntivi
        """
        # Carica metadata
        device = kwargs.get('device', 'cpu')
        metadata = torch.load(f"{path_prefix}_metadata.pt", map_location=device, weights_only=True)
        
        # Crea processo se passato come stringa
        if isinstance(process, str):
            from .stochastic_interface import ProcessFactory
            process_obj = ProcessFactory.create(process)
        else:
            process_obj = process
        
        # Crea istanza
        obj = cls(
            process=process_obj,
            short_term_maturities=metadata['short_term_maturities'].to(device),
            short_term_logK=metadata['short_term_logK'].to(device),
            mid_term_maturities=metadata['mid_term_maturities'].to(device),
            mid_term_logK=metadata['mid_term_logK'].to(device),
            long_term_maturities=metadata['long_term_maturities'].to(device),
            long_term_logK=metadata['long_term_logK'].to(device),
            short_term_threshold=metadata['short_term_threshold'],
            mid_term_threshold=metadata['mid_term_threshold'],
            **kwargs
        )
        
        # Carica i pesi dei 3 modelli
        short_sd = torch.load(f"{path_prefix}_short.pt", map_location=device, weights_only=True)
        mid_sd = torch.load(f"{path_prefix}_mid.pt", map_location=device, weights_only=True)
        long_sd = torch.load(f"{path_prefix}_long.pt", map_location=device, weights_only=True)
        
        if load_norm_stats:
            obj.short_term_pricer.load_state_dict(short_sd, strict=False)
            obj.mid_term_pricer.load_state_dict(mid_sd, strict=False)
            obj.long_term_pricer.load_state_dict(long_sd, strict=False)
        else:
            # Carica solo i pesi del modello
            for pricer, sd in [(obj.short_term_pricer, short_sd),
                              (obj.mid_term_pricer, mid_sd),
                              (obj.long_term_pricer, long_sd)]:
                model_state = {k: v for k, v in sd.items() 
                             if k not in ['theta_mean', 'theta_std', 'iv_mean', 'iv_std']}
                pricer.load_state_dict(model_state, strict=False)
        
        obj.eval()
        return obj
    
    def to(self, device):
        """Override per gestire correttamente il device di tutti i pricer."""
        self.device = device
        self.short_term_pricer = self.short_term_pricer.to(device)
        self.mid_term_pricer = self.mid_term_pricer.to(device)
        self.long_term_pricer = self.long_term_pricer.to(device)
        return super().to(device)
    
    def eval(self):
        """Metti tutti i pricer in eval mode."""
        super().eval()
        self.short_term_pricer.eval()
        self.mid_term_pricer.eval()
        self.long_term_pricer.eval()
        return self
    
    def train(self, mode=True):
        """Metti tutti i pricer in train mode."""
        super().train(mode)
        self.short_term_pricer.train(mode)
        self.mid_term_pricer.train(mode)
        self.long_term_pricer.train(mode)
        return self