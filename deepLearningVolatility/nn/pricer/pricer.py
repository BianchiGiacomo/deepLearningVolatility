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
from deepLearningVolatility.stochastic.stochastic_interface import ProcessFactory, StochasticProcess

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
            # Boundaries for T
            self.boundary_values['T_min'] = iv_surface[0, :]  # First row
            self.boundary_values['T_max'] = iv_surface[-1, :]  # Last row
            # Boundaries for k
            self.boundary_values['k_min'] = iv_surface[:, 0]  # First column
            self.boundary_values['k_max'] = iv_surface[:, -1]  # Last column
            # Corners
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
        
        # Manage Scalar input
        scalar_input = False
        if T.ndim == 0:
            T = T.reshape(1)
            k = k.reshape(1)
            scalar_input = True
        
        # Create points for interpolation
        points = np.column_stack([T, k])
        
        if self.extrapolation == 'flat':
            # Apply flat extrapolation
            result = np.zeros(len(points))
            
            for i, (t, k_val) in enumerate(points):
                # Check if the point is inside the Bounds
                if self.T_min <= t <= self.T_max and self.k_min <= k_val <= self.k_max:
                    # Interpola normalmente
                    result[i] = self.interpolator(points[i:i+1])[0]
                else:
                    # Flat extrapolation
                    if t < self.T_min and k_val < self.k_min:
                        result[i] = self.boundary_values['corner_TminKmin']
                    elif t < self.T_min and k_val > self.k_max:
                        result[i] = self.boundary_values['corner_TminKmax']
                    elif t > self.T_max and k_val < self.k_min:
                        result[i] = self.boundary_values['corner_TmaxKmin']
                    elif t > self.T_max and k_val > self.k_max:
                        result[i] = self.boundary_values['corner_TmaxKmax']
                    elif t < self.T_min:
                        k_idx = np.searchsorted(self.k_grid, k_val)
                        if k_idx == 0:
                            result[i] = self.boundary_values['T_min'][0]
                        elif k_idx >= len(self.k_grid):
                            result[i] = self.boundary_values['T_min'][-1]
                        else:
                            # Linear interpolation
                            w = (k_val - self.k_grid[k_idx-1]) / (self.k_grid[k_idx] - self.k_grid[k_idx-1])
                            result[i] = (1-w) * self.boundary_values['T_min'][k_idx-1] + w * self.boundary_values['T_min'][k_idx]
                    elif t > self.T_max:
                        k_idx = np.searchsorted(self.k_grid, k_val)
                        if k_idx == 0:
                            result[i] = self.boundary_values['T_max'][0]
                        elif k_idx >= len(self.k_grid):
                            result[i] = self.boundary_values['T_max'][-1]
                        else:
                            w = (k_val - self.k_grid[k_idx-1]) / (self.k_grid[k_idx] - self.k_grid[k_idx-1])
                            result[i] = (1-w) * self.boundary_values['T_max'][k_idx-1] + w * self.boundary_values['T_max'][k_idx]
                    elif k_val < self.k_min:
                        T_idx = np.searchsorted(self.T_grid, t)
                        if T_idx == 0:
                            result[i] = self.boundary_values['k_min'][0]
                        elif T_idx >= len(self.T_grid):
                            result[i] = self.boundary_values['k_min'][-1]
                        else:
                            w = (t - self.T_grid[T_idx-1]) / (self.T_grid[T_idx] - self.T_grid[T_idx-1])
                            result[i] = (1-w) * self.boundary_values['k_min'][T_idx-1] + w * self.boundary_values['k_min'][T_idx]
                    else:  # k_val > self.k_max
                        T_idx = np.searchsorted(self.T_grid, t)
                        if T_idx == 0:
                            result[i] = self.boundary_values['k_max'][0]
                        elif T_idx >= len(self.T_grid):
                            result[i] = self.boundary_values['k_max'][-1]
                        else:
                            w = (t - self.T_grid[T_idx-1]) / (self.T_grid[T_idx] - self.T_grid[T_idx-1])
                            result[i] = (1-w) * self.boundary_values['k_max'][T_idx-1] + w * self.boundary_values['k_max'][T_idx]
        else:
            # Standard interpolation (RBF manages extrapolation)
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
        
        # Identify valid points (not zero and reasonable)
        valid_mask = (iv_smile > 0.01) & (iv_smile < 2.0) & ~np.isnan(iv_smile)
        n_valid = valid_mask.sum()
        
        if n_valid < min_valid_points:
            # Too few valid points, use Fallback
            if n_valid > 0:
                # Use the average of the valid points
                fallback_vol = iv_smile[valid_mask].mean()
            return np.full_like(iv_smile, fallback_vol), len(iv_smile) - n_valid
        
        if n_valid == len(iv_smile):
            # All points are valid
            return iv_smile, 0
        
        # Interpolate using only valid points
        logK_valid = logK[valid_mask]
        iv_valid = iv_smile[valid_mask]
        
        try:
            if method == 'pchip':
                # PCIP preserves the shape and avoids oscillations
                interpolator = PchipInterpolator(logK_valid, iv_valid, extrapolate=False)
            else:
                # Linear o cubic
                interpolator = interp1d(logK_valid, iv_valid, kind=method, 
                                      bounds_error=False, fill_value='extrapolate')
            
            # Repair not valid points
            iv_repaired = iv_smile.copy()
            invalid_mask = ~valid_mask
            iv_repaired[invalid_mask] = interpolator(logK[invalid_mask])
            
            # Manage extrapolation with flat extension
            # For points to the left of the valid range
            left_invalid = invalid_mask & (logK < logK_valid.min())
            if left_invalid.any():
                iv_repaired[left_invalid] = iv_valid[0]
            
            # For points to the right of the valid range
            right_invalid = invalid_mask & (logK > logK_valid.max())
            if right_invalid.any():
                iv_repaired[right_invalid] = iv_valid[-1]
            
            # Ensures reasonable values
            iv_repaired = np.clip(iv_repaired, 0.01, 2.0)
            
            # If there are still Nan (it shouldn't happen), use Fallback
            nan_mask = np.isnan(iv_repaired)
            if nan_mask.any():
                iv_repaired[nan_mask] = fallback_vol
            
            return iv_repaired, invalid_mask.sum()
            
        except Exception as e:
            # Fallback in case of error
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
            
            if n_rep > n_strikes * 0.5:  # More than 50% repaired
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

#--------------------------------------------------------------
class NeuralSurfacePricer(nn.Module):
    """
    Generic interface for neural network–based volatility surface pricers.
    """

    def __init__(self, device='cpu', r=0.0):
        super().__init__()
        self.device = torch.device(device)
        self.r = r  # Risk-free rate
        
        # Normalization statistics
        self.register_buffer('theta_mean', None)
        self.register_buffer('theta_std', None)
        self.register_buffer('iv_mean', None)
        self.register_buffer('iv_std', None)
        
    def set_normalization_stats(self, theta_mean, theta_std, iv_mean, iv_std):
        """Set normalization statistics"""
        self.register_buffer('theta_mean', theta_mean.to(self.device))
        self.register_buffer('theta_std', theta_std.to(self.device))
        self.register_buffer('iv_mean', torch.tensor(iv_mean, device=self.device))
        self.register_buffer('iv_std', torch.tensor(iv_std, device=self.device))
    
    def normalize_theta(self, theta):
        """Normalize model parameters theta"""
        if self.theta_mean is None or self.theta_std is None:
            return theta  # If there are no statots, return not normalized
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
        """Override to correctly manage the device"""
        self.device = device
        return super().to(device)
    
    def save_model_only(self, path: str):
        """Save only model weights, excluding normalization stats."""
        # Create a copy of the State Dict
        state = self.state_dict()
        
        # Remove normalization statistics
        keys_to_remove = []
        for key in state.keys():
            if key in ['theta_mean', 'theta_std', 'iv_mean', 'iv_std', 'T_mean', 'T_std', 'k_mean', 'k_std']:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            state.pop(key)
        
        # Save only the weights of the model
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
        # Delegation to a common implementation
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
    def compute_barrier_adjusted_price_vectorized(S_paths, K, T, r, dt, process=None):
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
        
        # Use the process if provided, otherwise Fallbacks to local implementation
        if process and hasattr(process, 'handle_absorption'):
            absorption_times, absorbed_mask = process.handle_absorption(S_paths, dt)
        else:
            absorption_times, absorbed_mask = ZeroAbsorptionHandler.find_absorption_times(S_paths, dt)
        
        # Survived paths
        survived_mask = ~absorbed_mask
        n_survived = survived_mask.sum().item()
        
        if n_survived == 0:
            return 0.0, 0
        
        # Calculate Payoff for all Paths (0 for absorbed ones)
        ST = S_paths[:, -1]
        payoff = (ST - K).clamp(min=0.0)
        
        # Reset the payoff for absorbed paths
        payoff = payoff * survived_mask.float()
        
        # Average price on all paths
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
        
        iv_list = []
        absorbed_ratios = []
        
        for S_paths, K, T in zip(S_paths_list, K_list, T_list):
            n_paths = S_paths.shape[0]
            
            # Calculate price with apsorption management
            price, n_survived = ZeroAbsorptionHandler.compute_barrier_adjusted_price_vectorized(
                S_paths, K, T, r, dt, process
            )
            
            absorbed_ratio = 1.0 - (n_survived / n_paths)
            absorbed_ratios.append(absorbed_ratio)
            
            # Calculate IV
            if price > 1e-10 and absorbed_ratio < 0.95:
                try:
                    bs = BlackScholes(
                        EuropeanOption(
                            BrownianStock(),
                            strike=float(K),
                            maturity=float(T)
                        )
                    )
                    iv = bs.implied_volatility(
                        log_moneyness=torch.log(torch.tensor(spot/K)),
                        time_to_maturity=T,
                        price=price
                    )
                    iv_list.append(float(iv))
                except:
                    # Estimation of volatility from the Returns of the surviving paths
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

#--------------------------------------------------------------
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
        self.fixed_regime_dt = None  # Can be set externally for fixed dt per regime
        self.out_dim = len(self.Ts) * len(self.logKs)
        
        # Neural network with dynamic architecture based on Num_params
        self.net = build_network(
            process.num_params,
            self.out_dim, 
            hidden_layers, 
            activation
        ).to(device)
        
        # Interpolator
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
                    handle_absorption: bool = True,
                    q: float = 0.0) -> torch.Tensor:
        """
        Calculate IV on grid via Monte Carlo.
        """
        # Validate parameters
        is_valid, error_msg = self.process.validate_theta(theta)
        if not is_valid:
            raise ValueError(f"Invalid parameters: {error_msg}")
        
        # Determine dt based on regime or adaptive logic
        T_min = self.Ts.min().item()

        # Check if fixed regime dt is set (for multi-regime training)
        if hasattr(self, 'fixed_regime_dt') and self.fixed_regime_dt is not None:
            dt_base = self.fixed_regime_dt
        elif adaptive_dt:
            # Enhanced adaptive dt for short maturities
            if T_min <= 30/365:  # <= 1 month (SHORT regime)
                dt_base = 3e-5  # Ultra-fine discretization for stability
            elif T_min <= 0.1:
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
        
        # Initialize grid results
        iv = torch.zeros(len(self.Ts), len(self.logKs), device=self.device)
        
        if handle_absorption and self.process.supports_absorption:
            absorption_stats = torch.zeros(len(self.Ts), len(self.logKs), device=self.device)
        
        # Processes each tenor
        for iT, T in enumerate(self.Ts):
            T_val = T.item()
            disc = math.exp(-self.r * T_val)
            
            # Necessary steps
            n_steps_T = int(round(T_val / dt_base)) + 1
            
            # Adaptive paths number
            n_paths_T = n_paths
            if adaptive_paths:
                if T_val <= 0.05:
                    n_paths_T = int(n_paths * 5)
                elif T_val <= 0.1:
                    n_paths_T = int(n_paths * 3)
                elif T_val <= 0.25:
                    n_paths_T = int(n_paths * 1.5)
            
            # Pre-calculate all the strikes for this maturity (forward-invariant)
            F_val = spot * math.exp((self.r - q) * T_val)
            K_values = torch.tensor(F_val, device=self.device) * torch.exp(self.logKs)
            
            # Initialize accumulators for statistics
            payoff_sums = torch.zeros(len(self.logKs), device=self.device)
            n_valid_per_strike = torch.zeros(len(self.logKs), device=self.device)
            
            if control_variate:
                dS_sums = torch.zeros(len(self.logKs), device=self.device)
                payoff_dS_sums = torch.zeros(len(self.logKs), device=self.device)
                dS_sq_sums = torch.zeros(len(self.logKs), device=self.device)
            
            total_paths_processed = 0
            
            n_chunks = (n_paths_T + chunk_size - 1) // chunk_size
            
            for chunk_idx in range(n_chunks):
                current_chunk_size = min(chunk_size, n_paths_T - chunk_idx * chunk_size)
                
                # Simulate using the process
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
                
                # Manage apsorption if supported
                if handle_absorption and self.process.supports_absorption:
                    # The handle_absorption method always exists in BaseStochasticProcess
                    _, absorbed_mask = self.process.handle_absorption(S_chunk, dt_base)
                    survived_mask = ~absorbed_mask
                    n_survived_chunk = survived_mask.sum()
                else:
                    # All paths survive
                    survived_mask = torch.ones(current_chunk_size, dtype=torch.bool, device=self.device)
                    n_survived_chunk = current_chunk_size
                
                if n_survived_chunk > 0:
                    # Extract ST only for surviving paths
                    ST_chunk = S_chunk[:, -1]
                    ST_survived = ST_chunk[survived_mask]
                    
                    # Calculate payoffs for all strikes
                    payoffs = (ST_survived.unsqueeze(1) - K_values.unsqueeze(0)).clamp(min=0.0)
                    
                    # Accumulate statistics
                    payoff_sums += payoffs.sum(dim=0)
                    n_valid_per_strike += n_survived_chunk
                    
                    if control_variate:
                        dS = ST_survived - spot
                        dS_sums += dS.sum() * torch.ones(len(self.logKs), device=self.device)
                        
                        for jK in range(len(self.logKs)):
                            payoff_dS_sums[jK] += (payoffs[:, jK] * dS).sum()
                        
                        dS_sq_sums += (dS ** 2).sum() * torch.ones(len(self.logKs), device=self.device)
                
                total_paths_processed += current_chunk_size
                
                # Free memory
                del S_chunk
                if n_survived_chunk > 0:
                    del ST_chunk, ST_survived, payoffs
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Calculate IV for all strikes
            for jK, k in enumerate(self.logKs):
                # Strike related to the forward: K = F * exp(k)
                K = F_val * math.exp(k.item())
                n_valid = n_valid_per_strike[jK].item()
                
                # Fallback if no path survived
                if n_valid == 0:
                    # Use parameter-based default volatility
                    default_vol = self._get_default_volatility(theta)
                    iv[iT, jK] = default_vol
                    if handle_absorption and self.process.supports_absorption:
                        absorption_stats[iT, jK] = 1.0
                    continue
                
                # Normalized mean over the total number of paths
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
                
                # Ensure price above the intrinsic (PV di (F-K)^+)
                K = F_val * math.exp(k.item())
                intrinsic_value = disc * max(0.0, F_val - K)
                price_call = max(price_call, intrinsic_value + price_floor)
                
                # Absorption ratio
                if handle_absorption and self.process.supports_absorption:
                    absorbed_ratio = 1.0 - (n_valid / total_paths_processed)
                    absorption_stats[iT, jK] = absorbed_ratio
                
                # Calculate IV
                if price_call > 1e-10:
                    try:
                        bs = BlackScholes(
                            EuropeanOption(
                                BrownianStock(),
                                strike=float(K),
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
        
        # Post-processing with smile repair
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
        Gets a default volatility based on model parameters.
        """
        # For Rough Bergomi, use sqrt(xi0)
        # For other models, this logic will need to be adapted
        if hasattr(self.process, 'param_info'):
            param_info = self.process.param_info
            if 'xi0' in param_info.names:
                xi0_idx = param_info.names.index('xi0')
                return math.sqrt(theta[xi0_idx].item())
            elif 'sigma' in param_info.names:
                sigma_idx = param_info.names.index('sigma')
                return theta[sigma_idx].item()
        
        # General Default
        return 0.2
    
    # All other methods remain unchanged...
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
        Network fit on grid-based datasets.
        """
        # Flatten the surface
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
    
    def price_iv(self, theta: torch.Tensor,
                 denormalize_output: bool = True,
                 inputs_normalized: bool = False) -> torch.Tensor:
        """
        Calculates the IV surface for the given theta parameters.

        Args:
            theta: model parameters
            denormalize_output: If True, denormalizes the output
            inputs_normalized: set True if `theta` is already normalized
        """
        self.net.eval()
        
        # Use as-is if inputs are already normalized, otherwise normalize
        theta_for_forward = theta if inputs_normalized else self.normalize_theta(theta)
        iv_surface_norm = self.forward(theta_for_forward)
        
        # Denormalize output if required
        if denormalize_output:
            iv_surface = self.denormalize_iv(iv_surface_norm)
        else:
            iv_surface = iv_surface_norm
        
        # If a single theta is required, fit the interpolator
        if theta.shape[0] == 1 and not self._interpolator_fitted:
            self._fit_interpolator(iv_surface[0].detach())
        
        return iv_surface
    
    def _fit_interpolator(self, iv_surface: torch.Tensor):
        """Fit the interpolator onto a surface IV."""
        T_np = self.Ts.cpu().numpy()
        k_np = self.logKs.cpu().numpy()
        iv_np = iv_surface.cpu().numpy()
        
        self.interpolator.fit(T_np, k_np, iv_np)
        self._interpolator_fitted = True
    
    def interpolate_iv(self, T, k, theta=None):
        """
        Interpolate IV for arbitrary points (T, k).

        Args:
            T: maturity (scalar or array)
            k: log-moneyness (scalar or array)
            theta: model parameters (optional if already fitted)

        Returns:
            Interpolated IV
        """
        if theta is not None:
            # Calculate the surface area for this theta
            iv_surface = self.price_iv(theta.unsqueeze(0) if theta.dim() == 1 else theta)
            self._fit_interpolator(iv_surface[0])
        
        if not self._interpolator_fitted:
            raise ValueError("Interpolator not fitted. Provide theta or call price_iv first.")
        
        return self.interpolator(T, k)
    
    def get_repair_statistics(self):
        """
        Returns the statistics of the last repair.
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
        Saves the model weights.

        Args:
            path: file path
            include_norm_stats: If True, includes normalization statistics
        """
        if include_norm_stats:
            self.save_full(path)
        else:
            self.save_model_only(path)
    
    @classmethod
    def load(cls, path: str, maturities: torch.Tensor, logK: torch.Tensor, 
             load_norm_stats: bool = False, **kwargs):
        """
        Load a saved model.

        Args:
            path: file path
            maturities: array of maturities
            logK: array of log-moneyness
            load_norm_stats: if True, also attempts to load normalization statistics
            **kwargs: other constructor parameters
        """
        device = kwargs.get('device', maturities.device)
        obj = cls(maturities, logK, **kwargs)
        
        # Load the state dict
        sd = torch.load(path, map_location=device, weights_only=True)
        
        if load_norm_stats:
            # Load everything including statistics if any
            obj.load_state_dict(sd, strict=False)
        else:
            # Load model weights only
            model_state = {k: v for k, v in sd.items() 
                          if k not in ['theta_mean', 'theta_std', 'iv_mean', 'iv_std']}
            obj.load_state_dict(model_state, strict=False)
        
        obj.eval()
        return obj
    
    
class PointwiseNetworkPricer(NeuralSurfacePricer):
    """Models implied volatility for individual, arbitrary points.

    This class implements a neural pricer that learns a direct mapping from a
    combined input of model parameters and option characteristics to a single
    implied volatility value: `f(theta, T, K) -> IV`.

    This pointwise approach offers maximum flexibility, as it is not constrained
    to a predefined grid of maturities and strikes. It is particularly useful for
    pricing exotic options or options with non-standard tenors and for creating
    datasets based on randomized grids, as proposed by Baschetti et al. The
    trade-off for this flexibility is that generating a full surface requires
    multiple forward passes of the network, one for each point.

    Attributes:
        process (StochasticProcess): The underlying stochastic model used for
            generating training data.
        net (nn.Module): The neural network that maps the combined input of
            parameters, time-to-maturity, and log-moneyness to an IV point.
    """
    def __init__(self,
                 process: StochasticProcess,
                 hidden_layers: list = [128],
                 activation: str = 'ReLU',
                 dt: float = 1/365,
                 device: str = 'cpu',
                 r: float = 0.0,
                 enable_smile_repair: bool = True,
                 smile_repair_method: str = 'pchip'):
        super().__init__(device=device, r=r)
        
        self.process = process
        self.dt = dt
        
        # Net: num_params + 2 (T, k) -> 1 output
        self.net = build_network(
            process.num_params + 2,
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
                     chunk_size: int = None,
                     q: float = 0.0) -> float:
        """
        Calculate IV for a single point via Monte Carlo with variance reduction.

        Args:
            theta: model parameters (H, eta, rho, xi0)
            T: maturity
            logK: log-moneyness
            n_paths: number of paths
            spot: initial spot price
            price_floor: minimum price for numerical stability
            use_antithetic: use antithetical variables
            adaptive_dt: use smaller dt for short maturities
            control_variate: use Black-Scholes-based control variate
            chunk_size: process the MC in chunks to accommodate limited memory
        """
        # Validate parameters
        is_valid, error_msg = self.process.validate_theta(theta)
        if not is_valid:
            raise ValueError(f"Invalid parameters: {error_msg}")
        
        # Determine dt
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
        
        # Determine chunk_size
        if chunk_size is None:
            chunk_size = 10000 if self.device.type == 'cuda' else 20000
        
        # Accumulate statistics
        payoff_sum = 0.0
        dS_sum = 0.0
        payoff_dS_sum = 0.0
        payoff_sq_sum = 0.0
        dS_sq_sum = 0.0
        n_processed = 0
        n_valid_total = 0
        
        # Forward-invariant strike: K = F * exp(k), with F = spot * exp((r - q) T)
        F = spot * math.exp((self.r - q) * T)
        K = F * math.exp(logK)
        disc = math.exp(-self.r * T)
        
        n_chunks = (n_paths + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(n_chunks):
            current_chunk_size = min(chunk_size, n_paths - chunk_idx * chunk_size)
            
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
            
            # Manage absorption if supported
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
            
            # Calculate payoffs and accumulate statistics
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
            
            # Free memory
            del S_chunk, ST_chunk
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # If no valid path, return default volatility
        if n_valid_total == 0:
            return self._get_default_volatility(theta)
        
        # Calculate final price by normalizing on the total number of paths processed
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
        
        # Ensure price above the intrinsic (PV di (F-K)^+)
        intrinsic_value = disc * max(0.0, F - K)
        price_call = max(price_call, intrinsic_value + price_floor)
        
        # Calculate IV
        if price_call > 1e-10:
            try:
                bs = BlackScholes(
                    EuropeanOption(
                        BrownianStock(),
                        strike=float(K),
                        maturity=float(T)
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
        Gets a default volatility based on model parameters.
        """
        if hasattr(self.process, 'param_info'):
            param_info = self.process.param_info
            # Search for volatility-related parameters
            for vol_name in ['xi0', 'sigma', 'theta']:
                if vol_name in param_info.names:
                    idx = param_info.names.index(vol_name)
                    value = theta[idx].item()
                    # For xi0 and theta (variance), take sqrt
                    if vol_name in ['xi0', 'theta']:
                        return math.sqrt(value)
                    else:
                        return value
        
        # General default
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
                    chunk_size: int = None,
                    q: float = 0.0) -> torch.Tensor:
        """
        Calculate IV on grid via Monte Carlo.
        """
        # Validate parameters
        is_valid, error_msg = self.process.validate_theta(theta)
        if not is_valid:
            raise ValueError(f"Invalid parameters: {error_msg}")
        
        # Determines chunk_size if not specified
        if chunk_size is None:
            chunk_size = 10000 if self.device.type == 'cuda' else 20000
        
        # Initialize results grid
        iv = torch.zeros(len(maturities), len(logK), device=self.device)
        
        # Process each deadline separately
        for iT, T in enumerate(maturities):
            T_val = T.item()
            disc = math.exp(-self.r * T_val)
            # Forward for this maturity
            F_val = spot * math.exp((self.r - q) * T_val)
            
            # Determine dt for this deadline
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
            
            # Accumulate statistics for each strike
            strike_stats = {jK: {'payoff_sum': 0.0, 'dS_sum': 0.0, 'payoff_dS_sum': 0.0,
                                'payoff_sq_sum': 0.0, 'dS_sq_sum': 0.0, 
                                'n_processed': 0, 'n_valid': 0}
                           for jK in range(len(logK))}
            
            # Process in chunks
            n_chunks = (n_paths + chunk_size - 1) // chunk_size
            
            for chunk_idx in range(n_chunks):
                current_chunk_size = min(chunk_size, n_paths - chunk_idx * chunk_size)
                
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
                
                # Manage absorption
                if self.process.supports_absorption:
                    valid_mask = ST_chunk > 0
                else:
                    valid_mask = torch.ones_like(ST_chunk, dtype=torch.bool)
                
                n_valid_chunk = valid_mask.sum().item()
                
                # Accumulate statistics for each strike
                for jK, k in enumerate(logK):
                    # Strike relative to the forward: K = F * exp(k)
                    K = F_val * math.exp(k.item())
                    
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
                
                # Free memory
                del S_chunk, ST_chunk
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Calculate IV for each strike
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
                
                # Ensure price above the intrinsic (PV di (F-K)^+)
                K = F_val * math.exp(k.item())
                intrinsic_value = disc * max(0.0, F_val - K)
                price_call = max(price_call, intrinsic_value + price_floor)
                
                # Warning if many paths absorbed
                if self.process.supports_absorption:
                    absorption_ratio = 1.0 - (n_valid / n_total)
                    if absorption_ratio > 0.2:
                        print(f"Warning: {absorption_ratio:.1%} paths absorbed at T={T_val:.3f}, K={K:.3f}")
                
                # Calculate IV
                if price_call > 1e-10:
                    try:
                        bs = BlackScholes(
                            EuropeanOption(
                                BrownianStock(),
                                strike=float(K),
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
        """Train on point-wise datasets."""
        
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
        """Set normalization statistics"""
        self.register_buffer('T_mean', T_mean.to(self.device))
        self.register_buffer('T_std', T_std.to(self.device))
        self.register_buffer('k_mean', torch.tensor(k_mean, device=self.device))
        self.register_buffer('k_std', torch.tensor(k_std, device=self.device))
    
    def normalize_T(self, T):
        """Normalize maturity"""
        if self.T_mean is None or self.T_std is None:
            return T
        return (T - self.T_mean) / self.T_std
    
    def normalize_k(self, k):
        """Normalize log-moneyness"""
        if self.k_mean is None or self.k_std is None:
            return k
        return (k - self.k_mean) / self.k_std

    def price_iv_grid(self, theta: torch.Tensor, maturities: torch.Tensor, 
                  logK: torch.Tensor, denormalize_output: bool = True,
                  inputs_normalized: bool = False) -> torch.Tensor:
        """
        Computes full IV grid.

        Args:
            theta: model parameters (NOT normalized)
            maturities: maturity array (NOT normalized)
            logK: log-moneyness array (NOT normalized)
            inputs_normalized: set True if `theta` is already normalized
        """
        self.net.eval()
        
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        B, P = theta.shape[0], theta.shape[-1]
        T_len = maturities.numel()
        K_len = logK.numel()

        # Mesh and broadcast
        mat_mesh, k_mesh = torch.meshgrid(maturities.reshape(-1), logK.reshape(-1), indexing='ij')
        theta_exp = theta[:, None, None, :].expand(B, T_len, K_len, P)
        mat_exp   = mat_mesh.unsqueeze(0).expand(B, T_len, K_len)
        k_exp     = k_mesh.unsqueeze(0).expand(B, T_len, K_len)

        # Flatten
        theta_flat = theta_exp.reshape(-1, P)
        T_flat     = mat_exp.reshape(-1)
        k_flat     = k_exp.reshape(-1)

        with torch.no_grad():
            iv_flat = self.price_iv(theta_flat, T_flat, k_flat,
                                    denormalize_output=denormalize_output,
                                    inputs_normalized=inputs_normalized)
        return iv_flat.view(B, T_len, K_len)
    
    
    def price_iv(self, theta: torch.Tensor, T: torch.Tensor, k: torch.Tensor, 
                 denormalize_output: bool = True,
                 inputs_normalized: bool = False) -> torch.Tensor:
        """
        Calculate IV for specific points (theta, T, k).

        Args:
            theta: model parameters
            T: maturity
            k: log-moneyness
            denormalize_output: if True, denormalizes the output
            inputs_normalized: set True if (theta, T, k) are already normalized
        """
        self.net.eval()
        
        if inputs_normalized:
            # Use tensors as-is (already normalized)
            iv_norm = self.forward(theta, T, k)
        else:
            # Normalize then forward
            iv_norm = self.forward(
                self.normalize_theta(theta),
                self.normalize_T(T),
                self.normalize_k(k)
            )
        
        # Denormalize output if required
        if denormalize_output:
            return self.denormalize_iv(iv_norm)
        else:
            return iv_norm
        
    
    def get_repair_statistics(self):
        """
        Returns statistics of the last smile repair
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
        Saves the model weights.

        Args:
            path: file path
            include_norm_stats: If True, includes normalization statistics
        """
        if include_norm_stats:
            self.save_full(path)
        else:
            self.save_model_only(path)
    
    @classmethod
    def load(cls, path: str, process: Union[str, StochasticProcess], 
             load_norm_stats: bool = False, **kwargs):
        """
        Load a saved model.

        Args:
            path: file path
            process: process name or StochasticProcess instance
            load_norm_stats: if True, also attempts to load statistics
            **kwargs: constructor parameters
        """
        device = kwargs.get('device', 'cpu')
        
        # Create process if passed as a string
        if isinstance(process, str):
            process_obj = ProcessFactory.create(process)
        else:
            process_obj = process
        
        obj = cls(process=process_obj, **kwargs)
        
        # Load the state dict
        sd = torch.load(path, map_location=device, weights_only=True)
        
        if load_norm_stats:
            # Load everything including statistics if any
            obj.load_state_dict(sd, strict=False)
        else:
            # Load model weights only
            model_state = {k: v for k, v in sd.items() 
                          if k not in ['theta_mean', 'theta_std', 'iv_mean', 'iv_std', 
                                       'T_mean', 'T_std', 'k_mean', 'k_std']}
            obj.load_state_dict(model_state, strict=False)
        
        obj.eval()
        return obj
    
    
class MultiRegimeGridPricer(NeuralSurfacePricer):
    """Models the implied volatility surface using specialized-regime networks.

    This class implements a sophisticated hybrid approach that recognizes that
    volatility dynamics behave differently across various time horizons. Instead
    of using a single monolithic network, it employs three distinct
    `GridNetworkPricer` instances, each specialized for a specific maturity regime:
    short-term, mid-term, and long-term.

    This architecture allows the framework to capture the nuanced term structure
    of volatility more accurately. For instance, it can better model the steep,
    explosive skew for short-term options while simultaneously fitting the
    flatter, more stable smiles of long-term options. When a prediction is
    needed, the class intelligently routes the request to the appropriate
    specialized network based on the option's maturity.

    Attributes:
        short_term_pricer (GridNetworkPricer): The pricer trained on short-dated options.
        mid_term_pricer (GridNetworkPricer): The pricer trained on mid-dated options.
        long_term_pricer (GridNetworkPricer): The pricer trained on long-dated options.
        short_term_threshold (float): The maturity (in years) separating the short and mid regimes.
        mid_term_threshold (float): The maturity (in years) separating the mid and long regimes.
    """
    
    def __init__(self,
                 process: StochasticProcess,
                 # Parameters for the 3 regimes
                 short_term_maturities: torch.Tensor,
                 short_term_logK: torch.Tensor,
                 mid_term_maturities: torch.Tensor,
                 mid_term_logK: torch.Tensor,
                 long_term_maturities: torch.Tensor,
                 long_term_logK: torch.Tensor,
                 # Thresholds for dividing regimes (in years)
                 short_term_threshold: float = 0.25,
                 mid_term_threshold: float = 1.0,
                 dt: float = 1/365,
                 device: str = 'cpu',
                 r: float = 0.0,
                 # Architecture parameters for regime
                 short_term_hidden: list = [128, 64],
                 mid_term_hidden: list = [128, 64],
                 long_term_hidden: list = [128, 64],
                 short_term_activation: str = 'ReLU',
                 mid_term_activation: str = 'ReLU',
                 long_term_activation: str = 'ReLU',
                 # Interpolation parameters
                 interpolation_method: str = 'thin_plate_spline',
                 extrapolation: str = 'flat',
                 short_term_dt: float = None,
                 mid_term_dt: float = None,
                 long_term_dt: float = None):
        
        super().__init__(device=device, r=r)
        
        # Save process and thresholds
        self.process = process
        self.short_term_threshold = short_term_threshold
        self.mid_term_threshold = mid_term_threshold
        
        # Create the 3 separate pricers using GridNetworkPricer
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
        
        if short_term_dt is not None:
            self.short_term_pricer.fixed_regime_dt = short_term_dt
        if mid_term_dt is not None:
            self.mid_term_pricer.fixed_regime_dt = mid_term_dt
        if long_term_dt is not None:
            self.long_term_pricer.fixed_regime_dt = long_term_dt
        
        self.regime_dt_config = {
            'short': short_term_dt or 3e-5,
            'mid': mid_term_dt or 1e-4,
            'long': long_term_dt or 1/365
        }
        # Save all maturities and strikes for reference
        self.all_maturities = torch.cat([
            short_term_maturities,
            mid_term_maturities,
            long_term_maturities
        ]).unique().sort()[0]
        
        # For the final interpolation
        self.global_interpolator = VolatilityInterpolator(
            method=interpolation_method,
            extrapolation=extrapolation
        )
        self._global_interpolator_fitted = False
    
    def _get_regime(self, T: float) -> str:
        """Determine the maturity-based regime."""
        if T <= self.short_term_threshold:
            return 'short'
        elif T <= self.mid_term_threshold:
            return 'mid'
        else:
            return 'long'
    
    def _get_pricer_for_maturity(self, T: float) -> GridNetworkPricer:
        """Returns the appropriate pricer for the given maturity."""
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
        Sets the normalization statistics for all regimes.
        Note: theta_mean and theta_std are common to all regimes.
        """
        # Set theta statistics for all pricers
        self.short_term_pricer.set_normalization_stats(
            theta_mean, theta_std, short_iv_mean, short_iv_std
        )
        self.mid_term_pricer.set_normalization_stats(
            theta_mean, theta_std, mid_iv_mean, mid_iv_std
        )
        self.long_term_pricer.set_normalization_stats(
            theta_mean, theta_std, long_iv_mean, long_iv_std
        )
        
        # Also save in parent class for consistency
        super().set_normalization_stats(
            theta_mean, theta_std, 
            (short_iv_mean + mid_iv_mean + long_iv_mean) / 3,  # Approximate average
            (short_iv_std + mid_iv_std + long_iv_std) / 3
        )
    
    def fit(self,
            # Short term data
            theta_train_short: torch.Tensor,
            iv_train_short: torch.Tensor,
            theta_val_short: torch.Tensor = None,
            iv_val_short: torch.Tensor = None,
            # Mid-term data
            theta_train_mid: torch.Tensor = None,
            iv_train_mid: torch.Tensor = None,
            theta_val_mid: torch.Tensor = None,
            iv_val_mid: torch.Tensor = None,
            # Long term data
            theta_train_long: torch.Tensor = None,
            iv_train_long: torch.Tensor = None,
            theta_val_long: torch.Tensor = None,
            iv_val_long: torch.Tensor = None,
            # Training parameters
            n_paths: int = 4096,
            epochs: int = 30,
            batch_size: int = 512,
            lr: float = 1e-3,
            lr_decay: float = 0.95,  # Learning rate decay between regimes
            verbose: bool = True):
        """
        Fit the three pricers sequentially or in parallel.
        If mid/long data is not provided, use the same thetas as the short.
        """
        
        # If not provided, use the same thetas for all regimes
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
        
        # Mid-term fit with reduced learning rate
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
        
        # Long-term fit with further reduced learning rate
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
        Calculate the IV surfaces for all regimes.

        Returns:
        dict with keys 'short', 'mid', 'long' containing the respective surfaces
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
        Computes a unified IV surface on a common grid.

        Args:
            theta: model parameters
            maturities: desired maturities (if None, use all)
            logK: desired log-moneyness (if None, use union of all)
            denormalize_output: whether to denormalize the output

        Returns:
            Unified Surface [batch_size, len(maturities), len(logK)]
        """
        if maturities is None:
            maturities = self.all_maturities
        
        if logK is None:
            # Union of all logK of the 3 regimes
            all_logK = torch.cat([
                self.short_term_pricer.logKs,
                self.mid_term_pricer.logKs,
                self.long_term_pricer.logKs
            ]).unique().sort()[0]
            logK = all_logK
        
        batch_size = theta.shape[0]
        n_T = len(maturities)
        n_K = len(logK)
        
        # Initialize output
        unified_surface = torch.zeros(batch_size, n_T, n_K, device=self.device)
        
        # For each batch
        for b in range(batch_size):
            theta_b = theta[b:b+1]
            
            # Get surfaces from the 3 regimes
            surfaces = self.price_iv(theta_b, denormalize_output)
            
            # Fit interpolators for each regime
            self.short_term_pricer._fit_interpolator(surfaces['short'][0].detach())
            self.mid_term_pricer._fit_interpolator(surfaces['mid'][0].detach())
            self.long_term_pricer._fit_interpolator(surfaces['long'][0].detach())
            
            # Interpolate for each point of the unified grid
            for i, T in enumerate(maturities):
                T_val = T.item()
                pricer = self._get_pricer_for_maturity(T_val)
                
                for j, k in enumerate(logK):
                    k_val = k.item()
                    
                    # Use the appropriate regime interpolator
                    unified_surface[b, i, j] = pricer.interpolate_iv(T_val, k_val)
        
        return unified_surface
    
    def interpolate_iv(self, T, k, theta=None):
        """
        Interpolate IV for arbitrary points (T, k) using the appropriate regime.

        Args:
            T: maturity (scalar or array)
            k: log-moneyness (scalar or array)
            theta: model parameters (optional if already fitted)

        Returns:
            Interpolated IV
        """
        # Convert to numpy if necessary
        T_np = np.asarray(T)
        k_np = np.asarray(k)
        scalar_input = T_np.ndim == 0
        
        if scalar_input:
            T_np = T_np.reshape(1)
            k_np = k_np.reshape(1)
        
        # If theta is given, calculate the surfaces
        if theta is not None:
            surfaces = self.price_iv(theta.unsqueeze(0) if theta.dim() == 1 else theta)
            
            # Fit interpolators
            self.short_term_pricer._fit_interpolator(surfaces['short'][0].detach())
            self.mid_term_pricer._fit_interpolator(surfaces['mid'][0].detach())
            self.long_term_pricer._fit_interpolator(surfaces['long'][0].detach())
        
        # Interpolate using the appropriate regime for each point
        results = np.zeros_like(T_np, dtype=float)
        
        for i, (t, k_val) in enumerate(zip(T_np, k_np)):
            pricer = self._get_pricer_for_maturity(t)
            results[i] = pricer.interpolate_iv(t, k_val)
        
        if scalar_input:
            return results[0]
        return results
    
    def save(self, path_prefix: str, include_norm_stats: bool = False):
        """
        Save all 3 models.

        Args:
            path_prefix: File prefix (e.g., 'model' -> 'model_short.pt', etc.)
            include_norm_stats: Whether to include normalization statistics
        """
        self.short_term_pricer.save(f"{path_prefix}_short.pt", include_norm_stats)
        self.mid_term_pricer.save(f"{path_prefix}_mid.pt", include_norm_stats)
        self.long_term_pricer.save(f"{path_prefix}_long.pt", include_norm_stats)
        
        # Also saves metadata about regimes
        metadata = {
            'short_term_threshold': self.short_term_threshold,
            'mid_term_threshold': self.mid_term_threshold,
            'short_term_maturities': self.short_term_pricer.Ts.cpu(),
            'short_term_logK': self.short_term_pricer.logKs.cpu(),
            'mid_term_maturities': self.mid_term_pricer.Ts.cpu(),
            'mid_term_logK': self.mid_term_pricer.logKs.cpu(),
            'long_term_maturities': self.long_term_pricer.Ts.cpu(),
            'long_term_logK': self.long_term_pricer.logKs.cpu(),
            'regime_dt_config': self.regime_dt_config
        }
        torch.save(metadata, f"{path_prefix}_metadata.pt")
    
    @classmethod
    def load(cls, path_prefix: str, process: Union[str, StochasticProcess],
             load_norm_stats: bool = False, **kwargs):
        """
        Load a saved multi-regime model.

        Args:
            path_prefix: File prefix
            process: Process name or StochasticProcess instance
            load_norm_stats: Whether to load statistics
            **kwargs: Additional parameters
        """
        # Upload metadata
        device = kwargs.get('device', 'cpu')
        metadata = torch.load(f"{path_prefix}_metadata.pt", map_location=device, weights_only=True)
        if 'regime_dt_config' in metadata:
            kwargs['short_term_dt'] = metadata['regime_dt_config'].get('short')
            kwargs['mid_term_dt'] = metadata['regime_dt_config'].get('mid')  
            kwargs['long_term_dt'] = metadata['regime_dt_config'].get('long')
        # Create process if passed as a string
        if isinstance(process, str):
            process_obj = ProcessFactory.create(process)
        else:
            process_obj = process
        
        # Create instance
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
        
        # Load the weights of the 3 models
        short_sd = torch.load(f"{path_prefix}_short.pt", map_location=device, weights_only=True)
        mid_sd = torch.load(f"{path_prefix}_mid.pt", map_location=device, weights_only=True)
        long_sd = torch.load(f"{path_prefix}_long.pt", map_location=device, weights_only=True)
        
        if load_norm_stats:
            obj.short_term_pricer.load_state_dict(short_sd, strict=False)
            obj.mid_term_pricer.load_state_dict(mid_sd, strict=False)
            obj.long_term_pricer.load_state_dict(long_sd, strict=False)
        else:
            # Load model weights only
            for pricer, sd in [(obj.short_term_pricer, short_sd),
                              (obj.mid_term_pricer, mid_sd),
                              (obj.long_term_pricer, long_sd)]:
                model_state = {k: v for k, v in sd.items() 
                             if k not in ['theta_mean', 'theta_std', 'iv_mean', 'iv_std']}
                pricer.load_state_dict(model_state, strict=False)
        
        obj.eval()
        return obj
    
    def to(self, device):
        """Override to correctly manage the device of all pricers."""
        self.device = device
        self.short_term_pricer = self.short_term_pricer.to(device)
        self.mid_term_pricer = self.mid_term_pricer.to(device)
        self.long_term_pricer = self.long_term_pricer.to(device)
        return super().to(device)
    
    def eval(self):
        """Put all pricers in eval mode."""
        super().eval()
        self.short_term_pricer.eval()
        self.mid_term_pricer.eval()
        self.long_term_pricer.eval()
        return self
    
    def train(self, mode=True):
        """Put all pricers in train mode."""
        super().train(mode)
        self.short_term_pricer.train(mode)
        self.mid_term_pricer.train(mode)
        self.long_term_pricer.train(mode)
        return self