"""
Wrapper for the Rough Heston model.
"""
import torch
from torch import Tensor
from typing import Optional, Tuple, Dict, Any

from deepLearningVolatility.stochastic.stochastic_interface import (
    BaseStochasticProcess, SimulationOutput, ParameterInfo, ProcessFactory
)
from deepLearningVolatility.stochastic.rough_heston import generate_rough_heston


class RoughHestonProcess(BaseStochasticProcess):
    """
    Wrapper for the Rough Heston model.
    
    Theta parameters:
        - H: Hurst parameter (0 < H < 0.5)
        - nu: Volatility of volatility (σ in the paper)
        - rho: Correlation
        - kappa: Mean reversion speed (λ in the paper)
        - theta_var: Long-term variance level
    """
    
    def __init__(self, spot: float = 1.0):
        super().__init__(spot)
        self._supports_absorption = True  # Rough Heston can reach zero!
    
    @property
    def num_params(self) -> int:
        return 5  # H, nu, rho, kappa, theta
    
    @property
    def param_info(self) -> ParameterInfo:
        return ParameterInfo(
            names=['H', 'nu', 'rho', 'kappa', 'theta_var'],
            bounds=[
                (0.05, 0.45),    # H (Hurst parameter)
                (0.1, 1.0),      # nu (vol of vol)
                (-0.95, -0.1),   # rho (correlation)
                (0.1, 2.0),      # kappa (mean reversion)
                (0.01, 0.1)      # theta_var (long-term variance)
            ],
            defaults=[0.1, 0.3, -0.7, 0.3, 0.02],
            descriptions=[
                'Hurst parameter (roughness)',
                'Volatility of volatility',
                'Correlation between price and variance',
                'Mean reversion speed',
                'Long-term variance level'
            ]
        )
    
    @property
    def supports_absorption(self) -> bool:
        return self._supports_absorption
    
    @property
    def requires_variance_state(self) -> bool:
        return True
    
    def get_default_init_state(self) -> Tuple[float, ...]:
        """Per Rough Heston, usa theta_var come varianza iniziale di default."""
        return (self.spot, self.param_info.defaults[4])  # (S0, V0=theta_var)
    
    def simulate(self,
                 theta: Tensor,
                 n_paths: int,
                 n_steps: int,
                 dt: float,
                 init_state: Optional[Tuple[float, ...]] = None,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 antithetic: bool = False,
                 **kwargs) -> SimulationOutput:
        """
        Simulate the Rough Heston process.
        """
        # Validate parameters
        is_valid, error_msg = self.validate_theta(theta)
        if not is_valid:
            raise ValueError(f"Invalid parameters: {error_msg}")
        
        # Extract parameters
        H, nu, rho, kappa, theta_var = theta.tolist()
        
        # Prepare initial state
        init_state = self.prepare_init_state(init_state)
        spot_init, var_init = init_state
        
        # Parameters for simulation
        sim_kwargs = {
            'n_paths': n_paths,
            'n_steps': n_steps,
            'init_state': (spot_init, var_init),
            'H': H,
            'nu': nu,
            'rho': rho,
            'kappa': kappa,
            'theta': theta_var,
            'dt': dt,
            'device': device,
            'dtype': dtype
        }
        
        # Add optional parameters if supported
        import inspect
        sig = inspect.signature(generate_rough_heston)
        if 'antithetic' in sig.parameters:
            sim_kwargs['antithetic'] = antithetic
        
        # Filter extra supported kwargs
        for key, value in kwargs.items():
            if key in sig.parameters and key not in sim_kwargs:
                sim_kwargs[key] = value
        
        # Simulate
        try:
            spot_variance_tuple = generate_rough_heston(**sim_kwargs)
            
            # Create output
            auxiliary = {
                'H': torch.tensor(H),
                'roughness': torch.tensor(H),
                'mean_reversion': torch.tensor(kappa),
                'long_term_var': torch.tensor(theta_var)
            }
            
            return SimulationOutput(
                spot=spot_variance_tuple.spot,
                variance=spot_variance_tuple.variance,
                auxiliary=auxiliary
            )
            
        except Exception as e:
            raise RuntimeError(f"Rough Heston simulation failed: {str(e)}") from e
    
    def handle_absorption(self, 
                         paths: Tensor, 
                         dt: float,
                         threshold: float = 1e-10) -> Tuple[Tensor, Tensor]:
        """
        Specific handling for Rough Heston.
        The model can reach zero both due to roughness and the Heston nature.
        
        For Rough Heston with very small H, it may be necessary
        to handle adaptive thresholds.
        """
        # Possible optimization: adaptive threshold based on H
        if hasattr(self, '_last_H') and self._last_H < 0.1:
            threshold = max(threshold, 1e-8)
        
        # Use the default implementation which is already optimized
        return super().handle_absorption(paths, dt, threshold)


# Register the process in the factory
ProcessFactory.register(
    "rough_heston",
    RoughHestonProcess,
    aliases=["roughheston", "rough-heston", "rough_heston_process", "roughhestonprocess"]
)