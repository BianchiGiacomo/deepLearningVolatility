"""
Wrapper for the Merton Jump Diffusion model.
"""
import torch
from torch import Tensor
from typing import Optional, Tuple

from deepLearningVolatility.stochastic.stochastic_interface import (
    BaseStochasticProcess, SimulationOutput, ParameterInfo, ProcessFactory
)
from deepLearningVolatility.stochastic.merton_jump import generate_merton_jump


class MertonJumpProcess(BaseStochasticProcess):
    """
    Wrapper for the Merton Jump Diffusion model.
    
    Theta parameters:
        - mu: Drift
        - sigma: Volatility
        - jump_per_year: Average number of jumps per year
        - jump_mean: Mean of jump size
        - jump_std: Std dev of jump size
    """
    
    def __init__(self, spot: float = 1.0):
        super().__init__(spot)
    
    @property
    def num_params(self) -> int:
        return 5
    
    @property
    def param_info(self) -> ParameterInfo:
        return ParameterInfo(
            names=['mu', 'sigma', 'jump_per_year', 'jump_mean', 'jump_std'],
            bounds=[
                (-0.5, 0.5),     # mu
                (0.05, 0.5),     # sigma
                (0.0, 200.0),    # jump_per_year
                (-0.1, 0.1),     # jump_mean
                (0.01, 0.2)      # jump_std
            ],
            defaults=[0.0, 0.2, 68.2, 0.0, 0.02],
            descriptions=[
                'Drift coefficient',
                'Volatility',
                'Average number of jumps per year',
                'Mean of jump sizes',
                'Standard deviation of jump sizes'
            ]
        )
    
    @property
    def supports_absorption(self) -> bool:
        return False  # Merton does not touch zero (log-normal jumps)
    
    @property
    def requires_variance_state(self) -> bool:
        return False  # Only price state
    
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
        Simulate the Merton Jump Diffusion process.
        """
        # Validate parameters
        is_valid, error_msg = self.validate_theta(theta)
        if not is_valid:
            raise ValueError(f"Invalid parameters: {error_msg}")
        
        # Extract parameters
        mu, sigma, jump_per_year, jump_mean, jump_std = theta.tolist()
        
        # Prepare initial state
        init_state = self.prepare_init_state(init_state)
        
        # Simulate
        try:
            spot_paths = generate_merton_jump(
                n_paths=n_paths,
                n_steps=n_steps,
                init_state=init_state,
                mu=mu,
                sigma=sigma,
                jump_per_year=jump_per_year,
                jump_mean=jump_mean,
                jump_std=jump_std,
                dt=dt,
                dtype=dtype,
                device=device,
                **kwargs  # engine, etc.
            )
            
            # Merton returns only spot paths
            return SimulationOutput(
                spot=spot_paths,
                variance=None,
                auxiliary={'model': 'merton_jump'}
            )
            
        except Exception as e:
            raise RuntimeError(f"Merton Jump simulation failed: {str(e)}") from e


# Register the process
ProcessFactory.register(
    'merton_jump', MertonJumpProcess,
    aliases=['merton', 'mertonjump', 'merton-jump', 'merton_jump_process', 'mertonjumpprocess']
)