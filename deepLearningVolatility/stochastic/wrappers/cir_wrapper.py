"""
Wrapper for the Cox-Ingersoll-Ross process.
"""
import torch
from torch import Tensor
from typing import Optional, Tuple

from deepLearningVolatility.stochastic.stochastic_interface import (
    BaseStochasticProcess, SimulationOutput, ParameterInfo, ProcessFactory
)
from deepLearningVolatility.stochastic.cir import generate_cir

class CIRProcess(BaseStochasticProcess):
    """
    Wrapper for the CIR process (used for rates/volatility).
    
    Theta parameters:
        - kappa: Mean reversion speed
        - theta: Long-term mean
        - sigma: Volatility of volatility
    """
    
    def __init__(self, spot: float = 0.04):
        super().__init__(spot)
    
    @property
    def num_params(self) -> int:
        return 3
    
    @property
    def param_info(self) -> ParameterInfo:
        return ParameterInfo(
            names=['kappa', 'theta', 'sigma'],
            bounds=[
                (0.1, 5.0),      # kappa
                (0.01, 0.2),     # theta
                (0.01, 1.0)      # sigma
            ],
            defaults=[1.0, 0.04, 0.2],
            descriptions=[
                'Mean reversion speed',
                'Long-term mean',
                'Volatility of volatility'
            ]
        )
    
    @property
    def supports_absorption(self) -> bool:
        return True  # CIR can reach zero
    
    @property
    def requires_variance_state(self) -> bool:
        return False  # CIR is single-factor
    
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
        """Simulate the CIR process."""
        
        # Validate parameters
        is_valid, error_msg = self.validate_theta(theta)
        if not is_valid:
            raise ValueError(f"Invalid parameters: {error_msg}")
        
        # Extract parameters
        kappa, theta_param, sigma = theta.tolist()
        
        # Prepare initial state
        init_state = self.prepare_init_state(init_state)
        
         # Note: CIR uses the QE method which does not directly support antithetic
        # But we can pass the flag for future implementations
        if antithetic:
            print("Warning: CIR implementation doesn't support antithetic variables yet")
        
        # Simulate
        try:
            paths = generate_cir(
                n_paths=n_paths,
                n_steps=n_steps,
                init_state=init_state,
                kappa=kappa,
                theta=theta_param,
                sigma=sigma,
                dt=dt,
                dtype=dtype,
                device=device
            )
            
            return SimulationOutput(
                spot=paths,
                variance=None,
                auxiliary={'model': 'CIR', 'method': 'QE-M'}
            )
            
        except Exception as e:
            raise RuntimeError(f"CIR simulation failed: {str(e)}") from e


# Register the process
ProcessFactory.register(
    'cir', CIRProcess,
    aliases=['cir_process', 'cirprocess', 'cox_ingersoll_ross',
             'cox-ingersoll-ross', 'coxingersollross']
)