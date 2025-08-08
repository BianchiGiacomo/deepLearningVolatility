"""
Wrapper per il moto Browniano geometrico.
"""
import torch
from torch import Tensor
from typing import Optional, Tuple

from deepLearningVolatility.stochastic.stochastic_interface import (
    BaseStochasticProcess, SimulationOutput, ParameterInfo, ProcessFactory
)
from deepLearningVolatility.stochastic.brownian import generate_geometric_brownian

class GeometricBrownianProcess(BaseStochasticProcess):
    """
    Wrapper per il moto Browniano geometrico (Black-Scholes).
    
    Parametri theta:
        - mu: Drift
        - sigma: Volatility
    """
    
    def __init__(self, spot: float = 1.0):
        super().__init__(spot)
    
    @property
    def num_params(self) -> int:
        return 2
    
    @property
    def param_info(self) -> ParameterInfo:
        return ParameterInfo(
            names=['mu', 'sigma'],
            bounds=[
                (-0.5, 0.5),     # mu
                (0.01, 1.0)      # sigma
            ],
            defaults=[0.0, 0.2],
            descriptions=[
                'Drift coefficient',
                'Volatility'
            ]
        )
    
    @property
    def supports_absorption(self) -> bool:
        return False  # GBM non tocca mai zero
    
    @property
    def requires_variance_state(self) -> bool:
        return False
    
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
        """Simula il moto Browniano geometrico."""
        
        # Valida parametri
        is_valid, error_msg = self.validate_theta(theta)
        if not is_valid:
            raise ValueError(f"Invalid parameters: {error_msg}")
        
        # Estrai parametri
        mu, sigma = theta.tolist()
        
        # Prepara stato iniziale
        init_state = self.prepare_init_state(init_state)
        
        # Engine per random numbers
        if antithetic:
            from deepLearningVolatility.stochastic import randn_antithetic
            engine = lambda *size, dtype=None, device=None: randn_antithetic(
                *size, dtype=dtype, device=device
            )
        else:
            engine = torch.randn
        
        # Simula
        try:
            paths = generate_geometric_brownian(
                n_paths=n_paths,
                n_steps=n_steps,
                init_state=init_state,
                mu=mu,
                sigma=sigma,
                dt=dt,
                dtype=dtype,
                device=device,
                engine=engine  # Passa l'engine
            )
            
            return SimulationOutput(
                spot=paths,
                variance=None
            )
            
        except Exception as e:
            raise RuntimeError(f"Geometric Brownian simulation failed: {str(e)}") from e


# Registra il processo
ProcessFactory.register('geometric_brownian', GeometricBrownianProcess)
ProcessFactory.register('gbm', GeometricBrownianProcess)  # Alias
ProcessFactory.register('black_scholes', GeometricBrownianProcess)  # Alias