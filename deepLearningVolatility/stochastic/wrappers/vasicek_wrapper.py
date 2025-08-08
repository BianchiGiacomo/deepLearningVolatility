"""
Wrapper per il modello di Vasicek (tasso d'interesse).
"""
import torch
from torch import Tensor
from typing import Optional, Tuple

from deepLearningVolatility.stochastic.stochastic_interface import (
    BaseStochasticProcess, SimulationOutput, ParameterInfo, ProcessFactory
)
from deepLearningVolatility.stochastic.vasicek import generate_vasicek

class VasicekProcess(BaseStochasticProcess):
    """
    Wrapper per il modello di Vasicek (mean-reverting process).
    
    Parametri theta:
        - kappa: Mean reversion speed
        - theta: Long-term mean
        - sigma: Volatility
    """
    
    def __init__(self, spot: float = 0.04):  # Default al 4% per tasso
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
                (0.0, 0.2),      # theta (0-20% per tassi)
                (0.001, 0.1)     # sigma
            ],
            defaults=[1.0, 0.04, 0.04],
            descriptions=[
                'Mean reversion speed',
                'Long-term mean level',
                'Volatility'
            ]
        )
    
    @property
    def supports_absorption(self) -> bool:
        return False  # Vasicek può andare negativo, non tocca zero
    
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
        """Simula il processo di Vasicek."""
        
        # Valida parametri
        is_valid, error_msg = self.validate_theta(theta)
        if not is_valid:
            raise ValueError(f"Invalid parameters: {error_msg}")
        
        # Estrai parametri
        kappa, theta_param, sigma = theta.tolist()
        
        # Prepara stato iniziale
        init_state = self.prepare_init_state(init_state)
        
        # Simula
        try:
            paths = generate_vasicek(
                n_paths=n_paths,
                n_steps=n_steps,
                init_state=init_state,
                kappa=kappa,
                theta=theta_param,
                sigma=sigma,
                dt=dt,
                dtype=dtype,
                device=device,
                antithetic=antithetic
            )
            
            return SimulationOutput(
                spot=paths,  # Per Vasicek, "spot" rappresenta il tasso
                variance=None
            )
            
        except Exception as e:
            raise RuntimeError(f"Vasicek simulation failed: {str(e)}") from e


# Registra il processo
ProcessFactory.register('vasicek', VasicekProcess)