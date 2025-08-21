"""
Wrapper per il modello di Heston.
"""
import torch
from torch import Tensor
from typing import Optional, Tuple

from deepLearningVolatility.stochastic.stochastic_interface import (
    BaseStochasticProcess, SimulationOutput, ParameterInfo, ProcessFactory
)
from deepLearningVolatility.stochastic.heston import generate_heston



class HestonProcess(BaseStochasticProcess):
    """
    Wrapper per il modello di Heston.
    
    Parametri theta:
        - kappa: Mean reversion speed
        - theta: Long-term variance
        - sigma: Volatility of variance
        - rho: Correlation
    """
    
    def __init__(self, spot: float = 1.0):
        super().__init__(spot)
    
    @property
    def num_params(self) -> int:
        return 4
    
    @property
    def param_info(self) -> ParameterInfo:
        return ParameterInfo(
            names=['kappa', 'theta', 'sigma', 'rho'],
            bounds=[
                (0.1, 5.0),      # kappa
                (0.01, 0.5),     # theta
                (0.1, 1.0),      # sigma
                (-0.95, 0.95)    # rho
            ],
            defaults=[1.0, 0.04, 0.2, -0.7],
            descriptions=[
                'Mean reversion speed',
                'Long-term variance',
                'Volatility of variance',
                'Correlation between spot and variance'
            ]
        )
    
    @property
    def supports_absorption(self) -> bool:
        return True  # Heston può toccare zero
    
    @property
    def requires_variance_state(self) -> bool:
        return True
    
    def get_default_init_state(self) -> Tuple[float, ...]:
        """Per Heston, usa theta come varianza iniziale di default."""
        return (self.spot, self.param_info.defaults[1])  # (S0, V0=theta)
    
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
        Simula il processo di Heston.
        """
        # Valida parametri
        is_valid, error_msg = self.validate_theta(theta)
        if not is_valid:
            raise ValueError(f"Invalid parameters: {error_msg}")
        
        # Estrai parametri
        kappa, theta_param, sigma, rho = theta.tolist()
        
        # Prepara stato iniziale
        init_state = self.prepare_init_state(init_state)
        
        # Simula
        try:
            spot_variance_tuple = generate_heston(
                n_paths=n_paths,
                n_steps=n_steps,
                init_state=init_state,
                kappa=kappa,
                theta=theta_param,
                sigma=sigma,
                rho=rho,
                dt=dt,
                dtype=dtype,
                device=device
            )
            
            # generate_heston ritorna un namedtuple (spot, variance)
            return SimulationOutput(
                spot=spot_variance_tuple.spot,
                variance=spot_variance_tuple.variance,
                auxiliary={'volatility': spot_variance_tuple.volatility}
            )
            
        except Exception as e:
            raise RuntimeError(f"Heston simulation failed: {str(e)}") from e


# Registra il processo
ProcessFactory.register(
    'heston', HestonProcess,
    aliases=['heston_process', 'hestonprocess']
)
