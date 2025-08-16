"""
Wrapper per il processo Cox-Ingersoll-Ross.
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
    Wrapper per il processo CIR (usato per tassi/volatilità).
    
    Parametri theta:
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
        return True  # CIR può toccare zero
    
    @property
    def requires_variance_state(self) -> bool:
        return False  # CIR è single-factor
    
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
        """Simula il processo CIR."""
        
        # Valida parametri
        is_valid, error_msg = self.validate_theta(theta)
        if not is_valid:
            raise ValueError(f"Invalid parameters: {error_msg}")
        
        # Estrai parametri
        kappa, theta_param, sigma = theta.tolist()
        
        # Prepara stato iniziale
        init_state = self.prepare_init_state(init_state)
        
        # Nota: CIR usa il metodo QE che non supporta direttamente antithetic
        # Ma possiamo passare il flag per future implementazioni
        if antithetic:
            print("Warning: CIR implementation doesn't support antithetic variables yet")
        
        # Simula
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


# Registra il processo
ProcessFactory.register('cir', CIRProcess)
ProcessFactory.register('cox_ingersoll_ross', CIRProcess)  # Alias
ProcessFactory.register(
    'cir', CIRProcess,
    aliases=['cir_process', 'cirprocess', 'cox_ingersoll_ross',
             'cox-ingersoll-ross', 'coxingersollross']
)