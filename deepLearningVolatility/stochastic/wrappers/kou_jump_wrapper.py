"""
Wrapper per il modello Kou Jump Diffusion.
"""
import torch
from torch import Tensor
from typing import Optional, Tuple

from deepLearningVolatility.stochastic.stochastic_interface import (
    BaseStochasticProcess, SimulationOutput, ParameterInfo, ProcessFactory
)
from deepLearningVolatility.stochastic.kou_jump import generate_kou_jump


class KouJumpProcess(BaseStochasticProcess):
    """
    Wrapper per il modello Kou Jump Diffusion.
    
    Parametri theta:
        - sigma: Volatility
        - mu: Drift
        - jump_per_year: Average number of jumps per year
        - jump_mean_up: Mean of up jumps (exponential)
        - jump_mean_down: Mean of down jumps (exponential)
        - jump_up_prob: Probability of up jump given a jump occurs
    """
    
    def __init__(self, spot: float = 1.0):
        super().__init__(spot)
    
    @property
    def num_params(self) -> int:
        return 6
    
    @property
    def param_info(self) -> ParameterInfo:
        return ParameterInfo(
            names=['sigma', 'mu', 'jump_per_year', 'jump_mean_up', 'jump_mean_down', 'jump_up_prob'],
            bounds=[
                (0.05, 0.5),     # sigma
                (-0.5, 0.5),     # mu
                (0.0, 200.0),    # jump_per_year
                (0.01, 0.5),     # jump_mean_up (deve essere < 1)
                (0.01, 0.5),     # jump_mean_down
                (0.0, 1.0)       # jump_up_prob
            ],
            defaults=[0.2, 0.0, 68.0, 0.02, 0.05, 0.5],
            descriptions=[
                'Volatility',
                'Drift coefficient',
                'Average number of jumps per year',
                'Mean of up jumps (exponential)',
                'Mean of down jumps (exponential)',
                'Probability of up jump given a jump occurs'
            ]
        )
    
    @property
    def supports_absorption(self) -> bool:
        return True  # Kou può teoricamente toccare zero con down jumps estremi
    
    @property
    def requires_variance_state(self) -> bool:
        return False  # Solo stato del prezzo
    
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
        Simula il processo Kou Jump Diffusion.
        """
        # Valida parametri
        is_valid, error_msg = self.validate_theta(theta)
        if not is_valid:
            raise ValueError(f"Invalid parameters: {error_msg}")
        
        # Estrai parametri
        sigma, mu, jump_per_year, jump_mean_up, jump_mean_down, jump_up_prob = theta.tolist()
        
        # Validazione extra per Kou
        if jump_mean_up >= 1.0:
            raise ValueError(f"jump_mean_up must be < 1.0, got {jump_mean_up}")
        
        # Prepara stato iniziale
        init_state = self.prepare_init_state(init_state)
        
        # Simula
        try:
            spot_paths = generate_kou_jump(
                n_paths=n_paths,
                n_steps=n_steps,
                init_state=init_state,
                sigma=sigma,
                mu=mu,
                jump_per_year=jump_per_year,
                jump_mean_up=jump_mean_up,
                jump_mean_down=jump_mean_down,
                jump_up_prob=jump_up_prob,
                dt=dt,
                dtype=dtype,
                device=device,
                **kwargs  # engine, etc.
            )
            
            # Kou ritorna solo spot paths
            return SimulationOutput(
                spot=spot_paths,
                variance=None,
                auxiliary={
                    'model': 'kou_jump',
                    'asymmetric_jumps': True
                }
            )
            
        except Exception as e:
            raise RuntimeError(f"Kou Jump simulation failed: {str(e)}") from e


# Registra il processo
ProcessFactory.register('kou_jump', KouJumpProcess)
ProcessFactory.register('kou', KouJumpProcess)  # Alias