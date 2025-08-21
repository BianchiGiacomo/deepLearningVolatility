"""
Wrapper per il modello Rough Bergomi.
"""
import torch
from torch import Tensor
from typing import Optional, Tuple, Dict, Any

from deepLearningVolatility.stochastic.stochastic_interface import (
    BaseStochasticProcess, SimulationOutput, ParameterInfo, ProcessFactory
)
from deepLearningVolatility.stochastic.rough_bergomi import generate_rough_bergomi


class RoughBergomiProcess(BaseStochasticProcess):
    """
    Wrapper per il modello Rough Bergomi.
    
    Parametri theta:
        - H: Hurst parameter (0 < H < 0.5)
        - eta: Volatility of volatility
        - rho: Correlation
        - xi0: Initial variance
    """
    
    def __init__(self, spot: float = 1.0):
        super().__init__(spot)
        self._supports_absorption = True
    
    @property
    def num_params(self) -> int:
        return 4
    
    @property
    def param_info(self) -> ParameterInfo:
        return ParameterInfo(
            names=['H', 'eta', 'rho', 'xi0'],
            bounds=[
                (0.05, 0.45),    # H
                (0.5, 3.0),      # eta
                (-0.95, -0.1),   # rho
                (0.02, 0.16)     # xi0
            ],
            defaults=[0.1, 1.5, -0.7, 0.04],
            descriptions=[
                'Hurst parameter (roughness)',
                'Volatility of volatility',
                'Correlation between price and variance',
                'Initial variance'
            ]
        )
    
    @property
    def supports_absorption(self) -> bool:
        return self._supports_absorption
    
    @property
    def requires_variance_state(self) -> bool:
        return True
    
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
        Simula il processo Rough Bergomi.
        """
        # Valida parametri
        is_valid, error_msg = self.validate_theta(theta)
        if not is_valid:
            raise ValueError(f"Invalid parameters: {error_msg}")
        
        # Estrai parametri
        H, eta, rho, xi0 = theta.tolist()
        
        # Prepara stato iniziale
        init_state = self.prepare_init_state(init_state)
        spot_init, var_init = init_state
        
        # Parametri per la simulazione
        alpha = H - 0.5  # Conversione per rough_bergomi
        
        # Gestisci antithetic se la funzione lo supporta
        sim_kwargs = {
            'n_paths': n_paths,
            'n_steps': n_steps,
            'init_state': (spot_init, var_init),
            'alpha': alpha,
            'rho': rho,
            'eta': eta,
            'xi': var_init,  # xi iniziale = varianza iniziale
            'dt': dt,
            'device': device,
            'dtype': dtype
        }
        
        # Aggiungi parametri opzionali se supportati
        import inspect
        sig = inspect.signature(generate_rough_bergomi)
        if 'antithetic' in sig.parameters:
            sim_kwargs['antithetic'] = antithetic
        
        # Filtra kwargs extra supportati
        for key, value in kwargs.items():
            if key in sig.parameters and key not in sim_kwargs:
                sim_kwargs[key] = value
        
        # Simula
        try:
            spot, variance = generate_rough_bergomi(**sim_kwargs)
            
            # Crea output
            auxiliary = {
                'alpha': torch.tensor(alpha),
                'H': torch.tensor(H)
            }
            
            return SimulationOutput(
                spot=spot,
                variance=variance,
                auxiliary=auxiliary
            )
            
        except Exception as e:
            raise RuntimeError(f"Simulation failed: {str(e)}") from e
    
    def handle_absorption(self, 
                         paths: Tensor, 
                         dt: float,
                         threshold: float = 1e-10) -> Tuple[Tensor, Tensor]:
        """
        Gestione specifica per Rough Bergomi.
        Il modello può toccare zero a causa della roughness.
        """
        # Usa l'implementazione di default che è già ottimizzata
        return super().handle_absorption(paths, dt, threshold)


# Registra il processo nel factory
ProcessFactory.register(
    'rough_bergomi', RoughBergomiProcess,
    aliases=['roughbergomi', 'rough-bergomi', 'rough_bergomi_process', 'roughbergomiprocess']
)