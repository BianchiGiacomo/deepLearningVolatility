"""
Wrapper per il modello Rough Heston.
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
    Wrapper per il modello Rough Heston.
    
    Parametri theta:
        - H: Hurst parameter (0 < H < 0.5)
        - nu: Volatility of volatility (σ nel paper)
        - rho: Correlation
        - kappa: Mean reversion speed (λ nel paper)
        - theta_var: Long-term variance level
    """
    
    def __init__(self, spot: float = 1.0):
        super().__init__(spot)
        self._supports_absorption = True  # Rough Heston può toccare zero!
    
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
        Simula il processo Rough Heston.
        """
        # Valida parametri
        is_valid, error_msg = self.validate_theta(theta)
        if not is_valid:
            raise ValueError(f"Invalid parameters: {error_msg}")
        
        # Estrai parametri
        H, nu, rho, kappa, theta_var = theta.tolist()
        
        # Prepara stato iniziale
        init_state = self.prepare_init_state(init_state)
        spot_init, var_init = init_state
        
        # Parametri per la simulazione
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
        
        # Aggiungi parametri opzionali se supportati
        import inspect
        sig = inspect.signature(generate_rough_heston)
        if 'antithetic' in sig.parameters:
            sim_kwargs['antithetic'] = antithetic
        
        # Filtra kwargs extra supportati
        for key, value in kwargs.items():
            if key in sig.parameters and key not in sim_kwargs:
                sim_kwargs[key] = value
        
        # Simula
        try:
            spot_variance_tuple = generate_rough_heston(**sim_kwargs)
            
            # Crea output
            auxiliary = {
                'H': torch.tensor(H),
                'roughness': torch.tensor(H),  # Alias per consistenza
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
        Gestione specifica per Rough Heston.
        Il modello può toccare zero sia per la roughness che per la natura di Heston.
        
        Per Rough Heston con H molto piccolo, potrebbe essere necessario
        gestire threshold adattive.
        """
        # Possibile ottimizzazione: threshold adattiva basata su H
        if hasattr(self, '_last_H') and self._last_H < 0.1:
            threshold = max(threshold, 1e-8)
        
        # Usa l'implementazione di default che è già ottimizzata
        return super().handle_absorption(paths, dt, threshold)


# Registra il processo nel factory
ProcessFactory.register('rough_heston', RoughHestonProcess)
ProcessFactory.register('roughheston', RoughHestonProcess)  # Alias
ProcessFactory.register('rough-heston', RoughHestonProcess)  # Alias