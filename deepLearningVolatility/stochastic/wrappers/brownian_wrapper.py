import torch
from torch import Tensor
from typing import Optional, Tuple

from deepLearningVolatility.stochastic.stochastic_interface import (
    BaseStochasticProcess, SimulationOutput, ParameterInfo, ProcessFactory
)
from deepLearningVolatility.stochastic.brownian import generate_brownian

class BrownianProcess(BaseStochasticProcess):
    """
    Arithmetic Brownian motion:
        dS_t = mu * dt + sigma * dW_t
    Parameter vector theta = (mu, sigma).
    """

    def __init__(self, spot: float = 0.0):
        # For arithmetic Brownian, "spot" is the initial level S0 of the process (default 0)
        super().__init__(spot)

    @property
    def num_params(self) -> int:
        return 2

    @property
    def param_info(self) -> ParameterInfo:
        return ParameterInfo(
            names=["mu", "sigma"],
            bounds=[(-0.5, 0.5), (1e-3, 1.0)], 
            defaults=[0.0, 0.2],
            descriptions=["Drift", "Volatility"]
        )

    @property
    def supports_absorption(self) -> bool:
        # Arithmetic Brownian freely crosses zero; we do not treat zero as absorbing
        return False

    @property
    def requires_variance_state(self) -> bool:
        return False

    def get_default_init_state(self) -> Tuple[float, ...]:
        # Just the initial level S0
        return (self.spot,)

    def simulate(
        self,
        theta: Tensor,
        n_paths: int,
        n_steps: int,
        dt: float,
        init_state: Optional[Tuple[float, ...]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        antithetic: bool = False,
        **kwargs
    ) -> SimulationOutput:
        # 1) validate parameters
        is_valid, err = self.validate_theta(theta)
        if not is_valid:
            raise ValueError(f"Invalid parameters: {err}")

        mu, sigma = theta.tolist()

        # 2) initial state
        init_state = self.prepare_init_state(init_state)  # -> (S0,)

        # 3) simulate (generator already supports antithetic directly)
        paths = generate_brownian(
            n_paths=n_paths,
            n_steps=n_steps,
            init_state=init_state,
            sigma=sigma,
            mu=mu,
            dt=dt,
            dtype=dtype,
            device=device,
            antithetic=antithetic
        )

        # 4) standardize output
        return SimulationOutput(spot=paths, variance=None)


# Register keys + aliases
ProcessFactory.register(
    "brownian", BrownianProcess,
    aliases=["arithmetic_brownian", "abw", "brownian_motion"]
)
