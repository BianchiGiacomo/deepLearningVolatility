"""
Unified interface for stochastic processes.
Provides a common abstraction for different pricing models.
"""
from abc import ABC, abstractmethod
from typing import Protocol, NamedTuple, Optional, Dict, Tuple, List, Any, Iterable
import torch
from torch import Tensor


class SimulationOutput(NamedTuple):
    """Standardized output of simulations."""
    spot: Tensor
    variance: Optional[Tensor] = None
    auxiliary: Optional[Dict[str, Tensor]] = None  # For extra data (e.g. jump counts)
    
    def __repr__(self) -> str:
        base = f"SimulationOutput(spot={self.spot.shape}"
        if self.variance is not None:
            base += f", variance={self.variance.shape}"
        if self.auxiliary:
            base += f", auxiliary={list(self.auxiliary.keys())}"
        return base + ")"


class ParameterInfo(NamedTuple):
    """Model parameter information."""
    names: List[str]
    bounds: List[Tuple[float, float]]
    defaults: List[float]
    descriptions: Optional[List[str]] = None


class StochasticProcess(Protocol):
    """Protocol for stochastic processes usable in the framework."""

    @property
    def num_params(self) -> int:
        """Number of model parameters."""
        ...

    @property
    def param_info(self) -> ParameterInfo:
        """Detailed parameter information."""
        ...

    @property
    def supports_absorption(self) -> bool:
        """Indicates if the process can touch zero."""
        return False

    @property
    def requires_variance_state(self) -> bool:
        """Indicates if the process requires an initial variance state."""
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
        """
        Simulate paths of the process.

        Args:
            theta: Model parameters
            n_paths: Number of paths to simulate
            n_steps: Number of time steps
            dt: Time interval
            init_state: Initial state (interpretation depends on the model)
            device: Torch device
            dtype: Torch data type
            antithetic: Whether to use antithetic variables
            **kwargs: Additional model-specific parameters

        Returns:
            SimulationOutput with the simulated paths
        """
        ...

    def validate_theta(self, theta: Tensor) -> Tuple[bool, Optional[str]]:
        """
        Validate the model parameters.

        Returns:
            (is_valid, error_message)
        """
        if theta.shape[-1] != self.num_params:
            return False, f"Expected {self.num_params} parameters, got {theta.shape[-1]}"

        # Bounds validation
        param_info = self.param_info
        theta_np = theta.cpu().numpy()

        for i, (name, (low, high)) in enumerate(zip(param_info.names, param_info.bounds)):
            if not (low <= theta_np.flat[i] <= high):
                return False, f"Parameter '{name}' = {theta_np.flat[i]} outside bounds [{low}, {high}]"

        return True, None

    def handle_absorption(self, 
                         paths: Tensor, 
                         dt: float,
                         threshold: float = 1e-10) -> Tuple[Tensor, Tensor]:
        """
        Handling of paths that touch zero (if supported).

        Returns:
            (absorption_times, absorbed_mask)
        """
        if not self.supports_absorption:
            # No absorption: all paths survive
            n_paths = paths.shape[0]
            return torch.full((n_paths,), float('inf')), torch.zeros(n_paths, dtype=torch.bool)

        # Default implementation (can be overridden)
        from deepLearningVolatility.nn.pricer import ZeroAbsorptionHandler
        return ZeroAbsorptionHandler.find_absorption_times(paths, dt)


class BaseStochasticProcess(ABC):
    """Base class with common implementations."""

    def __init__(self, spot: float = 1.0):
        self.spot = spot

    @property
    @abstractmethod
    def num_params(self) -> int:
        pass

    @property
    @abstractmethod
    def param_info(self) -> ParameterInfo:
        pass

    def validate_theta(self, theta: Tensor) -> Tuple[bool, Optional[str]]:
        """
        Validate the model parameters.

        Returns:
            (is_valid, error_message)
        """
        if theta.shape[-1] != self.num_params:
            return False, f"Expected {self.num_params} parameters, got {theta.shape[-1]}"

        # Bounds validation
        param_info = self.param_info
        theta_np = theta.cpu().numpy()

        for i, (name, (low, high)) in enumerate(zip(param_info.names, param_info.bounds)):
            value = theta_np.flat[i]
            if not (low <= value <= high):
                return False, f"Parameter '{name}' = {value} outside bounds [{low}, {high}]"

        return True, None

    def get_default_init_state(self) -> Tuple[float, ...]:
        """Default initial state."""
        if self.requires_variance_state:
            # For models with stochastic variance, use xi0 as initial variance
            return (self.spot, self.param_info.defaults[3])  # xi0 is the 4th parameter for RB
        return (self.spot,)

    def prepare_init_state(self, init_state: Optional[Tuple[float, ...]] = None) -> Tuple[float, ...]:
        """Prepare the initial state with validation."""
        if init_state is None:
            return self.get_default_init_state()

        # Basic validation
        expected_len = 2 if self.requires_variance_state else 1
        if len(init_state) != expected_len:
            raise ValueError(f"Expected init_state of length {expected_len}, got {len(init_state)}")

        return init_state

    def handle_absorption(self, 
                         paths: Tensor, 
                         dt: float,
                         threshold: float = 1e-10) -> Tuple[Tensor, Tensor]:
        """
        Handling of paths that touch zero (if supported).

        Returns:
            (absorption_times, absorbed_mask)
        """
        if not self.supports_absorption:
            # No absorption: all paths survive
            n_paths = paths.shape[0]
            return torch.full((n_paths,), float('inf')), torch.zeros(n_paths, dtype=torch.bool)

        # Default implementation (can be overridden)
        # Copy logic from ZeroAbsorptionHandler to avoid circular imports
        zero_mask = paths <= threshold
        cumsum = zero_mask.cumsum(dim=1)
        first_zero_mask = (cumsum == 1) & zero_mask

        padded_mask = torch.cat([first_zero_mask, 
                                torch.ones(paths.shape[0], 1, dtype=torch.bool, device=paths.device)], 
                               dim=1)

        absorption_indices = padded_mask.to(torch.float32).argmax(dim=1)
        n_steps = paths.shape[1]
        absorbed_mask = absorption_indices < n_steps
        absorption_times = absorption_indices.float() * dt

        return absorption_times, absorbed_mask


# Factory pattern for dynamic registration
class ProcessFactory:
    """Factory for creating stochastic processes with alias support
    and reverse map (class -> canonical key).
    """
    _registry: Dict[str, type] = {}    # canonical key or alias -> class
    _reverse:  Dict[type, str] = {}    # class -> canonical key

    @classmethod
    def register(cls, name: str, process_class: type, aliases: Iterable[str] = ()) -> None:
        """Register a new process.
        - name: canonical key (e.g. 'rough_heston')
        - aliases: alternative keys that point to the same class
        """
        key = name.strip().lower()
        cls._registry[key] = process_class
        cls._reverse[process_class] = key
        for a in aliases:
            aka = a.strip().lower()
            cls._registry[aka] = process_class  # all aliases resolve to the same class

    @classmethod
    def create(cls, name: str, **kwargs):
        """Create an instance of the requested process (canonical key or alias)."""
        k = name.strip().lower()
        process_class = cls._registry.get(k)
        if process_class is None:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown process '{name}'. Available: {available}")
        inst = process_class(**kwargs)
        # store the canonical key on the instance (useful for config saving)
        setattr(inst, "_factory_key", cls._reverse.get(process_class, k))
        return inst

    @classmethod
    def key_for_class(cls, process_class: type) -> Optional[str]:
        """Returns the canonical key for a registered class."""
        return cls._reverse.get(process_class)

    @classmethod
    def key_for_instance(cls, proc) -> Optional[str]:
        """Returns the canonical key for an instance (if known)."""
        # prefer the key stored in create(...)
        k = getattr(proc, "_factory_key", None)
        if k is not None:
            return k
        return cls._reverse.get(type(proc))

    @classmethod
    def list_available(cls) -> List[str]:
        """List of registered keys (including any aliases)."""
        return list(cls._registry.keys())