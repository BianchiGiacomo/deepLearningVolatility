"""
Interfaccia unificata per processi stocastici.
Fornisce un'astrazione comune per diversi modelli di pricing.
"""
from abc import ABC, abstractmethod
from typing import Protocol, NamedTuple, Optional, Dict, Tuple, List, Any, Iterable
import torch
from torch import Tensor


class SimulationOutput(NamedTuple):
    """Output standardizzato delle simulazioni."""
    spot: Tensor
    variance: Optional[Tensor] = None
    auxiliary: Optional[Dict[str, Tensor]] = None  # Per dati extra (es. jump counts)
    
    def __repr__(self) -> str:
        base = f"SimulationOutput(spot={self.spot.shape}"
        if self.variance is not None:
            base += f", variance={self.variance.shape}"
        if self.auxiliary:
            base += f", auxiliary={list(self.auxiliary.keys())}"
        return base + ")"


class ParameterInfo(NamedTuple):
    """Informazioni sui parametri del modello."""
    names: List[str]
    bounds: List[Tuple[float, float]]
    defaults: List[float]
    descriptions: Optional[List[str]] = None


class StochasticProcess(Protocol):
    """Protocollo per processi stocastici utilizzabili nel framework."""
    
    @property
    def num_params(self) -> int:
        """Numero di parametri del modello."""
        ...
    
    @property
    def param_info(self) -> ParameterInfo:
        """Informazioni dettagliate sui parametri."""
        ...
    
    @property
    def supports_absorption(self) -> bool:
        """Indica se il processo può toccare zero."""
        return False
    
    @property
    def requires_variance_state(self) -> bool:
        """Indica se il processo richiede uno stato di varianza iniziale."""
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
        Simula paths del processo.
        
        Args:
            theta: Parametri del modello
            n_paths: Numero di paths da simulare
            n_steps: Numero di step temporali
            dt: Intervallo temporale
            init_state: Stato iniziale (interpretazione dipende dal modello)
            device: Device torch
            dtype: Tipo dati torch
            antithetic: Se usare variabili antitetiche
            **kwargs: Parametri aggiuntivi specifici del modello
            
        Returns:
            SimulationOutput con i paths simulati
        """
        ...
    
    def validate_theta(self, theta: Tensor) -> Tuple[bool, Optional[str]]:
        """
        Valida i parametri del modello.
        
        Returns:
            (is_valid, error_message)
        """
        if theta.shape[-1] != self.num_params:
            return False, f"Expected {self.num_params} parameters, got {theta.shape[-1]}"
        
        # Validazione bounds
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
        Gestione paths che toccano zero (se supportato).
        
        Returns:
            (absorption_times, absorbed_mask)
        """
        if not self.supports_absorption:
            # Nessun absorption: tutti i paths sopravvivono
            n_paths = paths.shape[0]
            return torch.full((n_paths,), float('inf')), torch.zeros(n_paths, dtype=torch.bool)
        
        # Implementazione default (può essere sovrascritta)
        from deepLearningVolatility.nn.pricer import ZeroAbsorptionHandler
        return ZeroAbsorptionHandler.find_absorption_times(paths, dt)


class BaseStochasticProcess(ABC):
    """Classe base con implementazioni comuni."""
    
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
        Valida i parametri del modello.
        
        Returns:
            (is_valid, error_message)
        """
        if theta.shape[-1] != self.num_params:
            return False, f"Expected {self.num_params} parameters, got {theta.shape[-1]}"
        
        # Validazione bounds
        param_info = self.param_info
        theta_np = theta.cpu().numpy()
        
        for i, (name, (low, high)) in enumerate(zip(param_info.names, param_info.bounds)):
            value = theta_np.flat[i]
            if not (low <= value <= high):
                return False, f"Parameter '{name}' = {value} outside bounds [{low}, {high}]"
        
        return True, None
    
    def get_default_init_state(self) -> Tuple[float, ...]:
        """Stato iniziale di default."""
        if self.requires_variance_state:
            # Per modelli con varianza stocastica, usa xi0 come varianza iniziale
            return (self.spot, self.param_info.defaults[3])  # xi0 è il 4° parametro per RB
        return (self.spot,)
    
    def prepare_init_state(self, init_state: Optional[Tuple[float, ...]] = None) -> Tuple[float, ...]:
        """Prepara lo stato iniziale con validazione."""
        if init_state is None:
            return self.get_default_init_state()
        
        # Validazione base
        expected_len = 2 if self.requires_variance_state else 1
        if len(init_state) != expected_len:
            raise ValueError(f"Expected init_state of length {expected_len}, got {len(init_state)}")
        
        return init_state
    
    def handle_absorption(self, 
                         paths: Tensor, 
                         dt: float,
                         threshold: float = 1e-10) -> Tuple[Tensor, Tensor]:
        """
        Gestione paths che toccano zero (se supportato).
        
        Returns:
            (absorption_times, absorbed_mask)
        """
        if not self.supports_absorption:
            # Nessun absorption: tutti i paths sopravvivono
            n_paths = paths.shape[0]
            return torch.full((n_paths,), float('inf')), torch.zeros(n_paths, dtype=torch.bool)
        
        # Implementazione default (può essere sovrascritta)
        # Copiamo la logica da ZeroAbsorptionHandler per evitare import circolari
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


# Factory pattern per registrazione dinamica
class ProcessFactory:
    """Factory per la creazione di processi stocastici con supporto alias
    e mappa inversa (classe -> chiave canonica).
    """
    _registry: Dict[str, type] = {}    # chiave canonica o alias -> classe
    _reverse:  Dict[type, str] = {}    # classe -> chiave canonica

    @classmethod
    def register(cls, name: str, process_class: type, aliases: Iterable[str] = ()) -> None:
        """Registra un nuovo processo.
        - name: chiave canonica (es. 'rough_heston')
        - aliases: chiavi alternative che puntano alla stessa classe
        """
        key = name.strip().lower()
        cls._registry[key] = process_class
        cls._reverse[process_class] = key
        for a in aliases:
            aka = a.strip().lower()
            cls._registry[aka] = process_class  # tutti gli alias risolvono alla stessa classe

    @classmethod
    def create(cls, name: str, **kwargs):
        """Crea un'istanza del processo richiesto (chiave canonica o alias)."""
        k = name.strip().lower()
        process_class = cls._registry.get(k)
        if process_class is None:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown process '{name}'. Available: {available}")
        inst = process_class(**kwargs)
        # memorizza la chiave canonica sull'istanza (utile per salvataggio config)
        setattr(inst, "_factory_key", cls._reverse.get(process_class, k))
        return inst

    @classmethod
    def key_for_class(cls, process_class: type) -> Optional[str]:
        """Ritorna la chiave canonica per una classe registrata."""
        return cls._reverse.get(process_class)

    @classmethod
    def key_for_instance(cls, proc) -> Optional[str]:
        """Ritorna la chiave canonica per un'istanza (se nota)."""
        # preferisci la chiave memorizzata in create(...)
        k = getattr(proc, "_factory_key", None)
        if k is not None:
            return k
        return cls._reverse.get(type(proc))

    @classmethod
    def list_available(cls) -> List[str]:
        """Lista delle chiavi registrate (incluse eventuali alias)."""
        return list(cls._registry.keys())