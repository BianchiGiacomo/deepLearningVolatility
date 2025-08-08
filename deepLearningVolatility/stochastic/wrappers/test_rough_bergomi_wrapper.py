"""
Test per verificare che l'interfaccia funzioni correttamente.
"""
import torch
import pytest
from deepLearningVolatility.stochastic.stochastic_interface import (
    StochasticProcess, BaseStochasticProcess, SimulationOutput, ParameterInfo, ProcessFactory
)
from deepLearningVolatility.stochastic.wrappers.rough_bergomi_wrapper import RoughBergomiProcess


def test_rough_bergomi_wrapper():
    """Test del wrapper Rough Bergomi."""
    # Crea processo
    process = RoughBergomiProcess(spot=100.0)
    
    # Test proprietà
    assert process.num_params == 4
    assert process.supports_absorption == True
    assert process.requires_variance_state == True
    
    # Test param info
    param_info = process.param_info
    assert len(param_info.names) == 4
    assert param_info.names == ['H', 'eta', 'rho', 'xi0']
    
    # Test simulazione
    theta = torch.tensor([0.1, 1.5, -0.7, 0.04])
    result = process.simulate(
        theta=theta,
        n_paths=1000,
        n_steps=100,
        dt=1/252,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Verifica output
    assert result.spot.shape == (1000, 100)
    assert result.variance is not None
    assert result.variance.shape == (1000, 100)
    
    # Test validazione
    is_valid, _ = process.validate_theta(theta)
    assert is_valid
    
    # Test parametri invalidi
    bad_theta = torch.tensor([0.6, 1.5, -0.7, 0.04])  # H > 0.5
    is_valid, error = process.validate_theta(bad_theta)
    assert not is_valid
    assert 'H' in error


def test_process_factory():
    """Test del factory pattern."""
    # Registrazione già fatta nell'import
    
    # Crea tramite factory
    process = ProcessFactory.create('rough_bergomi', spot=100.0)
    assert isinstance(process, RoughBergomiProcess)
    
    # Test alias
    process2 = ProcessFactory.create('roughbergomi', spot=100.0)
    assert isinstance(process2, RoughBergomiProcess)
    
    # Lista processi disponibili
    available = ProcessFactory.list_available()
    assert 'rough_bergomi' in available
    
    # Test processo non esistente
    with pytest.raises(ValueError):
        ProcessFactory.create('non_existent_process')


if __name__ == '__main__':
    test_rough_bergomi_wrapper()
    test_process_factory()
    print("✅ All tests passed!")