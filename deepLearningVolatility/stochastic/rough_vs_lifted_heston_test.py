"""
Script di test per i modelli di volatilità rough implementati.
Verifica correttezza, performance e proprietà statistiche.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import dei modelli (assumendo che siano in moduli separati)
from deepLearningVolatility.stochastic.rough_heston_commented import generate_rough_heston_hqe
from deepLearningVolatility.stochastic.lifted_heston_commented import generate_lifted_heston
from deepLearningVolatility.stochastic.rough_bergomi import generate_rough_bergomi

# Per questo test, includiamo le funzioni direttamente
# (in produzione, importale dai moduli appropriati)


class ModelTester:
    """Classe per testare i modelli di volatilità rough."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.results = {}
        
    def test_basic_properties(self, model_name: str, generate_fn, params: Dict) -> Dict:
        """Testa le proprietà di base del modello."""
        print(f"\n{'='*50}")
        print(f"Testing {model_name}")
        print(f"{'='*50}")
        
        results = {}
        
        # Test 1: Verifica che il modello giri senza errori
        print("\n1. Testing basic execution...")
        try:
            output = generate_fn(
                n_paths=100,
                n_steps=50,
                device=self.device,
                **params
            )
            print("✓ Model runs without errors")
            results['runs'] = True
        except Exception as e:
            print(f"✗ Model failed with error: {e}")
            results['runs'] = False
            return results
            
        # Test 2: Verifica shape degli output
        print("\n2. Testing output shapes...")
        expected_shape = (100, 50)
        if output.spot.shape == expected_shape and output.variance.shape == expected_shape:
            print(f"✓ Output shapes correct: {output.spot.shape}")
            results['shapes_correct'] = True
        else:
            print(f"✗ Wrong shapes: spot={output.spot.shape}, variance={output.variance.shape}")
            results['shapes_correct'] = False
            
        # Test 3: Verifica positività
        print("\n3. Testing positivity...")
        spot_positive = (output.spot >= 0).all().item()
        var_positive = (output.variance >= 0).all().item()
        
        if spot_positive and var_positive:
            print("✓ All values are non-negative")
            results['positive'] = True
        else:
            print(f"✗ Negative values found: spot_positive={spot_positive}, var_positive={var_positive}")
            results['positive'] = False
            
        # Test 4: Statistiche di base
        print("\n4. Basic statistics:")
        print(f"   Spot - Mean: {output.spot.mean():.4f}, Std: {output.spot.std():.4f}")
        print(f"   Variance - Mean: {output.variance.mean():.4f}, Std: {output.variance.std():.4f}")
        
        results['spot_stats'] = {
            'mean': output.spot.mean().item(),
            'std': output.spot.std().item(),
            'min': output.spot.min().item(),
            'max': output.spot.max().item()
        }
        results['var_stats'] = {
            'mean': output.variance.mean().item(),
            'std': output.variance.std().item(),
            'min': output.variance.min().item(),
            'max': output.variance.max().item()
        }
        
        return results
    
    def test_performance(self, model_name: str, generate_fn, params: Dict, 
                        n_paths_list: List[int] = [100, 1000, 10000]) -> Dict:
        """Testa la performance del modello."""
        print(f"\n5. Performance testing for {model_name}...")
        
        timing_results = {}
        n_steps = 100
        
        for n_paths in n_paths_list:
            # Warm-up
            _ = generate_fn(n_paths=10, n_steps=10, device=self.device, **params)
            
            # Timing
            start_time = time.time()
            _ = generate_fn(n_paths=n_paths, n_steps=n_steps, device=self.device, **params)
            elapsed = time.time() - start_time
            
            timing_results[n_paths] = elapsed
            print(f"   {n_paths:6d} paths: {elapsed:.3f}s ({n_paths/elapsed:.0f} paths/s)")
            
        return timing_results
    
    def test_hurst_index(self, model_name: str, generate_fn, params: Dict, 
                        expected_H: float = 0.1) -> float:
        """Stima l'indice di Hurst dalla varianza simulata."""
        print(f"\n6. Hurst index estimation for {model_name}...")
        
        # Genera un path lungo
        output = generate_fn(
            n_paths=1000,
            n_steps=1000,
            **params
        )
        
        # Calcola la q-variazione per diversi lag
        lags = np.arange(1, 51)
        q_variations = []
        
        for lag in lags:
            var_diff = torch.diff(output.variance[:, ::lag], dim=1)
            q_var = torch.mean(torch.abs(var_diff)).item()
            q_variations.append(q_var)
        
        # Stima H tramite regressione log-log
        log_lags = np.log(lags * params.get('dt', 1/250))
        log_q_var = np.log(q_variations)
        
        # Regressione lineare
        coeffs = np.polyfit(log_lags, log_q_var, 1)
        estimated_H = coeffs[0]
        
        print(f"   Expected H: {expected_H:.3f}")
        print(f"   Estimated H: {estimated_H:.3f}")
        print(f"   Error: {abs(estimated_H - expected_H):.3f}")
        
        return estimated_H
    
    def plot_sample_paths(self, models: Dict) -> None:
        """Plotta sample path per confronto visivo."""
        fig, axes = plt.subplots(len(models), 2, figsize=(12, 4*len(models)))
        if len(models) == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (name, (generate_fn, params)) in enumerate(models.items()):
            # Genera un singolo path
            output = generate_fn(
                n_paths=1,
                n_steps=250,
                device=self.device,
                **params
            )
            
            # Time grid
            dt = params.get('dt', 1/250)
            times = np.arange(250) * dt
            
            # Plot spot
            axes[idx, 0].plot(times, output.spot[0].cpu().numpy())
            axes[idx, 0].set_title(f'{name} - Spot Price')
            axes[idx, 0].set_xlabel('Time (years)')
            axes[idx, 0].set_ylabel('Price')
            axes[idx, 0].grid(True, alpha=0.3)
            
            # Plot variance
            axes[idx, 1].plot(times, output.variance[0].cpu().numpy())
            axes[idx, 1].set_title(f'{name} - Variance')
            axes[idx, 1].set_xlabel('Time (years)')
            axes[idx, 1].set_ylabel('Variance')
            axes[idx, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def compare_distributions(self, models: Dict, T: float = 1.0) -> None:
        """Confronta le distribuzioni finali dei modelli."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        n_paths = 10000
        n_steps = int(T * 250)
        
        for name, (generate_fn, params) in models.items():
            output = generate_fn(
                n_paths=n_paths,
                n_steps=n_steps,
                device=self.device,
                **params
            )
            
            # Distribuzione finale del prezzo
            final_prices = output.spot[:, -1].cpu().numpy()
            ax1.hist(final_prices, bins=50, alpha=0.5, label=name, density=True)
            
            # Distribuzione finale della varianza
            final_variances = output.variance[:, -1].cpu().numpy()
            ax2.hist(final_variances, bins=50, alpha=0.5, label=name, density=True)
        
        ax1.set_title(f'Final Price Distribution (T={T})')
        ax1.set_xlabel('Price')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title(f'Final Variance Distribution (T={T})')
        ax2.set_xlabel('Variance')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def test_correlation(self, model_name: str, generate_fn, params: Dict) -> float:
        """Testa la correlazione tra spot e varianza."""
        print(f"\n7. Testing correlation for {model_name}...")
        
        output = generate_fn(
            n_paths=10000,
            n_steps=100,
            device=self.device,
            **params
        )
        
        # Calcola log-return e cambiamenti di varianza
        log_returns = torch.diff(torch.log(output.spot), dim=1)
        var_changes = torch.diff(output.variance, dim=1)
        
        # Correlazione media nel tempo
        correlations = []
        for t in range(log_returns.shape[1]):
            corr = torch.corrcoef(torch.stack([log_returns[:, t], var_changes[:, t]]))[0, 1]
            if not torch.isnan(corr):
                correlations.append(corr.item())
        
        avg_corr = np.mean(correlations)
        expected_corr = params.get('rho', -0.7)
        
        print(f"   Expected correlation: {expected_corr:.3f}")
        print(f"   Average correlation: {avg_corr:.3f}")
        print(f"   Error: {abs(avg_corr - expected_corr):.3f}")
        
        return avg_corr


def main():
    """Funzione principale per eseguire tutti i test."""
    
    # Inizializza il tester
    tester = ModelTester(device='cpu')  # Usa 'cuda' se disponibile
    
    # Parametri comuni
    common_params = {
        'H': 0.1,        # Hurst parameter
        'nu': 0.3,       # Vol of vol (sigma)
        'rho': -0.7,     # Correlation
        'dt': 1/250,     # Time step
    }
    
    # Definisci i modelli da testare
    models = {
        'Rough Heston (HQE)': (generate_rough_heston_hqe, {
            **common_params,
            'kappa': 0.3,
            'theta': 0.02,
        }),
        'Lifted Heston': (generate_lifted_heston, {
            'n_factors': 3,
            'kappa': 0.3,
            'theta': 0.02,
            'sigma': 0.3,
            'rho': -0.7,
            'dt': 1/250,
        }),
        # Aggiungi altri modelli se necessario
    }
    
    # Esegui test per ogni modello
    all_results = {}
    
    for model_name, (generate_fn, params) in models.items():
        # Test proprietà di base
        basic_results = tester.test_basic_properties(model_name, generate_fn, params)
        
        if basic_results.get('runs', False):
            # Test performance
            perf_results = tester.test_performance(model_name, generate_fn, params)
            
            # Test Hurst index (solo per modelli rough)
            if 'H' in params:
                hurst_estimate = tester.test_hurst_index(model_name, generate_fn, params, params['H'])
            
            # Test correlazione
            corr_estimate = tester.test_correlation(model_name, generate_fn, params)
            
            all_results[model_name] = {
                'basic': basic_results,
                'performance': perf_results,
                'hurst': hurst_estimate if 'H' in params else None,
                'correlation': corr_estimate
            }
    
    # Visualizzazioni
    print("\n\nGenerating visualizations...")
    
    # Sample paths
    tester.plot_sample_paths(models)
    
    # Confronto distribuzioni
    tester.compare_distributions(models)
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        print(f"  - Basic tests: {'PASSED' if results['basic']['runs'] else 'FAILED'}")
        print(f"  - Performance (10k paths): {results['performance'].get(10000, 'N/A'):.3f}s")
        if results.get('hurst') is not None:
            print(f"  - Hurst index error: {abs(results['hurst'] - 0.1):.3f}")
        print(f"  - Correlation error: {abs(results['correlation'] - (-0.7)):.3f}")


if __name__ == "__main__":
    # Nota: per eseguire questo script, assicurati che le funzioni
    # generate_rough_heston_hqe e generate_lifted_heston siano importabili
    
    # Per ora, stampiamo solo la struttura del test
    print("Test script structure:")
    print("1. Basic properties (execution, shapes, positivity)")
    print("2. Performance benchmarking")
    print("3. Hurst index estimation")
    print("4. Correlation testing")
    print("5. Visual comparisons")
    print("\nTo run: import the model functions and execute main()")
    
    main()  # Decommentare quando le funzioni sono disponibili