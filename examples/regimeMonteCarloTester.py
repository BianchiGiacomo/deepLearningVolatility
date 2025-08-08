import torch
import numpy as np
import matplotlib.pyplot as plt
from deepLearningVolatility.stochastic import generate_rough_bergomi
from deepLearningVolatility.nn.modules import BlackScholes
from deepLearningVolatility.instruments import EuropeanOption, BrownianStock
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class MCConfig:
    """Configurazione per Monte Carlo"""
    n_paths: int
    dt: float
    regime: str
    
    @property
    def steps_per_year(self):
        return int(1/self.dt)

class RegimeMonteCarloTester:
    """Tester completo per tutti i regimi di maturità"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Configurazioni di default per regime
        self.default_configs = {
            'short': [
                MCConfig(5000, 1/365, 'short'),      # Daily
                MCConfig(10000, 1/730, 'short'),     # 12h
                MCConfig(30000, 1/1460, 'short'),    # 6h
                MCConfig(50000, 1/2920, 'short'),    # 3h
            ],
            'mid': [
                MCConfig(5000, 1/365, 'mid'),        # Daily
                MCConfig(10000, 1/365, 'mid'),       # Daily
                MCConfig(20000, 1/730, 'mid'),       # 12h
                MCConfig(30000, 1/730, 'mid'),       # 12h
            ],
            'long': [
                MCConfig(5000, 1/365, 'long'),       # Daily
                MCConfig(10000, 1/365, 'long'),      # Daily
                MCConfig(20000, 1/365, 'long'),      # Daily
                MCConfig(5000, 1/252, 'long'),       # Business days
            ]
        }
        
    def analyze_theta_robustness(self, 
                               theta_list: List[List[float]],
                               maturities_dict: Dict[str, torch.Tensor],
                               logK_dict: Dict[str, torch.Tensor]):
        """
        Analizza la robustezza di diversi theta sui vari regimi
        """
        results = {}
        
        for theta_idx, theta in enumerate(theta_list):
            print(f"\n{'='*80}")
            print(f"Testing theta {theta_idx}: H={theta[0]:.3f}, eta={theta[1]:.3f}, "
                  f"rho={theta[2]:.3f}, xi0={theta[3]:.3f}")
            print(f"{'='*80}")
            
            results[f'theta_{theta_idx}'] = {}
            
            for regime in ['short', 'mid', 'long']:
                maturities = maturities_dict[regime]
                logK = logK_dict[regime]
                
                print(f"\n{regime.upper()} TERM REGIME")
                print(f"Maturities: {maturities.tolist()}")
                print(f"LogK range: [{logK.min():.3f}, {logK.max():.3f}]")
                
                # Test con configurazione ottimale per il regime
                config = self._get_optimal_config(regime, maturities[0].item())
                
                surface, stats = self._compute_surface_with_stats(
                    theta, maturities, logK, config
                )
                
                results[f'theta_{theta_idx}'][regime] = {
                    'surface': surface,
                    'stats': stats,
                    'config': config
                }
                
                # Stampa statistiche
                print(f"Config: {config.n_paths} paths, dt={config.dt:.5f} "
                      f"({config.steps_per_year} steps/year)")
                print(f"Time: {stats['time']:.2f}s")
                print(f"IV range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                print(f"IV mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
                
                if stats['zeros'] > 0:
                    print(f"⚠️ WARNING: {stats['zeros']} zero values!")
                if stats['extreme_values'] > 0:
                    print(f"⚠️ WARNING: {stats['extreme_values']} extreme values!")
        
        return results
    
    def _get_optimal_config(self, regime: str, min_maturity: float) -> MCConfig:
        """Ottiene configurazione ottimale basata sul regime e maturity minima"""
        if regime == 'short':
            if min_maturity < 14/365:  # < 2 settimane
                return MCConfig(50000, min(min_maturity/50, 1/2920), regime)
            elif min_maturity < 30/365:  # < 1 mese
                return MCConfig(30000, min(min_maturity/30, 1/1460), regime)
            else:
                return MCConfig(20000, 1/730, regime)
        elif regime == 'mid':
            return MCConfig(20000, 1/365, regime)
        else:  # long
            return MCConfig(10000, 1/365, regime)
    
    def _compute_surface_with_stats(self, theta, maturities, logK, config):
        """Calcola superficie con statistiche dettagliate"""
        start_time = time.time()
        
        H, eta, rho, xi0 = theta
        iv_surface = torch.zeros(len(maturities), len(logK), device=self.device)
        
        for i, T in enumerate(maturities):
            T_val = T.item()
            n_steps = max(2, int(np.ceil(T_val / config.dt)))
            
            try:
                # Genera paths con chunking per memoria
                chunk_size = min(10000, config.n_paths)
                all_ST = []
                
                for chunk_start in range(0, config.n_paths, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, config.n_paths)
                    chunk_paths = chunk_end - chunk_start
                    
                    S, _ = generate_rough_bergomi(
                        chunk_paths, n_steps,
                        init_state=(1.0, xi0),
                        alpha=H - 0.5,
                        rho=rho,
                        eta=eta,
                        xi=xi0,
                        dt=config.dt,
                        device=self.device,
                        antithetic= True
                    )
                    all_ST.append(S[:, -1])
                
                ST = torch.cat(all_ST)
                
                # Calcola IV per ogni strike
                for j, k in enumerate(logK):
                    K = np.exp(k.item())
                    payoff = (ST - K).clamp(min=0.0)
                    price = payoff.mean().item()
                    
                    if price > 1e-10:
                        try:
                            bs = BlackScholes(
                                EuropeanOption(
                                    BrownianStock(),
                                    strike=K,
                                    maturity=T_val
                                )
                            )
                            iv = bs.implied_volatility(
                                log_moneyness=-k.item(),
                                time_to_maturity=T_val,
                                price=price
                            )
                            iv_surface[i, j] = max(0.01, min(iv, 2.0))  # Clamp
                        except:
                            iv_surface[i, j] = np.sqrt(xi0)
                    else:
                        iv_surface[i, j] = 0.0  # Mark as problematic
                        
            except Exception as e:
                print(f"  Error for T={T_val}: {e}")
                iv_surface[i, :] = np.sqrt(xi0)
        
        elapsed = time.time() - start_time
        
        # Calcola statistiche
        valid_mask = iv_surface > 0.01
        valid_ivs = iv_surface[valid_mask]
        
        stats = {
            'time': elapsed,
            'mean': valid_ivs.mean().item() if valid_mask.any() else 0,
            'std': valid_ivs.std().item() if valid_mask.any() else 0,
            'min': valid_ivs.min().item() if valid_mask.any() else 0,
            'max': valid_ivs.max().item() if valid_mask.any() else 0,
            'zeros': (iv_surface == 0).sum().item(),
            'extreme_values': ((iv_surface < 0.01) | (iv_surface > 1.0)).sum().item()
        }
        
        return iv_surface, stats
    
    def test_all_regimes(self, theta, custom_configs=None):
        """Test completo per tutti i regimi con visualizzazione"""
        
        # Definisci maturità per regime
        maturities = {
            'short': torch.tensor([7/365, 14/365, 30/365]),
            'mid': torch.tensor([60/365, 90/365, 120/365, 180/365, 270/365]),
            'long': torch.tensor([1.0, 2.0, 3.0])
        }
        
        # Definisci log-moneyness per regime
        logK = {
            'short': torch.linspace(-0.1, 0.1, 11),  # Range più stretto
            'mid': torch.linspace(-0.3, 0.3, 13),
            'long': torch.linspace(-0.5, 0.5, 15)
        }
        
        configs = custom_configs or self.default_configs
        results = {}
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        for regime_idx, regime in enumerate(['short', 'mid', 'long']):
            print(f"\n{'='*60}")
            print(f"{regime.upper()} TERM TESTING")
            print(f"{'='*60}")
            
            results[regime] = {}
            
            # Test diverse configurazioni
            for config_idx, config in enumerate(configs[regime][:3]):
                print(f"\nTesting: {config.n_paths} paths, "
                      f"dt={config.dt:.5f} ({config.steps_per_year} steps/year)")
                
                surface, stats = self._compute_surface_with_stats(
                    theta, maturities[regime], logK[regime], config
                )
                
                results[regime][f'config_{config_idx}'] = {
                    'surface': surface,
                    'stats': stats,
                    'config': config
                }
                
                # Visualizza
                ax = axes[regime_idx, config_idx]
                im = ax.imshow(surface.cpu(), aspect='auto', origin='lower',
                             cmap='viridis', vmin=0, vmax=0.5)
                ax.set_title(f'{regime}: {config.n_paths}p, {config.steps_per_year}s/y')
                ax.set_xlabel('Strike idx')
                ax.set_ylabel('Maturity idx')
                
                # Aggiungi colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle(f'Theta: H={theta[0]:.3f}, eta={theta[1]:.3f}, '
                     f'rho={theta[2]:.3f}, xi0={theta[3]:.3f}')
        plt.tight_layout()
        plt.show()
        
        return results
    
    def plot_comparison(self, results_dict, regime='short'):
        """Confronta risultati per diversi theta"""
        n_thetas = len(results_dict)
        n_maturities = len(results_dict[list(results_dict.keys())[0]][regime]['surface'])
        
        fig, axes = plt.subplots(n_thetas, 3, figsize=(15, 4*n_thetas))
        if n_thetas == 1:
            axes = axes.reshape(1, -1)
        
        for theta_idx, (theta_key, theta_results) in enumerate(results_dict.items()):
            surface = theta_results[regime]['surface'].cpu()
            stats = theta_results[regime]['stats']
            
            # Surface plot
            ax1 = axes[theta_idx, 0]
            im = ax1.imshow(surface, aspect='auto', origin='lower', cmap='viridis')
            ax1.set_title(f'{theta_key} - Surface')
            ax1.set_xlabel('Strike')
            ax1.set_ylabel('Maturity')
            plt.colorbar(im, ax=ax1)
            
            # Smile for shortest maturity
            ax2 = axes[theta_idx, 1]
            smile = surface[0]
            ax2.plot(smile, 'o-')
            ax2.set_title('Smile (shortest maturity)')
            ax2.set_xlabel('Strike idx')
            ax2.set_ylabel('IV')
            ax2.grid(True, alpha=0.3)
            
            # Stats
            ax3 = axes[theta_idx, 2]
            ax3.text(0.1, 0.8, f"Mean: {stats['mean']:.4f}", transform=ax3.transAxes)
            ax3.text(0.1, 0.6, f"Std: {stats['std']:.4f}", transform=ax3.transAxes)
            ax3.text(0.1, 0.4, f"Range: [{stats['min']:.4f}, {stats['max']:.4f}]", 
                    transform=ax3.transAxes)
            ax3.text(0.1, 0.2, f"Zeros: {stats['zeros']}", transform=ax3.transAxes)
            ax3.text(0.1, 0.0, f"Time: {stats['time']:.2f}s", transform=ax3.transAxes)
            ax3.set_title('Statistics')
            ax3.axis('off')
        
        plt.suptitle(f'{regime.upper()} TERM Comparison')
        plt.tight_layout()
        plt.show()
    
    def recommend_adaptive_config(self, maturity, regime, theta):
        """Raccomanda configurazione adattiva basata su maturity e parametri"""
        H, eta, rho, xi0 = theta
        
        # Fattori di aggiustamento basati sui parametri
        h_factor = 1.0 + max(0, 0.2 - H) * 2  # Più paths per H piccolo
        rho_factor = 1.0 + abs(rho) * 0.5     # Più paths per alta correlazione
        xi_factor = 1.0 + max(0, xi0 - 0.1) * 2  # Più paths per alta vol
        
        total_factor = h_factor * rho_factor * xi_factor
        
        if regime == 'short':
            if maturity < 14/365:
                base_paths = 30000
                base_dt = maturity / 50
            elif maturity < 30/365:
                base_paths = 20000
                base_dt = maturity / 30
            else:
                base_paths = 15000
                base_dt = 1/730
        elif regime == 'mid':
            base_paths = 10000
            base_dt = 1/365
        else:  # long
            base_paths = 5000
            base_dt = 1/252
        
        recommended_paths = int(base_paths * total_factor)
        recommended_dt = min(base_dt, 1/365)
        
        return MCConfig(recommended_paths, recommended_dt, regime)


# Funzioni di test specifiche
def test_problematic_thetas():
    """Test theta problematici vs buoni"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tester = RegimeMonteCarloTester(device)
    
    # Theta da testare
    # thetas = [
    #     [0.05, 1.8, -0.5, 0.08],  # Problematico per strikes estremi
    #     [0.08628702, 3.19980052, -0.48315778, 0.19590783],  # Buono
    #     [0.1, 2.5, -0.9, 0.04],   # Low H, high correlation
    #     [0.4, 0.5, -0.2, 0.15],   # High H, low correlation
    # ]
    thetas = [
        [0.3, 1., -0.5, 0.01],
        [0.3, 1., -0.5, 0.05],
        [0.3, 1., -0.5, 0.1],
        [0.3, 1., -0.5, 0.16],
    ]
    # Definisci griglie appropriate per ogni regime
    maturities_dict = {
        'short': torch.tensor([7/365, 14/365, 30/365]),
        'mid': torch.tensor([ 60/365, 90/365, 120/365, 180/365, 270/365]),
        'long': torch.tensor([1.0, 2.0, 3.0])
    }
    
    # Usa range di strikes adattivi
    logK_dict = {}
    for i, theta in enumerate(thetas):
        H, eta, rho, xi0 = theta
        
        # Adatta range basato su parametri
        if H < 0.15 or abs(rho) > 0.8:
            # Parametri estremi: range più stretto
            logK_dict['short'] = torch.linspace(-0.1, 0.1, 11)
            logK_dict['mid'] = torch.linspace(-0.2, 0.2, 13)
            logK_dict['long'] = torch.linspace(-0.3, 0.3, 15)
        else:
            # Parametri normali
            logK_dict['short'] = torch.linspace(-0.2, 0.2, 15)
            logK_dict['mid'] = torch.linspace(-0.4, 0.4, 17)
            logK_dict['long'] = torch.linspace(-0.6, 0.6, 19)
    
    results = tester.analyze_theta_robustness(thetas, maturities_dict, logK_dict)
    
    # Confronta risultati per ogni regime
    for regime in ['short', 'mid', 'long']:
        tester.plot_comparison(results, regime)
    
    return results


def get_optimal_mc_params(theta, regime='short'):
    """
    Funzione helper per ottenere parametri MC ottimali
    
    Returns:
        dict: {'n_paths': int, 'dt': float, 'adaptive_dt': bool}
    """
    H, eta, rho, xi0 = theta
    
    # Parametri base per regime
    base_params = {
        'short': {'n_paths': 70_000, 'dt': 1/1460, 'adaptive_dt': True},
        'mid': {'n_paths': 70_000, 'dt': 1/730, 'adaptive_dt': True},
        'long': {'n_paths': 70_000, 'dt': 1/365, 'adaptive_dt': False}
    }
    
    params = base_params[regime].copy()
    
    # Aggiustamenti basati su theta
    if H < 0.15:
        params['n_paths'] = int(params['n_paths'] * 1.5)
        params['dt'] = params['dt'] / 2
    
    if abs(rho) > 0.8:
        params['n_paths'] = int(params['n_paths'] * 1.3)
    
    if xi0 > 0.15:
        params['n_paths'] = int(params['n_paths'] * 1.2)
    
    # Limiti massimi
    params['n_paths'] = min(params['n_paths'], 100000)
    params['dt'] = max(params['dt'], 1/5840)  # Min 1.5h
    
    return params


if __name__ == "__main__":
    # Test principale
    print("Testing problematic vs good thetas across all regimes...")
    results = test_problematic_thetas()
    
    # Esempio di uso della funzione helper
    theta_good = [0.08628702, 3.19980052, -0.48315778, 0.19590783]
    for regime in ['short', 'mid', 'long']:
        params = get_optimal_mc_params(theta_good, regime)
        print(f"\nOptimal params for {regime}: {params}")
        
        
        
    