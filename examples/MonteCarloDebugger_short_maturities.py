import torch
import numpy as np
import matplotlib.pyplot as plt
from deepLearningVolatility.stochastic import generate_rough_bergomi
from deepLearningVolatility.nn.modules import BlackScholes
from deepLearningVolatility.instruments import EuropeanOption, BrownianStock
import time

class MonteCarloDebugger:
    """Tool per debuggare Monte Carlo su short maturities"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device) if isinstance(device, str) else device
        
    def test_mc_parameters(self, 
                          theta,
                          maturities,
                          logK,
                          n_paths_list=[1_000, 10_000, 50_000, 100_000, 200_000],
                          dt_list=[1/365, 1/720, 1/1440, 1/2880],  # daily, 12h, 6h, 3h
                          use_antithetic=True,
                          control_variate=True,
                          spot=1.0):
        """
        Testa diverse combinazioni di parametri MC
        
        Args:
            theta: [H, eta, rho, xi0]
            maturities: array di maturità da testare
            logK: array di log-moneyness
            n_paths_list: lista di numero paths da testare
            dt_list: lista di time steps da testare
        """
        H, eta, rho, xi0 = theta
        
        results = {}
        
        print(f"Testing MC parameters for theta: H={H:.3f}, eta={eta:.3f}, rho={rho:.3f}, xi0={xi0:.3f}")
        print(f"Maturities: {maturities}")
        print(f"Log-moneyness range: [{logK.min():.3f}, {logK.max():.3f}]")
        print("="*80)
        
        for dt in dt_list:
            results[f'dt_{dt}'] = {}
            
            for n_paths in n_paths_list:
                print(f"\nTesting dt={dt:.6f} ({1/dt:.0f} steps/year), n_paths={n_paths}")
                
                start_time = time.time()
                
                # Calcola IV surface
                iv_surface = self._compute_iv_surface(
                    theta, maturities, logK, 
                    n_paths=n_paths, 
                    dt=dt,
                    use_antithetic=use_antithetic,
                    control_variate=control_variate,
                    spot=spot
                )
                
                elapsed = time.time() - start_time
                
                # Analizza risultati
                stats = {
                    'surface': iv_surface,
                    'time': elapsed,
                    'mean': iv_surface.mean().item(),
                    'std': iv_surface.std().item(),
                    'min': iv_surface.min().item(),
                    'max': iv_surface.max().item(),
                    'nan_count': torch.isnan(iv_surface).sum().item(),
                    'inf_count': torch.isinf(iv_surface).sum().item()
                }
                
                results[f'dt_{dt}'][f'paths_{n_paths}'] = stats
                
                print(f"  Time: {elapsed:.2f}s")
                print(f"  IV range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                print(f"  IV mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
                if stats['nan_count'] > 0 or stats['inf_count'] > 0:
                    print(f"  ⚠️ WARNING: {stats['nan_count']} NaN, {stats['inf_count']} Inf values!")
        
        return results
    
    def _compute_iv_surface(self, theta, maturities, logK, n_paths, dt, 
                           use_antithetic, control_variate, spot):
        """Calcola superficie IV con parametri specificati"""
        H, eta, rho, xi0 = theta
        
        iv_surface = torch.zeros(len(maturities), len(logK), device=self.device)
        
        for i, T in enumerate(maturities):
            T_val = T.item() if torch.is_tensor(T) else T
            n_steps = max(2, int(np.ceil(T_val / dt)))
            
            # Genera paths
            try:
                S, _ = generate_rough_bergomi(
                    n_paths, n_steps,
                    init_state=(spot, xi0),
                    alpha=H - 0.5,
                    rho=rho,
                    eta=eta,
                    xi=xi0,
                    dt=dt,
                    device=self.device,
                    antithetic = use_antithetic
                )
                
                ST = S[:, -1]
                
                # Calcola IV per ogni strike
                for j, k in enumerate(logK):
                    K = spot * np.exp(k.item() if torch.is_tensor(k) else k)
                    
                    # Payoff
                    payoff = (ST - K).clamp(min=0.0)
                    
                    if control_variate:
                        # Control variate usando delta S
                        dS = ST - spot
                        payoff_mean = payoff.mean()
                        dS_mean = dS.mean()
                        cov = ((payoff - payoff_mean) * (dS - dS_mean)).mean()
                        var_dS = ((dS - dS_mean) ** 2).mean()
                        
                        if var_dS > 1e-10:
                            beta = cov / var_dS
                            price = (payoff_mean - beta * dS_mean).item()
                        else:
                            price = payoff_mean.item()
                    else:
                        price = payoff.mean().item()
                    
                    # Discount
                    price = price * np.exp(-0.0 * T_val)  # r=0
                    
                    # Calcola IV
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
                                log_moneyness=-k.item() if torch.is_tensor(k) else -k,
                                time_to_maturity=T_val,
                                price=price
                            )
                            iv_surface[i, j] = iv
                        except:
                            iv_surface[i, j] = np.sqrt(xi0)
                    else:
                        iv_surface[i, j] = np.sqrt(xi0)
                        
            except Exception as e:
                print(f"  Error for T={T_val}: {e}")
                iv_surface[i, :] = np.sqrt(xi0)
        
        return iv_surface
    
    def visualize_results(self, results, maturities, logK, save_path=None):
        """Visualizza i risultati del test"""
        dt_values = sorted([float(k.split('_')[1]) for k in results.keys()])
        n_paths_values = sorted([int(k.split('_')[1]) for k in results[list(results.keys())[0]].keys()])
        
        # Plot 1: Superfici per diverse configurazioni
        fig1, axes = plt.subplots(len(dt_values), len(n_paths_values), 
                                 figsize=(4*len(n_paths_values), 3*len(dt_values)))
        
        if len(dt_values) == 1:
            axes = axes.reshape(1, -1)
        if len(n_paths_values) == 1:
            axes = axes.reshape(-1, 1)
        
        vmin = float('inf')
        vmax = float('-inf')
        
        # Prima trova il range globale
        for dt_key in results:
            for paths_key in results[dt_key]:
                surface = results[dt_key][paths_key]['surface']
                vmin = min(vmin, surface.min().item())
                vmax = max(vmax, surface.max().item())
        
        # Poi plotta
        for i, dt in enumerate(dt_values):
            for j, n_paths in enumerate(n_paths_values):
                ax = axes[i, j] if len(dt_values) > 1 and len(n_paths_values) > 1 else axes[max(i, j)]
                
                surface = results[f'dt_{dt}'][f'paths_{n_paths}']['surface'].cpu().numpy()
                
                im = ax.imshow(surface, aspect='auto', origin='lower', 
                             vmin=vmin, vmax=vmax, cmap='viridis')
                
                ax.set_title(f'dt={dt:.5f}, paths={n_paths}', fontsize=10)
                
                if j == 0:
                    ax.set_ylabel('Maturity idx')
                if i == len(dt_values) - 1:
                    ax.set_xlabel('Strike idx')
                
                # Aggiungi colorbar per ogni riga
                if j == len(n_paths_values) - 1:
                    plt.colorbar(im, ax=ax, label='IV')
        
        plt.suptitle('IV Surfaces for Different MC Parameters', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_surfaces.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        # Plot 2: Convergenza per maturity più corta
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        shortest_T_idx = 0  # Prima maturity
        colors = plt.cm.rainbow(np.linspace(0, 1, len(dt_values)))
        
        # Convergenza vs n_paths per diversi dt
        for i, (dt, color) in enumerate(zip(dt_values, colors)):
            means = []
            stds = []
            times = []
            
            for n_paths in n_paths_values:
                stats = results[f'dt_{dt}'][f'paths_{n_paths}']
                surface = stats['surface']
                means.append(surface[shortest_T_idx].mean().item())
                stds.append(surface[shortest_T_idx].std().item())
                times.append(stats['time'])
            
            ax1.plot(n_paths_values, means, 'o-', color=color, 
                    label=f'dt={dt:.5f} ({1/dt:.0f} steps/y)')
            ax1.fill_between(n_paths_values, 
                           np.array(means) - np.array(stds),
                           np.array(means) + np.array(stds),
                           alpha=0.2, color=color)
        
        ax1.set_xlabel('Number of Paths')
        ax1.set_ylabel('Mean IV (shortest maturity)')
        ax1.set_title(f'Convergence for T={maturities[shortest_T_idx]:.4f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Tempo di calcolo
        for i, (dt, color) in enumerate(zip(dt_values, colors)):
            times = []
            for n_paths in n_paths_values:
                times.append(results[f'dt_{dt}'][f'paths_{n_paths}']['time'])
            
            ax2.plot(n_paths_values, times, 'o-', color=color,
                    label=f'dt={dt:.5f}')
        
        ax2.set_xlabel('Number of Paths')
        ax2.set_ylabel('Computation Time (s)')
        ax2.set_title('Computation Time vs Parameters')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_convergence.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        # Plot 3: Smile per ogni configurazione
        # Calcola le dimensioni della griglia necessarie
        n_rows = len(dt_values)
        n_cols = len(n_paths_values)
        
        fig3, axes = plt.subplots(n_rows, n_cols, 
                                  figsize=(5 * n_cols, 4 * n_rows), 
                                  squeeze=False) # squeeze=False assicura che axes sia sempre 2D

        # Utilizza un doppio ciclo for per iterare su tutte le combinazioni
        for i, dt in enumerate(dt_values):
            for j, n_paths in enumerate(n_paths_values):
                ax = axes[i, j] # Accedi all'asse corretto
                
                # Estrai la superficie di volatilità per la configurazione corrente
                surface = results[f'dt_{dt}'][f'paths_{n_paths}']['surface']
                
                # Plotta gli smile per le prime 3 scadenze
                for maturity_idx, T in enumerate(maturities[:3]):
                    iv_smile = surface[maturity_idx].cpu().numpy()
                    logK_numpy = logK.cpu().numpy() if torch.is_tensor(logK) else logK
                    ax.plot(logK_numpy, iv_smile, 'o-', markersize=4, label=f'T={T:.3f}')
                
                ax.set_xlabel('Log-Moneyness')
                ax.set_ylabel('Implied Volatility')
                ax.set_title(f'dt={dt:.5f}, paths={n_paths}')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('IV Smiles for All Configurations', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Aggiusta per il suptitle
        
        if save_path:
            plt.savefig(f"{save_path}_all_smiles.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    def recommend_parameters(self, results, target_accuracy=0.001, max_time=10.0):
        """Raccomanda parametri ottimali basati sui risultati"""
        print("\n" + "="*80)
        print("PARAMETER RECOMMENDATIONS")
        print("="*80)
        
        recommendations = []
        
        for dt_key in results:
            dt = float(dt_key.split('_')[1])
            
            for paths_key in results[dt_key]:
                n_paths = int(paths_key.split('_')[1])
                stats = results[dt_key][paths_key]
                
                if stats['time'] <= max_time and stats['nan_count'] == 0:
                    # Stima accuracy usando std come proxy
                    estimated_accuracy = stats['std'] / np.sqrt(n_paths) * 10
                    
                    if estimated_accuracy <= target_accuracy:
                        recommendations.append({
                            'dt': dt,
                            'n_paths': n_paths,
                            'time': stats['time'],
                            'accuracy': estimated_accuracy,
                            'mean_iv': stats['mean']
                        })
        
        # Ordina per tempo
        recommendations.sort(key=lambda x: x['time'])
        
        if recommendations:
            best = recommendations[0]
            print(f"Best configuration for accuracy<{target_accuracy} and time<{max_time}s:")
            print(f"  dt = {best['dt']:.6f} ({1/best['dt']:.0f} steps/year)")
            print(f"  n_paths = {best['n_paths']}")
            print(f"  Expected time: {best['time']:.2f}s")
            print(f"  Estimated accuracy: {best['accuracy']:.6f}")
        else:
            print("No configuration meets the criteria. Consider:")
            print("  - Increasing max_time")
            print("  - Decreasing target_accuracy")
            print("  - Using more paths or smaller dt")
        
        return recommendations


# Esempio di utilizzo
def test_short_maturities():
    """Test specifico per short maturities problematiche"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    debugger = MonteCarloDebugger(device)
    
    # Parametri tipici del Rough Bergomi
    # theta = [0.25, 1.8, -0.5, 0.08]  # H, eta, rho, xi0
    theta = [ 0.08628702,  3.19980052, -0.48315778,  0.19590783]
    
    # Short maturities (< 3 mesi)
    short_maturities = torch.tensor([
        7/365,    # 1 settimana
        14/365,   # 2 settimane
        30/365,   # 1 mese
        60/365,   # 2 mesi
        90/365    # 3 mesi
    ])
    
    # Log-moneyness
    logK = torch.linspace(-0.2, 0.2, 15)
    
    # Test diverse configurazioni
    results = debugger.test_mc_parameters(
        theta=theta,
        maturities=short_maturities,
        logK=logK,
        # n_paths_list=[5000, 10000, 30000, 50000],
        dt_list=[1/365, 1/730, 1/1460, 1/2920],  # daily, 12h, 6h, 3h
        use_antithetic=True,
        control_variate=True
    )
    
    # Visualizza risultati
    debugger.visualize_results(results, short_maturities, logK, save_path="mc_debug")
    
    # Raccomandazioni
    recommendations = debugger.recommend_parameters(
        results, 
        target_accuracy=0.0005,  # Target accuracy più stringente per short mat
        max_time=30.0  # Max 30 secondi per superficie
    )
    
    return results, recommendations


# Test con parametri estremi
def test_extreme_cases():
    """Test casi estremi che potrebbero causare problemi"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    debugger = MonteCarloDebugger(device)
    
    # Casi estremi da testare
    test_cases = [
        # [H, eta, rho, xi0, description]
        [0.1, 2.5, -0.9, 0.04, "Low H, high correlation"],
        [0.4, 0.5, -0.2, 0.15, "High H, low correlation"], 
        [0.25, 3.0, -0.95, 0.02, "Extreme eta and rho"],
        [0.05, 1.0, -0.5, 0.16, "Boundary H and xi0"]
    ]
    
    very_short_mat = torch.tensor([30/365])  # Solo 1 settimana
    # logK = torch.tensor([-0.2, -0.1, 0.0, 0.1, 0.2])  # 5 strikes
    logK = torch.linspace(-0.1, 0.1, 15)
    
    for params in test_cases:
        theta = params[:4]
        desc = params[4]
        
        print(f"\n{'='*60}")
        print(f"Testing: {desc}")
        print(f"{'='*60}")
        
        results = debugger.test_mc_parameters(
            theta=theta,
            maturities=very_short_mat,
            logK=logK,
            n_paths_list=[10000, 100000],
            dt_list=[1/1460, 1/2920],  # 6h, 3h
            use_antithetic=True,
            control_variate=True
        )
        
        # Quick visualization
        plt.figure(figsize=(10, 4))
        
        for i, (dt_key, paths_dict) in enumerate(results.items()):
            for j, (paths_key, stats) in enumerate(paths_dict.items()):
                plt.subplot(1, len(results) * len(paths_dict), i * len(paths_dict) + j + 1)
                
                smile = stats['surface'][0].cpu().numpy()
                plt.plot(logK.numpy(), smile, 'o-')
                plt.xlabel('Log-K')
                plt.ylabel('IV')
                plt.title(f"{dt_key}, {paths_key}")
                plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'Smile for {desc}')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Esegui test principale
    # results, recommendations = test_short_maturities()
    
    # Opzionale: test casi estremi
    test_extreme_cases()
