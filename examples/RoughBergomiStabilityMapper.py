import torch
import numpy as np
import matplotlib.pyplot as plt
from deepLearningVolatility.stochastic import generate_rough_bergomi
from deepLearningVolatility.nn.modules import BlackScholes
from deepLearningVolatility.instruments import EuropeanOption, BrownianStock
import time
from tqdm import tqdm
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

class RoughBergomiStabilityMapper:
    """
    Mappa le regioni di stabilità per i parametri del modello Rough Bergomi
    """
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device) if isinstance(device, str) else device
        
    def compute_stability_score(self, theta, test_config):
        """
        Calcola uno score di stabilità per una combinazione di parametri
        
        Returns:
            float: score tra 0 (instabile) e 1 (stabile)
        """
        H, eta, rho, xi0 = theta
        
        # Check preliminari sui parametri
        alpha = H - 0.5
        
        # Verifica che alpha sia in un range valido per il modello
        # alpha molto negativo (H molto piccolo) può causare problemi numerici
        if alpha < -0.49:  # H < 0.01
            return 0.0
        
        # Verifica che la matrice di covarianza sarà definita positiva
        # Questo richiede che dt^(alpha+1) sia ben definito
        if alpha <= -1.0:
            return 0.0
            
        # Verifica altri parametri
        if abs(rho) >= 1.0 or eta <= 0 or xi0 <= 0:
            return 0.0
        
        maturities = test_config['maturities']
        strikes = test_config['strikes']
        n_paths = test_config['n_paths']
        dt = test_config['dt']
        
        try:
            # Per H molto bassi, usa dt più piccolo per stabilità numerica
            if H < 0.1:
                dt = min(dt, 1/2920)  # Max 3h steps
                
            # Genera una piccola superficie di test
            iv_surface = torch.zeros(len(maturities), len(strikes))
            zero_count = 0
            extreme_count = 0
            total_points = len(maturities) * len(strikes)
            
            for i, T in enumerate(maturities):
                n_steps = max(3, int(np.ceil(T / dt)))  # Minimo 3 steps
                
                # Per parametri estremi, prova con meno paths per il test
                test_paths = min(n_paths, 5000) if H < 0.1 else n_paths
                
                # Genera paths con gestione errori
                try:
                    # Assicura che tutti i parametri siano float32
                    S, V = generate_rough_bergomi(
                        test_paths, n_steps,
                        init_state=(float(1.0), float(xi0)),
                        alpha=float(alpha),
                        rho=float(rho),
                        eta=float(eta),
                        xi=float(xi0),
                        dt=float(dt),
                        device=self.device,
                        dtype=torch.float32,  # Forza float32
                        antithetic=True
                    )
                    
                    # Verifica che i paths siano validi
                    if torch.isnan(S).any() or torch.isinf(S).any():
                        return 0.0
                        
                    if (S <= 0).any():
                        return 0.0
                    
                    ST = S[:, -1]
                except Exception as e:
                    # Se la generazione fallisce per questo set di parametri
                    return 0.0
                
                for j, logK in enumerate(strikes):
                    K = np.exp(logK)
                    payoff = (ST - K).clamp(min=0.0)
                    price = payoff.mean().item()
                    
                    if price > 1e-10:
                        try:
                            bs = BlackScholes(
                                EuropeanOption(
                                    BrownianStock(),
                                    strike=K,
                                    maturity=T
                                )
                            )
                            iv = bs.implied_volatility(
                                log_moneyness=-logK,
                                time_to_maturity=T,
                                price=price
                            )
                            
                            # Controlla se IV è ragionevole
                            if 0.01 < iv < 2.0:
                                iv_surface[i, j] = iv
                            else:
                                extreme_count += 1
                                
                        except:
                            zero_count += 1
                    else:
                        zero_count += 1
            
            # Calcola metriche di stabilità
            valid_ivs = iv_surface[iv_surface > 0]
            
            if len(valid_ivs) == 0:
                return 0.0
            
            # Score basato su:
            # 1. Percentuale di valori validi
            valid_ratio = len(valid_ivs) / total_points
            
            # 2. Smoothness dello smile (variazione tra strikes adiacenti)
            smoothness_scores = []
            for i in range(len(maturities)):
                smile = iv_surface[i]
                valid_smile = smile[smile > 0]
                if len(valid_smile) > 1:
                    diffs = torch.diff(valid_smile)
                    smoothness = 1.0 / (1.0 + diffs.abs().mean().item())
                    smoothness_scores.append(smoothness)
            
            avg_smoothness = np.mean(smoothness_scores) if smoothness_scores else 0
            
            # 3. Range ragionevole di IV
            iv_std = valid_ivs.std().item()
            range_score = 1.0 / (1.0 + max(0, iv_std - 0.5))
            
            # Combina scores
            stability_score = valid_ratio * 0.5 + avg_smoothness * 0.3 + range_score * 0.2
            
            return stability_score
            
        except Exception as e:
            # print(f"Error computing stability: {e}")
            return 0.0
    
    def map_2d_stability(self, param1_name, param1_range, param2_name, param2_range,
                        fixed_params, n_grid=20, test_regime='short'):
        """
        Crea una mappa 2D di stabilità variando due parametri
        """
        # Configurazione di test basata sul regime
        if test_regime == 'short':
            test_config = {
                'maturities': [7/365, 14/365, 30/365],
                'strikes': np.linspace(-0.15, 0.15, 7),
                'n_paths': 50_000,
                'dt': 1/1460
            }
        elif test_regime == 'mid':
            test_config = {
                'maturities': [90/365, 180/365],
                'strikes': np.linspace(-0.3, 0.3, 7),
                'n_paths': 50_000,
                'dt': 1/365/2
            }
        else:  # long
            test_config = {
                'maturities': [1.0, 2.0],
                'strikes': np.linspace(-0.5, 0.5, 7),
                'n_paths': 50_000,
                'dt': 1/365
            }
        
        # Griglia di parametri
        param1_vals = np.linspace(param1_range[0], param1_range[1], n_grid)
        param2_vals = np.linspace(param2_range[0], param2_range[1], n_grid)
        
        stability_map = np.zeros((n_grid, n_grid))
        
        # Mapping nome parametro -> indice
        param_idx = {'H': 0, 'eta': 1, 'rho': 2, 'xi0': 3}
        
        print(f"Mapping stability for {param1_name} vs {param2_name} ({test_regime} regime)")
        print(f"Grid size: {n_grid}x{n_grid}")
        
        # Progress bar
        total_tests = n_grid * n_grid
        pbar = tqdm(total=total_tests, desc="Computing stability")
        
        for i, p1 in enumerate(param1_vals):
            for j, p2 in enumerate(param2_vals):
                # Costruisci theta
                theta = list(fixed_params)
                theta[param_idx[param1_name]] = p1
                theta[param_idx[param2_name]] = p2
                
                # Calcola stabilità
                score = self.compute_stability_score(theta, test_config)
                stability_map[i, j] = score
                
                pbar.update(1)
        
        pbar.close()
        
        return stability_map, param1_vals, param2_vals
    
    def plot_stability_map(self, stability_map, param1_vals, param2_vals,
                          param1_name, param2_name, title=None):
        """
        Visualizza la mappa di stabilità con zone colorate
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Usa colormap con soglie chiare
        # Rosso = instabile (< 0.3), Giallo = marginale (0.3-0.7), Verde = stabile (> 0.7)
        cmap = plt.cm.RdYlGn
        
        im = ax.imshow(stability_map.T, origin='lower', cmap=cmap,
                      extent=[param1_vals[0], param1_vals[-1],
                             param2_vals[0], param2_vals[-1]],
                      aspect='auto', vmin=0, vmax=1)
        
        # Aggiungi contorni per le zone
        contour_levels = [0.3, 0.7]
        contours = ax.contour(param1_vals, param2_vals, stability_map.T,
                            levels=contour_levels, colors='black', linewidths=2)
        ax.clabel(contours, inline=True, fontsize=10, fmt='%.1f')
        
        # Labels e titolo
        ax.set_xlabel(f'{param1_name}', fontsize=12)
        ax.set_ylabel(f'{param2_name}', fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14, pad=20)
        else:
            ax.set_title(f'Stability Map: {param1_name} vs {param2_name}', fontsize=14, pad=20)
        
        # Colorbar con labels
        cbar = plt.colorbar(im, ax=ax, label='Stability Score')
        cbar.set_ticks([0, 0.3, 0.7, 1.0])
        cbar.set_ticklabels(['Unstable', 'Marginal', 'Stable', 'Excellent'])
        
        # Aggiungi griglia
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Aggiungi annotazioni per zone problematiche
        self._annotate_problematic_zones(ax, stability_map, param1_vals, param2_vals)
        
        plt.tight_layout()
        return fig, ax
    
    def _annotate_problematic_zones(self, ax, stability_map, param1_vals, param2_vals):
        """Evidenzia zone particolarmente problematiche"""
        # Trova zone con score < 0.1
        very_bad = stability_map < 0.1
        
        if very_bad.any():
            # Trova cluster di punti problematici
            from scipy import ndimage
            labeled, num_features = ndimage.label(very_bad)
            
            for i in range(1, num_features + 1):
                mask = labeled == i
                if mask.sum() > 4:  # Solo cluster significativi
                    # Trova bounding box
                    rows, cols = np.where(mask)
                    min_row, max_row = rows.min(), rows.max()
                    min_col, max_col = cols.min(), cols.max()
                    
                    # Converti in coordinate parametri
                    p1_min = param1_vals[min_row]
                    p1_max = param1_vals[max_row]
                    p2_min = param2_vals[min_col]
                    p2_max = param2_vals[max_col]
                    
                    # Disegna rettangolo
                    rect = Rectangle((p1_min, p2_min), 
                                   p1_max - p1_min, 
                                   p2_max - p2_min,
                                   linewidth=2, edgecolor='red',
                                   facecolor='none', linestyle='--')
                    ax.add_patch(rect)
    
    def full_parameter_analysis(self, n_grid=15):
        """
        Analisi completa di tutte le combinazioni di parametri
        """
        # Parametri di default "sicuri"
        default_params = {
            'H': 0.25,
            'eta': 1.5,
            'rho': -0.5,
            'xi0': 0.1
        }
        
        # Range dei parametri
        param_ranges = {
            'H': (0.05, 0.45),
            'eta': (0.5, 3.0),
            'rho': (-0.95, -0.1),
            'xi0': (0.02, 0.20)
        }
        
        # Combinazioni da testare
        param_pairs = [
            ('H', 'eta'),
            ('H', 'rho'),
            ('H', 'xi0'),
            ('eta', 'rho'),
            ('eta', 'xi0'),
            ('rho', 'xi0')
        ]
        
        results = {}
        
        for regime in ['short', 'mid', 'long']:
            print(f"\n{'='*60}")
            print(f"Analyzing {regime.upper()} TERM regime")
            print(f"{'='*60}")
            
            regime_results = {}
            
            # Crea subplot per questo regime
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for idx, (param1, param2) in enumerate(param_pairs):
                print(f"\nMapping {param1} vs {param2}...")
                
                # Parametri fissi
                fixed = default_params.copy()
                del fixed[param1]
                del fixed[param2]
                fixed_list = [fixed.get('H', 0.25), 
                            fixed.get('eta', 1.5),
                            fixed.get('rho', -0.5), 
                            fixed.get('xi0', 0.1)]
                
                # Calcola mappa
                stability_map, p1_vals, p2_vals = self.map_2d_stability(
                    param1, param_ranges[param1],
                    param2, param_ranges[param2],
                    fixed_list, n_grid, regime
                )
                
                regime_results[f'{param1}_vs_{param2}'] = {
                    'map': stability_map,
                    'param1_vals': p1_vals,
                    'param2_vals': p2_vals
                }
                
                # Plot su subplot
                ax = axes[idx]
                im = ax.imshow(stability_map.T, origin='lower', cmap='RdYlGn',
                             extent=[p1_vals[0], p1_vals[-1],
                                    p2_vals[0], p2_vals[-1]],
                             aspect='auto', vmin=0, vmax=1)
                
                # Contorni
                contours = ax.contour(p1_vals, p2_vals, stability_map.T,
                                    levels=[0.3, 0.7], colors='black', linewidths=1.5)
                
                ax.set_xlabel(param1)
                ax.set_ylabel(param2)
                ax.set_title(f'{param1} vs {param2}')
                ax.grid(True, alpha=0.3)
                
                # Mini colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            plt.suptitle(f'Stability Maps - {regime.upper()} TERM', fontsize=16)
            plt.tight_layout()
            plt.show()
            
            results[regime] = regime_results
        
        return results
    
    def find_safe_regions(self, results, threshold=0.7):
        """
        Trova regioni sicure comuni a tutti i regimi
        """
        print("\n" + "="*60)
        print("SAFE PARAMETER REGIONS (score > {:.1f})".format(threshold))
        print("="*60)
        
        safe_regions = {}
        
        for pair_name in ['H_vs_eta', 'H_vs_rho', 'H_vs_xi0', 
                         'eta_vs_rho', 'eta_vs_xi0', 'rho_vs_xi0']:
            
            print(f"\n{pair_name.replace('_', ' ')}:")
            
            # Combina mappe da tutti i regimi
            combined_map = None
            
            for regime in ['short', 'mid', 'long']:
                regime_map = results[regime][pair_name]['map']
                if combined_map is None:
                    combined_map = regime_map
                else:
                    combined_map = np.minimum(combined_map, regime_map)
            
            # Trova regioni sicure
            safe_mask = combined_map > threshold
            
            if safe_mask.any():
                # Estrai ranges
                param1_vals = results['short'][pair_name]['param1_vals']
                param2_vals = results['short'][pair_name]['param2_vals']
                
                safe_i, safe_j = np.where(safe_mask)
                
                param1_safe_range = [param1_vals[safe_i].min(), param1_vals[safe_i].max()]
                param2_safe_range = [param2_vals[safe_j].min(), param2_vals[safe_j].max()]
                
                param1_name = pair_name.split('_vs_')[0]
                param2_name = pair_name.split('_vs_')[1]
                
                print(f"  {param1_name}: [{param1_safe_range[0]:.3f}, {param1_safe_range[1]:.3f}]")
                print(f"  {param2_name}: [{param2_safe_range[0]:.3f}, {param2_safe_range[1]:.3f}]")
                
                safe_regions[pair_name] = {
                    'param1_range': param1_safe_range,
                    'param2_range': param2_safe_range,
                    'combined_map': combined_map
                }
            else:
                print("  No safe regions found!")
        
        return safe_regions
    
    def suggest_parameter_constraints(self, safe_regions):
        """
        Suggerisce vincoli sui parametri basati sull'analisi
        """
        print("\n" + "="*60)
        print("SUGGESTED PARAMETER CONSTRAINTS")
        print("="*60)
        
        # Raccogli tutti i range sicuri per ogni parametro
        param_constraints = {
            'H': [], 'eta': [], 'rho': [], 'xi0': []
        }
        
        for pair_name, region_data in safe_regions.items():
            if region_data:
                param1_name = pair_name.split('_vs_')[0]
                param2_name = pair_name.split('_vs_')[1]
                
                param_constraints[param1_name].append(region_data['param1_range'])
                param_constraints[param2_name].append(region_data['param2_range'])
        
        # Trova intersezione dei range sicuri
        final_constraints = {}
        
        for param, ranges in param_constraints.items():
            if ranges:
                # Prendi l'intersezione più conservativa
                min_val = max(r[0] for r in ranges)
                max_val = min(r[1] for r in ranges)
                
                if min_val < max_val:
                    final_constraints[param] = (min_val, max_val)
                    print(f"\n{param}:")
                    print(f"  Safe range: [{min_val:.3f}, {max_val:.3f}]")
                else:
                    print(f"\n{param}: No consistent safe range found")
        
        # Suggerimenti specifici
        print("\n" + "-"*40)
        print("SPECIFIC RECOMMENDATIONS:")
        print("-"*40)
        
        if 'H' in final_constraints:
            if final_constraints['H'][0] > 0.15:
                print("- Avoid very low Hurst parameter (H < 0.15)")
        
        if 'rho' in final_constraints:
            if final_constraints['rho'][1] < -0.8:
                print("- Avoid extreme negative correlation (rho < -0.8)")
        
        if 'xi0' in final_constraints:
            if final_constraints['xi0'][0] > 0.05:
                print("- Ensure sufficient initial volatility (xi0 > 0.05)")
        
        return final_constraints


# Funzione principale per eseguire l'analisi completa
def run_stability_analysis(device='cpu', n_grid=20):
    """
    Esegue un'analisi completa di stabilità
    """
    mapper = RoughBergomiStabilityMapper(device)
    
    # 1. Analisi completa
    print("Starting full parameter stability analysis...")
    results = mapper.full_parameter_analysis(n_grid=n_grid)
    
    # 2. Trova regioni sicure
    safe_regions = mapper.find_safe_regions(results, threshold=0.7)
    
    # 3. Suggerisci vincoli
    constraints = mapper.suggest_parameter_constraints(safe_regions)
    
    # 4. Test di verifica con parametri suggeriti
    if constraints:
        print("\n" + "="*60)
        print("VERIFICATION TEST")
        print("="*60)
        
        # Crea un theta nel centro della regione sicura
        safe_theta = []
        param_names = ['H', 'eta', 'rho', 'xi0']
        default_vals = [0.25, 1.5, -0.5, 0.1]
        
        for i, param in enumerate(param_names):
            if param in constraints:
                safe_val = (constraints[param][0] + constraints[param][1]) / 2
                safe_theta.append(safe_val)
            else:
                safe_theta.append(default_vals[i])
        
        print(f"Testing safe parameters: H={safe_theta[0]:.3f}, eta={safe_theta[1]:.3f}, "
              f"rho={safe_theta[2]:.3f}, xi0={safe_theta[3]:.3f}")
        
        # Test su tutti i regimi
        for regime in ['short', 'mid', 'long']:
            test_config = {
                'short': {
                    'maturities': [7/365, 14/365, 30/365],
                    'strikes': np.linspace(-0.2, 0.2, 11),
                    'n_paths': 50_000,
                    'dt': 1/1460
                },
                'mid': {
                    'maturities': [90/365, 180/365, 365/365],
                    'strikes': np.linspace(-0.4, 0.4, 13),
                    'n_paths': 50_000,
                    'dt': 1/365/2
                },
                'long': {
                    'maturities': [1.0, 2.0, 3.0],
                    'strikes': np.linspace(-0.5, 0.5, 15),
                    'n_paths': 50_000,
                    'dt': 1/365
                }
            }[regime]
            
            score = mapper.compute_stability_score(safe_theta, test_config)
            print(f"  {regime} term stability score: {score:.3f}")
    
    return results, safe_regions, constraints


if __name__ == "__main__":
    # Esegui analisi
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test veloce con griglia piccola
    print("Running quick stability analysis...")
    results, safe_regions, constraints = run_stability_analysis(device, n_grid=10)
    
    # Per analisi più dettagliata, usa n_grid=20 o 30
    # results, safe_regions, constraints = run_stability_analysis(device, n_grid=20)