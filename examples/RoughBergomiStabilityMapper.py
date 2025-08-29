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
    Maps the stability regions for the parameters of the Rough Bergomi model
    """
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device) if isinstance(device, str) else device
        
    def compute_stability_score(self, theta, test_config):
        """
        Computes a stability score for a parameter combination

        Returns:
            float: score between 0 (unstable) and 1 (stable)
        """
        H, eta, rho, xi0 = theta
        
        # Preliminary checks on parameters
        alpha = H - 0.5
        
        # Check that alpha is in a valid range for the model
        # Very negative alpha (very small H) can cause numerical issues
        if alpha < -0.49:  # H < 0.01
            return 0.0
        
        # Check that the covariance matrix will be positive definite
        # This requires that dt^(alpha+1) is well defined
        if alpha <= -1.0:
            return 0.0
            
        # Check other parameters
        if abs(rho) >= 1.0 or eta <= 0 or xi0 <= 0:
            return 0.0
        
        maturities = test_config['maturities']
        strikes = test_config['strikes']
        n_paths = test_config['n_paths']
        dt = test_config['dt']
        
        try:
            # For very low H, use smaller dt for numerical stability
            if H < 0.1:
                dt = min(dt, 1/2920)  # Max 3h steps
                
            # Generate a small test surface
            iv_surface = torch.zeros(len(maturities), len(strikes))
            zero_count = 0
            extreme_count = 0
            total_points = len(maturities) * len(strikes)
            
            for i, T in enumerate(maturities):
                n_steps = max(3, int(np.ceil(T / dt)))  # Minimum 3 steps
                
                # For extreme parameters, try with fewer paths for the test
                test_paths = min(n_paths, 5000) if H < 0.1 else n_paths
                
                # Generate paths with error handling
                try:
                    # Ensure all parameters are float32
                    S, V = generate_rough_bergomi(
                        test_paths, n_steps,
                        init_state=(float(1.0), float(xi0)),
                        alpha=float(alpha),
                        rho=float(rho),
                        eta=float(eta),
                        xi=float(xi0),
                        dt=float(dt),
                        device=self.device,
                        dtype=torch.float32,  # Force float32
                        antithetic=True
                    )
                    
                    # Check that the paths are valid
                    if torch.isnan(S).any() or torch.isinf(S).any():
                        return 0.0
                        
                    # if (S <= 0).any():
                    #     return 0.0
                    
                    ST = S[:, -1]
                except Exception as e:
                    # If generation fails for this parameter set
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
                            
                            if 0.01 < iv < 2.0:
                                iv_surface[i, j] = iv
                            else:
                                extreme_count += 1
                                
                        except:
                            zero_count += 1
                    else:
                        zero_count += 1
            
            # Compute stability metrics
            valid_ivs = iv_surface[iv_surface > 0]
            
            if len(valid_ivs) == 0:
                return 0.0
            
            # Score based on:
            # 1. Percentage of valid values
            valid_ratio = len(valid_ivs) / total_points
            
            # 2. Smile smoothness (variation between adjacent strikes)
            smoothness_scores = []
            for i in range(len(maturities)):
                smile = iv_surface[i]
                valid_smile = smile[smile > 0]
                if len(valid_smile) > 1:
                    diffs = torch.diff(valid_smile)
                    smoothness = 1.0 / (1.0 + diffs.abs().mean().item())
                    smoothness_scores.append(smoothness)
            
            avg_smoothness = np.mean(smoothness_scores) if smoothness_scores else 0
            
            # 3. Reasonable IV range
            iv_std = valid_ivs.std().item()
            range_score = 1.0 / (1.0 + max(0, iv_std - 0.5))
            
            # Combine scores
            stability_score = valid_ratio * 0.5 + avg_smoothness * 0.3 + range_score * 0.2
            
            return stability_score
            
        except Exception as e:
            # print(f"Error computing stability: {e}")
            return 0.0
    
    def map_2d_stability(self, param1_name, param1_range, param2_name, param2_range,
                        fixed_params, n_grid=20, test_regime='short'):
        """
        Creates a 2D stability map by varying two parameters
        """
        # Test configuration based on regime
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
        
        # Parameter grid
        param1_vals = np.linspace(param1_range[0], param1_range[1], n_grid)
        param2_vals = np.linspace(param2_range[0], param2_range[1], n_grid)
        
        stability_map = np.zeros((n_grid, n_grid))
        
        # Mapping parameter name -> index
        param_idx = {'H': 0, 'eta': 1, 'rho': 2, 'xi0': 3}
        
        print(f"Mapping stability for {param1_name} vs {param2_name} ({test_regime} regime)")
        print(f"Grid size: {n_grid}x{n_grid}")
        
        # Progress bar
        total_tests = n_grid * n_grid
        pbar = tqdm(total=total_tests, desc="Computing stability")
        
        for i, p1 in enumerate(param1_vals):
            for j, p2 in enumerate(param2_vals):
                # Build theta
                theta = list(fixed_params)
                theta[param_idx[param1_name]] = p1
                theta[param_idx[param2_name]] = p2
                
                # Compute stability
                score = self.compute_stability_score(theta, test_config)
                stability_map[i, j] = score
                
                pbar.update(1)
        
        pbar.close()
        
        return stability_map, param1_vals, param2_vals
    
    def plot_stability_map(self, stability_map, param1_vals, param2_vals,
                          param1_name, param2_name, title=None):
        """
        Displays the stability map with colored zones
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Use colormap with clear thresholds
        # Red = unstable (< 0.3), Yellow = marginal (0.3-0.7), Green = stable (> 0.7)
        cmap = plt.cm.RdYlGn
        
        im = ax.imshow(stability_map.T, origin='lower', cmap=cmap,
                      extent=[param1_vals[0], param1_vals[-1],
                             param2_vals[0], param2_vals[-1]],
                      aspect='auto', vmin=0, vmax=1)
        
        # Add contours for the zones
        contour_levels = [0.3, 0.7]
        contours = ax.contour(param1_vals, param2_vals, stability_map.T,
                            levels=contour_levels, colors='black', linewidths=2)
        ax.clabel(contours, inline=True, fontsize=10, fmt='%.1f')
        
        # Labels and title
        ax.set_xlabel(f'{param1_name}', fontsize=12)
        ax.set_ylabel(f'{param2_name}', fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14, pad=20)
        else:
            ax.set_title(f'Stability Map: {param1_name} vs {param2_name}', fontsize=14, pad=20)
        
        # Colorbar with labels
        cbar = plt.colorbar(im, ax=ax, label='Stability Score')
        cbar.set_ticks([0, 0.3, 0.7, 1.0])
        cbar.set_ticklabels(['Unstable', 'Marginal', 'Stable', 'Excellent'])
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add annotations for problematic zones
        self._annotate_problematic_zones(ax, stability_map, param1_vals, param2_vals)
        
        plt.tight_layout()
        return fig, ax
    
    def _annotate_problematic_zones(self, ax, stability_map, param1_vals, param2_vals):
        """Highlight particularly problematic zones"""
        # Find zones with score < 0.1
        very_bad = stability_map < 0.1
        
        if very_bad.any():
            # Find clusters of problematic points
            from scipy import ndimage
            labeled, num_features = ndimage.label(very_bad)
            
            for i in range(1, num_features + 1):
                mask = labeled == i
                if mask.sum() > 4:  # Only significant clusters
                    # Find bounding box
                    rows, cols = np.where(mask)
                    min_row, max_row = rows.min(), rows.max()
                    min_col, max_col = cols.min(), cols.max()
                    
                    # Convert to parameter coordinates
                    p1_min = param1_vals[min_row]
                    p1_max = param1_vals[max_row]
                    p2_min = param2_vals[min_col]
                    p2_max = param2_vals[max_col]
                    
                    # Draw rectangle
                    rect = Rectangle((p1_min, p2_min), 
                                   p1_max - p1_min, 
                                   p2_max - p2_min,
                                   linewidth=2, edgecolor='red',
                                   facecolor='none', linestyle='--')
                    ax.add_patch(rect)
    
    def full_parameter_analysis(self, n_grid=15):
        """
        Complete analysis of all parameter combinations
        """
        # Default "safe" parameters
        default_params = {
            'H': 0.25,
            'eta': 1.5,
            'rho': -0.5,
            'xi0': 0.1
        }
        
        # Parameter ranges
        param_ranges = {
            'H': (0.05, 0.45),
            'eta': (0.5, 3.0),
            'rho': (-0.95, -0.1),
            'xi0': (0.02, 0.20)
        }
        
        # Pairs to test
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
            
            # Create subplot for this regime
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for idx, (param1, param2) in enumerate(param_pairs):
                print(f"\nMapping {param1} vs {param2}...")
                
                # Fixed parameters
                fixed = default_params.copy()
                del fixed[param1]
                del fixed[param2]
                fixed_list = [fixed.get('H', 0.25), 
                            fixed.get('eta', 1.5),
                            fixed.get('rho', -0.5), 
                            fixed.get('xi0', 0.1)]
                
                # Compute map
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
                
                # Plot on subplot
                ax = axes[idx]
                im = ax.imshow(stability_map.T, origin='lower', cmap='RdYlGn',
                             extent=[p1_vals[0], p1_vals[-1],
                                    p2_vals[0], p2_vals[-1]],
                             aspect='auto', vmin=0, vmax=1)
                
                # Contours
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
        Finds safe regions common to all regimes
        """
        print("\n" + "="*60)
        print("SAFE PARAMETER REGIONS (score > {:.1f})".format(threshold))
        print("="*60)
        
        safe_regions = {}
        
        for pair_name in ['H_vs_eta', 'H_vs_rho', 'H_vs_xi0', 
                         'eta_vs_rho', 'eta_vs_xi0', 'rho_vs_xi0']:
            
            print(f"\n{pair_name.replace('_', ' ')}:")
            
            # Combine maps from all regimes
            combined_map = None
            
            for regime in ['short', 'mid', 'long']:
                regime_map = results[regime][pair_name]['map']
                if combined_map is None:
                    combined_map = regime_map
                else:
                    combined_map = np.minimum(combined_map, regime_map)
            
            # Find safe regions
            safe_mask = combined_map > threshold
            
            if safe_mask.any():
                # Extract ranges
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
        Suggests parameter constraints based on the analysis
        """
        print("\n" + "="*60)
        print("SUGGESTED PARAMETER CONSTRAINTS")
        print("="*60)
        
        # Collect all safe ranges for each parameter
        param_constraints = {
            'H': [], 'eta': [], 'rho': [], 'xi0': []
        }
        
        for pair_name, region_data in safe_regions.items():
            if region_data:
                param1_name = pair_name.split('_vs_')[0]
                param2_name = pair_name.split('_vs_')[1]
                
                param_constraints[param1_name].append(region_data['param1_range'])
                param_constraints[param2_name].append(region_data['param2_range'])
        
        # Find intersection of safe ranges
        final_constraints = {}
        
        for param, ranges in param_constraints.items():
            if ranges:
                # Take the most conservative intersection
                min_val = max(r[0] for r in ranges)
                max_val = min(r[1] for r in ranges)
                
                if min_val < max_val:
                    final_constraints[param] = (min_val, max_val)
                    print(f"\n{param}:")
                    print(f"  Safe range: [{min_val:.3f}, {max_val:.3f}]")
                else:
                    print(f"\n{param}: No consistent safe range found")
        
        # Specific recommendations
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


def run_stability_analysis(device='cpu', n_grid=20):
    """
    Runs a complete stability analysis
    """
    mapper = RoughBergomiStabilityMapper(device)
    
    # 1. Complete analysis
    print("Starting full parameter stability analysis...")
    results = mapper.full_parameter_analysis(n_grid=n_grid)
    
    # 2. Find safe regions
    safe_regions = mapper.find_safe_regions(results, threshold=0.7)
    
    # 3. Suggest constraints
    constraints = mapper.suggest_parameter_constraints(safe_regions)
    
    # 4. Verification test with suggested parameters
    if constraints:
        print("\n" + "="*60)
        print("VERIFICATION TEST")
        print("="*60)
        
        # Create a theta in the center of the safe region
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
        
        # Test on all regimes
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
    # Run analysis
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Quick test with small grid
    print("Running quick stability analysis...")
    results, safe_regions, constraints = run_stability_analysis(device, n_grid=10)
    
    # For more detailed analysis, use n_grid=20 or 30
    # results, safe_regions, constraints = run_stability_analysis(device, n_grid=20)