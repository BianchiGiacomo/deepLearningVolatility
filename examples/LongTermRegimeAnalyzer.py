import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')
from deepLearningVolatility.nn.pricer import GridNetworkPricer
from deepLearningVolatility.stochastic.stochastic_interface import ProcessFactory
from deepLearningVolatility.stochastic.wrappers.rough_bergomi_wrapper import RoughBergomiProcess

class LongTermRegimeAnalyzer:
    """
    Specifically analyzes the behavior of the long term regime
    with a focus on absorption and stability
    """
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Specific configuration for long term
        self.long_maturities = torch.tensor([1.0, 1.5, 2.0, 3.0, 5.0], device=device)
        self.long_logK = torch.linspace(-0.5, 0.5, 15, device=device)
        
    def test_single_configuration(self, pricer, theta, n_paths=50000, 
                                 handle_absorption=True, verbose=True):
        """
        Tests a single parameter configuration
        """
        H, eta, rho, xi0 = theta
        
        if verbose:
            print(f"\nTesting theta: H={H:.3f}, eta={eta:.3f}, rho={rho:.3f}, xi0={xi0:.3f}")
            print("-" * 60)
        
        # Test without absorption handling
        iv_no_abs = pricer._mc_iv_grid(
            torch.tensor(theta, device=self.device),
            n_paths=n_paths,
            handle_absorption=False,
            adaptive_dt=True,
            chunk_size=10000
        )
        
        # Test with absorption handling
        iv_with_abs = pricer._mc_iv_grid(
            torch.tensor(theta, device=self.device),
            n_paths=n_paths,
            handle_absorption=True,
            adaptive_dt=True,
            chunk_size=10000
        )
        
        # Get absorption statistics if available
        absorption_stats = None
        if hasattr(pricer, 'last_absorption_stats'):
            absorption_stats = pricer.last_absorption_stats.cpu().numpy()
        
        results = {
            'theta': theta,
            'iv_no_absorption': iv_no_abs.cpu().numpy(),
            'iv_with_absorption': iv_with_abs.cpu().numpy(),
            'absorption_stats': absorption_stats,
            'difference': (iv_with_abs - iv_no_abs).cpu().numpy()
        }
        
        if verbose:
            self._print_summary(results)
        
        return results
    
    def _print_summary(self, results):
        """Stampa summary dei risultati"""
        iv_no_abs = results['iv_no_absorption']
        iv_with_abs = results['iv_with_absorption']
        abs_stats = results['absorption_stats']
        
        print(f"IV Statistics (no absorption handling):")
        print(f"  Range: [{iv_no_abs.min():.4f}, {iv_no_abs.max():.4f}]")
        print(f"  Mean: {iv_no_abs.mean():.4f} ± {iv_no_abs.std():.4f}")
        print(f"  Zero values: {(iv_no_abs == 0).sum()}")
        
        print(f"\nIV Statistics (with absorption handling):")
        print(f"  Range: [{iv_with_abs.min():.4f}, {iv_with_abs.max():.4f}]")
        print(f"  Mean: {iv_with_abs.mean():.4f} ± {iv_with_abs.std():.4f}")
        print(f"  Zero values: {(iv_with_abs == 0).sum()}")
        
        if abs_stats is not None:
            print(f"\nAbsorption Statistics:")
            print(f"  Mean absorption: {abs_stats.mean():.1%}")
            print(f"  Max absorption: {abs_stats.max():.1%}")
            print(f"  Points with >50% absorption: {(abs_stats > 0.5).sum()}")
            
            # Analysis by maturity
            for i, T in enumerate(self.long_maturities):
                abs_row = abs_stats[i, :]
                print(f"  T={T:.1f}y: mean={abs_row.mean():.1%}, max={abs_row.max():.1%}")
    
    def plot_comparison(self, results, save_path=None):
        """Visualizza confronto tra metodi"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        theta = results['theta']
        fig.suptitle(f'Long Term Analysis - H={theta[0]:.3f}, eta={theta[1]:.3f}, '
                     f'rho={theta[2]:.3f}, xi0={theta[3]:.3f}', fontsize=16)
        
        # 1. Surface without absorption handling
        ax = axes[0, 0]
        im1 = ax.imshow(results['iv_no_absorption'], aspect='auto', origin='lower',
                       cmap='viridis', vmin=0, vmax=0.6)
        ax.set_title('IV Surface - No Absorption Handling')
        ax.set_xlabel('Strike Index')
        ax.set_ylabel('Maturity Index')
        plt.colorbar(im1, ax=ax)
        
        # 2. Surface with absorption handling
        ax = axes[0, 1]
        im2 = ax.imshow(results['iv_with_absorption'], aspect='auto', origin='lower',
                       cmap='viridis', vmin=0, vmax=0.6)
        ax.set_title('IV Surface - With Absorption Handling')
        ax.set_xlabel('Strike Index')
        ax.set_ylabel('Maturity Index')
        plt.colorbar(im2, ax=ax)
        
        # 3. Difference
        ax = axes[0, 2]
        diff = results['difference']
        im3 = ax.imshow(diff, aspect='auto', origin='lower',
                       cmap='RdBu_r', vmin=-0.1, vmax=0.1)
        ax.set_title('Difference (With - Without)')
        ax.set_xlabel('Strike Index')
        ax.set_ylabel('Maturity Index')
        plt.colorbar(im3, ax=ax)
        
        # 4. Absorption statistics
        if results['absorption_stats'] is not None:
            ax = axes[1, 0]
            im4 = ax.imshow(results['absorption_stats'], aspect='auto', origin='lower',
                           cmap='Reds', vmin=0, vmax=1)
            ax.set_title('Absorption Ratio')
            ax.set_xlabel('Strike Index')
            ax.set_ylabel('Maturity Index')
            cbar = plt.colorbar(im4, ax=ax)
            cbar.set_label('Fraction Absorbed')
        
        # 5. Smile comparison for the longest maturity
        ax = axes[1, 1]
        mat_idx = -1 
        smile_no_abs = results['iv_no_absorption'][mat_idx, :]
        smile_with_abs = results['iv_with_absorption'][mat_idx, :]
        
        logK = self.long_logK.cpu().numpy()
        ax.plot(logK, smile_no_abs, 'o-', label='No Absorption', linewidth=2)
        ax.plot(logK, smile_with_abs, 's-', label='With Absorption', linewidth=2)
        ax.set_xlabel('Log-Moneyness')
        ax.set_ylabel('Implied Volatility')
        ax.set_title(f'Smile Comparison - T={self.long_maturities[mat_idx]:.1f}Y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Absorption by maturity
        if results['absorption_stats'] is not None:
            ax = axes[1, 2]
            mean_abs_by_mat = results['absorption_stats'].mean(axis=1)
            maturities = self.long_maturities.cpu().numpy()
            
            ax.bar(range(len(maturities)), mean_abs_by_mat, alpha=0.7)
            ax.set_xticks(range(len(maturities)))
            ax.set_xticklabels([f'{T:.1f}Y' for T in maturities])
            ax.set_ylabel('Mean Absorption Ratio')
            ax.set_title('Absorption by Maturity')
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(y=0.5, color='r', linestyle='--', label='50% threshold')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def parameter_sweep_long_term(self, pricer, n_grid=10):
        """
        Parameter sweep specific for long term
        """
        # Ranges adapted for long term (more conservative)
        H_range = np.linspace(0.1, 0.4, n_grid)  # Avoid too low H
        xi0_range = np.linspace(0.05, 0.15, n_grid)  # Higher xi0 for long term
        
        # More stable fixed parameters
        eta_fixed = 1.5
        rho_fixed = -0.5
        
        results_grid = np.zeros((n_grid, n_grid, 4))  # Store: mean_iv, std_iv, zeros, absorption
        
        print(f"Parameter sweep for long term regime")
        print(f"H range: [{H_range[0]:.3f}, {H_range[-1]:.3f}]")
        print(f"xi0 range: [{xi0_range[0]:.3f}, {xi0_range[-1]:.3f}]")
        print(f"Fixed: eta={eta_fixed:.2f}, rho={rho_fixed:.2f}")
        print("-" * 60)
        
        pbar = tqdm(total=n_grid*n_grid, desc="Parameter sweep")
        
        for i, H in enumerate(H_range):
            for j, xi0 in enumerate(xi0_range):
                theta = [H, eta_fixed, rho_fixed, xi0]
                
                try:
                    # Test with absorption handling
                    iv_surface = pricer._mc_iv_grid(
                        torch.tensor(theta, device=self.device),
                        n_paths=10000,  # Fewer paths for fast sweep
                        handle_absorption=True,
                        adaptive_dt=True
                    )
                    
                    # Calculate metrics
                    iv_np = iv_surface.cpu().numpy()
                    results_grid[i, j, 0] = iv_np[iv_np > 0].mean() if (iv_np > 0).any() else 0
                    results_grid[i, j, 1] = iv_np[iv_np > 0].std() if (iv_np > 0).any() else 0
                    results_grid[i, j, 2] = (iv_np == 0).sum() / iv_np.size  # Fraction of zeros
                    
                    if hasattr(pricer, 'last_absorption_stats'):
                        results_grid[i, j, 3] = pricer.last_absorption_stats.mean().item()
                    
                except Exception as e:
                    print(f"\nError at H={H:.3f}, xi0={xi0:.3f}: {e}")
                    results_grid[i, j, :] = np.nan
                
                pbar.update(1)
        
        pbar.close()
        
        return results_grid, H_range, xi0_range
    
    def plot_parameter_sweep(self, results_grid, H_range, xi0_range, save_path=None):
        """Displays results of the parameter sweep"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics = ['Mean IV', 'Std IV', 'Zero Fraction', 'Mean Absorption']
        cmaps = ['viridis', 'plasma', 'Reds', 'Reds']
        
        for idx, (ax, metric, cmap) in enumerate(zip(axes.flat, metrics, cmaps)):
            data = results_grid[:, :, idx].T
            
            im = ax.imshow(data, origin='lower', aspect='auto', cmap=cmap,
                          extent=[H_range[0], H_range[-1], xi0_range[0], xi0_range[-1]])
            
            ax.set_xlabel('H')
            ax.set_ylabel('xi0')
            ax.set_title(f'{metric} - Long Term')
            
            plt.colorbar(im, ax=ax)
            
            # Add contours for problematic values
            if idx == 2:  # Zero fraction
                contours = ax.contour(H_range, xi0_range, data, levels=[0.1, 0.3, 0.5],
                                    colors='black', linewidths=1)
                ax.clabel(contours, inline=True, fontsize=8)
            elif idx == 3:  # Absorption
                contours = ax.contour(H_range, xi0_range, data, levels=[0.3, 0.5, 0.7],
                                    colors='black', linewidths=1)
                ax.clabel(contours, inline=True, fontsize=8)
        
        plt.suptitle('Long Term Regime - Parameter Analysis', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def recommend_safe_parameters_long(self, results_grid, H_range, xi0_range, 
                                     max_absorption=0.3, max_zeros=0.1):
        """
        Recommends safe parameters for long term
        """
        print("\n" + "="*60)
        print("SAFE PARAMETER RECOMMENDATIONS FOR LONG TERM")
        print("="*60)
        
        # Find safe regions
        safe_mask = (results_grid[:, :, 2] < max_zeros) & \
                   (results_grid[:, :, 3] < max_absorption) & \
                   ~np.isnan(results_grid[:, :, 0])
        
        if safe_mask.any():
            safe_H_idx, safe_xi0_idx = np.where(safe_mask)
            
            H_min = H_range[safe_H_idx].min()
            H_max = H_range[safe_H_idx].max()
            xi0_min = xi0_range[safe_xi0_idx].min()
            xi0_max = xi0_range[safe_xi0_idx].max()
            
            print(f"Safe ranges (absorption < {max_absorption:.0%}, zeros < {max_zeros:.0%}):")
            print(f"  H:   [{H_min:.3f}, {H_max:.3f}]")
            print(f"  xi0: [{xi0_min:.3f}, {xi0_max:.3f}]")
            
            # Find optimal configuration
            safe_results = results_grid[safe_mask]
            
            # Optimize for: low absorption, few zeros, stable IV
            score = -safe_results[:, 3] - safe_results[:, 2] - 0.1 * safe_results[:, 1]
            best_idx = np.argmax(score)
            
            best_flat_idx = np.where(safe_mask.ravel())[0][best_idx]
            best_i, best_j = np.unravel_index(best_flat_idx, safe_mask.shape)
            
            print(f"\nOptimal configuration:")
            print(f"  H = {H_range[best_i]:.3f}")
            print(f"  xi0 = {xi0_range[best_j]:.3f}")
            print(f"  Mean absorption: {results_grid[best_i, best_j, 3]:.1%}")
            print(f"  Zero fraction: {results_grid[best_i, best_j, 2]:.1%}")
            
        else:
            print("No safe region found with current criteria!")
            print("Consider:")
            print("  - Relaxing absorption threshold")
            print("  - Increasing minimum xi0")
            print("  - Reducing strike range for long maturities")


# Main function to run the analysis
def analyze_long_term_regime(device='cpu'):
    """
    Analisi completa del regime long term
    """
    # Create pricer for long term
    from deepLearningVolatility.stochastic import generate_rough_bergomi
    
    analyzer = LongTermRegimeAnalyzer(device)
    
    # Create a GridNetworkPricer for long term
    # (assuming you already have the class with absorption modifications)
    rb_process = ProcessFactory.create('rough_bergomi', spot=1.0)
    long_pricer = GridNetworkPricer(
        maturities=analyzer.long_maturities,
        logK=analyzer.long_logK,
        process=rb_process,
        hidden_layers=[64, 32],  # Rete più piccola per test
        device=device
    )
    
    # Test 1: Single configuration comparison
    print("="*60)
    print("TEST 1: Single Configuration Analysis")
    print("="*60)
    
    test_thetas = [
        [0.25, 1.8, -0.5, 0.08],
        [0.15, 2.0, -0.7, 0.15],
        [0.3, 1.5, -0.4, 0.15],
    ]
    
    all_results = []
    for theta in test_thetas:
        results = analyzer.test_single_configuration(
            long_pricer, theta, n_paths=30000, verbose=True
        )
        all_results.append(results)
        analyzer.plot_comparison(results)
    
    # Test 2: Parameter sweep
    print("\n" + "="*60)
    print("TEST 2: Parameter Sweep")
    print("="*60)
    
    results_grid, H_range, xi0_range = analyzer.parameter_sweep_long_term(
        long_pricer, n_grid=15
    )
    
    analyzer.plot_parameter_sweep(results_grid, H_range, xi0_range)
    analyzer.recommend_safe_parameters_long(results_grid, H_range, xi0_range)
    
    return analyzer, all_results, results_grid


# Example usage
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Run analysis
    analyzer, results, param_grid = analyze_long_term_regime(device)
    
    # Save results
    import pickle
    with open('long_term_analysis_results.pkl', 'wb') as f:
        pickle.dump({
            'single_configs': results,
            'parameter_grid': param_grid,
            'device': device
        }, f)
    
    print("\nAnalysis complete! Results saved to 'long_term_analysis_results.pkl'")