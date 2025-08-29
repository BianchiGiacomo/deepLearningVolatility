import torch
import numpy as np
import matplotlib.pyplot as plt
from deepLearningVolatility.stochastic import generate_rough_bergomi
from deepLearningVolatility.nn.modules import BlackScholes
from deepLearningVolatility.instruments import EuropeanOption, BrownianStock
import time
from scipy import stats as scipy_stats

class MonteCarloDebuggerOptimized:
    """Optimized tool for debugging Monte Carlo on short maturities"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device) if isinstance(device, str) else device

    def adaptive_dt_selection(self, T, H, eta, base_steps_per_year=365):
        """
        Selects dt based on defined regimes:
        - SHORT: T ≤ 30 days (1 month) -> dt = 3e-5
        - MID: 30 days < T ≤ 1 year -> dt = 1e-4
        - LONG: T > 1 year -> dt = 1/365
        """
        T_years = T.item() if torch.is_tensor(T) else T
        
        # Determine the regime and corresponding fixed dt
        if T_years <= 30/365:  # SHORT: ≤ 1 month
            dt = 3e-5
            regime = "SHORT"
        elif T_years <= 1.0:  # MID: ≤ 1 year
            dt = 5e-4
            regime = "MID"
        else:  # LONG: > 1 year
            dt = 1/365
            regime = "LONG"
        
        # Calculate number of steps
        n_steps = max(2, int(T_years / dt))
        
        # Regime log for debugging
        T_days = T_years * 365
        print(f"    Regime: {regime} (T={T_days:.1f} days) -> dt={dt:.2e}, n_steps={n_steps}")
        
        return dt, n_steps
    
    def compute_iv_with_confidence(self, theta, T, logK, n_paths=50000, 
                                   n_batches=10, confidence_level=0.95):
        """
        Calcola IV con intervalli di confidenza usando batching
        """
        H, eta, rho, xi0 = theta
        spot = 1.0
        
        # Adaptive dt selection
        dt, n_steps = self.adaptive_dt_selection(T, H, eta)
        
        print(f"  T={T:.4f}: usando dt={dt:.6f} ({n_steps} steps)")
        
        batch_ivs = []
        batch_size = n_paths // n_batches
        
        for batch in range(n_batches):
            # Generate paths for this batch
            S, _ = generate_rough_bergomi(
                batch_size, n_steps,
                init_state=(spot, xi0),
                alpha=H - 0.5,
                rho=rho,
                eta=eta,
                xi=xi0,
                dt=dt,
                device=self.device,
                antithetic=True
            )
            
            ST = S[:, -1]
            
            # Calcola IV per ogni strike
            batch_iv = torch.zeros_like(logK)
            
            for j, k in enumerate(logK):
                K = spot * np.exp(k.item() if torch.is_tensor(k) else k)
                
                # Compute price with control variate
                payoff = (ST - K).clamp(min=0.0)
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
                
                # Compute IV
                if price > 1e-10:
                    try:
                        bs = BlackScholes(
                            EuropeanOption(
                                BrownianStock(),
                                strike=K,
                                maturity=T.item() if torch.is_tensor(T) else T
                            )
                        )
                        iv = bs.implied_volatility(
                            log_moneyness=-k.item() if torch.is_tensor(k) else -k,
                            time_to_maturity=T.item() if torch.is_tensor(T) else T,
                            price=price
                        )
                        batch_iv[j] = iv
                    except:
                        batch_iv[j] = np.sqrt(xi0)
                else:
                    batch_iv[j] = np.sqrt(xi0)
            
            batch_ivs.append(batch_iv)
        
        # Compute statistics
        batch_ivs = torch.stack(batch_ivs)
        mean_iv = batch_ivs.mean(dim=0)
        std_iv = batch_ivs.std(dim=0)
        
        # Confidence intervals
        z_score = scipy_stats.norm.ppf((1 + confidence_level) / 2)
        ci_lower = mean_iv - z_score * std_iv / np.sqrt(n_batches)
        ci_upper = mean_iv + z_score * std_iv / np.sqrt(n_batches)
        
        return mean_iv, std_iv, ci_lower, ci_upper, dt, n_steps
    
    def analyze_convergence(self, theta, maturities, logK, 
                           path_schedule=[10000, 25000, 50000, 100000, 200000]):
        """
        Analyzes convergence as the number of paths varies
        with adaptive dt for each maturity
        """
        H, eta, rho, xi0 = theta
        
        results = {}
        
        print(f"\nAnalyzing convergence for theta: H={H:.3f}, eta={eta:.3f}, rho={rho:.3f}, xi0={xi0:.3f}")
        print("="*80)
        
        for T in maturities:
            T_val = T.item() if torch.is_tensor(T) else T
            results[T_val] = {}
            
            print(f"\nMaturity T = {T_val:.4f} years ({T_val*365:.1f} days)")
            
            # Determine optimal dt for this maturity
            dt_opt, n_steps_opt = self.adaptive_dt_selection(T_val, H, eta)
            print(f"Optimal dt = {dt_opt:.6f} ({n_steps_opt} steps)")
            
            for n_paths in path_schedule:
                start_time = time.time()
                
                mean_iv, std_iv, ci_lower, ci_upper, dt_used, n_steps_used = \
                    self.compute_iv_with_confidence(theta, T, logK, n_paths, n_batches=10)
                
                elapsed = time.time() - start_time
                
                # Compute quality metrics
                avg_ci_width = (ci_upper - ci_lower).mean().item()
                max_ci_width = (ci_upper - ci_lower).max().item()
                
                results[T_val][n_paths] = {
                    'mean_iv': mean_iv,
                    'std_iv': std_iv,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'dt': dt_used,
                    'n_steps': n_steps_used,
                    'time': elapsed,
                    'avg_ci_width': avg_ci_width,
                    'max_ci_width': max_ci_width,
                    'coefficient_variation': (std_iv / mean_iv).mean().item()
                }
                
                print(f"  n_paths={n_paths:6d}: time={elapsed:5.2f}s, "
                      f"avg_CI_width={avg_ci_width:.4f}, CV={results[T_val][n_paths]['coefficient_variation']:.4f}")
        
        return results
    
    def compare_discretization_schemes(self, theta, T, logK, n_paths=50000):
        """
        Compares the dt of the three regimes for a given maturity
        """
        H, eta, rho, xi0 = theta
        spot = 1.0
        T_val = T.item() if torch.is_tensor(T) else T
        
        # Test only the dt of the defined regimes
        schemes = {
            'regime_short_dt': (3e-5, int(T_val / 3e-5)),
            'regime_mid_dt': (1e-4, int(T_val / 1e-4)),
            'regime_long_dt': (1/365, int(T_val / (1/365))),
            'adaptive': self.adaptive_dt_selection(T_val, H, eta)
        }
        
        # Filter schemes with too many steps
        schemes = {k: v for k, v in schemes.items() if v[1] <= 10000}
        
        results = {}
        
        print(f"\nComparing regime-specific dt values for T={T_val:.4f} ({T_val*365:.1f} days)")
        print("="*60)
        
        for scheme_name, (dt, n_steps) in schemes.items():
            if n_steps > 10000 and scheme_name == 'ultra_fine':
                print(f"Skipping {scheme_name} (too many steps)")
                continue
                
            print(f"\n{scheme_name}: dt={dt:.8f}, n_steps={n_steps}")
            
            start_time = time.time()
            
            try:
                # Generate paths
                S, _ = generate_rough_bergomi(
                    n_paths, n_steps,
                    init_state=(spot, xi0),
                    alpha=H - 0.5,
                    rho=rho,
                    eta=eta,
                    xi=xi0,
                    dt=dt,
                    device=self.device,
                    antithetic=True
                )
                
                ST = S[:, -1]
                
                # Compute IV
                iv_smile = torch.zeros_like(logK)
                
                for j, k in enumerate(logK):
                    K = spot * np.exp(k.item() if torch.is_tensor(k) else k)
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
                                log_moneyness=-k.item() if torch.is_tensor(k) else -k,
                                time_to_maturity=T_val,
                                price=price
                            )
                            iv_smile[j] = iv
                        except:
                            iv_smile[j] = np.sqrt(xi0)
                    else:
                        iv_smile[j] = np.sqrt(xi0)
                
                elapsed = time.time() - start_time
                
                results[scheme_name] = {
                    'smile': iv_smile,
                    'dt': dt,
                    'n_steps': n_steps,
                    'time': elapsed,
                    'mean_iv': iv_smile.mean().item(),
                    'std_iv': iv_smile.std().item()
                }
                
                print(f"  Time: {elapsed:.2f}s, Mean IV: {results[scheme_name]['mean_iv']:.4f}")
                
            except Exception as e:
                print(f"  Error: {e}")
                results[scheme_name] = None
        
        return results
    
    def visualize_analysis(self, convergence_results, discretization_results, 
                          maturities, logK, save_path=None):
        """
        Visualizes analysis results
        """
        # Plot 1: Convergence with confidence intervals
        n_maturities = len(maturities)
        fig1, axes = plt.subplots(1, n_maturities, figsize=(5*n_maturities, 5))
        
        if n_maturities == 1:
            axes = [axes]
        
        for idx, T in enumerate(maturities):
            T_val = T.item() if torch.is_tensor(T) else T
            ax = axes[idx]
            
            if T_val not in convergence_results:
                continue
            
            # Take the last (best) result for the plot
            n_paths_list = sorted(convergence_results[T_val].keys())
            best_n_paths = n_paths_list[-1]
            result = convergence_results[T_val][best_n_paths]
            
            mean_iv = result['mean_iv'].cpu().numpy()
            ci_lower = result['ci_lower'].cpu().numpy()
            ci_upper = result['ci_upper'].cpu().numpy()
            logK_np = logK.cpu().numpy() if torch.is_tensor(logK) else logK
            
            ax.plot(logK_np, mean_iv, 'b-', linewidth=2, label='Mean IV')
            ax.fill_between(logK_np, ci_lower, ci_upper, alpha=0.3, color='blue', 
                           label=f'95% CI (n={best_n_paths})')
            
            ax.set_xlabel('Log-Moneyness')
            ax.set_ylabel('Implied Volatility')
            ax.set_title(f'T={T_val:.4f} ({T_val*365:.1f} days)\ndt={result["dt"]:.6f}, steps={result["n_steps"]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('IV Smiles with Confidence Intervals', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_confidence.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        # Plot 2: Discretization scheme comparison
        if discretization_results:
            fig2, ax = plt.subplots(figsize=(10, 6))
            
            colors = plt.cm.rainbow(np.linspace(0, 1, len(discretization_results)))
            logK_np = logK.cpu().numpy() if torch.is_tensor(logK) else logK
            
            for (scheme_name, result), color in zip(discretization_results.items(), colors):
                if result is not None:
                    smile = result['smile'].cpu().numpy()
                    ax.plot(logK_np, smile, 'o-', color=color, 
                           label=f'{scheme_name} (dt={result["dt"]:.6f})', 
                           markersize=5)
            
            ax.set_xlabel('Log-Moneyness')
            ax.set_ylabel('Implied Volatility')
            ax.set_title('Comparison of Discretization Schemes')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(f"{save_path}_discretization.png", dpi=150, bbox_inches='tight')
            plt.show()
        
        # Plot 3: Convergence metrics
        fig3, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for T in maturities:
            T_val = T.item() if torch.is_tensor(T) else T
            if T_val not in convergence_results:
                continue
                
            n_paths_list = sorted(convergence_results[T_val].keys())
            times = [convergence_results[T_val][n]['time'] for n in n_paths_list]
            avg_ci_widths = [convergence_results[T_val][n]['avg_ci_width'] for n in n_paths_list]
            max_ci_widths = [convergence_results[T_val][n]['max_ci_width'] for n in n_paths_list]
            cvs = [convergence_results[T_val][n]['coefficient_variation'] for n in n_paths_list]
            
            label = f'T={T_val:.3f}'
            
            axes[0,0].plot(n_paths_list, times, 'o-', label=label)
            axes[0,1].plot(n_paths_list, avg_ci_widths, 'o-', label=label)
            axes[1,0].plot(n_paths_list, max_ci_widths, 'o-', label=label)
            axes[1,1].plot(n_paths_list, cvs, 'o-', label=label)
        
        axes[0,0].set_xlabel('Number of Paths')
        axes[0,0].set_ylabel('Computation Time (s)')
        axes[0,0].set_title('Computation Time')
        axes[0,0].set_xscale('log')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        axes[0,1].set_xlabel('Number of Paths')
        axes[0,1].set_ylabel('Average CI Width')
        axes[0,1].set_title('Average Confidence Interval Width')
        axes[0,1].set_xscale('log')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        axes[1,0].set_xlabel('Number of Paths')
        axes[1,0].set_ylabel('Max CI Width')
        axes[1,0].set_title('Maximum Confidence Interval Width')
        axes[1,0].set_xscale('log')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        axes[1,1].set_xlabel('Number of Paths')
        axes[1,1].set_ylabel('Coefficient of Variation')
        axes[1,1].set_title('Average Coefficient of Variation')
        axes[1,1].set_xscale('log')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.suptitle('Convergence Metrics', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_metrics.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    def recommend_optimal_parameters(self, convergence_results, target_ci_width=0.005, max_time=30):
        """
        Recommends optimal parameters based on results
        """
        print("\n" + "="*80)
        print("OPTIMAL PARAMETER RECOMMENDATIONS")
        print("="*80)
        
        recommendations = {}
        
        for T_val in convergence_results:
            print(f"\nMaturity T = {T_val:.4f} years ({T_val*365:.1f} days):")
            
            best_config = None
            best_score = float('inf')
            
            for n_paths in convergence_results[T_val]:
                result = convergence_results[T_val][n_paths]
                
                # Check constraints
                if result['time'] <= max_time and result['avg_ci_width'] <= target_ci_width:
                    # Score: balance time and accuracy
                    score = result['time'] + 100 * result['avg_ci_width']
                    
                    if score < best_score:
                        best_score = score
                        best_config = {
                            'n_paths': n_paths,
                            'dt': result['dt'],
                            'n_steps': result['n_steps'],
                            'time': result['time'],
                            'avg_ci_width': result['avg_ci_width'],
                            'max_ci_width': result['max_ci_width']
                        }
            
            if best_config:
                print(f"  ✓ Recommended configuration:")
                print(f"    - n_paths: {best_config['n_paths']}")
                print(f"    - dt: {best_config['dt']:.8f}")
                print(f"    - n_steps: {best_config['n_steps']}")
                print(f"    - Expected time: {best_config['time']:.2f}s")
                print(f"    - Avg CI width: {best_config['avg_ci_width']:.5f}")
                print(f"    - Max CI width: {best_config['max_ci_width']:.5f}")
                recommendations[T_val] = best_config
            else:
                print(f"  ⚠ No configuration meets criteria (CI<{target_ci_width}, time<{max_time}s)")
                print(f"    Consider increasing n_paths or relaxing constraints")
        
        return recommendations


# Test functions
def test_short_regime():
    """Test specifico per il regime SHORT (≤ 30 giorni)"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    debugger = MonteCarloDebuggerOptimized(device)
    
    # Typical parameters for SHORT
    # theta = [0.100, 0.500, -0.700, 0.040]  # H, eta, rho, xi0
    theta = [0.35, 2.5, -0.5, 0.15]
    
    # SHORT maturities: all ≤ 30 days
    maturities = torch.tensor([
        7/365,    # 1 settimana
        14/365,   # 2 settimane
        21/365,   # 3 settimane
        30/365    # 1 mese (confine del regime)
    ])
    
    # Log-moneyness restricted for short maturities
    logK = torch.linspace(-0.15, 0.15, 11)
    
    print("="*80)
    print("TESTING SHORT REGIME (≤ 30 days)")
    print("Expected dt = 3e-5 for all maturities")
    print("="*80)
    
    # Test 1: Convergence analysis
    convergence_results = debugger.analyze_convergence(
        theta, 
        maturities, 
        logK,
        path_schedule=[10000, 25000, 50000, 100000]
    )
    
    # Test 2: Discretization comparison (for the shortest maturity)
    discretization_results = debugger.compare_discretization_schemes(
        theta,
        maturities[0],  # 1 settimana
        logK,
        n_paths=50000
    )
    
    # Visualization
    debugger.visualize_analysis(
        convergence_results,
        discretization_results,
        maturities,
        logK,
        save_path="short_regime_analysis"
    )
    
    # Recommendations
    recommendations = debugger.recommend_optimal_parameters(
        convergence_results,
        target_ci_width=0.003,
        max_time=60
    )
    
    return convergence_results, discretization_results, recommendations


def test_mid_regime():
    """Test specific for MID regime (1-6 months)"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    debugger = MonteCarloDebuggerOptimized(device)
    
    # Typical MID regime parameters
    theta = [0.25, 1.2, -0.6, 0.13]  # H, eta, rho, xi0
    
    # MID maturities (1-6 months)
    mid_maturities = torch.tensor([
        30/365,   # 1 month
        60/365,   # 2 months
        90/365,   # 3 months
        120/365,  # 4 months
        180/365   # 6 months
    ])
    
    # Log-moneyness
    logK = torch.linspace(-0.3, 0.3, 15)
    
    print("="*80)
    print("TESTING MID REGIME PARAMETERS")
    print("="*80)
    
    # Test 1: Convergence analysis
    convergence_results = debugger.analyze_convergence(
        theta, 
        mid_maturities, 
        logK,
        path_schedule=[10000, 25000, 50000]
    )
    
    # Test 2: Discretization comparison for 2-month maturity
    discretization_results = debugger.compare_discretization_schemes(
        theta,
        mid_maturities[1],  # 2 months
        logK,
        n_paths=50000
    )
    
    # Visualization
    debugger.visualize_analysis(
        convergence_results,
        discretization_results,
        mid_maturities,
        logK,
        save_path="mid_regime_analysis"
    )
    
    # Recommendations
    recommendations = debugger.recommend_optimal_parameters(
        convergence_results,
        target_ci_width=0.005,  # Slightly relaxed for MID
        max_time=45
    )
    
    print("\n" + "="*80)
    print("MID REGIME OPTIMAL CONFIGURATION:")
    print("="*80)
    print(f"Recommended dt: 1e-4 (10,000 microseconds)")
    print(f"Recommended n_paths: 30,000-50,000")
    print(f"Expected computation time: ~15-30s per surface")
    
    return convergence_results, discretization_results, recommendations


def test_long_regime():
    """Test specific for LONG regime (6 months - 2 years)"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    debugger = MonteCarloDebuggerOptimized(device)
    
    # Typical LONG regime parameters
    theta = [0.35, 2.5, -0.5, 0.15]  # H, eta, rho, xi0
    
    # LONG maturities (6 months - 2 years)
    long_maturities = torch.tensor([
        1.0,      # 1.5 anni
        2.0,      # 2 anni
        3.0,      # 3 anni
        5.0       # 5 anni
    ])
    
    # Log-moneyness (wider for longer maturities)
    logK = torch.linspace(-0.5, 0.5, 15)
    
    print("="*80)
    print("TESTING LONG REGIME PARAMETERS")
    print("="*80)
    
    # Test 1: Convergence analysis
    convergence_results = debugger.analyze_convergence(
        theta, 
        long_maturities, 
        logK,
        path_schedule=[5000, 10000, 20000, 30000]
    )
    
    # Test 2: Discretization comparison for 1-year maturity
    discretization_results = debugger.compare_discretization_schemes(
        theta,
        long_maturities[0],  # 1 year
        logK,
        n_paths=50000
    )
    
    # Visualization
    debugger.visualize_analysis(
        convergence_results,
        discretization_results,
        long_maturities,
        logK,
        save_path="long_regime_analysis"
    )
    
    # Recommendations
    recommendations = debugger.recommend_optimal_parameters(
        convergence_results,
        target_ci_width=0.01,  # More relaxed for LONG
        max_time=60
    )
    
    print("\n" + "="*80)
    print("LONG REGIME OPTIMAL CONFIGURATION:")
    print("="*80)
    print(f"Recommended dt: 1/365 (daily)")
    print(f"Recommended n_paths: 20,000-30,000")
    print(f"Expected computation time: ~10-20s per surface")
    
    return convergence_results, discretization_results, recommendations


def test_all_regimes_comparison():
    """Compare all three regimes side by side"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    debugger = MonteCarloDebuggerOptimized(device)
    
    # Common theta for comparison
    theta = [0.15, 1.5, -0.6, 0.09]
    
    # Representative maturity from each regime
    test_maturities = torch.tensor([
        14/365,   # SHORT: 2 weeks
        90/365,   # MID: 3 months  
        365/365   # LONG: 1 year
    ])
    
    logK = torch.linspace(-0.3, 0.3, 15)
    
    # Test each maturity with different dt values
    dt_values = [3e-5, 1e-4, 1/365, 1/730]
    n_paths = 30000
    
    results = {}
    
    for T in test_maturities:
        T_days = int(T.item() * 365)
        results[T_days] = {}
        
        for dt in dt_values:
            n_steps = max(2, int(T.item() / dt))
            
            if n_steps > 10000:  # Skip if too many steps
                continue
            
            print(f"\nTesting T={T_days} days with dt={dt:.6f} ({n_steps} steps)")
            
            mean_iv, std_iv, ci_lower, ci_upper, _, _ = \
                debugger.compute_iv_with_confidence(
                    theta, T, logK, n_paths, n_batches=10
                )
            
            results[T_days][dt] = {
                'mean_iv': mean_iv,
                'std_iv': std_iv,
                'ci_width': (ci_upper - ci_lower).mean().item()
            }
    
    # Print comparison table
    print("\n" + "="*80)
    print("DT COMPARISON ACROSS REGIMES")
    print("="*80)
    print(f"{'Maturity':<15} {'dt':<15} {'Mean IV':<15} {'CI Width':<15}")
    print("-"*60)
    
    for T_days, dt_results in results.items():
        for dt, stats in dt_results.items():
            print(f"{T_days} days{'':<7} {dt:<15.6f} {stats['mean_iv'].mean():<15.4f} {stats['ci_width']:<15.5f}")
    
    return results


# Usage example
if __name__ == "__main__":
    # Test each regime
    print("\n" + "="*80)
    print("MULTI-REGIME MONTE CARLO OPTIMIZATION")
    print("="*80)
    
    # 1. SHORT regime test
    short_conv, short_disc, short_rec = test_short_regime()
    
    # 2. MID regime test
    mid_conv, mid_disc, mid_rec = test_mid_regime()
    
    # 3. LONG regime test
    long_conv, long_disc, long_rec = test_long_regime()
    
    # 4. Comparison across regimes
    # comparison = test_all_regimes_comparison()
    
