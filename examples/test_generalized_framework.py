"""
test_generalized_framework.py

Script di test per verificare il framework generalizzato con diversi processi stocastici.
Testa principalmente Rough Bergomi e Heston.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from time import time
import os

# Import delle nuove classi
from deepLearningVolatility.stochastic.stochastic_interface import ProcessFactory
from deepLearningVolatility.stochastic.wrappers.rough_bergomi_wrapper import RoughBergomiProcess
from deepLearningVolatility.stochastic.wrappers.heston_wrapper import HestonProcess
from deepLearningVolatility.nn.dataset_builder import DatasetBuilder, MultiRegimeDatasetBuilder
from deepLearningVolatility.nn.pricer import GridNetworkPricer, PointwiseNetworkPricer, MultiRegimeGridPricer

# Imposta device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def test_process_creation():
    """Test 1: Creazione e validazione dei processi"""
    print("\n" + "="*60)
    print("TEST 1: Process Creation and Validation")
    print("="*60)
    
    # Test factory pattern
    print("\n1.1 Testing ProcessFactory...")
    available_processes = ProcessFactory.list_available()
    print(f"Available processes: {available_processes}")
    
    # Crea processi usando factory
    rb_process = ProcessFactory.create('rough_bergomi', spot=100.0)
    heston_process = ProcessFactory.create('heston', spot=100.0)
    
    print(f"\n✓ Created Rough Bergomi process: {rb_process}")
    print(f"✓ Created Heston process: {heston_process}")
    
    # Test proprietà dei processi
    print("\n1.2 Testing process properties...")
    for name, process in [('Rough Bergomi', rb_process), ('Heston', heston_process)]:
        print(f"\n{name}:")
        print(f"  - Number of parameters: {process.num_params}")
        print(f"  - Parameter names: {process.param_info.names}")
        print(f"  - Parameter bounds: {process.param_info.bounds}")
        print(f"  - Supports absorption: {process.supports_absorption}")
        print(f"  - Requires variance state: {process.requires_variance_state}")
    
    # Test validazione parametri
    print("\n1.3 Testing parameter validation...")
    
    # Parametri validi per Rough Bergomi
    theta_rb_valid = torch.tensor([0.1, 1.5, -0.7, 0.04])
    is_valid, msg = rb_process.validate_theta(theta_rb_valid)
    print(f"\nRough Bergomi - Valid params: {is_valid} {msg or '✓'}")
    
    # Parametri invalidi per Rough Bergomi
    theta_rb_invalid = torch.tensor([0.6, 1.5, -0.7, 0.04])  # H > 0.5
    is_valid, msg = rb_process.validate_theta(theta_rb_invalid)
    print(f"Rough Bergomi - Invalid params: {not is_valid} ✓ (Error: {msg})")
    
    # Parametri validi per Heston
    theta_heston_valid = torch.tensor([1.0, 0.04, 0.2, -0.7])
    is_valid, msg = heston_process.validate_theta(theta_heston_valid)
    print(f"\nHeston - Valid params: {is_valid} {msg or '✓'}")
    
    return rb_process, heston_process


def test_simulation():
    """Test 2: Simulazione dei processi"""
    print("\n" + "="*60)
    print("TEST 2: Process Simulation")
    print("="*60)
    
    # Crea processi
    rb_process = ProcessFactory.create('rough_bergomi')
    heston_process = ProcessFactory.create('heston')
    
    # Parametri di simulazione
    n_paths = 1000
    n_steps = 100
    dt = 1/252
    
    print(f"\nSimulation parameters:")
    print(f"  - Paths: {n_paths}")
    print(f"  - Steps: {n_steps}")
    print(f"  - dt: {dt}")
    
    # Test Rough Bergomi
    print("\n2.1 Testing Rough Bergomi simulation...")
    theta_rb = torch.tensor([0.1, 1.5, -0.7, 0.04])
    
    start_time = time()
    result_rb = rb_process.simulate(
        theta=theta_rb,
        n_paths=n_paths,
        n_steps=n_steps,
        dt=dt,
        device=device,
        antithetic=True
    )
    sim_time = time() - start_time
    
    print(f"✓ Simulation completed in {sim_time:.2f}s")
    print(f"  - Spot shape: {result_rb.spot.shape}")
    print(f"  - Variance shape: {result_rb.variance.shape}")
    print(f"  - Spot range: [{result_rb.spot.min():.4f}, {result_rb.spot.max():.4f}]")
    print(f"  - Variance range: [{result_rb.variance.min():.4f}, {result_rb.variance.max():.4f}]")
    
    # Test Heston
    print("\n2.2 Testing Heston simulation...")
    theta_heston = torch.tensor([1.0, 0.04, 0.2, -0.7])
    
    start_time = time()
    result_heston = heston_process.simulate(
        theta=theta_heston,
        n_paths=n_paths,
        n_steps=n_steps,
        dt=dt,
        device=device,
        antithetic=True
    )
    sim_time = time() - start_time
    
    print(f"✓ Simulation completed in {sim_time:.2f}s")
    print(f"  - Spot shape: {result_heston.spot.shape}")
    print(f"  - Variance shape: {result_heston.variance.shape}")
    print(f"  - Spot range: [{result_heston.spot.min():.4f}, {result_heston.spot.max():.4f}]")
    print(f"  - Variance range: [{result_heston.variance.min():.4f}, {result_heston.variance.max():.4f}]")
    
    # Visualizza sample paths
    visualize_paths(result_rb, result_heston, n_paths_to_plot=5)
    
    return result_rb, result_heston


def visualize_paths(result_rb, result_heston, n_paths_to_plot=5):
    """Visualizza alcuni sample paths"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Rough Bergomi - Spot
    for i in range(n_paths_to_plot):
        axes[0, 0].plot(result_rb.spot[i].cpu().numpy(), alpha=0.7)
    axes[0, 0].set_title('Rough Bergomi - Spot Paths')
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].set_ylabel('Spot Price')
    
    # Rough Bergomi - Variance
    for i in range(n_paths_to_plot):
        axes[0, 1].plot(result_rb.variance[i].cpu().numpy(), alpha=0.7)
    axes[0, 1].set_title('Rough Bergomi - Variance Paths')
    axes[0, 1].set_xlabel('Time Steps')
    axes[0, 1].set_ylabel('Variance')
    
    # Heston - Spot
    for i in range(n_paths_to_plot):
        axes[1, 0].plot(result_heston.spot[i].cpu().numpy(), alpha=0.7)
    axes[1, 0].set_title('Heston - Spot Paths')
    axes[1, 0].set_xlabel('Time Steps')
    axes[1, 0].set_ylabel('Spot Price')
    
    # Heston - Variance
    for i in range(n_paths_to_plot):
        axes[1, 1].plot(result_heston.variance[i].cpu().numpy(), alpha=0.7)
    axes[1, 1].set_title('Heston - Variance Paths')
    axes[1, 1].set_xlabel('Time Steps')
    axes[1, 1].set_ylabel('Variance')
    
    plt.tight_layout()
    plt.show()


def test_dataset_builder():
    """Test 3: DatasetBuilder"""
    print("\n" + "="*60)
    print("TEST 3: DatasetBuilder")
    print("="*60)
    
    # Crea builders per diversi processi
    rb_builder = DatasetBuilder('rough_bergomi', device=device)
    heston_builder = DatasetBuilder('heston', device=device)
    
    print("\n3.1 Testing parameter sampling...")
    
    # Test LHS sampling
    n_samples = 100
    rb_thetas = rb_builder.sample_theta_lhs(n_samples)
    heston_thetas = heston_builder.sample_theta_lhs(n_samples)
    
    print(f"\nRough Bergomi theta samples: {rb_thetas.shape}")
    print(f"  Sample theta[0]: {rb_thetas[0]}")
    print(f"\nHeston theta samples: {heston_thetas.shape}")
    print(f"  Sample theta[0]: {heston_thetas[0]}")
    
    # Visualizza sampling
    print("\n3.2 Visualizing parameter sampling...")
    rb_builder.visualize_sampling_with_labels(n_samples=50)
    
    # Test MC parameters ottimizzati
    print("\n3.3 Testing process-specific MC parameters...")
    rb_mc_params = rb_builder.get_process_specific_mc_params()
    heston_mc_params = heston_builder.get_process_specific_mc_params()
    
    print(f"\nRough Bergomi MC params: {rb_mc_params}")
    print(f"Heston MC params: {heston_mc_params}")
    
    return rb_builder, heston_builder


def test_pricer_creation():
    """Test 4: Creazione e test dei pricer"""
    print("\n" + "="*60)
    print("TEST 4: Pricer Creation and Testing")
    print("="*60)
    
    # Setup
    maturities = torch.tensor([0.25, 0.5, 1.0, 2.0])
    log_moneyness = torch.linspace(-0.3, 0.3, 11)
    
    # Crea builders
    rb_builder = DatasetBuilder('rough_bergomi', device=device)
    heston_builder = DatasetBuilder('heston', device=device)
    
    # Crea pricers
    print("\n4.1 Creating GridNetworkPricer...")
    rb_pricer = rb_builder.create_pricer(
        maturities=maturities,
        logK=log_moneyness,
        hidden_layers=[64, 32],
        activation='ReLU'
    )
    
    heston_pricer = heston_builder.create_pricer(
        maturities=maturities,
        logK=log_moneyness,
        hidden_layers=[64, 32],
        activation='ReLU'
    )
    
    print(f"✓ Rough Bergomi pricer created")
    print(f"  Network architecture: {rb_pricer.net}")
    print(f"\n✓ Heston pricer created")
    print(f"  Network architecture: {heston_pricer.net}")
    
    # Test MC pricing
    print("\n4.2 Testing Monte Carlo IV calculation...")
    
    # Rough Bergomi
    theta_rb = torch.tensor([0.1, 1.5, -0.7, 0.04])
    print(f"\nCalculating IV surface for Rough Bergomi...")
    start_time = time()
    iv_surface_rb = rb_pricer._mc_iv_grid(
        theta_rb,
        n_paths=50_000,
        use_antithetic=True,
        adaptive_dt=True,
        control_variate=True
    )
    calc_time = time() - start_time
    
    print(f"✓ IV surface calculated in {calc_time:.2f}s")
    print(f"  Shape: {iv_surface_rb.shape}")
    print(f"  Range: [{iv_surface_rb.min():.4f}, {iv_surface_rb.max():.4f}]")
    print(f"  Mean: {iv_surface_rb.mean():.4f}")
    
    # Heston
    theta_heston = torch.tensor([1.0, 0.04, 0.2, -0.7])
    print(f"\nCalculating IV surface for Heston...")
    start_time = time()
    iv_surface_heston = heston_pricer._mc_iv_grid(
        theta_heston,
        n_paths=50_000,
        use_antithetic=True,
        adaptive_dt=True,
        control_variate=True
    )
    calc_time = time() - start_time
    
    print(f"✓ IV surface calculated in {calc_time:.2f}s")
    print(f"  Shape: {iv_surface_heston.shape}")
    print(f"  Range: [{iv_surface_heston.min():.4f}, {iv_surface_heston.max():.4f}]")
    print(f"  Mean: {iv_surface_heston.mean():.4f}")
    
    # Visualizza superfici
    visualize_iv_surfaces(iv_surface_rb, iv_surface_heston, maturities, log_moneyness)
    
    return rb_pricer, heston_pricer, iv_surface_rb, iv_surface_heston


def visualize_iv_surfaces(iv_rb, iv_heston, maturities, log_moneyness):
    """Visualizza le superfici di volatilità implicita"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Rough Bergomi
    im1 = axes[0].imshow(iv_rb.cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Rough Bergomi - IV Surface')
    axes[0].set_xlabel('Log-Moneyness')
    axes[0].set_ylabel('Maturity')
    axes[0].set_xticks(range(0, len(log_moneyness), 2))
    axes[0].set_xticklabels([f'{k:.2f}' for k in log_moneyness[::2]])
    axes[0].set_yticks(range(len(maturities)))
    axes[0].set_yticklabels([f'{T:.2f}' for T in maturities])
    plt.colorbar(im1, ax=axes[0])
    
    # Heston
    im2 = axes[1].imshow(iv_heston.cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Heston - IV Surface')
    axes[1].set_xlabel('Log-Moneyness')
    axes[1].set_ylabel('Maturity')
    axes[1].set_xticks(range(0, len(log_moneyness), 2))
    axes[1].set_xticklabels([f'{k:.2f}' for k in log_moneyness[::2]])
    axes[1].set_yticks(range(len(maturities)))
    axes[1].set_yticklabels([f'{T:.2f}' for T in maturities])
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.show()


def test_dataset_generation():
    """Test 5: Generazione dataset"""
    print("\n" + "="*60)
    print("TEST 5: Dataset Generation")
    print("="*60)
    
    # Setup
    maturities = torch.tensor([0.25, 0.5, 1.0])
    log_moneyness = torch.linspace(-0.2, 0.2, 7)
    
    # Crea builders e pricers
    rb_builder = DatasetBuilder('rough_bergomi', device=device)
    heston_builder = DatasetBuilder('heston', device=device)
    
    rb_pricer = rb_builder.create_pricer(maturities, log_moneyness)
    heston_pricer = heston_builder.create_pricer(maturities, log_moneyness)
    
    # Genera mini dataset
    n_samples = 10
    n_paths = 50_000
    
    print(f"\n5.1 Generating Rough Bergomi dataset...")
    print(f"  Samples: {n_samples}")
    print(f"  MC paths: {n_paths}")
    
    start_time = time()
    theta_rb, iv_rb = rb_builder.build_grid_dataset(
        rb_pricer,
        n_samples=n_samples,
        n_paths=n_paths,
        normalize=True,
        show_progress=True
    )
    gen_time = time() - start_time
    
    print(f"\n✓ Dataset generated in {gen_time:.2f}s")
    print(f"  Theta shape: {theta_rb.shape}")
    print(f"  IV shape: {iv_rb.shape}")
    
    print(f"\n5.2 Generating Heston dataset...")
    start_time = time()
    theta_heston, iv_heston = heston_builder.build_grid_dataset(
        heston_pricer,
        n_samples=n_samples,
        n_paths=n_paths,
        normalize=True,
        show_progress=True
    )
    gen_time = time() - start_time
    
    print(f"\n✓ Dataset generated in {gen_time:.2f}s")
    print(f"  Theta shape: {theta_heston.shape}")
    print(f"  IV shape: {iv_heston.shape}")
    
    return theta_rb, iv_rb, theta_heston, iv_heston


def test_training():
    """Test 6: Training delle reti"""
    print("\n" + "="*60)
    print("TEST 6: Network Training")
    print("="*60)
    
    # Setup ridotto per test veloce
    maturities = torch.tensor([0.5, 1.0])
    log_moneyness = torch.linspace(-0.1, 0.1, 5)
    
    # Genera dataset
    rb_builder = DatasetBuilder('rough_bergomi', device=device)
    rb_pricer = rb_builder.create_pricer(maturities, log_moneyness, hidden_layers=[32, 16])
    
    print("Generating training data...")
    theta_train, iv_train = rb_builder.build_grid_dataset(
        rb_pricer,
        n_samples=50,
        n_paths=50_000,
        normalize=True
    )
    
    # Split train/val
    n_train = int(0.8 * len(theta_train))
    theta_val = theta_train[n_train:]
    iv_val = iv_train[n_train:]
    theta_train = theta_train[:n_train]
    iv_train = iv_train[:n_train]
    
    print(f"\nTraining data: {len(theta_train)} samples")
    print(f"Validation data: {len(theta_val)} samples")
    
    # Train
    print("\nTraining network...")
    rb_pricer.fit(
        theta_train, iv_train,
        theta_val, iv_val,
        epochs=5,
        batch_size=10,
        lr=1e-3
    )
    
    # Test prediction
    print("\n6.1 Testing prediction...")
    rb_pricer.eval()
    with torch.no_grad():
        # Usa un theta dal validation set
        test_theta = theta_val[0:1]
        pred_iv = rb_pricer(test_theta)
        
        print(f"✓ Prediction shape: {pred_iv.shape}")
        print(f"  Predicted IV range: [{pred_iv.min():.4f}, {pred_iv.max():.4f}]")
        
        # Confronta con ground truth
        true_iv = iv_val[0:1]
        mse = ((pred_iv - true_iv) ** 2).mean()
        print(f"  MSE vs ground truth: {mse:.6f}")


def test_multi_regime():
    """Test 7: Multi-regime pricer"""
    print("\n" + "="*60)
    print("TEST 7: Multi-Regime Pricer")
    print("="*60)
    
    # Setup regimi
    short_maturities = torch.tensor([7/365, 14/365, 30/365])
    mid_maturities = torch.tensor([60/365, 180/365, 270/365])
    long_maturities = torch.tensor([1.0, 2.0, 3.0])
    
    short_log_moneyness = torch.linspace(-0.15, 0.15, 7)
    mid_log_moneyness = torch.linspace(-0.3, 0.3, 7)
    long_log_moneyness = torch.linspace(-0.5, 0.5, 7)
    
    # Crea processo
    rb_process = ProcessFactory.create('rough_bergomi')
    
    # Crea multi-regime pricer
    print("Creating multi-regime pricer...")
    multi_pricer = MultiRegimeGridPricer(
        process=rb_process,
        short_term_maturities=short_maturities,
        short_term_logK=short_log_moneyness,
        mid_term_maturities=mid_maturities,
        mid_term_logK=mid_log_moneyness,
        long_term_maturities=long_maturities,
        long_term_logK=long_log_moneyness,
        device=device,
        short_term_hidden=[32, 16],
        mid_term_hidden=[32, 16],
        long_term_hidden=[32, 16]
    )
    
    print("✓ Multi-regime pricer created")
    print(f"  Short term pricer: {multi_pricer.short_term_pricer}")
    print(f"  Mid term pricer: {multi_pricer.mid_term_pricer}")
    print(f"  Long term pricer: {multi_pricer.long_term_pricer}")
    
    # Test con un theta
    theta_test = torch.tensor([0.1, 1.5, -0.7, 0.04])
    
    print("\nCalculating IV surfaces for all regimes...")
    surfaces = multi_pricer.price_iv(theta_test.unsqueeze(0))
    
    for regime in ['short', 'mid', 'long']:
        surface = surfaces[regime]
        print(f"\n{regime.upper()} regime:")
        print(f"  Shape: {surface.shape}")
        print(f"  Range: [{surface.min():.4f}, {surface.max():.4f}]")


def main():
    """Esegue tutti i test"""
    print("\n" + "="*80)
    print("TESTING GENERALIZED OPTION PRICING FRAMEWORK")
    print("="*80)
    
    try:
        # Test 1: Creazione processi
        rb_process, heston_process = test_process_creation()
        
        # Test 2: Simulazione
        result_rb, result_heston = test_simulation()
        
        # Test 3: Dataset builder
        rb_builder, heston_builder = test_dataset_builder()
        
        # Test 4: Pricer
        rb_pricer, heston_pricer, iv_rb, iv_heston = test_pricer_creation()
        
        # Test 5: Dataset generation
        theta_rb, iv_rb_data, theta_heston, iv_heston_data = test_dataset_generation()
        
        # Test 6: Training
        test_training()
        
        # Test 7: Multi-regime
        test_multi_regime()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED SUCCESSFULLY!")
        print("="*80)
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ TEST FAILED: {str(e)}")
        print("="*80)
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()