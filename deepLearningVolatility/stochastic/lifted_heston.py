"""
Implementazione corretta del modello Lifted Heston con schema Moment Matching.
Basato sul Capitolo 3.1 della tesi di Bertolo.
"""
import torch
import numpy as np
from typing import Optional, Tuple, NamedTuple
from deepLearningVolatility._utils.typing import TensorOrScalar
from deepLearningVolatility.stochastic._utils import cast_state


class SpotVarianceTuple(NamedTuple):
    spot: torch.Tensor
    variance: torch.Tensor


def generate_lifted_heston(
    n_paths: int,
    n_steps: int,
    init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
    n_factors: int = 3,      # Numero di fattori N nell'approssimazione
    kappa: float = 0.3,      # λ nella tesi (mean reversion)
    theta: float = 0.02,     # θ nella tesi (long-term variance)
    sigma: float = 0.3,      # σ nella tesi (vol of vol)
    rho: float = -0.7,       # ρ nella tesi (correlazione)
    dt: float = 1 / 250,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> SpotVarianceTuple:
    """
    Genera traiettorie del modello Lifted Heston usando lo schema
    Moment Matching di Bayer e Breneis (Sezione 3.1 della tesi).
    
    Il processo segue le equazioni (3.3)-(3.4):
    dS_t = √V_t S_t (ρ dW_t + √(1-ρ²) dB_t)
    dV^i_t = -x_i(V^i_t - v^i_0)dt + (θ - λV_t)dt + σ√V_t dW_t
    V_t = Σ w_i V^i_t
    """
    if init_state is None:
        init_state = (1.0, theta)
    
    init_state = cast_state(init_state, dtype=dtype, device=device)
    S0, V0 = init_state[0], init_state[1]

    # Nodi e pesi dalla Tabella 4.1 della tesi per T=1, H=0.1
    if n_factors == 2:
        nodes = torch.tensor([0.05, 8.7171], dtype=dtype, device=device)
        weights = torch.tensor([0.7673, 3.2294], dtype=dtype, device=device)
    elif n_factors == 3:
        nodes = torch.tensor([0.03333, 2.2416, 46.831], dtype=dtype, device=device)
        weights = torch.tensor([0.5554, 1.1111, 6.0858], dtype=dtype, device=device)
    else:
        raise ValueError("n_factors deve essere 2 o 3 per questo esempio.")

    # Inizializza v^i_0 secondo la tesi
    v0_i = torch.full((n_factors,), V0 / n_factors, dtype=dtype, device=device)
    w_bar = weights.sum()

    # Inizializza i processi
    S = torch.zeros(n_paths, n_steps + 1, dtype=dtype, device=device)
    V_i = torch.zeros(n_paths, n_steps + 1, n_factors, dtype=dtype, device=device)
    Y_i = torch.zeros(n_paths, n_steps + 1, n_factors, dtype=dtype, device=device)
    
    S[:, 0] = S0
    V_i[:, 0, :] = v0_i
    Y_i[:, 0, :] = 0  # Y^i_0 = 0

    # Matrici per lo splitting scheme (Sezione 3.1.1)
    A = -kappa * weights.unsqueeze(0).repeat(n_factors, 1) - torch.diag(nodes)
    b = theta * torch.ones(n_factors, dtype=dtype, device=device) + torch.diag(nodes) @ v0_i
    
    # Pre-calcola matrici per efficienza
    exp_A_half = torch.linalg.matrix_exp(A * dt / 2)
    A_inv = torch.linalg.inv(A)
    exp_A_half_minus_I = exp_A_half - torch.eye(n_factors, dtype=dtype, device=device)

    # Genera rumore
    dW = torch.randn(n_paths, n_steps, dtype=dtype, device=device) * np.sqrt(dt)
    dB = torch.randn(n_paths, n_steps, dtype=dtype, device=device) * np.sqrt(dt)
    
    # Costanti per Moment Matching (Sezione 3.1.1)
    C = (6 + np.sqrt(3)) / 4
    
    # Loop temporale principale
    for i in range(n_steps):
        # Varianza corrente
        V_current = (V_i[:, i, :] @ weights).clamp(min=1e-8)
        
        # Step 1: Strang Splitting per la varianza (Eq. 3.8)
        
        # a. Mezzo passo deterministico (drift)
        V_half = (V_i[:, i, :] @ exp_A_half.T) + (exp_A_half_minus_I @ A_inv @ b)
        
        # b. Passo stocastico completo (Moment Matching)
        x = (V_half @ weights).clamp(min=1e-8)  # x = w · y nella tesi
        z = sigma**2 * w_bar**2 * dt
        
        # Calcola i tre possibili stati secondo la Sezione 3.1.1
        discriminant = (3 * x + C**2 * z) * z
        sqrt_disc = torch.sqrt(discriminant.clamp(min=0))
        
        x1 = x + C * z - sqrt_disc
        x2 = x + (C - 3/4) * z
        x3 = x + C * z + sqrt_disc
        
        # Calcola i momenti
        m1 = x
        m2 = x**2 + x*z
        m3 = x**3 + 3*x**2*z + 1.5*x*z**2
        
        # Calcola le probabilità (evita divisione per zero)
        eps = 1e-9
        denom1 = x1 * (x3 - x1) * (x2 - x1) + eps
        denom2 = x2 * (x3 - x2) * (x1 - x2) + eps
        denom3 = x3 * (x1 - x3) * (x2 - x3) + eps
        
        p1 = (m1*x2*x3 - m2*(x2+x3) + m3) / denom1
        p2 = (m1*x1*x3 - m2*(x1+x3) + m3) / denom2
        p3 = (m1*x1*x2 - m2*(x1+x2) + m3) / denom3
        
        # Normalizza le probabilità
        probs = torch.stack([p1, p2, p3], dim=1).clamp(min=0)
        probs = probs / probs.sum(dim=1, keepdim=True)
        
        # Campiona gli stati
        states = torch.stack([x1, x2, x3], dim=1)
        indices = torch.multinomial(probs, 1).squeeze(-1)
        Y_hat = states[torch.arange(n_paths), indices]
        
        # Ricostruisci Q̂ e V_i secondo la Sezione 3.1.1
        Q_hat = (Y_hat - x) / w_bar
        V_after_stoch = V_half + Q_hat.unsqueeze(-1)
        
        # c. Altro mezzo passo deterministico
        V_i[:, i+1, :] = (V_after_stoch @ exp_A_half.T) + (exp_A_half_minus_I @ A_inv @ b)
        V_i[:, i+1, :] = V_i[:, i+1, :].clamp(min=0)
        
        # Step 2: Aggiorna Y secondo Eq. (3.10)
        Y_i[:, i+1, :] = Y_i[:, i, :] + (dt/2) * (V_i[:, i, :] + V_i[:, i+1, :])
        
        # Step 3: Simula il prezzo usando Leapfrog splitting (Sezione 3.1.2)
        V_next = (V_i[:, i+1, :] @ weights).clamp(min=1e-8)
        
        # Parametri per S^W secondo Eq. (3.11) con scelta c_i dalla Sezione 3.1.2
        c1 = rho / sigma
        a = -(nodes[0] * v0_i[0] + theta) * c1
        b1 = c1 * nodes[0] + kappa * weights[0] * c1 - 0.5 * weights[0] * rho**2
        
        # Log-prezzo per S^W
        log_S_W = a * dt + b1 * (Y_i[:, i+1, 0] - Y_i[:, i, 0]) + c1 * (V_i[:, i+1, 0] - V_i[:, i, 0])
        S_W = S[:, i] * torch.exp(log_S_W)
        
        # S^B (parte indipendente)
        S_B = S[:, i] * torch.exp(
            np.sqrt(1 - rho**2) * torch.sqrt(V_current * dt) * dB[:, i] - 
            0.5 * V_current * (1 - rho**2) * dt
        )
        
        # Randomized Leapfrog splitting (Eq. 3.12)
        U = torch.rand(n_paths, device=device)

        # Calcola il log-ritorno per la parte W (lo avevi già)
        log_return_W = a * dt + b1 * (Y_i[:, i+1, 0] - Y_i[:, i, 0]) + c1 * (V_i[:, i+1, 0] - V_i[:, i, 0])
        
        # Calcola il log-ritorno per la parte B
        log_return_B = np.sqrt(1 - rho**2) * torch.sqrt(V_current.clamp(min=0)) * dB[:, i] - \
                       0.5 * V_current * (1 - rho**2) * dt
        
        # Combina i due contributi
        S[:, i+1] = S[:, i] * torch.exp(log_return_W + log_return_B)

    # Calcola la varianza finale
    V_final = (V_i @ weights.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
    
    return SpotVarianceTuple(S[:, :-1], V_final[:, :-1])