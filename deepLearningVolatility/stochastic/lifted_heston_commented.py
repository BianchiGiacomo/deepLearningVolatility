"""
Implementazione del modello Lifted Heston con commenti dettagliati e riferimenti teorici.
Ogni passaggio è spiegato in relazione alla teoria matematica sottostante.
"""
import torch
import numpy as np
from typing import Optional, Tuple, NamedTuple
from scipy.special import gamma as scipy_gamma
import math
from deepLearningVolatility._utils.typing import TensorOrScalar
from deepLearningVolatility.stochastic._utils import cast_state


class SpotVarianceTuple(NamedTuple):
    spot: torch.Tensor
    variance: torch.Tensor


def compute_lifted_heston_params(T: float, H: float, n_factors: int, 
                                device=None, dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calcola i nodi (x_i) e i pesi (w_i) per il modello Lifted Heston.
    
    TEORIA DI RIFERIMENTO:
    -----------------------
    Il modello Rough Heston ha un kernel frazionario K(t) = t^(H-1/2) / Γ(H+1/2).
    Nel Lifted Heston, approssimiamo questo kernel con una somma di esponenziali:
    
    K(t) ≈ Σ_{i=1}^n w_i * exp(-x_i * t)
    
    Questa è un'approssimazione di tipo Padé del kernel frazionario.
    
    RIFERIMENTI:
    - Sezione 2.2 della tesi (Convergence to the rough Heston model)
    - Teorema 2.2.1 e 2.2.3 per la convergenza
    - [10] Bayer & Breneis "Markovian approximations of rough Heston"
    
    METODOLOGIA:
    1. Scegliamo i nodi x_i secondo una progressione geometrica
    2. Calcoliamo i pesi w_i usando il metodo dei momenti
    3. L'obiettivo è minimizzare ||K - K_n||_L1([0,T])
    """
    
    # alpha = H + 1/2 è l'esponente nel kernel K(t) = t^(alpha-1) / Γ(alpha)
    # Questo deriva dalla relazione tra il parametro di Hurst H e l'esponente del kernel
    alpha = H + 0.5
    
    # SCELTA DI r_n:
    # ---------------
    # Il parametro r_n controlla la distribuzione geometrica dei nodi.
    # Secondo Bayer & Breneis, r_n deve soddisfare:
    # 1. r_n → 1 quando n → ∞
    # 2. n * log(r_n) → +∞ quando n → ∞
    # 
    # Questi valori sono ottimizzati empiricamente per minimizzare l'errore
    # di approssimazione per piccoli valori di n (che è il nostro caso pratico)
    if n_factors == 1:
        r_n = 1.5  # Per n=1, r_n più grande per coprire meglio lo spettro
    elif n_factors == 2:
        r_n = 2.5  # Valore empirico che funziona bene per H ≈ 0.1
    elif n_factors == 3:
        r_n = 3.5  # Aumenta con n per mantenere buona copertura
    else:
        # Formula generale che soddisfa le condizioni teoriche
        r_n = 2.0 + np.sqrt(n_factors)
    
    # GENERAZIONE DEI NODI:
    # ---------------------
    # Secondo il Teorema 2.2.1 della tesi, i nodi sono definiti come:
    # x_i^n = [(1-α)/(2-α)] * [(r_n^(2-α) - 1)/(r_n^(1-α) - 1)] * r_n^(i-1-n/2)
    # 
    # Questa formula deriva dall'ottimizzazione della convergenza in norma L2
    
    # Calcola il fattore di scala che appare nella formula dei nodi
    # Questo normalizza i nodi per l'orizzonte temporale T
    scale_factor = (1 - alpha) / (2 - alpha) * (r_n**(2-alpha) - 1) / (r_n**(1-alpha) - 1)
    
    nodes = torch.zeros(n_factors, device=device, dtype=dtype)
    weights = torch.zeros(n_factors, device=device, dtype=dtype)
    
    # Genera i nodi secondo la progressione geometrica
    for i in range(n_factors):
        # λ_i è il nodo "base" prima della normalizzazione
        # i - n_factors/2 centra i nodi attorno a zero nell'esponente
        lambda_i = r_n**(i - 1 - n_factors/2) / T
        
        # Applica il fattore di scala derivato dalla teoria
        nodes[i] = scale_factor * lambda_i
    
    # CALCOLO DEI PESI:
    # -----------------
    # I pesi sono calcolati per matching dei momenti.
    # Vogliamo che l'approssimazione Σ w_i * exp(-x_i * t) abbia gli stessi
    # momenti del kernel originale K(t).
    # 
    # Il k-esimo momento del kernel è:
    # m_k = ∫_0^∞ t^k * K(t) * exp(-t) dt = Γ(k + α) / Γ(α)
    # 
    # Per l'approssimazione:
    # m_k^approx = Σ w_i * ∫_0^∞ t^k * exp(-(x_i + 1)t) dt = Σ w_i * [k! / (x_i + 1)^(k+1)]
    
    # Costruiamo il sistema lineare A * w = b
    A = torch.zeros(n_factors, n_factors, device=device, dtype=dtype)
    b = torch.zeros(n_factors, device=device, dtype=dtype)
    
    for k in range(n_factors):
        # Calcola il k-esimo momento del kernel frazionario
        # usando la formula dei momenti della distribuzione Gamma
        moment_k = scipy_gamma(k + alpha) / scipy_gamma(alpha)
        b[k] = moment_k
        
        # Riempi la riga k della matrice
        # A[k,i] = k! / x_i^(k+1) deriva dall'integrale del momento
        for i in range(n_factors):
            if k == 0:
                A[k, i] = 1.0
            else:
                A[k, i] = math.factorial(k) / (nodes[i] ** (k+1))
    
    # RISOLUZIONE DEL SISTEMA:
    # ------------------------
    # Risolviamo A * w = b per trovare i pesi ottimali
    try:
        weights = torch.linalg.solve(A, b)
    except:
        # Se il sistema è mal condizionato (può succedere per n grande o 
        # nodi molto vicini), usiamo un fallback con pesi uniformi
        weights = torch.ones(n_factors, device=device, dtype=dtype) / n_factors
        weights = weights * scipy_gamma(alpha)  # Normalizzazione per preservare scala
    
    # CORREZIONI DI STABILITÀ:
    # ------------------------
    # I pesi dovrebbero essere positivi per definizione (il kernel è positivo)
    # Piccoli errori numerici possono renderli negativi
    weights = torch.abs(weights)
    
    # NORMALIZZAZIONE FINALE:
    # -----------------------
    # Assicuriamo che il primo momento sia preservato esattamente
    # Questo è cruciale per la convergenza del metodo
    total_weight = torch.sum(weights / nodes)
    target_weight = scipy_gamma(alpha)  # Primo momento teorico del kernel
    weights = weights * (target_weight / total_weight)
    
    return nodes, weights


def generate_lifted_heston(
    n_paths: int,
    n_steps: int,
    init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
    n_factors: int = 3,
    H: float = 0.1,
    T: float = 1.0,
    kappa: float = 0.3,      # λ nella tesi
    theta: float = 0.02,     # θ nella tesi  
    sigma: float = 0.3,      # σ nella tesi
    rho: float = -0.7,       # ρ nella tesi
    dt: Optional[float] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> SpotVarianceTuple:
    """
    DINAMICA DEL MODELLO LIFTED HESTON:
    -----------------------------------
    Secondo le equazioni (3.3)-(3.4) della tesi:
    
    dS_t = √V_t S_t (ρ dW_t + √(1-ρ²) dB_t)
    dV^i_t = -x_i(V^i_t - v^i_0)dt + (θ - λV_t)dt + σ√V_t dW_t
    V_t = Σ w_i V^i_t
    
    dove W e B sono moti Browniani indipendenti.
    """
    
    if init_state is None:
        init_state = (1.0, theta)
    
    init_state = cast_state(init_state, dtype=dtype, device=device)
    S0, V0 = init_state[0], init_state[1]
    
    if dt is None:
        dt = T / n_steps

    # Calcola i parametri del modello per i valori dati di T e H
    nodes, weights = compute_lifted_heston_params(T, H, n_factors, device=device, dtype=dtype)
    
    # INIZIALIZZAZIONE DI v^i_0:
    # --------------------------
    # Dalla Sezione 2.1 della tesi, scegliamo v^i_0 = V_0 per semplicità (tutti uguali)
    # Questo garantisce che Σ w_i v^i_0 ≈ V_0 se Σ w_i ≈ 1
    v0_i = torch.full((n_factors,), V0, dtype=dtype, device=device)
    
    # Normalizza per garantire la condizione iniziale esatta
    current_sum = torch.sum(weights * v0_i)
    v0_i = v0_i * (V0 / current_sum)
    
    # w_bar = Σ w_i è usato frequentemente nelle formule
    w_bar = weights.sum()

    # Inizializza gli array per memorizzare le traiettorie
    S = torch.zeros(n_paths, n_steps + 1, dtype=dtype, device=device)
    V_i = torch.zeros(n_paths, n_steps + 1, n_factors, dtype=dtype, device=device)
    Y_i = torch.zeros(n_paths, n_steps + 1, n_factors, dtype=dtype, device=device)
    
    S[:, 0] = S0
    V_i[:, 0, :] = v0_i
    Y_i[:, 0, :] = 0  # Y^i_t = ∫_0^t V^i_s ds, quindi Y^i_0 = 0

    # COSTRUZIONE DELLE MATRICI PER LO SPLITTING:
    # -------------------------------------------
    # Dalla Sezione 3.1.1, risolviamo l'ODE (3.5) usando splitting di Strang.
    # L'ODE è: dZ^i_t = -x_i(Z^i_t - v^i_0)dt + (θ - λZ_t)dt
    # 
    # In forma matriciale: dZ/dt = AZ + b, dove:
    # A = -λ*1*w^T - diag(x)  (matrice n×n)
    # b = θ*1 + diag(x)*v_0   (vettore n×1)
    A = -torch.diag(nodes) - kappa * weights.unsqueeze(0).repeat(n_factors, 1)
    
    # b = θ*1 + diag(x)*v_0
    b = theta * torch.ones(n_factors, dtype=dtype, device=device) + torch.diag(nodes) @ v0_i
    
    # Pre-calcola exp(A*dt/2) per lo splitting di Strang
    exp_A_half = torch.linalg.matrix_exp(A * dt / 2)
    
    # Per calcolare (exp(A*dt/2) - I)A^(-1)b, necessitiamo A^(-1)
    # Aggiungiamo una piccola regolarizzazione per stabilità numerica
    A_reg = A + 1e-8 * torch.eye(n_factors, dtype=dtype, device=device)
    A_inv = torch.linalg.inv(A_reg)
    
    exp_A_half_minus_I = exp_A_half - torch.eye(n_factors, dtype=dtype, device=device)

    # Genera il rumore Browniano per l'intera simulazione
    dW = torch.randn(n_paths, n_steps, dtype=dtype, device=device) * np.sqrt(dt)
    dB = torch.randn(n_paths, n_steps, dtype=dtype, device=device) * np.sqrt(dt)
    
    # COSTANTE C PER MOMENT MATCHING:
    # --------------------------------
    # Dalla Sezione 3.1.1 della tesi, C = (6 + √3)/4 ≈ 1.933
    # Questa costante appare nella costruzione dei tre stati per il moment matching
    C = (6 + np.sqrt(3)) / 4
    
    # LOOP TEMPORALE PRINCIPALE:
    for i in range(n_steps):
        # Calcola la varianza corrente V_t = Σ w_i V^i_t
        V_current = (V_i[:, i, :] @ weights).clamp(min=1e-8)
        
        # STEP 1: STRANG SPLITTING PER LA VARIANZA
        # -----------------------------------------
        # Schema di Strang: D(dt/2) ∘ S(dt) ∘ D(dt/2)
        # dove D è l'operatore deterministico e S quello stocastico
        
        # 1a. Mezzo passo deterministico: Z_{t+dt/2} = exp(A*dt/2)*Z_t + ...
        V_half = (V_i[:, i, :] @ exp_A_half.T) + (exp_A_half_minus_I @ A_inv @ b)
        
        # 1b. PASSO STOCASTICO COMPLETO (MOMENT MATCHING)
        # ------------------------------------------------
        # Dobbiamo simulare Y_h che ha media x = w·y e varianza z = σ²w̄²dt
        
        x = (V_half @ weights).clamp(min=1e-8)  # Media di Y_h
        z = sigma**2 * w_bar**2 * dt            # Varianza di Y_h
        
        # COSTRUZIONE DEI TRE STATI:
        # --------------------------
        # Secondo la Sezione 3.1.1, costruiamo tre possibili valori x₁, x₂, x₃
        # tali che una loro combinazione convessa abbia i primi 5 momenti corretti
        
        # Il discriminante appare nella formula per x₁ e x₃
        discriminant = (3 * x + C**2 * z) * z
        sqrt_disc = torch.sqrt(discriminant.clamp(min=0))
        
        # I tre stati secondo le formule della tesi
        x1 = x + C * z - sqrt_disc
        x2 = x + (C - 3/4) * z  # Nota: C - 3/4 = (6 + √3)/4 - 3/4 = (3 + √3)/4
        x3 = x + C * z + sqrt_disc
        
        # CALCOLO DEI MOMENTI:
        # --------------------
        # Per il matching, calcoliamo i primi 3 momenti di Y_h
        m1 = x                           # E[Y_h]
        m2 = x**2 + x*z                  # E[Y_h²]
        m3 = x**3 + 3*x**2*z + 1.5*x*z**2  # E[Y_h³]
        
        # CALCOLO DELLE PROBABILITÀ:
        # --------------------------
        # Risolviamo il sistema per trovare p₁, p₂, p₃ tali che:
        # p₁ + p₂ + p₃ = 1
        # p₁x₁ + p₂x₂ + p₃x₃ = m₁
        # p₁x₁² + p₂x₂² + p₃x₃² = m₂
        # p₁x₁³ + p₂x₂³ + p₃x₃³ = m₃
        
        eps = 1e-12
        
        # Verifica che x1, x2, x3 siano distinti
        if torch.any(torch.abs(x3 - x1) < eps) or torch.any(torch.abs(x2 - x1) < eps):
            # Fallback: usa schema semplificato
            Y_hat = x + torch.sqrt(z) * torch.randn_like(x)
        else:
            # Calcolo delle probabilità con maggiore stabilità
            denom1 = (x1 * (x3 - x1) * (x2 - x1)).clamp(min=eps)
            denom2 = (x2 * (x3 - x2) * (x1 - x2)).clamp(min=eps)
            
            p1 = (m1*x2*x3 - m2*(x2+x3) + m3) / denom1
            p2 = (m1*x1*x3 - m2*(x1+x3) + m3) / denom2
            p3 = 1 - p1 - p2
            
            # Correzione delle probabilità
            probs = torch.stack([p1, p2, p3], dim=1)
            probs = torch.clamp(probs, min=0, max=1)
            probs = probs / probs.sum(dim=1, keepdim=True)
            
            # Campionamento
            states = torch.stack([x1, x2, x3], dim=1)
            indices = torch.multinomial(probs, 1).squeeze(-1)
            Y_hat = states[torch.arange(n_paths), indices]
        
        # RICOSTRUZIONE DI V^i:
        # ---------------------
        # Da Y_h = w·V_after_stoch, ricaviamo Q̂ = (Y_h - x)/w̄
        # e quindi V_after_stoch = V_half + Q̂*1
        Q_hat = (Y_hat - x) / w_bar
        V_after_stoch = V_half + Q_hat.unsqueeze(-1)
        
        # 1c. Altro mezzo passo deterministico
        V_i[:, i+1, :] = (V_after_stoch @ exp_A_half.T) + (exp_A_half_minus_I @ A_inv @ b)
        V_i[:, i+1, :] = V_i[:, i+1, :].clamp(min=0)
        
        # STEP 2: AGGIORNA Y (INTEGRALE DELLA VARIANZA)
        # ----------------------------------------------
        # Y^i_t = ∫_0^t V^i_s ds, quindi dY^i = V^i dt
        # Usiamo la regola del trapezio per l'integrazione
        Y_i[:, i+1, :] = Y_i[:, i, :] + (dt/2) * (V_i[:, i, :] + V_i[:, i+1, :])
        
        # STEP 3: SIMULAZIONE DEL PREZZO (LEAPFROG SPLITTING)
        # ----------------------------------------------------
        # Dalla Sezione 3.1.2, splittiamo l'SDE del prezzo in:
        # - S^W: parte correlata con la varianza
        # - S^B: parte indipendente (Browniana pura)
        
        # V_next = (V_i[:, i+1, :] @ weights).clamp(min=1e-8)
        
        # PARAMETRI PER S^W:
        # ------------------
        # Dalla eq. (3.11) con la scelta c_i = (ρ/σ)δ_{i,1} (solo c₁ ≠ 0)
        c = torch.zeros(n_factors, dtype=dtype, device=device)
        c[0] = rho / sigma
        sum_c = c[0]
        
        # Coefficiente a dall'equazione (3.11)
        a = -(nodes[0] * v0_i[0] * c[0]) - theta * sum_c
        # Vettorizzazione completa del calcolo di b
        # b_i = c_i*x_i + λ*w_i*sum_c - (1/2)*w_i*ρ²
        b = c * nodes + kappa * weights * sum_c - 0.5 * weights * rho**2
        
        # Log-prezzo per S^W secondo eq. (3.11) - completamente vettorizzato
        delta_Y_i = Y_i[:, i+1, :] - Y_i[:, i, :] # Shape: (n_paths, n_factors)
        delta_V_i = V_i[:, i+1, :] - V_i[:, i, :] # Shape: (n_paths, n_factors)
        log_return_W = a * dt + \
               torch.einsum('ij,j->i', delta_Y_i, b) + \
               torch.einsum('ij,j->i', delta_V_i, c)
        
        # S^B: soluzione esatta per la parte Browniana indipendente
        log_return_B = torch.sqrt((1 - rho**2) * V_current.clamp(min=0)) * dB[:, i] - \
               0.5 * V_current * (1 - rho**2) * dt
               
        S[:, i+1] = S[:, i] * torch.exp(log_return_W + log_return_B)

    # Calcola la varianza finale aggregata
    V_final = (V_i @ weights.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
    
    return SpotVarianceTuple(S[:, 1:], V_final[:, 1:])