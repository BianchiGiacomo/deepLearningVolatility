# Deep Learning Volatility

Framework for **pricing** and **calibration** of volatility surfaces with neural networks.
It includes dataset generators (*random grid*), neural pricers (grid, pointwise, multi‑regime),
Monte Carlo engines for rough/classical models, and post‑processing tools (surface interpolation and smile repair).

> Status: research project in progress (APIs may change).

---

## Key features

- **Neural pricers**
  - Grid‑based (dense surface on a T×K grid)
  - Pointwise (single queries (	$\theta$, T, K)) with *random grid* and time buckets
  - Multi‑regime (short/mid/long) with automatic routing
- **Data generation**: Monte Carlo with **absorption** handling for rough models (long‑term regime)
- **Supported processes** (excerpt): Rough Bergomi, Rough Heston, Lifted Heston, GBM, Local Vol, jump‑diffusion processes (Kou/Merton)
- **Post‑processing**: surface interpolation and *smile repair* modules
- **Examples**: scripts for stability analysis, MC debugging, and long‑term absorption

---

## Requirements

- Python **>= 3.8.1**
- PyTorch **>= 1.9, < 3.0**
- Numpy, Matplotlib, TQDM
- (Optional) Poetry for environment management

---

## Installation

### Poetry (recommended)
```bash
git clone https://github.com/BianchiGiacomo/deepLearningVolatility.git
cd deepLearningVolatility
poetry install
poetry shell
```

### pip (alternative)
```bash
git clone https://github.com/BianchiGiacomo/deepLearningVolatility.git
cd deepLearningVolatility
pip install -e .
```

---

## Quickstart

You can run the ready‑to‑use scripts in the `examples/` folder. For instance:

```bash
# Long‑term absorption analysis (1Y–5Y) and smile comparison at T=5Y
python examples/LongTermRegimeAnalyzer.py

# Monte Carlo debugger for short maturities
python examples/MonteCarloDebugger_short_maturities.py
```

Parameters and thresholds can be tweaked directly inside the scripts.

---

## Project structure (excerpt)

```
deepLearningVolatility/
├─ instruments/                # Products and payoffs
├─ nn/
│  ├─ dataset_builder/         # Dataset generators (random grid, time buckets)
│  ├─ modules/bs/              # Black–Scholes modules
│  └─ pricer/                  # Neural pricers (grid/pointwise/multi‑regime)
├─ stochastic/
│  ├─ engine.py                # Monte Carlo engine
│  ├─ rough_bergomi.py, rough_heston.py, heston.py, ...
│  └─ wrappers/                # Ready‑to‑use process wrappers
├─ examples/                   # Ready‑to‑run scripts
├─ images/                     # Demonstrative output figures
├─ docs/                       # Documentation (if present)
└─ tests/                      # Tests
```

---

## References

- Grid‑based approach (Horváth–Muguruza–Tomas, 2021)
- Pointwise with *random grid* (Baschetti–Bormetti–Rossi, 2024)
- Project docs and notes on **absorption handling** (`docs/` and `images/`)

See also `examples/LongTermRegimeAnalyzer.py` to reproduce additional cases mentioned in the docs.

---

## Contributing

Contributions and issues are welcome.
Open an issue or a pull request describing motivation, impact, and minimal tests.
Please keep code style consistent (`black`, `isort`) and pass checks (`flake8`, `mypy`).

---

## License

MIT License – see the `LICENSE` file.
