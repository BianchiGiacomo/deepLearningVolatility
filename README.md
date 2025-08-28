# Deep Learning Volatility

Framework per il **pricing** e la **calibrazione** di superfici di volatilità con reti neurali.
Include generatori di dataset (*random grid*), pricers neurali (grid, pointwise, multi‑regime),
motori Monte Carlo per modelli rough/classici e strumenti di post‑processing (interpolazione e smile repair).

> Stato: progetto di ricerca in evoluzione (API soggette a cambiamenti).

---

## Caratteristiche principali

- **Neural pricers**
  - Grid‑based (superficie densa su griglia T×K)
  - Pointwise (query singole \((\theta, T, K)\)) con *random grid* e bucket temporali
  - Multi‑regime (short/mid/long) con routing automatico
- **Generazione dati**: Monte Carlo con gestione dell'**absorption** per modelli rough (regime long‑term)
- **Processi supportati** (estratto): Rough Bergomi, Rough Heston, Heston/Lifted Heston, GBM, Local Vol, processi jump‑diffusion (Kou/Merton)
- **Post‑processing**: interpolazione della superficie e moduli di *smile repair*
- **Esempi**: script per analisi di stabilità, debug MC, long‑term absorption

---

## Requisiti

- Python **>= 3.8.1**
- PyTorch **>= 1.9, < 3.0**
- Numpy, Matplotlib, TQDM
- (Opz.) SciPy, Seaborn, strumenti di sviluppo/qualità (pytest, black, isort, flake8, mypy)

> Nota: se usi GPU, installa prima la build di **torch** compatibile con la tua versione CUDA.

---

## Installazione

### Poetry (consigliato)
```bash
git clone https://github.com/BianchiGiacomo/deepLearningVolatility.git
cd deepLearningVolatility
poetry install
poetry shell
```

### pip (alternativa)
```bash
git clone https://github.com/BianchiGiacomo/deepLearningVolatility.git
cd deepLearningVolatility
pip install -e .
```

---

## Quickstart

### Analisi long‑term (absorption handling)
Esegue una simulazione Monte Carlo su scadenze 1Y–5Y e produce:
1) confronto dello **smile** a T=5Y (con/senza absorption),
2) **Absorption by maturity** con soglia configurabile.

```bash
python examples/LongTermRegimeAnalyzer.py
```

Parametri e soglia possono essere modificati nello script.

### Altri esempi
Altri script dimostrativi sono disponibili nella cartella `examples/` (stabilità per scadenze corte, debug MC, ecc.).

---

## Struttura del progetto (estratto)

```
deepLearningVolatility/
├─ instruments/                # Prodotti e payoff
├─ nn/
│  ├─ dataset_builder/         # Generatori dataset (random grid, bucket temporali)
│  ├─ modules/bs/              # Moduli Black–Scholes
│  └─ pricer/                  # Pricers neurali (grid/pointwise/multi‑regime)
├─ stochastic/
│  ├─ engine.py                # Motore Monte Carlo
│  ├─ rough_bergomi.py, rough_heston.py, heston.py, ...
│  └─ wrappers/                # Wrappers pronti per i processi
├─ examples/                   # Script pronti all'uso
├─ images/                     # Figure di output dimostrative
├─ docs/                       # Documentazione (se presente)
└─ tests/                      # Test
```

---

## Test e qualità

Sono predisposti i tool di qualità e test (vedi `pyproject.toml`):
```bash
pytest -q
flake8 deepLearningVolatility
mypy deepLearningVolatility
black --check deepLearningVolatility
isort --check-only deepLearningVolatility
```

---

## Riferimenti

- Approccio grid‑based (Horváth–Muguruza–Tomas, 2021)
- Pointwise con *random grid* (Baschetti–Bormetti–Rossi, 2024)
- Documentazione del progetto e note su **absorption handling** (cartella `docs/` e `images/`)

Per riprodurre i casi extra citati nella documentazione, vedi `examples/LongTermRegimeAnalyzer.py`.

---

## Contribuire

Contributi e segnalazioni sono benvenuti.
Apri una issue o una pull request descrivendo motivazione, impatto e test minimi.
Mantieni lo stile del codice (`black`, `isort`) e rispetta i controlli (`flake8`, `mypy`).

---

## Licenza

MIT License – vedi file `LICENSE`.
