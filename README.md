# Delta-Vega Hedging with Swing Options

**Quantitative Simulation and Analysis of Delta-Vega Hedging using Variance Swaps under the Heston Model**

---

## Overview

This repository provides a comprehensive, simulation-based study of delta and vega hedging strategies for European options, focusing on the use of variance swaps as vega hedges. The study is performed under the Heston stochastic volatility model, with all analyses and experiments contained in a single Jupyter notebook: **`Delta-Vega Notebook.ipynb`**.

The notebook is designed for **quantitative analysts** and practitioners who wish to:

- Understand and simulate delta and vega risk management in equity options.
- Analyze the impact of variance swaps as vega hedging instruments.
- Explore the effects of transaction costs, model parameters, and market events (such as earnings jumps).
- Visualize hedging performance and risk metrics under a range of market scenarios.

All computational finance routines (simulation, pricing, and analytics) are implemented in **`myUtils.py`**, allowing the notebook to focus on scenario construction, results, and interpretation.

---

## Structure

### 1. `Delta-Vega Notebook.ipynb`

This notebook is the main entry point, providing:

- **Introduction:** Delta hedging, the Heston model (dynamics, pricing, sensitivities).
- **Variance Swaps:** Payoff, strike and notional calculation, running value.
- **Scenario Analysis:** 
    - Impact of vega hedging (with/without variance swaps).
    - Effects of different maturities, volatilities, vol-of-vol, and hedging frequencies.
    - Cost assumptions and real-world market frictions.
    - Visualization of results: P&L distributions, hedging performance, risk metrics (VaR/CVaR).
- **Earnings Gap Modeling:** Simulates price jumps and analyzes their impact on hedging.
- **Conclusion:** Summarizes quantitative findings and practical considerations.

> **Note:** All heavy-lifting (simulation, pricing, analytics) is delegated to routines in `myUtils.py`. For full reproducibility, ensure both the notebook and `myUtils.py` are present in your working directory.

---

### 2. `myUtils.py`

This module provides advanced, production-grade quantitative finance tools, including:

- **Stochastic Simulation:** Heston model path generation (Quadratic-Exponential scheme, earnings shocks).
- **Option Pricing:** 
    - Heston model option pricing and Greeks (via QuantLib and analytical methods).
    - Black-Scholes option pricing and Greeks for benchmarking.
- **Variance Swaps:**
    - Analytical and discrete strike calculation under Heston dynamics.
    - Variance swap notional calculation and mark-to-market.
    - Delta-Vega hedge ratio computation.
- **Risk and Performance Analytics:** 
    - P&L simulation and distribution analysis.
    - Bootstrap confidence intervals for P&L, VaR, CVaR.
    - Normality diagnostics for risk distributions.
- **Utilities:** Fast vectorized routines (NumPy, Numba, joblib), advanced plotting (matplotlib, seaborn), and robust statistical tools.

**Designed for extensibility and use outside the notebook as well.**

---

## Usage

1. **Clone the repository:**
    ```bash
    git clone https://github.com/ithakis/Delta-Vega-Hedging-with-Variance-Swaps.git
    cd Delta-Vega-Hedging-with-Variance-Swaps
    ```

2. **Install dependencies:**
    - Python 3.9+ recommended.
    - Required packages: `numpy`, `scipy`, `matplotlib`, `seaborn`, `QuantLib`, `py_vollib_vectorized`, `bootstrapped`, `numba`, `joblib`, `scienceplots`, `tqdm`, `jupyter`.
    - Install via pip:
        ```bash
        pip install numpy scipy matplotlib seaborn QuantLib-Python py_vollib_vectorized bootstrapped numba joblib scienceplots tqdm jupyter
        ```

3. **Run the notebook:**
    ```bash
    jupyter notebook "Delta-Vega Notebook.ipynb"
    ```
    - All scenarios and figures are reproducible. 
    - The notebook imports `myUtils.py` for all computations.

---

## For Quantitative Analysts

- **Code Transparency:** All models and formulas are fully documented in `myUtils.py`. You can trace, modify, or extend any routine.
- **Reproducibility:** The notebook is designed for reproducibility. Change simulation parameters directly in the notebook to explore your own scenarios.
- **Performance:** Leveraging Numba, joblib, and QuantLib for high-performance simulations.
- **Research-Grade:** Analytical derivations are included as docstrings; empirical results are statistically evaluated (bootstrap CIs, normality tests, etc.).

---

## Citation

If you use this codebase or its results in your research or professional work, please cite or acknowledge this repository:

```
@misc{DeltaVegaHedging,
  author = {ithakis},
  title = {Delta-Vega Hedging with Swing Options},
  year = {2025},
  url = {https://github.com/ithakis/Delta-Vega-Hedging-with-Variance-Swaps}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

For questions or collaboration, please open an issue or contact the repository maintainer via GitHub.
