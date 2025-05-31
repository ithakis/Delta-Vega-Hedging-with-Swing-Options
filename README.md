# Delta-Vega Hedging with Swing Options

This repository implements models and algorithms for **Delta-Vega Hedging with Swing Options**, providing tools for pricing, risk management, and hedging strategies in energy and financial markets where swing options are prevalent.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

Swing options are financial derivatives that offer the holder multiple exercise rights over a period, commonly used in energy markets. Delta and Vega hedging are risk management techniques aimed at offsetting price and volatility risk, respectively. This project provides Python implementations for modeling, simulating, and hedging swing options using Delta and Vega strategies.

## Features

- Pricing of swing options using various models
- Simulation of underlying asset paths
- Delta and Vega calculation and dynamic hedging
- Backtesting of hedging strategies
- Visualization tools for risk and performance analysis

## Prerequisites

- Python **3.12** is required. The code may not be compatible with earlier Python versions.
- Recommended to use a virtual environment (e.g., `venv`, `conda`).

### Python Packages

The main dependencies are listed below. Please refer to the `requirements.txt` file for the full list and specific versions.

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `jupyter` (for running notebooks, if provided)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ithakis/Delta-Vega-Hedging-with-Swing-Options.git
   cd Delta-Vega-Hedging-with-Swing-Options
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- Refer to example scripts or Jupyter notebooks in the repository for usage examples.
- To run a script:
  ```bash
  python path/to/your_script.py
  ```
- For interactive exploration, start JupyterLab:
  ```bash
  jupyter lab
  ```

## Project Structure

```
Delta-Vega-Hedging-with-Swing-Options/
├── data/                # Sample datasets and data utilities
├── notebooks/           # Jupyter notebooks with exploratory analysis and examples
├── src/                 # Source code for models, hedging, and analysis
├── tests/               # Unit tests
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
```

## Contributing

Contributions are welcome! Please open issues and submit pull requests for bug fixes, enhancements, or new features.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

---

*For questions or support, please open an issue in this repository.*
