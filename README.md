# robustness-of-quantum-algorithms
This repository contains the scripts to reproduce the results of the paper: 

<cite>Berberich, Julian, et al. "Robustness of quantum algorithms: Worst-case fidelity bounds and implications for design" (2025).</cite>

## Structure of the repository

- `Composite-Pulses` contains the Matlab scripts to reproduce the results corresponding to the studies on composite pulses
- `QFT-Transpilation` contains the Python scripts and data to reproduce the results corresponding to the studies on the QFT-Transpilation

## Requirements and Installation

To install the required Python packages, run:

```pip install -r requirements.txt```

## Running the QFT-Transpilation Scripts

Run `main.py` from the directory `QFT-Transpilation` to execute the experiment. To recreate the plots from the paper, run the notebook `plots_qft.ipynb` form the `Plots` subdirectory.