# robustness-of-quantum-algorithms
This repository contains the scripts to reproduce the results of the paper: 

<cite>Berberich, Julian, et al. "Robustness of quantum algorithms: Worst-case fidelity bounds and implications for design" (2025).</cite>

## Structure of the repository

- `Composite-Pulses` contains the Matlab scripts to reproduce the results corresponding to the studies on composite pulses. These experiments are implemented in Matlab.
- `QFT_and_adder` contains the Python scripts and data to reproduce the results corresponding to the studies on the QFT-Transpilation and the modular adder circuit. These experiments are implemented in Python.
- `QFT_experimental` contains the Python scripts and data to reproduce the results corresponding to the studies on the experimental results on the QFT circuit using QPU from IBM. These experiments are implemented in Python.

## Requirements and Installation

To install the required Python packages, run:

```pip install -r requirements.txt```

## Running the QFT-Transpilation Scripts

Run `main_qft.py` from the directory `QFT_and_adder` to execute the experiment. This script will create a `.pkl` file, which will be saved in the subdirectory `Results`. To recreate the plots from the paper, run the notebook `plots_qft.ipynb` form the `Plots` subdirectory.

## Running the Modular Adder Scripts

Run `main_adder.py` from the directory `QFT_and_adder` to execute the experiment. To run the experiment on the scalin gof different partition sizes on the norm based bound off $\gamma$, run `main_adder_different_partitioning_sizes.py`. These scripts will create `.pkl` files, which will be saved in the subdirectory `Results`. To recreate the plots from the paper, run the notebook `plots_adder.ipynb` form the `Plots` subdirectory.

## Running the Experimental QFT Scripts

Run `main_qft_experimental.py` from the directory `QFT_experimental` to execute the experiments for real processor execution. Note that a QiskitRuntimeService instance is required here. Run `main_qft_theoretical` from the directoty `QFT_experimental` to execute the theoretical experiments. This script will create a `.csv` file, which will be saved in the subdirectory `Theoretical_Results`. To recreate the plots from the paper, run the notebook `plots_experimental.ipynb` from the `Plots` subdirectory. 
