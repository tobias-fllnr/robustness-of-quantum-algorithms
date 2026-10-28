import pickle
import glob
import os
import pandas as pd
import pickle
from datetime import datetime
import numpy as np
from qiskit import qpy
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from datetime import datetime
import functions as fc


def run_robustness_hardware_loaded_backend(
    transpiled_circuit: QuantumCircuit,
    hardware_snapshot: dict,
    hardware_properties,
    delta: float,
    obs: SparsePauliOp,
    b2_iterations: int,
    sim_iterations: int,
) -> dict:
    results = {}
    hardware_profile = hardware_snapshot["hardware_profile"]
    basis_gates = hardware_snapshot["basis_gates"]
    gate_qubit_counts = hardware_snapshot["gate_qubit_counts"]
    noise_model = fc.get_noise_model_profile_backend(
        hardware_profile, basis_gates, gate_qubit_counts
    )
    num_qubits, depth, num_gates = fc.append_stats(
        results=results, circuit=transpiled_circuit
    )
    A_bar_list, V_bar_list, S_ideal = fc.get_ideal_properties(
        circuit=transpiled_circuit
    )
    B_dict, gamma_b2, gamma_b4, weight_mat, phys_to_sub_index = (
        fc.perform_theoretical_analysis(
            results=results,
            circuit=transpiled_circuit,
            V_bar_list=V_bar_list,
            noise_model=noise_model,
            hardware_profile=hardware_profile,
            delta=delta,
            b2_iterations=b2_iterations,
        )
    )
    readout_variation = fc._calc_readout_variation(
        properties=hardware_properties, phys_to_sub_index=phys_to_sub_index
    )
    fc.append_theoretical_bounds(
        results,
        num_qubits,
        delta,
        depth,
        gamma_b4,
        gamma_b2,
        readout_variation,
        obs,
    )
    fc.perform_simulations(
        results, B_dict, weight_mat, delta, A_bar_list, S_ideal, sim_iterations
    )
    return results


def combine_data(folder, filename):
    """
    Combine individual result files into a single CSV file.
    """
    rows = []
    print(folder)
    file_pattern = os.path.join(folder, "result_*.pkl")
    files = sorted(glob.glob(file_pattern))
    file_count = len(files)
    if file_count == 0:
        print("No files found. Exiting.")
        return
    confirm = input(
        f"Are you sure you want to process and DELETE {file_count} files? (y/n): "
    ).lower()
    if confirm != "y":
        print("Operation cancelled by user.")
        return
    for file in files:
        data = pickle.load(open(file, "rb"))
        rows.append(data)
        os.remove(file)
    df = pd.DataFrame(rows)
    output_dir = os.path.join(folder, filename)
    csv_file = df.to_csv(output_dir, index=False)
    print(f"Combined data saved to {output_dir}")


if __name__ == "__main__":
    num_qubits = 3
    deltas = list(np.logspace(-5, -1, num=13))
    alg = "qft"
    hardware = "superconducting"
    optimization_levels = [0, 1, 2, 3]
    processor = "ibm_fez"
    use_fake_backend = False
    observable = "Z" * int(num_qubits)
    obs_qiskit = SparsePauliOp(observable)

    for delta in deltas:
        for optimization_level in optimization_levels:
            params = {
                "num_qubits": int(num_qubits),
                "delta": float(delta),
                "alg_name": str(alg),
                "hardware": str(hardware),
                "processor": str(processor),
                "optimization_level": int(optimization_level),
            }
            circuit_directory = "Experimental_Data"
            experiment_directory = (
                f"Experimental_Data/{alg}_{num_qubits}_{processor}_{optimization_level}"
            )
            with open(
                os.path.join(experiment_directory, "transpiled_circuit.qpy"), "rb"
            ) as f:
                transpiled_circuit = qpy.load(f)[0]
            with open(
                os.path.join(experiment_directory, "backend_properties.pkl"), "rb"
            ) as f:
                backend_properties = pickle.load(f)
            with open(
                os.path.join(experiment_directory, "hardware_snapshots_metadata.pkl"),
                "rb",
            ) as f:
                hardware_snapshot_metadata = pickle.load(f)
            results = run_robustness_hardware_loaded_backend(
                transpiled_circuit=transpiled_circuit,
                hardware_snapshot=hardware_snapshot_metadata,
                hardware_properties=backend_properties,
                delta=delta,
                obs=obs_qiskit,
                b2_iterations=100,
                sim_iterations=10000,
            )
            results.update(params)
            outdir = "Theoretical_Results"
            os.makedirs(outdir, exist_ok=True)
            filename = os.path.join(
                outdir,
                f"result_num_qubits{num_qubits}_delta{delta}_alg_name{alg}_hardware{hardware}_processor{processor}_optimization_level{optimization_level}.pkl",
            )
            with open(filename, "wb") as f:
                pickle.dump(results, f)
    dir = "Theoretical_Results"
    current_time = datetime.now().strftime("%m%d%H%M")
    filename = f"theoretical_transpilation_results_{current_time}.csv"
    combine_data(dir, filename)
