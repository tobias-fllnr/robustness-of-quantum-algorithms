import functions as fc
import numpy as np
import pickle
from datetime import datetime
from bqskit.ext import qiskit_to_bqskit
from qiskit.circuit.library.arithmetic import adders
from qiskit import QuantumCircuit, transpile
import time


def build_circuit(n_qubits: int) -> QuantumCircuit:
    """Build a modular adder circuit with two registers of n_qubits each.
    Args:
        n_qubits (int): Number of qubits per register.
    Returns:
        QuantumCircuit: The constructed modular adder circuit.
    """
    qc = QuantumCircuit(n_qubits*2)
    # Add the ModularAdderGate to the circuit
    adder_gate = adders.ModularAdderGate(num_state_qubits=n_qubits)
    qc.append(adder_gate, range(n_qubits*2))

    return qc

def transpile_circuit_qiskit(circuit: QuantumCircuit) -> QuantumCircuit:
    """Transpile a circuit to the gate set of Henon QPUs using Qiskit.
    Args:
        circuit (QuantumCircuit): The circuit to transpile.
    Returns:
        QuantumCircuit: The transpiled circuit.
    """
    basis_gates = ['cz', 'rx', 'rz', 'rzz', 'sx', 'x']
    transpiled_circuit = transpile(circuit, basis_gates=basis_gates, optimization_level=3, seed_transpiler=42)
    return transpiled_circuit



if __name__ == "__main__":


    n_qubits = [25] # Number of qubits per register
    error_list = ["x", "y", "z", "coherent_control"]
    deltas = list(np.logspace(-7, -1, num=25))
    n_cut_list = [2, 3, 4, 5, 6]  # Max qubits per partition

    all_results = {}
    total_iterations = len(n_qubits) * len(deltas) * len(error_list)

    total_time_start = time.time()
    for n_q in n_qubits:
        circuit_original = build_circuit(n_qubits=n_q)
        circuit = transpile_circuit_qiskit(circuit=circuit_original)
        num_gates = len(circuit.data)
        for n_cut in n_cut_list:
            partitioned_circuits = fc.circuit_partitioning(circuit=circuit, n_cut=n_cut)
            num_circuit_partitions = len(partitioned_circuits)
            print(f"Number of circuit partitions: {num_circuit_partitions}", flush=True)

            layer_unitaries_list_list = []
            subcircuit_bqskit_list = []
            v_list_list = []
            N = 0

            for subcircuit in partitioned_circuits:
                subcircuit_bqskit = fc.qiskit_to_bqskit(subcircuit)
                layer_unitaries_list_subcircuit = fc.get_layer_per_gate_unitaries_bqskit(subcircuit_bqskit)
                layer_unitaries_list_list.append(layer_unitaries_list_subcircuit)
                subcircuit_bqskit_list.append(subcircuit_bqskit)
                v_list = fc.calculate_V(U_ideal_list=layer_unitaries_list_subcircuit)
                v_list_list.append(v_list)
                N += len(layer_unitaries_list_subcircuit)
            print(f"Number of layers: {N}", flush=True)
            print("--------------------------------", flush=True)

            for error in error_list:
                time_start_error = time.time()
                gamma_b4 = 0.0
                gamma_b2 = 0.0
                K_max = 0.0
                for i, layer_unitaries_list_subcircuit in enumerate(layer_unitaries_list_list):
                    b_array, K = fc.compute_local_error_b_array(error_type=error, circuit=subcircuit_bqskit_list[i], layer_unitaries_list=layer_unitaries_list_subcircuit)
                    K_max = max(K_max, K)
                    v_list = v_list_list[i]
                    m = fc.compute_M(V_list=v_list, B_array=b_array)

                    layers_subcircuit = len(layer_unitaries_list_subcircuit)
                    if error == "coherent_control":
                        gamma_b4_subcircuit = fc.calculate_norm_based_gamma_bound(M=m, l=1, num_unitaries=layers_subcircuit)
                    else:
                        gamma_b4_subcircuit = fc.calculate_norm_based_gamma_bound(M=m, l=1, num_unitaries=layers_subcircuit)
                    gamma_b4 += (layers_subcircuit*gamma_b4_subcircuit)
                gamma_b2 /= N*K_max
                gamma_b4 /= N*K_max
                    
                print("--------------------------------", flush=True)
                print(f"Time to compute error parameters for error {error} with {n_q*2} qubits: {time.time() - time_start_error:.2f} seconds", flush=True)
                print("--------------------------------", flush=True)

                for delt in deltas:

                    worst_case_fidelity_bound_b2 = fc.worst_case_fidelity_bound(delta=delt, N=N, gamma=gamma_b2)
                    worst_case_fidelity_bound_b4 = fc.worst_case_fidelity_bound(delta=delt, N=N, gamma=gamma_b4)


                    key = (n_q, delt, error, n_cut)
                    results = {"depth": N,
                                "num_gates": num_gates,
                                "num_circuit_partitions": num_circuit_partitions,
                                "gamma_b2": gamma_b2,
                                "gamma_b4": gamma_b4,
                                "worst_case_fidelity_bound_b2": worst_case_fidelity_bound_b2,
                                "worst_case_fidelity_bound_b4": worst_case_fidelity_bound_b4
                                }
                    all_results[key] = results

    print("--------------------------------", flush=True)
    print(f"Total time for all simulations: {time.time() - total_time_start:.2f} seconds", flush=True)

    current_time = datetime.now().strftime("%m%d%H%M")
    with open(f'./QFT_and_adder/Results/Adder/results{current_time}.pkl', 'wb') as f:
        pickle.dump(all_results, f)

