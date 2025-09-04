import functions as fc
import numpy as np
import pickle
from datetime import datetime
from bqskit.ext import qiskit_to_bqskit


def qft_transpilation(num_q: int, gate_sets_list: list):
    """
    Transpile a QFT circuit for different gate sets and print the results.
    
    Args:
        num_q (int): Number of qubits for the QFT circuit.
        gate_sets_list (list): List of gate set names to transpile against.
    """
    qft_circuit = fc.get_qft_circuit(n_qubits=num_q)
    for name in gate_sets_list:
        print(f"Transpiling Gate Set from {name}")
        gate_set = fc.get_gate_set(name)
        transpiled_circuit = fc.transpile_circuit_bqskit(circuit=qft_circuit, gate_set=gate_set, optimization_level=2)
        fc.save_circuit_to_qasm(transpiled_circuit, f"qft_{num_q}_qubits_{name}")

    qft_circuit = qiskit_to_bqskit(qft_circuit)
    fc.save_circuit_to_qasm(qft_circuit, f"qft_{num_q}_qubits_original")


def build_circuit(name: str, n_qubits: int):
    if name == "google":
        qft_circuit = fc.get_qft_circuit(n_qubits=n_qubits)
        gate_set = fc.get_gate_set(name)
        return fc.transpile_circuit_bqskit(qft_circuit, gate_set, optimization_level=2)
    return fc.load_circuit_from_qasm(f"qft_{n_qubits}_qubits_{name}")

    
def get_original_unitary(n_qubits: int):

    circ = fc.load_circuit_from_qasm(f"qft_{n_qubits}_qubits_original")
    return fc.get_circuit_unitary(circ)


def compute_error_b_array(error_type: str, circuit, layer_unitaries_list, l):

    if error_type in ("x", "y", "z"):
        b_array = fc.compute_B_array(layer_unitaries_list, l, noise_generator=error_type)
    elif error_type == "coherent_control":
        b_array = fc.compute_B_array_coherent_controll(circuit)
    else:
        raise ValueError(f"Unknown error type: {error_type}")


    return b_array



if __name__ == "__main__":

    n_qubits = [3]
    dev_names = ["rigetti", "ibm", "google", "quantinuum"]
    deltas = list(np.logspace(-5, -1, num=17))
    error_list = ["x", "y", "z", "coherent_control"]

    # Transpile the QFT circuit into the gate sets
    qft_transpilation(num_q=n_qubits[0], gate_sets_list=dev_names)

    # Plot the circuits
    gate_sets_plot = ["ibm", "rigetti", "quantinuum", "google", "original"]
    for name in gate_sets_plot:
        fc.plot_circuit_diagrams(name=name, n_qubits=n_qubits[0])

    b2_iterations = 1000

    all_results = {}
    total_iterations = len(n_qubits) * len(dev_names) * len(deltas) * len(error_list)
    iteration = 0

    for n_q in n_qubits:
        for dev_name in dev_names:
            circuit = build_circuit(name=dev_name, n_qubits=n_q)
            unitary_circuit = fc.get_circuit_unitary(circuit=circuit)
            circuit_original = get_original_unitary(n_qubits=n_q)
            unitary_original = fc.get_circuit_unitary(circuit=circuit_original)
            transpilation_fidelity = fc.fidelity(unitary1=unitary_original, unitary2=unitary_circuit)
            layer_unitaries_list = fc.get_layer_unitaries_bqskit(circuit)
            N = len(layer_unitaries_list)
            num_gates = circuit.num_operations
            for error in error_list:
                b_array = compute_error_b_array(error_type=error, circuit=circuit, layer_unitaries_list=layer_unitaries_list, l=n_q)
                v_list = fc.calculate_V(U_ideal_list=layer_unitaries_list)
                m = fc.compute_M(V_list=v_list, B_array=b_array)
                if error == "coherent_control":
                    gamma_b4 = fc.calculate_norm_based_gamma_bound(M=m, l=1, num_unitaries=N)
                else:
                    gamma_b4 = fc.calculate_norm_based_gamma_bound(M=m, l=n_q, num_unitaries=N)
                gamma_b2, mean_gamma_b2, std_gamma_b2 = fc.compute_gamma_optimization(v_list, b_array, iterations=b2_iterations)
                noiseless_unitary = fc.get_circuit_unitary(circuit=circuit)
                for delt in deltas:
                    print(f"Iteration {iteration+1}/{total_iterations}: n_q={n_q}, dev={dev_name}, delta={delt:.1e}, error={error}", flush=True)
                    iteration += 1
                    if error == "coherent_control":
                        bound_coherent_control = fc.compute_fidelity_bound_cce_paper(unitary_list=layer_unitaries_list, delta=delt)
                    else:
                        bound_coherent_control = None
                    worst_case_fidelity_bound_b2 = fc.worst_case_fidelity_bound(delta=delt, N=N, gamma=gamma_b2)
                    worst_case_fidelity_bound_b4 = fc.worst_case_fidelity_bound(delta=delt, N=N, gamma=gamma_b4)
                    fidelities = []
                    runs = 10000
                    for i in range(runs):
                        noise_unitary_list = fc.get_noise_unitary_list_from_B_array(B_array=b_array, delta=delt, seed=i)

                        total_noisy_unitary = fc.get_total_circuit_unitary(noiseless_unitaries=layer_unitaries_list, noise_unitaries=noise_unitary_list)
                        fidelity = fc.fidelity(unitary1=noiseless_unitary, unitary2=total_noisy_unitary)
                        fidelities.append(fidelity)
                    average_fidelity = sum(fidelities) / len(fidelities)
                    std_fidelity = np.std(fidelities)
                    worst_fidelity = min(fidelities)

                    key = (n_q, dev_name, delt, error)
                    results = {"depth": N,
                               "num_gates": num_gates,
                               "transpilation_fidelity": transpilation_fidelity,
                               "gamma_b2": gamma_b2,
                               "mean_gamma_b2": mean_gamma_b2,
                               "std_gamma_b2": std_gamma_b2,
                               "gamma_b4": gamma_b4,
                               "worst_case_fidelity_bound_b2": worst_case_fidelity_bound_b2,
                               "worst_case_fidelity_bound_b4": worst_case_fidelity_bound_b4,
                               "fidelity_bound_coherent_control": bound_coherent_control,
                               "average_fidelity": average_fidelity,
                               "std_fidelity": std_fidelity,
                               "worst_fidelity": worst_fidelity}
                    all_results[key] = results

    current_time = datetime.now().strftime("%m%d%H%M")
    with open(f'./Results/QFT/results{current_time}.pkl', 'wb') as f:
        pickle.dump(all_results, f)