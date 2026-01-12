import numpy as np
import networkx as nx
from qiskit.synthesis.qft import synth_qft_full
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from scipy.linalg import logm
from bqskit.compiler.compile import compile
from bqskit.ext import qiskit_to_bqskit
from bqskit.compiler.machine import MachineModel
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.parameterized.rz import RZGate
from bqskit.ir.gates.parameterized.rx import RXGate
from bqskit.ir.gates.parameterized.rzz import RZZGate
from bqskit.ir.gates.parameterized.u1q import U1qPi2Gate
from bqskit.ir.gates.parameterized.u1q import U1qPiGate
from bqskit.ir.gates.constant.cz import CZGate
from bqskit.ir.gates.constant.iswap import ISwapGate
from bqskit.ir.gates.constant.sx import SXGate
from bqskit.ir.gates.constant.x import XGate
from bqskit.ir.gates.constant.sqrtiswap import SqrtISwapGate
from bqskit.ir.gates.constant.sycamore import SycamoreGate
from bqskit.ir.gates.parameterized.phasedxz import PhasedXZGate
from scipy.linalg import expm
from scipy.optimize import minimize

def save_circuit_to_qasm(circuit: Circuit, filename: str):
    """
    Saves a Circuit to a QASM file.
    
    Args:
        circuit (QuantumCircuit): The quantum circuit to save.
        filename (str): The name of the file to save the circuit to.
    """
    circuit.save(f'./QFT_and_adder/Circuits/{filename}.qasm')

def load_circuit_from_qasm(filename: str) -> Circuit:
    """
    Loads a Circuit from a QASM file.
    
    Args:
        filename (str): The name of the file to load the circuit from.
        
    Returns:
        QuantumCircuit: The loaded quantum circuit.
    """
    return Circuit.from_file(f'./QFT_and_adder/Circuits/{filename}.qasm')

def get_circuit_unitary(circuit: Circuit) -> np.ndarray:
    """
    Returns the unitary matrix of a BQSKit circuit.
    
    Args:
        circuit (Circuit): Input quantum circuit.
        
    Returns:
        np.ndarray: The unitary matrix of the circuit.
    """
    return circuit.get_unitary()

def get_total_circuit_unitary(noiseless_unitaries: list, noise_unitaries: list) -> np.ndarray:
    """
    Computes the total unitary of a circuit given the noiseless and noise unitary matrices.
    
    Args:
        noiseless_unitaries (list): List of noiseless unitary matrices for each layer.
        noise_unitaries (list): List of noise unitary matrices for each layer.
        
    Returns:
        np.ndarray: The total unitary matrix of the circuit.
    """
    total_unitary = np.eye(noiseless_unitaries[0].shape[0], dtype=complex)
    for U_n, U_e in zip(noiseless_unitaries, noise_unitaries):
        total_unitary =  U_n @ U_e @ total_unitary
    return total_unitary

def get_noise_unitary_list_from_B_array(B_array: np.ndarray, delta: float, K: float, seed: int=None) -> list:
    """
    Generate a list of noise unitary matrices from a B_array.
    Args:
        B_array (np.ndarray): A 4D array of shape (num_layers, l, dim, dim) representing the B_{j,i} matrices.
        delta (float): Maximum noise level.
        K (float): Scaling factor for the noise Hamiltonian.
        seed (int, optional): Random seed for reproducibility.
    Returns:
        List[np.ndarray]: A list of noise unitary matrices, one for each layer.
    """
    num_layers, l, dim , _ = B_array.shape
    np.random.seed(seed)
    noise_unitaries = []
    for i in range(num_layers):
        theta = np.random.uniform(-delta/l, delta/l, size=l) # delta/l to ensure condition on infinity norm of theta_j
        # theta = np.array([delta/l]* l)  # Use fixed value for testing
        # theta = np.random.choice([-delta/l, delta/l], size=l)
        H_e = np.zeros((dim, dim), dtype=complex)
        for j in range(l):
            H_e += theta[j] * B_array[i, j]
        H_e /= K  # Scale by K
        # Turn H_e into a unitary
        noise_unitary = expm(-1j * H_e)
        noise_unitaries.append(noise_unitary)
    return noise_unitaries

def get_gate_set(gate_set_name: str) -> set:
    """
    Returns a set of gates corresponding to the specified gate set name.
    Quantinuum (former Honeywall) now has RZZ instead of ZZ gate.
    Args:
        gate_set_name (str): Name of the gate set (e.g., "ibm", "rigetti", "quantinuum", "google").
    Returns:
        set: A set of gates available in the specified gate set.
    """
    gate_sets = {"ibm": {SXGate(), XGate(), RZGate(), CZGate(), RZZGate(), RXGate()},
                "rigetti": {SXGate(), XGate(), RZGate(), ISwapGate()},
                "quantinuum": {U1qPiGate, U1qPi2Gate, RZGate(), RZZGate()},
                "google": {PhasedXZGate(), RZGate(), SycamoreGate(), CZGate(), SqrtISwapGate()}
                }
    return gate_sets.get(gate_set_name, set())

def get_qft_circuit(n_qubits: int) -> QuantumCircuit:
    """
    Generates a Quantum Fourier Transform (QFT) circuit for a given number of qubits.
    
    Args:
        n_qubits (int): The number of qubits for the QFT circuit.
    
    Returns:
        QuantumCircuit: The QFT circuit.
    """
    qft_circuit = synth_qft_full(n_qubits, do_swaps=True)
    return qft_circuit

def transpile_circuit_bqskit(circuit: QuantumCircuit, gate_set: set, optimization_level: int=3, seed: int=42) -> Circuit:
    """
    Transpiles a Qiskit QuantumCircuit to a BQSKit Circuit using a specified gate set.
    Args:
        circuit (QuantumCircuit): The Qiskit circuit to transpile.
        gate_set (set): The set of gates to use for transpilation.
        optimization_level (int): Level of optimization for the transpilation.
    Returns:
        QuantumCircuit: The transpiled Qiskit circuit.
    """
    circuit = qiskit_to_bqskit(circuit)
    model = MachineModel(circuit.num_qudits, gate_set=gate_set)
    out_circuit = compile(circuit, model=model, optimization_level=optimization_level, seed=seed)
    return out_circuit

def fidelity(unitary1: np.ndarray, unitary2: np.ndarray) -> float:
    """
    Computes the fidelity between two unitary matrices.
    
    Args:
        unitary1 (np.ndarray): The first unitary matrix.
        unitary2 (np.ndarray): The second unitary matrix.
        
    Returns:
        float: The fidelity between the two unitaries.
    """
    d = unitary1.shape[0]
    return np.abs(np.trace(unitary1.conj().T @ unitary2)/d)**2

def unitaries_to_generators(unitary_list: list) -> list:
    """
    Given a list of unitary Operators (or matrices), return the corresponding Hermitian generators.

    Args:
        unitary_list (List[Operator or np.ndarray]): List of unitary operators.

    Returns:
        List[np.ndarray]: List of Hermitian generators H_i such that U_i = exp(-i H_i).
    """
    generators = []
    for U in unitary_list:
        if isinstance(U, Operator):
            U = U.data  # Convert to numpy array
        H = 1j * logm(U)  # Hermitian generator
        # Ensure Hermiticity numerically
        H = (H + H.conj().T) / 2
        # Remove the trace (global phase component)
        H -= np.trace(H) / H.shape[0] * np.eye(H.shape[0])
        generators.append(H)
    return generators

def compute_M(V_list: list[np.ndarray], B_array: np.ndarray) -> np.ndarray:
    """
    Compute matrix M from a list of V_j matrices and a 4D array of B_{j,i}.
    
    Args:
        V_list (list of np.ndarray): List of N unitaries V_j of shape (dim, dim)
        B_array (np.ndarray): Array of shape (N, l, dim, dim)
        
    Returns:
        np.ndarray: The matrix M of shape (dim*dim, N*l)
    """
    N, l, _ , _ = B_array.shape
    M_blocks = []

    for j in range(N):
        Vj = V_list[j]
        Vj_dag = Vj.conj().T
        Mj_columns = []

        for i in range(l):
            Bji = B_array[j, i]
            transformed = Vj_dag @ Bji @ Vj
            vec = transformed.flatten(order='F')  # Column-wise flattening
            Mj_columns.append(vec)

        Mj = np.column_stack(Mj_columns)
        M_blocks.append(Mj)

    M_full = np.hstack(M_blocks) / N
    return M_full

def calculate_V(U_ideal_list: list[np.ndarray]) -> list[np.ndarray]:
    """
    Efficiently computes V_k = U_{k-1} * V_{k-1} with V_1 = I.
    
    Parameters:
        U_list (list of np.ndarray): list of unitaries [U1, U2, ..., UN]
    
    Returns:
        list of np.ndarray: [V1, V2, ..., VN]
    """
    N = len(U_ideal_list)
    dim = U_ideal_list[0].shape[0]
    
    V_list = [np.eye(dim, dtype=complex)]  # V1 = I
    
    for k in range(1, N):
        V_prev = V_list[-1]
        U_prev = U_ideal_list[k - 1]
        V_k = U_prev @ V_prev
        V_list.append(V_k)
    
    return V_list

def matrix_2_norm(A: np.ndarray) -> float:
    """
    Compute the matrix 2-norm (spectral norm) of a given n x m matrix A.
    
    Args:
        A (np.ndarray): A 2D numpy array of shape (n, m)
        
    Returns:
        float: The matrix 2-norm (largest singular value)
    """
    return np.linalg.norm(A, ord=2)

def calculate_norm_based_gamma_bound(M: np.ndarray, l: int, num_unitaries: int) -> float:
    """
    Calculate the norm-based gamma bound for the given matrix M and unitaries U_list.
    
    Args:
        M (np.ndarray): The matrix M of shape (dim*dim, N*l)
        l (int): Number of independent errors per layer
        num_unitaries (int): Number of unitaries in the U_list        
    Returns:
        float: The computed gamma bound
    """
    gamma_bound = np.sqrt(num_unitaries/l)* matrix_2_norm(M)
    return gamma_bound

def worst_case_fidelity_bound(delta: float, N: int, gamma: float) -> float:
    """
    Computes the worst-case fidelity lower bound based on the provided parameters.
    
    Args:
        delta (float): The delta error bound.
        N (int): The number of steps or layers (e.g., circuit depth).
        gamma (float): Additional error contribution (e.g., from noise).
        
    Returns:
        float: The worst-case fidelity lower bound.
    """
    term = (0.5 * (N - 1) * delta + gamma)
    bound = 1 - (delta**2) * (N**2) * (term**2)
    return bound

def G_theta(theta: np.ndarray, V_list: list[np.ndarray], B_array: np.ndarray) -> np.ndarray:
    """
    Computes G(θ) for a flattened θ vector of shape (N*l,)
    Args:
        theta (np.ndarray): Flattened θ vector of shape (N*l,).
        V_list (list[np.ndarray]): List of unitary matrices V_j.
        B_array (np.ndarray): Array of shape (N, l, d, d) representing B_{j,i} matrices.
    Returns:
        np.ndarray: The matrix G(θ) of shape (d, d).
    """
    N, l, d, _ = B_array.shape
    G = np.zeros((d, d), dtype=complex)
    theta = theta.reshape((N, l))  # reshape into [N, l]

    for j in range(N):
        Hj = sum(theta[j, k] * B_array[j, k] for k in range(l))
        Gj = V_list[j].conj().T @ Hj @ V_list[j]
        G += Gj
    G = G / N
    return G

def compute_gamma_optimization(V_list: list[np.ndarray], B_array: np.ndarray, method: str="L-BFGS-B", seed: int=42, iterations: int=1000) -> tuple:
    """
    Compute gamma by maximizing spectral norm of G(θ) over ∥θ∥_∞ ≤ 1/ℓ

    This function uses optimization to find the maximum spectral norm of G(θ)
    by varying θ within the bounds defined by the infinity norm constraint.
    Args:
        V_list (list[np.ndarray]): List of unitary matrices V_j.
        B_array (np.ndarray): Array of shape (N, l, d, d) representing B_{j,i} matrices.
        method (str): Optimization method to use (default: "L-BFGS-B").
        seed (int): Random seed for reproducibility.
    Returns:
        tuple: (best_gamma, mean_gamma, std_gamma) where best_gamma is the maximum
               spectral norm found, mean_gamma is the average over all optimization runs,
               and std_gamma is the standard deviation.
    """
    np.random.seed(seed)
    N, l, d, _ = B_array.shape
    dim = N * l

    def objective(theta_flat):
        G = G_theta(theta_flat, V_list, B_array)
        return -np.linalg.norm(G, ord=2)  # Maximize -> minimize negative

    bounds = [(-1/l, 1/l) for _ in range(dim)]

    # Try from a few random starting points and pick the best
    best_gamma = 0.0
    all_gammas = []
    for _ in range(iterations):
        x0 = np.random.uniform(-1/l, 1/l, size=dim)
        result = minimize(objective, x0=x0, bounds=bounds, method=method)
        if result.success:
            gamma = -result.fun
            all_gammas.append(gamma)
            best_gamma = max(best_gamma, gamma)
    mean_gamma = np.mean(all_gammas)
    std_gamma = np.std(all_gammas)

    return best_gamma, mean_gamma, std_gamma

def compute_fidelity_bound_cce_paper(unitary_list: list[np.ndarray], delta: float):
    """
    Computes the fidelity bound based on the coherent control error paper.
    This function computes the bound as described in the paper by calculating
    the 2-norm of the Hermitian generators of the unitary matrices.
    Args:
        unitary_list (list[np.ndarray]): List of unitary matrices.
        delta (float): Maximum error bound.
    Returns:
        float: The computed fidelity bound.
    """

    generators = unitaries_to_generators(unitary_list)
    bound = [matrix_2_norm(H) for H in generators]
    bound = sum(bound)**2
    bound = 1 - delta**2 * bound
    bound = bound**2 # square to match fidelity of cce paper with our definition of fidelity: But is problematic, when bound < 0
    return bound

def plot_circuit_diagrams(name: str, n_qubits: int):
    """
    Plots the circuit diagram for a given experiment and saves it as a PDF.
    Args:
        name (str): Name of the quantum hardware or simulator (e.g., "ibm", "rigetti", "google", "quantinuum", "original).
        n_qubits (int): Number of qubits in the circuit.
    """

    output_dir = f"./QFT_and_adder/Plots/Circuits/circuit_diagram_qft_{n_qubits}_qubits_{name}.pdf"
    qasm_file = f"./QFT_and_adder/Circuits/qft_{n_qubits}_qubits_{name}.qasm"
    if name in ["rigetti", "original"]:
        # Load the circuit from a .qasm file
        qc = QuantumCircuit.from_qasm_file(qasm_file)
    
        qc.draw(output="mpl", style="bw", fold=14, filename=output_dir)

    elif name == "ibm":
        # Replace sx with rx(pi/2) for plotting to avoid confusion
        qc = QuantumCircuit.from_qasm_file(qasm_file)
        new_qc = QuantumCircuit(qc.num_qubits)
        for instr, qargs, cargs in qc.data:
            if instr.name == "sx":
                # Replace SX with RX(pi/2)
                new_qc.rx(np.pi/2, qargs[0])
            else:
                # Keep all other instructions
                new_qc.append(instr, qargs, cargs)
        new_qc.draw(output="mpl", style="bw", fold=14, filename=output_dir)
    
    elif name == "google":
        # Define custom gates

        def pxz_gate(theta, phi, lam):
            """Return a single-qubit pxz gate as a U3 rotation."""
            qc = QuantumCircuit(1)
            qc.u(theta, phi, lam, 0)
            return qc.to_gate(label="pxz")

        def syc_gate():
            """Placeholder 2-qubit syc gate without extra 0/1 labels."""
            qc = QuantumCircuit(2, name="syc")
            return qc.to_gate(label="syc")

        def sqisw_gate():
            """Placeholder 2-qubit sqisw gate without extra 0/1 labels."""
            qc = QuantumCircuit(2, name="sqisw")
            return qc.to_gate(label="sqisw")

        with open(qasm_file, "r") as f:
            lines = f.readlines()


        # Create a circuit with 3 qubits (adjust if needed)
        qc = QuantumCircuit(3)
        q = qc.qubits

        for line in lines:
            line = line.strip()
            if line.startswith("pxz"):
                params, qubit = line.split(") ")
                params = params[4:]  # remove 'pxz('
                theta, phi, lam = [float(x) for x in params.split(",")]
                q_index = int(qubit.replace("q[","").replace("]","").replace(";",""))
                qc.append(pxz_gate(theta, phi, lam), [q[q_index]])

            elif line.startswith("syc") or line.startswith("sqisw"):
                gate_func = syc_gate if line.startswith("syc") else sqisw_gate
                qubits_str = line.split(" ")[1:]  # everything after gate name
                qubits_str = " ".join(qubits_str).replace(";", "")  # remove semicolon
                q_indices = [int(x.replace("q[", "").replace("]", "").strip()) for x in qubits_str.split(",")]
                qc.append(gate_func(), [q[i] for i in q_indices])
            

        # Draw the circuit
        qc.draw(output="mpl", style="bw", fold=14, filename=output_dir)

    elif name == "quantinuum":
        # Define gates
        def U1q_gate(theta, phi):
            qc = QuantumCircuit(1)
            qc.u(theta, phi, 0, 0)
            return qc.to_gate(label="U1q")

        def rz_gate(theta):
            qc = QuantumCircuit(1)
            qc.rz(theta, 0)
            return qc.to_gate(label="RZ")

        def rzz_gate():
            qc = QuantumCircuit(2)
            return qc.to_gate(label="RZZ")

        with open(qasm_file, "r") as f:
            lines = f.readlines()

        # Create circuit
        qc = QuantumCircuit(3)
        q = qc.qubits  # for easy indexing

        # Parse QASM lines
        for line in lines:
            line = line.strip()
            
            if line.startswith("U1q"):
                # Extract parameters inside parentheses
                start = line.index("(") + 1
                end = line.index(")")
                params_str = line[start:end]
                theta, phi = [float(x) for x in params_str.split(",")]
                # Extract qubit index
                q_index = int(line.split("q[")[1].split("]")[0])
                qc.append(U1q_gate(theta, phi), [q[q_index]])

            elif line.startswith("rzz"):
                qubits_str = line.split(" ")[1:]  # everything after gate name
                qubits_str = " ".join(qubits_str).replace(";", "")  # remove semicolon
                q_indices = [int(x.replace("q[", "").replace("]", "").strip()) for x in qubits_str.split(",")]
                qc.append(rzz_gate(), [q[i] for i in q_indices])

            elif line.startswith("rz"):
                start = line.index("(") + 1
                end = line.index(")")
                theta = float(line[start:end])
                q_index = int(line.split("q[")[1].split("]")[0])
                qc.append(rz_gate(theta), [q[q_index]])


        # Draw the circuit
        qc.draw(output="mpl", style="bw", fold=14, filename=output_dir)

def get_layer_per_gate_unitaries_bqskit(circuit: Circuit) -> list:
    """
    Return list of unitary matrices for each gate in the circuit.
    Args:
        circuit (Circuit): Input quantum circuit.
    Returns:
        List[np.ndarray]: List of unitary matrices, one per gate.
    """
    num_qudits = circuit.num_qudits
    unitary_list = []

    for op in circuit:
        subcircuit = Circuit(num_qudits)
        subcircuit.append_gate(op.gate, op.location, op.params)
        unitary = subcircuit.get_unitary()
        unitary_list.append(unitary)

    return unitary_list

def compute_local_error_b_array(error_type: str, circuit: Circuit, layer_unitaries_list: list) -> np.ndarray:
    """
    Computes the B_{j,i} matrices for a given error type.
    Args:
        error_type (str): Type of error ("x", "y", "z", "coherent_control").
        circuit (Circuit): Input quantum circuit.
        layer_unitaries_list (list of np.ndarray): List of unitaries for each layer.
        l (int): Number of independent errors per layer.
    Returns:
        np.ndarray: A 4D array B_{j,i} of shape (N, l, dim, dim) where N is the number of layers.
    """
    if error_type in ("x", "y", "z"):
        b_array = compute_B_array_local_error(circuit, noise_generator=error_type)
        K = 1.0  # For local errors, K is 1
    elif error_type == "coherent_control":
        b_array, K = compute_B_array_coherent_controll_local_error(layer_unitaries_list)
    else:
        raise ValueError(f"Unknown error type: {error_type}")
    
    return b_array, K
    
def compute_B_array_local_error(circuit: Circuit, noise_generator: str = "z") -> np.ndarray:
    """
    Computes the B_{j,i} matrices for a BQSKit circuit, where each gate is treated individually
    Args:
        circuit (Circuit): Input quantum circuit.
        noise_generator (str): Type of noise operator ("z", "x", "y").
    Returns:
        np.ndarray: A 4D array B_{j,i} of shape (num_operations, 1, dim, dim) where num_operations is the number of gates in the circuit.
    """
    N = circuit.num_operations
    num_qubits = circuit.num_qudits
    dim = 2**num_qubits
    
    # Initialize B_array with zeros
    B_array = np.zeros((N, 1, dim, dim), dtype=complex)

    if noise_generator == "z":
        noise_matrix = np.array([[1, 0], [0, -1]], dtype=complex)
    elif noise_generator == "x":
        noise_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
    elif noise_generator == "y":
        noise_matrix = np.array([[0, -1j], [1j, 0]], dtype=complex)
    else:
        raise ValueError(f"Unsupported noise generator: {noise_generator}")
    
    for i, op in enumerate(circuit):
        j = op.location
        num_active_qubits = len(j)
        if num_active_qubits == 1:
            j = j[0]
            noise_tensor = np.kron(np.eye(2**j, dtype=complex), np.kron(noise_matrix, np.eye(2**(num_qubits-j-1), dtype=complex)))
        
        elif num_active_qubits == 2:
            j1 = j[0]
            j2 = j[1]
            if j1 > j2:
                j1, j2 = j2, j1
            noise_tensor = np.kron(np.eye(2**j1, dtype=complex), np.kron(noise_matrix, np.kron(np.eye(2**(j2-j1-1), dtype=complex), np.kron(noise_matrix, np.eye(2**(num_qubits-j2-1), dtype=complex)))))
        
        else:
            raise ValueError("Only single-qubit and two-qubit gates are supported for local error model.")

        B_array[i, 0] = noise_tensor
    
    return B_array

def compute_B_array_coherent_controll_local_error(layer_unitaries_list: list) -> np.ndarray:
    """
    Computes the B_{j,i} matrices for a list of layer unitaries, where each layer is treated individually
    Args:
        layer_unitaries_list (list of np.ndarray): List of unitaries for each layer.
    Returns:
        np.ndarray: A 4D array B_{j,i} of shape (N, 1, dim, dim) where N is the number of layers.
    """
    N = len(layer_unitaries_list)
    dim = layer_unitaries_list[0].shape[0]
    
    # Initialize B_array with zeros
    B_array = np.zeros((N, 1, dim, dim), dtype=complex)

    for i, gate in enumerate(layer_unitaries_list):
        H_e = 1j * logm(gate)
        # ensure hermitianity numerically
        H_e = (H_e + H_e.conj().T) / 2

        # Fix: Remove the trace (global phase component)
        H_e -= np.trace(H_e) / H_e.shape[0] * np.eye(H_e.shape[0])

        B_array[i, 0] = H_e

    # Calculate K as the maximum 2-norm of B_{i,0} for rescaling (to satisfy ∥B_{i}∥_2 ≤ 1)
    K = max(matrix_2_norm(B_array[i, 0]) for i in range(N))
    
    return B_array, K

def circuit_partitioning(circuit: QuantumCircuit, n_cut: int) -> list[QuantumCircuit]:
    """
    Partitions a Qiskit circuit into subcircuits such that each subcircuit 
    acts on at most n_cut qubits.
    
    Args:
        circuit (QuantumCircuit): The input circuit to be partitioned.
        n_cut (int): The maximum number of active qubits allowed in a single subcircuit.
        
    Returns:
        list[QuantumCircuit]: A list of partitioned subcircuits.
        
    Raises:
        ValueError: If a single instruction acts on more qubits than n_cut.
    """
    subcircuits = []
    
    # Initialize the first subcircuit with the same registers as the original
    current_subcircuit = QuantumCircuit(circuit.qubits, circuit.clbits)
    active_qubits = set()
    
    for instruction in circuit.data:
        # instruction is a CircuitInstruction object (operation, qargs, cargs)
        op_qubits = set(instruction.qubits)
        
        # logical check: If a single gate is larger than the cut size, it's impossible
        if len(op_qubits) > n_cut:
            raise ValueError(
                f"Instruction {instruction.operation.name} acts on {len(op_qubits)} "
                f"qubits, which exceeds the limit of n_cut={n_cut}."
            )
        
        # Check if adding this instruction exceeds the n_cut limit
        potential_active_qubits = active_qubits.union(op_qubits)
        
        if len(potential_active_qubits) <= n_cut:
            # It fits: add to current subcircuit and update active set
            current_subcircuit.append(instruction)
            active_qubits = potential_active_qubits
        else:
            # It doesn't fit: push current subcircuit and start a new one
            subcircuits.append(remove_unused_wires(current_subcircuit))
            
            # Start new subcircuit
            current_subcircuit = QuantumCircuit(circuit.qubits, circuit.clbits)
            current_subcircuit.append(instruction)
            active_qubits = op_qubits
            
    # Append the final subcircuit if it contains any operations
    if len(current_subcircuit) > 0:
        subcircuits.append(remove_unused_wires(current_subcircuit))
        
    return subcircuits

def remove_unused_wires(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Removes unused quantum wires from a Qiskit QuantumCircuit.
    
    Args:
        qc (QuantumCircuit): The input circuit containing potentially empty wires.
        
    Returns:
        QuantumCircuit: A new circuit with only the active quantum bits, 
                        maintaining the relative order of the original qubits.
    """
    # 1. Identify all qubits that are actually used in instructions
    used_qubits = set()
    for instruction in qc.data:
        for qubit in instruction.qubits:
            used_qubits.add(qubit)
            
    # 2. Sort them by their original index to preserve relative ordering
    #    (e.g., if q0 and q2 are used, q0 becomes new q0, q2 becomes new q1)
    sorted_used_qubits = sorted(list(used_qubits), key=lambda q: qc.find_bit(q).index)
    
    # 3. Create a mapping from old Qubit objects to new qubit indices
    #    key: old_qubit_object, value: new_index_integer
    qubit_map = {old_q: i for i, old_q in enumerate(sorted_used_qubits)}
    
    # 4. Create the new circuit
    #    We preserve the number of classical bits to ensure measurements still work
    new_qc = QuantumCircuit(len(sorted_used_qubits), qc.num_clbits)
    
    # 5. Copy instructions to the new circuit using the mapping
    for instruction in qc.data:
        # Map the old qubits to the new circuit's qubits
        new_qubits = [new_qc.qubits[qubit_map[q]] for q in instruction.qubits]
        
        # Map classical bits directly by index (preserving their structure)
        new_clbits = [new_qc.clbits[qc.find_bit(c).index] for c in instruction.clbits]
        
        # Append the operation with the remapped bits
        new_qc.append(instruction.operation, new_qubits, new_clbits)
        
    return new_qc