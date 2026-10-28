from __future__ import annotations
import numpy as np
import scipy as sp
from itertools import combinations
from dataclasses import dataclass
from typing import Literal
from collections import defaultdict
from qiskit import QuantumCircuit
from qiskit.synthesis.qft import synth_qft_full
from qiskit.quantum_info import (
    Operator,
    SuperOp,
    process_fidelity,
    SparsePauliOp,
)
from qiskit.converters import circuit_to_dag
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import (
    FakeBrisbane,
    FakeTorino,
    FakeKingston,
    FakeMarrakesh,
    FakeFez,
    FakeBoston,
    FakeMiami,
)
from scipy.optimize import minimize

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
sm = np.array([[0, 1], [0, 0]], dtype=complex)
sp = np.array([[0, 0], [1, 0]], dtype=complex)
L_z = np.array([[1, 0], [0, -1]], dtype=complex) / np.sqrt(2)
XX = np.kron(X, X)
YY = np.kron(Y, Y)
ZZ = np.kron(Z, Z)
XY = np.kron(X, Y)
XZ = np.kron(X, Z)
YZ = np.kron(Y, Z)

ROTATION_AXES = {
    "x": X,
    "sx": X,
    "y": Y,
    "sy": Y,
    "rz": None,
    "id": None,
    "cz": np.kron(Z, Z),
    "cx": np.kron(I2, X),
    "ecr": np.kron(Z, X),
}


@dataclass(frozen=True)
class ErrorChannel:
    """
    One independent physical error source.

    Parameters
    ----------
    label    : name, used for logging/plotting
    operator : normalised operator
    kind     : incoherent or coherent
    contexts : circuit context, where error source is applied
    weight   : physical magnitude of error
    n_qubits : number of qubits the error acts on
    """

    label: str
    operator: np.ndarray
    kind: Literal["incoherent", "coherent"]
    contexts: list[str]
    weights: dict
    n_qubits: int = 1


def create_all_z_observables(num_qubits: int) -> tuple:
    """
    Creates all observables which can be possibly measured at the end of an
    n - qubit quantum circuit. Only creating observables measurable in the
    computational basis.
    """
    list_observables = []
    meta_observables = []
    for k in range(1, num_qubits + 1):
        for subset in combinations(range(num_qubits), k):
            pauli_obs = ["I"] * num_qubits
            for q in subset:
                pauli_obs[q] = "Z"
            pauli_string = "".join(pauli_obs[::-1])
            osbervable = SparsePauliOp.from_list([(pauli_string, 1.0)])
            list_observables.append(osbervable)
            meta_observables.append(
                {"k": k, "qubits": subset, "pauli_str": pauli_string}
            )
    return list_observables, meta_observables


def _create_qft_circuit(n_qubits: int) -> QuantumCircuit:
    """
    Generates a Quantum Fourier Transform (QFT) circuit for a given number of qubits.

    Args:
        n_qubits (int): The number of qubits for the QFT circuit.

    Returns:
        QuantumCircuit: The QFT circuit.
    """
    qft_circuit = synth_qft_full(n_qubits, do_swaps=True)
    return qft_circuit


def _get_ibm_backend(
    backend_name: str, service: QiskitRuntimeService = None, use_fake: bool = False
):
    """
    Fetch IBM backend by name.

    Args:
        backend_name: Backend name (e.g., 'ibm_fez')
        service: QiskitRuntimeService instance
        use_fake: If True, use a fake backend for testing

    Returns:
        Qiskit backend object
    """
    if use_fake:
        target_name = backend_name.lower()
        if not target_name.startswith("ibm_"):
            target_name = f"ibm_{target_name}"
        fake_backends_map = {
            "ibm_fez": FakeFez,
            "ibm_marrakesh": FakeMarrakesh,
            "ibm_kingston": FakeKingston,
            "ibm_brisbane": FakeBrisbane,
            "ibm_torino": FakeTorino,
            "ibm_miami": FakeMiami,
            "ibm_boston": FakeBoston,
        }
        if target_name not in fake_backends_map:
            raise ValueError(
                f"Fake backend '{backend_name}' not recognized. Available options: {list(fake_backends_map.keys())}"
            )
        backend = fake_backends_map[target_name]()
        return backend
    else:
        return service.backend(backend_name)


def _extract_hardware_profile_gates(backend):
    profile = {
        "T1": {},
        "T2": {},
        "topology": {},
        "rot_axis": {},
    }
    properties = backend.properties()
    calibration_time = properties.last_update_date
    gate_set = backend.basis_gates
    gates_1q = []
    gates_2q = []

    for gate in gate_set:
        op = backend.target.operation_from_name(gate)
        if op.num_qubits == 1:
            gates_1q.append(gate)
        elif op.num_qubits == 2:
            gates_2q.append(gate)
        dict_ = {}
        profile[f"{gate}_gate_err"] = dict_
        profile[f"{gate}_rot_err"] = dict_
        profile[f"{gate}_gate_duration"] = dict_
        profile["rot_axis"][gate] = ROTATION_AXES.get(gate)
    if backend.coupling_map:
        profile["topology"] = list(backend.coupling_map)
    for qubit in range(backend.num_qubits):
        try:
            props = backend.qubit_properties(qubit)
            T1 = props.t1
            T2 = props.t2
            profile["T1"][qubit] = T1 if T1 is not None and T1 > 0 else None
            profile["T2"][qubit] = T2 if T2 is not None and T2 > 0 else None

        except (NotImplementedError, AttributeError):
            print(f"No T1/T2 can be extracted for qubit {qubit}")
            profile["T1"][qubit] = None
            profile["T2"][qubit] = None
    for gate_1q_name in gates_1q:
        for qubit in range(backend.num_qubits):
            target = backend.target[gate_1q_name].get((qubit,))
            if target:
                profile[f"{gate_1q_name}_gate_err"][qubit] = (
                    target.error if target.error is not None else 0.0
                )
                profile[f"{gate_1q_name}_rot_err"][qubit] = (
                    np.sqrt(6 * target.error) if target.error is not None else 0.0
                )
                profile[f"{gate_1q_name}_gate_duration"][qubit] = (
                    target.duration if target.duration is not None else 0.0
                )
    for gate_2q_name in gates_2q:
        for edge in profile["topology"]:
            target = backend.target[gate_2q_name].get(edge)
            if target:
                profile[f"{gate_2q_name}_gate_err"][edge] = (
                    target.error if target.error is not None else 0.0
                )
                profile[f"{gate_2q_name}_rot_err"][edge] = (
                    np.sqrt(6 * target.error) if target.error is not None else 0.0
                )
                profile[f"{gate_2q_name}_gate_duration"][edge] = (
                    target.duration if target.duration is not None else 0.0
                )
    return profile, calibration_time


def get_noise_model_profile_backend(
    hardware_profile: dict, basis_gates: list, gate_qubit_counts: dict
):
    """
    Creates the summary of noisy channels based of loaded hardware snapshot.
    """
    descr, channels = _make_sc_err_profile_gates_from_snapshot(
        hardware_profile, basis_gates, gate_qubit_counts
    )
    return {
        "type": "hardware",
        "architecture": "superconducting",
        "noise_profile_descr": descr,
        "noise_profile_channels": channels,
    }


def _make_sc_err_profile_gates_from_snapshot(
    hardware_profile,
    basis_gates,
    gate_qubit_counts,
):
    T1 = hardware_profile["T1"]
    T2 = hardware_profile["T2"]
    T_phi = {}
    for q in T1:
        t1 = T1[q]
        t2 = T2[q]
        if t1 is None or t2 is None or t1 <= 0 or t2 <= 0:
            T_phi[q] = None
            continue
        inv = 1.0 / t2 - 1.0 / (2.0 * t1)
        if inv > 0:
            T_phi[q] = 1.0 / inv
        else:
            T_phi[q] = np.inf
    channels = []
    for gate in basis_gates:
        n_qubits = gate_qubit_counts[gate]
        p_dep = {}
        for loc, gate_err in hardware_profile[f"{gate}_gate_err"].items():
            if gate_err is None:
                continue
            p_dep[loc] = gate_err
        if n_qubits == 1:
            for P, lbl in [
                (X, "X"),
                (Y, "Y"),
                (Z, "Z"),
            ]:
                channels.append(
                    ErrorChannel(
                        label=f"depol_{gate}_{lbl}",
                        operator=P,
                        kind="incoherent",
                        contexts=frozenset({gate}),
                        weights={q: p / 3 for q, p in p_dep.items()},
                        n_qubits=1,
                    )
                )
        elif n_qubits == 2:
            paulis_2q = [
                np.kron(A, B)
                for A in [I2, X, Y, Z]
                for B in [I2, X, Y, Z]
                if not (A is I2 and B is I2)
            ]

            labels_2q = [
                f"{a}{b}" for a in "IXYZ" for b in "IXYZ" if not (a == "I" and b == "I")
            ]
            for P, lbl in zip(paulis_2q, labels_2q):
                channels.append(
                    ErrorChannel(
                        label=f"depol_{gate}_{lbl}",
                        operator=P,
                        kind="incoherent",
                        contexts=frozenset({gate}),
                        weights={loc: p / 15 for loc, p in p_dep.items()},
                        n_qubits=2,
                    )
                )
        duration = hardware_profile[f"{gate}_gate_duration"]
        if not duration:
            continue
        weights_T1 = {}
        weights_Tphi = {}
        for loc, dt in duration.items():
            if isinstance(loc, int):
                qubits = [loc]
            else:
                qubits = list(loc)
            q = qubits[0]
            t1 = T1.get(q)
            tphi = T_phi.get(q)
            if t1 is not None and t1 > 0:
                weights_T1[loc] = dt / t1
            if tphi is not None and tphi > 0 and np.isfinite(tphi):
                weights_Tphi[loc] = dt / tphi
            elif tphi == np.inf:
                weights_Tphi[loc] = 0
        if weights_T1:
            channels.append(
                ErrorChannel(
                    label=f"T1_{gate}",
                    operator=sm,
                    kind="incoherent",
                    contexts=frozenset({gate}),
                    weights=weights_T1,
                    n_qubits=n_qubits,
                )
            )
        if weights_Tphi:
            channels.append(
                ErrorChannel(
                    label=f"Tphi_{gate}",
                    operator=Z,
                    kind="incoherent",
                    contexts=frozenset({gate}),
                    weights=weights_Tphi,
                    n_qubits=n_qubits,
                )
            )
        axis = hardware_profile["rot_axis"].get(gate)
        if axis is not None:
            channels.append(
                ErrorChannel(
                    label=f"rot_{gate}",
                    operator=axis,
                    kind="coherent",
                    contexts=frozenset({gate}),
                    weights=hardware_profile[f"{gate}_rot_err"],
                    n_qubits=n_qubits,
                )
            )
    description = (
        "Superconducting noise model: "
        "T1 relaxation, pure dephasing, "
        "depolarising (1Q+2Q), over-rotation."
    )
    return description, channels


def append_stats(results: dict, circuit: QuantumCircuit):
    active_qubits = set()
    for instruction in circuit.data:
        for qubit in instruction.qubits:
            active_qubits.add(qubit)
    num_qubits = len(active_qubits)
    depth = circuit.depth()
    num_gates = circuit.size()
    results["num_qubits"] = num_qubits
    results["depth"] = depth
    results["num_gates"] = num_gates
    return num_qubits, depth, num_gates


def _get_layer_superoperators(circuit: QuantumCircuit) -> list:
    """
    Given an ideal Qiskit quantum circuit, returns the corresponding list of
    superoperators per layer.
    """
    A_bar_list = []
    dag = circuit_to_dag(circuit)
    active_qubit_objects = set()
    for node in dag.op_nodes():
        active_qubit_objects.update(node.qargs)
    active_qubits_list = list(active_qubit_objects)
    n_active_qubits = len(active_qubits_list)
    qubit_indices = {q: i for i, q in enumerate(active_qubits_list)}
    for layer in dag.layers():
        layer_dag = layer["graph"]
        layer_circuit = QuantumCircuit(n_active_qubits)
        for node in layer_dag.op_nodes():
            mapped_qargs = [qubit_indices[q] for q in node.qargs]
            layer_circuit.append(node.op, mapped_qargs, node.cargs)
        unitary = Operator(layer_circuit)
        superop = SuperOp(unitary)
        A_bar_list.append(superop)
    return A_bar_list


def _calculate_V(A_bar_list: list) -> list:
    """
    Calculates the overall superoperators V for a quantum circuit for every layer j in [1, ..., L]
    by multiplying the superoperators of each moment.
    Formula: V_bar_j = A_bar_j * V_bar_{j-1} with V_bar_0 = I.
    """
    if not A_bar_list:
        return []
    V_bar_list = []
    dim = np.shape(A_bar_list[0])[0]
    identity_matrix = np.eye(dim, dtype=complex)
    V_bar = SuperOp(identity_matrix)
    V_bar_list.append(V_bar)
    for A_bar in A_bar_list:
        if len(V_bar_list) == len(A_bar_list):
            break
        V_bar = A_bar.compose(V_bar)
        V_bar_list.append(V_bar)
    return V_bar_list


def _get_total_ideal_superop_circuit(noiseless_superop_list: list) -> SuperOp:
    """
    Given a list of superoperators per layer, calculates the total ideal
    superoperator for the entire quantum circuit by composing them sequentially.
    """
    if not noiseless_superop_list:
        raise ValueError("The input superoperator list is empty.")
    dim = np.shape(noiseless_superop_list[0])[0]
    identity_matrix = np.eye(dim, dtype=complex)
    S_total = SuperOp(identity_matrix)
    for qiskit_A_bar in noiseless_superop_list:
        S_total = qiskit_A_bar.compose(S_total)
    return S_total


def get_ideal_properties(circuit: QuantumCircuit):
    """
    Calculates the list of ideal superoperator per layer ($\bar{A_j}$) and the concatenated superoperator
    up to this layer($\bar{V_j} = \bar{A_j} * \bar{V_{j-1}}$) per layer. Also calculates the superoperator of
    the whole quantum circuit.
    """
    A_bar_list = _get_layer_superoperators(circuit)
    V_bar_list = _calculate_V(A_bar_list)
    S_ideal = _get_total_ideal_superop_circuit(A_bar_list)
    return A_bar_list, V_bar_list, S_ideal


def _build_adjacency(coupling_map):
    adj = defaultdict(set)
    for a, b in coupling_map:
        adj[a].add(b)
    return dict(adj)


def _context_layers(circuit: QuantumCircuit) -> list[dict]:
    active_physical_qubits = sorted(
        {
            circuit.find_bit(q).index
            for instruction in circuit.data
            for q in instruction.qubits
        }
    )
    num_sub_qubits = len(active_physical_qubits)
    phys_to_sub_idx = {p: i for i, p in enumerate(active_physical_qubits)}
    dag = circuit_to_dag(circuit)
    qubit_context_layers = []
    qubit_to_op_layers = []
    for cycle_index, layer in enumerate(dag.layers()):
        qubit_context = {}
        qubit_to_op = {}
        for node in layer["graph"].op_nodes():
            phys_qubits = [circuit.find_bit(q).index for q in node.qargs]
            sub_qubits = [
                phys_to_sub_idx[q] for q in phys_qubits if q in phys_to_sub_idx
            ]
            gate_name = node.op.name
            for q, phys_q in zip(sub_qubits, phys_qubits):
                qubit_context[q] = {
                    "layer": cycle_index,
                    "gate_name": gate_name,
                    "phys_qubit": phys_qubits[sub_qubits.index(q)],
                    "sub_qubit": q,
                    "target_qubits": sub_qubits,
                }
            if len(sub_qubits) > 0:
                key = tuple(sub_qubits)
                qubit_to_op[key] = {
                    "layer": cycle_index,
                    "gate_name": gate_name,
                    "phys_qubits": phys_qubits,
                    "sub_qubits": sub_qubits,
                }
        qubit_context_layers.append(qubit_context)
        qubit_to_op_layers.append(qubit_to_op)
    return (
        active_physical_qubits,
        qubit_context_layers,
        phys_to_sub_idx,
        num_sub_qubits,
        qubit_to_op_layers,
    )


def _get_active_qubits_layers(
    circuit: QuantumCircuit, num_sub_qubits: int
) -> list[set]:
    active_qubits_per_layer = []
    (
        active_physical_qubits,
        qubit_context_layers,
        _,
        num_sub_qubits,
        qubit_to_op_layers,
    ) = _context_layers(circuit)
    depth = len(qubit_context_layers)
    for idx, layer in enumerate(qubit_context_layers):
        if idx == 0:
            active_qubits_per_layer.append(
                {qubit: "" for qubit in range(num_sub_qubits)}
            )
        else:
            previous_layer = qubit_context_layers[idx - 1]
            active_qubits_layer = {}
            for q in range(num_sub_qubits):
                if q in previous_layer:
                    active_qubits_layer[q] = {
                        "activeness": "active",
                        "gate": previous_layer[q]["gate_name"],
                    }
                else:
                    active_qubits_layer[q] = {"activeness": "inactive"}
            active_qubits_per_layer.append(active_qubits_layer)
    return active_qubits_per_layer


def _embed_operator(op: np.ndarray, qubits: list[int], num_qubits: int) -> np.ndarray:
    if len(qubits) == 1:
        return _kron_1q(op, qubits[0], num_qubits)
    elif len(qubits) == 2:
        return _kron_2q(op, qubits[0], qubits[1], num_qubits)
    else:
        raise ValueError("No other gates than 1 qubit and 2 qubit gates supported")


def _kron_1q(op: np.ndarray, qubit: int, n: int) -> np.ndarray:
    """
    Embeds one qubit operator inside n-qubits space.
    """
    ops = [np.eye(2, dtype=complex)] * n
    ops[qubit] = op
    result = ops[0]
    for o in ops[1:]:
        result = np.kron(result, o)
    return result


def _kron_2q(op: np.ndarray, q1: int, q2: int, n: int) -> np.ndarray:
    """Embeds 2-qubit operator."""
    dim = 2**n
    full = np.zeros((dim, dim), dtype=complex)
    if op.shape == (2, 2):
        op = np.kron(op, op)
    for col in range(dim):
        bits = [(col >> (n - 1 - k)) & 1 for k in range(n)]
        local_col = (bits[q1] << 1) | bits[q2]
        for local_row in range(4):
            amp = op[local_row, local_col]
            new_bits = bits.copy()
            new_bits[q1] = (local_row >> 1) & 1
            new_bits[q2] = local_row & 1
            row = 0
            for b in new_bits:
                row = (row << 1) | b
            full[row, col] += amp
    return full


def _B_incoh(L: np.ndarray) -> np.ndarray:
    """Returns Lindblad generator in superoperator space for a given jump operator L."""
    d = L.shape[0]
    Id = np.eye(d, dtype=complex)
    LdL = L.conj().T @ L
    return 2 * np.kron(L.conj(), L) - np.kron(Id, LdL) - np.kron(LdL.T, Id)


def _B_coh(H: np.ndarray) -> np.ndarray:
    """Returns Lindblad generator in superoperator space for a given Hamiltonian H."""
    d = H.shape[0]
    Id = np.eye(d, dtype=complex)
    return np.kron(Id, H) - np.kron(H.T, Id)


def _get_neighbors(phys_qubits, gate_phys_qubits, adjacency):
    neigh = set()
    for q in gate_phys_qubits:
        adjacent_qubits = adjacency.get(q, ())
        for qubit in adjacent_qubits:
            if qubit in gate_phys_qubits or qubit not in phys_qubits:
                continue
            else:
                neigh.add(qubit)
    return neigh


def _check_for_crosstalk(
    phys_qubits, target_qubits, ajdancency, active_qubits, phys_to_sub_idx
):
    neighs = _get_neighbors(phys_qubits, target_qubits, ajdancency)
    #
    crosstalk_1q = []
    crosstalk_2q = []
    added = False
    for neigh in neighs:
        sub_neigh = phys_to_sub_idx[neigh]
        if active_qubits[sub_neigh]["activeness"] == "active":
            if not added:
                crosstalk_2q.append(
                    {
                        "qubits": list(
                            phys_to_sub_idx[target_qubit]
                            for target_qubit in target_qubits
                        ),
                        "gate": "",
                    }
                )
                added = True
            crosstalk_1q.append(
                {"qubit": sub_neigh, "gate": active_qubits[sub_neigh]["gate"]}
            )
    return crosstalk_1q, crosstalk_2q


def _compute_B_matrices(
    circuit: QuantumCircuit, noise_model: dict, hardware_profile: dict
) -> tuple:
    B_dict = {}
    num_errs_per_layer = []
    coupling_map = hardware_profile["topology"]
    adjacency = _build_adjacency(coupling_map)
    error_channels = noise_model["noise_profile_channels"]
    dag = circuit_to_dag(circuit)
    (
        active_physical_qubits,
        qubit_context_layers,
        phys_to_sub_idx,
        num_sub_qubits,
        qubit_to_op_layers,
    ) = _context_layers(circuit)
    active_qubits_per_layer = _get_active_qubits_layers(circuit, num_sub_qubits)
    for cycle_idx, layer in enumerate(dag.layers()):
        error_idx = 0
        processed_gates = set()
        layer_errors = []
        active_qubits = active_qubits_per_layer[cycle_idx]
        gate_physical_sub_qubits = set()
        gate_sub_qubits = set()
        for gate_qubits, gate_info in qubit_to_op_layers[cycle_idx].items():
            target_qubits = list(gate_qubits)
            physical_target_qubits = [
                key
                for value in target_qubits
                for key, val in phys_to_sub_idx.items()
                if val == value
            ]
            phys_qubits = [
                phys_q
                for phys_q, sub_q in phys_to_sub_idx.items()
                if sub_q in target_qubits
            ]
            gate_physical_sub_qubits.update(physical_target_qubits)
            gate_sub_qubits.update(target_qubits)
            gate_name = gate_info["gate_name"]
            gate_key = tuple(sorted(target_qubits)) + (gate_name,)
            if gate_key in processed_gates:
                continue
            processed_gates.add(gate_key)
            for channel in error_channels:
                if gate_name not in channel.contexts:
                    continue
                if channel.n_qubits != len(target_qubits):
                    continue
                op = _embed_operator(
                    channel.operator,
                    target_qubits,
                    num_sub_qubits,
                )
                B = _B_incoh(op) if channel.kind == "incoherent" else _B_coh(op)
                layer_errors.append(
                    {
                        "B": B,
                        "layer": cycle_idx,
                        "error_idx": error_idx,
                        "error_channel": channel,
                        "target_qubits": physical_target_qubits,
                        "type": "native",
                    }
                )
                error_idx += 1
            if len(target_qubits) == 1:
                continue
            crosstalk_1q, crosstalk_2q = _check_for_crosstalk(
                active_physical_qubits,
                phys_qubits,
                adjacency,
                active_qubits,
                phys_to_sub_idx,
            )
            if crosstalk_1q and crosstalk_2q:
                for depol_channel in error_channels:
                    if depol_channel.n_qubits == 2 and "depol" in depol_channel.label:
                        op = _embed_operator(
                            depol_channel.operator,
                            target_qubits,
                            num_sub_qubits,
                        )
                        B = _B_incoh(op)
                        layer_errors.append(
                            {
                                "B": B,
                                "layer": cycle_idx,
                                "error_idx": error_idx,
                                "error_channel": depol_channel,
                                "target_qubits": physical_target_qubits,
                                "source_qubits": crosstalk_1q,
                                "type": "crosstalk_2q",
                            }
                        )
                        error_idx += 1
                for neigh in crosstalk_1q:
                    gate = neigh["gate"]
                    if gate == "cz":
                        gate = "rz"
                    for depol_channel in error_channels:
                        if (
                            depol_channel.n_qubits == 1
                            and "depol" in depol_channel.label
                            and gate in depol_channel.contexts
                        ):
                            qubit = neigh["qubit"]
                            op = _embed_operator(
                                depol_channel.operator,
                                [qubit],
                                num_sub_qubits,
                            )
                            B = _B_incoh(op)
                            layer_errors.append(
                                {
                                    "B": B,
                                    "layer": cycle_idx,
                                    "error_idx": error_idx,
                                    "error_channel": depol_channel,
                                    "target_qubits": [qubit],
                                    "source_qubits": crosstalk_1q,
                                    "type": f"crosstalk_1q_{gate}",
                                }
                            )
                            error_idx += 1
        idle_qubits = set(range(num_sub_qubits)) - gate_sub_qubits
        idle_physical_qubits = set(active_physical_qubits) - gate_physical_sub_qubits
        gate = "id"
        if idle_qubits:
            for channel in error_channels:
                if "T1_id" in channel.label or "Tphi_id" in channel.label:
                    for idx, idle_qubit in enumerate(idle_qubits):
                        physical_qubit = list(idle_physical_qubits)[idx]
                        op = _embed_operator(
                            channel.operator, [idle_qubit], num_sub_qubits
                        )
                        B = _B_incoh(op)
                        layer_errors.append(
                            {
                                "B": B,
                                "layer": cycle_idx,
                                "error_idx": error_idx,
                                "error_channel": channel,
                                "target_qubits": [physical_qubit],
                                "type": "idle",
                            }
                        )
                        error_idx += 1
                else:
                    continue
        B_dict[cycle_idx] = layer_errors
        num_errs_per_layer.append(error_idx)
    return B_dict, num_errs_per_layer, phys_to_sub_idx


def _compute_M_bound(
    V_bar_list: list[SuperOp], B_dict: dict, num_errs_per_layer: list[int]
):
    """
    Calculates the M matrices as M = 1/N*[M_1, M_2, ..., M_N] with
    M_j = [vec(\dagger{V_j}B_j,1 V_j), vec(\dagger{V_j}B_j,2 V_j),...,vec(\dagger{V_j}B_j,l_j V_j)].
    """
    N = len(num_errs_per_layer)
    M_blocks = []
    for j in range(N):
        Vj = V_bar_list[j].data
        Vj_dag = Vj.conj().T
        lj = num_errs_per_layer[j]
        layer_errors = B_dict.get(j, [])
        Mj_columns = []
        for i in range(lj):
            Bji = layer_errors[i]["B"]
            if isinstance(Bji, SuperOp):
                Bji = Bji.data
            transformed = Vj_dag @ Bji @ Vj
            Mj_columns.append(transformed.flatten(order="F"))
        if Mj_columns:
            Mj = np.column_stack(Mj_columns)
        else:
            dim_squared = Vj.shape[0]
            Mj = np.empty((dim_squared, 0), dtype=complex)
        M_blocks.append(Mj)
    M_full = np.hstack(M_blocks) / N
    return M_full


def _calculate_general_norm_based_gamma_bound(M_full: np.ndarray, B_dict: dict):
    weights = []
    norm_weight_matrix = []
    for layer_idx in sorted(B_dict.keys()):
        layer_errors = B_dict[layer_idx]
        raw_weights_layer = []
        for err in layer_errors:
            w = err["error_channel"].weights
            if np.isscalar(w):
                raw_weights_layer.append(float(w))
            elif isinstance(w, dict):
                targets = err.get("target_qubits", None)
                if len(targets) == 2:
                    targets = tuple(targets)
                elif len(targets) == 1:
                    targets = targets[0]
                else:
                    ValueError("Target qubits must be 2 or 1")
                try:
                    raw_weights_layer.append(float(w[targets]))
                except KeyError:
                    try:
                        raw_weights_layer.append(float(w[targets[::-1]]))
                    except KeyError:
                        pass
            else:
                raw_weights_layer.append(float(w))
        raw_weights_layer = np.asarray(raw_weights_layer, dtype=float)
        s = raw_weights_layer.sum()
        if s < 0:
            raise ValueError(f"Layer {layer_idx} has invalid weight sum: {s}")
        elif s == 0:
            norm_weights_layer = np.zeros_like(raw_weights_layer)
        else:
            norm_weights_layer = raw_weights_layer / s
        norm_weight_matrix.append(norm_weights_layer)
        weights.extend(norm_weights_layer.tolist())
    weights = np.asarray(weights, dtype=float)
    num_errors = sum(len(w_layer) for w_layer in norm_weight_matrix)
    if M_full.shape[1] != num_errors:
        raise ValueError(
            f"Mismatch: M has {M_full.shape[1]} columns "
            f"but got {num_errors} weights"
        )
    spectral_norm = np.linalg.norm(M_full, ord=2)
    weight_factor = np.sqrt(sum(np.sum(w_layer**2) for w_layer in norm_weight_matrix))
    gamma_bound = spectral_norm * weight_factor
    return (
        gamma_bound,
        norm_weight_matrix,
    )


def _G_theta_optimized(
    param: np.ndarray, array_products: np.ndarray, num_errs_per_layer: list[int]
):
    N = len(num_errs_per_layer)
    array_flat = array_products.reshape(len(array_products), -1)
    G = param @ array_flat
    G = G.reshape(array_products.shape[1:]) / N
    return G


def _compute_gamma_optimization(
    V_bar_list: list[SuperOp],
    B_dict: dict,
    num_errs_per_layer: list[int],
    weight_mat: list[list],
    method: str = "L-BFGS-B",
    seed: int = 42,
    iterations: int = 100,
):
    """
    optimizes gamma with dynamical bounds.
    """
    np.random.seed(seed)
    N = len(num_errs_per_layer)
    bounds = []
    for j in range(N):
        l_j = num_errs_per_layer[j]
        layer_errors = B_dict.get(j, [])
        for k in range(l_j):
            channel = layer_errors[k]["error_channel"]
            limit = weight_mat[j][k]
            if channel.kind == "incoherent":
                bounds.append((0.0, limit))
            elif channel.kind == "coherent":
                bounds.append((-limit, limit))
    dim = len(bounds)
    if dim == 0:
        return 0.0, 0.0, 0.0
    array_products = []
    for j, V_bar_j in enumerate(V_bar_list):
        Vj = V_bar_j.data
        layer_errors = B_dict.get(j, [])
        B_arrays = np.asarray([error_channel["B"] for error_channel in layer_errors])
        array_prod = Vj.conj().T @ B_arrays @ Vj
        array_products.append(array_prod)
    array_products = np.concatenate(array_products, axis=0)

    def objective(param_flat):
        G = _G_theta_optimized(param_flat, array_products, num_errs_per_layer)
        return -np.linalg.norm(G, ord=2)

    best_gamma = 0.0
    all_gammas = []
    successful_results = 0
    for _ in range(iterations):
        x0 = np.zeros(dim)
        for idx, (low, high) in enumerate(bounds):
            x0[idx] = np.random.uniform(low, high)

        result = minimize(
            objective,
            x0=x0,
            bounds=bounds,
            method=method,
            options={
                "maxiter": 1000,
                "maxfun": 1000000,
            },
        )
        if result.success:
            successful_results += 1
            gamma = -result.fun
            all_gammas.append(gamma)
            best_gamma = max(best_gamma, gamma)
        else:
            print(f"Minimization failed due to {result.message}")
    print(
        f"{successful_results} out of {iterations} were successful: {100*successful_results/iterations}%"
    )
    if all_gammas:
        mean_gamma = np.mean(all_gammas)
        std_gamma = np.std(all_gammas)
    else:
        mean_gamma, std_gamma = 0.0, 0.0

    return best_gamma, mean_gamma, std_gamma


def perform_theoretical_analysis(
    results: dict,
    circuit: QuantumCircuit,
    V_bar_list: list[np.ndarray],
    noise_model: dict,
    hardware_profile: dict,
    b2_iterations: int,
):
    """
    performs theoretical robustness analysis with gamma_b2 bound and gamma_b4 bound.
    """
    B_dict, num_errs_per_layer, phys_to_sub_index = _compute_B_matrices(
        circuit=circuit, noise_model=noise_model, hardware_profile=hardware_profile
    )
    M_bound = _compute_M_bound(
        V_bar_list=V_bar_list, B_dict=B_dict, num_errs_per_layer=num_errs_per_layer
    )
    gamma_b4, norm_weight_matrix = _calculate_general_norm_based_gamma_bound(
        M_bound, B_dict
    )
    gamma_b2, mean_gamma_b2, std_gamma_b2 = _compute_gamma_optimization(
        V_bar_list,
        B_dict,
        num_errs_per_layer,
        norm_weight_matrix,
        iterations=b2_iterations,
    )
    results["gamma_b4"] = gamma_b4
    results["gamma_b2"] = gamma_b2
    results["mean_gamma_b2"] = mean_gamma_b2
    results["std_gamma_b2"] = std_gamma_b2

    return B_dict, gamma_b2, gamma_b4, norm_weight_matrix, phys_to_sub_index


def _calc_readout_variation(
    properties,
    phys_to_sub_index: dict,
):
    product = 1
    for qubit in phys_to_sub_index.keys():
        prob_01 = properties.qubit_property(qubit, "prob_meas0_prep1")[0]
        prob_10 = properties.qubit_property(qubit, "prob_meas1_prep0")[0]
        term = 1 - max(prob_01, prob_10)
        product = product * term
    return 1 - product


def _incoherent_worst_case_fidelity_bound(
    n: int, delta: float, N: int, gamma: float
) -> float:
    """
    Calculates the worst-case fidelity bound for noise induced by incoherent or coherent errors or a combination.
    """
    bracket_term = ((N - 1) / 2) * delta + 2 ** (n / 2) * gamma
    term = 2 ** (n / 2) * delta * N * bracket_term
    fidelity = 1 - term
    return fidelity


def _calc_total_variation_bound(n, delta, N, gamma, readout_variation):
    bracket_term = ((N - 1) / 2) * delta + 2 ** (n / 2) * gamma
    term = 2 ** (n / 2) * delta * N * bracket_term
    circuit_variation = 0.5 * term
    total_variation = circuit_variation + readout_variation
    return total_variation, circuit_variation


def _calc_observable_bound(obs: SparsePauliOp, total_variation: float):
    norm = np.linalg.norm(obs.to_matrix(), ord=np.inf)
    term = 2 * norm * total_variation
    return term


def append_theoretical_bounds(
    results,
    num_qubits,
    delta,
    depth,
    gamma_b4,
    gamma_b2,
    readout_variation,
    obs,
):
    worst_case_fidelity_bound_b4 = _incoherent_worst_case_fidelity_bound(
        num_qubits, delta, depth, gamma_b4
    )
    worst_case_total_variation_b4, circuit_variation_b4 = _calc_total_variation_bound(
        num_qubits, delta, depth, gamma_b4, readout_variation
    )
    worst_case_observable_bound_b4 = _calc_observable_bound(
        obs, worst_case_total_variation_b4
    )
    worst_case_fidelity_bound_b2 = _incoherent_worst_case_fidelity_bound(
        num_qubits, delta, depth, gamma_b2
    )
    worst_case_total_variation_b2, circuit_variation_b2 = _calc_total_variation_bound(
        num_qubits, delta, depth, gamma_b2, readout_variation
    )
    worst_case_observable_bound_b2 = _calc_observable_bound(
        obs, worst_case_total_variation_b2
    )
    results["worst_case_fidelity_bound_b4"] = worst_case_fidelity_bound_b4
    worst_case_infidelity_bound_b4 = 1 - worst_case_fidelity_bound_b4
    results["worst_case_infidelity_bound_b4"] = worst_case_infidelity_bound_b4
    results["worst_case_total_variation_b4"] = worst_case_total_variation_b4
    results["circuit_variation_b4"] = circuit_variation_b4
    results["readout_variation"] = readout_variation
    results["worst_case_observable_bound_b4"] = worst_case_observable_bound_b4
    results["worst_case_fidelity_bound_b2"] = worst_case_fidelity_bound_b2
    worst_case_infidelity_bound_b2 = 1 - worst_case_fidelity_bound_b2
    results["worst_case_infidelity_bound_b2"] = worst_case_infidelity_bound_b2
    results["worst_case_total_variation_b2"] = worst_case_total_variation_b2
    results["circuit_variation_b2"] = circuit_variation_b2
    results["worst_case_observable_bound_b2"] = worst_case_observable_bound_b2


def _get_noise_superop_list_from_B_dict(
    B_dict: dict[int, list[dict]],
    weight_mat: list[list],
    delta: float,
    seed: int = None,
) -> list[SuperOp]:
    np.random.seed(seed)
    dim = None
    for layer_errors in B_dict.values():
        if layer_errors:
            dim = layer_errors[0]["B"].shape[0]
            break
    if dim is None:
        raise ValueError("B_dict contains no error entries to process.")
    noise_superop_list = []
    for layer in sorted(B_dict.keys()):
        layer_errors = B_dict[layer]
        lj = len(layer_errors)
        M_e = np.zeros((dim, dim), dtype=complex)
        if lj == 0:
            noise_superop_list.append(np.eye(dim, dtype=complex))
            continue
        for k, error_entry in enumerate(layer_errors):
            B_matrix = error_entry["B"]
            channel = error_entry["error_channel"]
            limit = weight_mat[layer][k]
            if channel.kind == "incoherent":
                param = np.random.uniform(0, delta * limit)
                M_e += param * B_matrix
            elif channel.kind == "coherent":
                param = np.random.uniform(-delta * limit, delta * limit)
                M_e += -1j * param * B_matrix
            else:
                raise ValueError(f"Unknown channel kind: {channel.kind}")
        noise_superop = sp.linalg.expm(M_e)
        noise_superop = SuperOp(noise_superop)
        noise_superop_list.append(noise_superop)
    return noise_superop_list


def _get_total_superop_circuit(noiseless_superop_list, noise_superop_list):
    dim = np.shape(noiseless_superop_list[0])[0]
    identity_matrix = np.eye(dim, dtype=complex)
    S_total_after = SuperOp(identity_matrix)
    for A_bar, A_e in zip(noiseless_superop_list, noise_superop_list):
        S_total_after = A_e.compose(A_bar).compose(S_total_after)
    return S_total_after


def _calc_fidelity_superop(ideal_superop: SuperOp, noisy_superop: SuperOp) -> tuple:
    """
    Calculates the fidelity (and infidelity) between an superoperator representation of an ideal quantum operation
    and a superoperator representation of a noisy quantum operation.
    """
    S1 = ideal_superop.data
    S2 = noisy_superop.data
    try:
        fidelity = process_fidelity(noisy_superop, ideal_superop)
        infidelity = 1 - fidelity
    except np.linalg.LinAlgError as e:
        print(e)
    return fidelity, infidelity


def perform_simulations(
    results,
    B_dict,
    weight_mat,
    delta,
    A_bar_list,
    S_total_ideal,
    sim_iterations,
):
    fidelities = []
    infidelities = []
    for i in range(sim_iterations):
        noise_superop_list = _get_noise_superop_list_from_B_dict(
            B_dict, weight_mat, delta, seed=i
        )
        S_total_after = _get_total_superop_circuit(A_bar_list, noise_superop_list)
        fidelity, infidelity = _calc_fidelity_superop(S_total_ideal, S_total_after)
        fidelities.append(fidelity)
        infidelities.append(infidelity)
    average_fidelity = np.mean(fidelities)
    results["average_fidelity_simulated"] = average_fidelity
    average_infidelity = np.mean(infidelities)
    results["average_infidelity_simulated"] = average_infidelity
    std_fidelity = np.std(fidelities)
    results["std_fidelity_simulated"] = std_fidelity
    std_infidelity = np.std(infidelities)
    results["std_infidelity_simulated"] = std_infidelity
    worst_fidelity = min(fidelities)
    results["worst_fidelity_simulated"] = worst_fidelity
    worst_infidelity = max(infidelities)
    results["worst_infidelity_simulated"] = worst_infidelity
