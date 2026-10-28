import os
from qiskit import qpy
import functions as fc
import pickle
from datetime import datetime
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit.quantum_info import Statevector
from qiskit.transpiler import generate_preset_pass_manager


def execute_jobs_different_transpilations(service, num_qubits, optimization_level):
    alg_name = "qft"
    all_z_observables, _ = fc.create_all_z_observables(num_qubits=num_qubits)
    quantum_algorithm = fc._create_qft_circuit(n_qubits=num_qubits)
    experiment_dir = f"Experimental_Data"
    os.makedirs(experiment_dir, exist_ok=True)
    psi_ideal = Statevector.from_instruction(quantum_algorithm)
    ideal_expectation_values = [
        np.real(psi_ideal.expectation_value(observable))
        for observable in all_z_observables
    ]
    for backend_name in [
        "ibm_fez",
    ]:
        backend = fc._get_ibm_backend(
            backend_name=backend_name, service=service, use_fake=False
        )
        properties = backend.properties()
        pm = generate_preset_pass_manager(
            backend=backend, optimization_level=optimization_level
        )
        transpiled_circuit = pm.run(quantum_algorithm)
        experiment_dir = f"Experimental_Data/{alg_name}_{num_qubits}_{backend_name}_{optimization_level}"
        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, "transpiled_circuit.qpy"), "wb") as f:
            qpy.dump(transpiled_circuit, f)
        with open(os.path.join(experiment_dir, "backend_properties.pkl"), "wb") as f:
            pickle.dump(properties, f)
        profile, calibration_time = fc._extract_hardware_profile_gates(backend=backend)
        hardware_snapshot_metadata = {
            "backend": backend_name,
            "is_fake": False,
            "num_qubits": num_qubits,
            "optimization_level": optimization_level,
            "basis_gates": list(backend.basis_gates),
            "gate_qubit_counts": {
                gate: backend.target.operation_from_name(gate).num_qubits
                for gate in backend.basis_gates
            },
            "calibration_time": calibration_time,
            "hardware_profile": profile,
            "saving_timestamp": datetime.now(),
        }
        with open(
            os.path.join(experiment_dir, "hardware_snapshots_metadata.pkl"), "wb"
        ) as f:
            pickle.dump(hardware_snapshot_metadata, f)
        mapped_observables = [
            observable.apply_layout(transpiled_circuit.layout)
            for observable in all_z_observables
        ]
        estimator = Estimator(backend)
        estimator.options.resilience_level = 0
        estimator.options.twirling.enable_gates = False
        estimator.options.twirling.enable_measure = False
        estimator.options.dynamical_decoupling.enable = False
        estimator.options.default_shots = 100000
        confirm = input(
            f"Runnin job of algorithm {alg_name} (Qubits: {num_qubits}) with fake backend = {use_fake} on Backend {backend_name}, are you sure to proceed? (y/n): "
        ).lower()
        if confirm != "y":
            print("Job submission cancelled by user")
            continue
        job = estimator.run([(transpiled_circuit, mapped_observables)])
        print(f"Job submitted with id {job.job_id()}")
        job_result = job.result()
        print(f"Results retrieved from job with id {job.job_id()}")
        pub_result = job.result()[0]
        expectation_values = pub_result.data.evs
        statistical_errors = pub_result.data.stds
        absolute_errors = np.abs(
            np.array(expectation_values) - np.array(ideal_expectation_values)
        )
        results = {
            "job_id": {job.job_id()},
            "ideal": {
                "expectation_values": np.asarray(ideal_expectation_values),
            },
            "hardware": {
                "expectation_values": np.asarray(expectation_values),
                "stds": np.asarray(statistical_errors),
                "observable_difference": np.asarray(absolute_errors),
            },
        }
        with open(os.path.join(experiment_dir, "results.pkl"), "wb") as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    service = QiskitRuntimeService()
    num_qubits = 3
    algorithm = "qft"
    for optimization_level in [0, 1, 2, 3]:
        execute_jobs_different_transpilations(service, num_qubits, optimization_level)
