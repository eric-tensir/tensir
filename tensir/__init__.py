from tensir.logger import TensirLogger

def start(molecule, basis_set, num_qubits, provider, 
          backend_name="simulator", exact_energy=None,
          is_simulator=True, qubit_mapping="parity"):
    return TensirLogger(
        molecule=molecule,
        basis_set=basis_set,
        num_qubits=num_qubits,
        provider=provider,
        backend_name=backend_name,
        is_simulator=is_simulator,
        qubit_mapping=qubit_mapping,
        exact_energy=exact_energy,
    )
