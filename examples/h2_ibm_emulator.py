import numpy as np
import tensir
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA

# ── Hamiltonian ──────────────────────────────────────────────
hamiltonian = SparsePauliOp.from_list([
    ("II", -1.0523732),
    ("IZ",  0.3979374),
    ("ZI", -0.3979374),
    ("ZZ", -0.0112801),
    ("XX",  0.1809312),
])

pool = [
    SparsePauliOp.from_list([("XY", 1.0), ("YX", -1.0)]),
    SparsePauliOp.from_list([("XX", 1.0), ("YY",  1.0)]),
    SparsePauliOp.from_list([("ZX", 1.0), ("IY", -1.0)]),
    SparsePauliOp.from_list([("ZY", 1.0), ("IX",  1.0)]),
]

# ── Tensir ───────────────────────────────────────────────────
run = tensir.start(
    molecule="H2",
    basis_set="STO-3G",
    num_qubits=2,
    provider="ibm",
    backend_name="aer_simulator",
    exact_energy=-1.85727503,
)

# ── ADAPT-VQE ────────────────────────────────────────────────
def compute_gradient(sv, op):
    comm = (hamiltonian @ op - op @ hamiltonian).simplify()
    return abs(sv.expectation_value(comm))

selected_ops, opt_params = [], []
estimator = StatevectorEstimator()
result = None

for iteration in range(1, 10):
    qc_now = QuantumCircuit(2)
    qc_now.x(0)
    for op, p in zip(selected_ops, opt_params):
        qc_now.append(PauliEvolutionGate(op, time=p, synthesis=LieTrotter()), range(2))
    sv = Statevector(qc_now.decompose())

    grads = [compute_gradient(sv, op) for op in pool]
    best_idx = int(np.argmax(grads))
    best_grad = grads[best_idx]

    if best_grad < 1e-3:
        break

    selected_ops.append(pool[best_idx])
    n = len(selected_ops)
    params = ParameterVector("t", n)
    qc = QuantumCircuit(2)
    qc.x(0)
    for op, p in zip(selected_ops, params):
        qc.append(PauliEvolutionGate(op, time=p, synthesis=LieTrotter()), range(2))

    vqe = VQE(estimator, qc.decompose(), COBYLA(maxiter=500))
    vqe.initial_point = np.zeros(n)
    result = vqe.compute_minimum_eigenvalue(hamiltonian)
    opt_params = list(result.optimal_point)

    run.log(iteration, best_grad, pool[best_idx], float(result.eigenvalue), opt_params)

run.done(float(result.eigenvalue), "examples/outputs/h2_ibm_emulator.json")
print(f"Energy: {result.eigenvalue:.8f} Ha | Error: {abs(result.eigenvalue - (-1.85727503))*1000:.4f} mHa")

