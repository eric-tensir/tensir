import numpy as np
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

# ── Operator pool ────────────────────────────────────────────
pool = [
    SparsePauliOp.from_list([("XY", 1.0), ("YX", -1.0)]),
    SparsePauliOp.from_list([("XX", 1.0), ("YY",  1.0)]),
    SparsePauliOp.from_list([("ZX", 1.0), ("IY", -1.0)]),
    SparsePauliOp.from_list([("ZY", 1.0), ("IX",  1.0)]),
]

# ── Gradient ─────────────────────────────────────────────────
def compute_gradient(sv, op):
    comm = (hamiltonian @ op - op @ hamiltonian).simplify()
    return abs(sv.expectation_value(comm))

# ── ADAPT-VQE loop ───────────────────────────────────────────
selected_ops = []
opt_params   = []
threshold    = 1e-3
estimator    = StatevectorEstimator()
result       = None

print("Starting ADAPT-VQE — H2 STO-3G IBM emulator\n")

for iteration in range(1, 10):
    qc_now = QuantumCircuit(2)
    qc_now.x(0)
    for op, p in zip(selected_ops, opt_params):
        evo = PauliEvolutionGate(op, time=p, synthesis=LieTrotter())
        qc_now.append(evo, range(2))
    sv = Statevector(qc_now.decompose())

    grads     = [compute_gradient(sv, op) for op in pool]
    best_idx  = int(np.argmax(grads))
    best_grad = grads[best_idx]
    print(f"Iter {iteration} | max gradient = {best_grad:.6f}")

    if best_grad < threshold:
        print("Converged.")
        break

    selected_ops.append(pool[best_idx])
    n      = len(selected_ops)
    params = ParameterVector("t", n)

    qc = QuantumCircuit(2)
    qc.x(0)
    for op, p in zip(selected_ops, params):
        evo = PauliEvolutionGate(op, time=p, synthesis=LieTrotter())
        qc.append(evo, range(2))
    ansatz = qc.decompose()

    vqe = VQE(estimator, ansatz, COBYLA(maxiter=500))
    vqe.initial_point = np.zeros(n)
    result = vqe.compute_minimum_eigenvalue(hamiltonian)
    opt_params = list(result.optimal_point)
    print(f"         energy    = {result.eigenvalue:.8f} Ha")

print(f"\nFinal energy : {result.eigenvalue:.8f} Ha")
print(f"Exact        : -1.85727503 Ha")
print(f"Error        : {abs(result.eigenvalue - (-1.85727503))*1000:.4f} mHa")
print(f"Operators    : {len(selected_ops)}")

