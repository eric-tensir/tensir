"""
H2 ADAPT-VQE on QuEra Bloqade Digital Emulator (local, no API needed).

Uses Bloqade's qasm2.extended dialect with the pauli_exponential helper
from the Bloqade docs for the variational ansatz, and StackMemorySimulator
for exact statevector simulation. This is the "exact simulator" reference
run — no noise, no shots — to be compared against Gemini neutral-atom
hardware once access becomes available.
"""

import math
import numpy as np
import tensir
from bloqade import qasm2
from bloqade.pyqrack import StackMemorySimulator

# ── Pauli exponentiation helpers (verbatim from Bloqade docs) ──
@qasm2.extended
def zzzz_gadget(targets: tuple[qasm2.Qubit, ...], gamma: float):
    for i in range(len(targets) - 1):
        qasm2.cx(targets[i], targets[i + 1])
    qasm2.rz(targets[-1], gamma)
    for j in range(len(targets) - 1):
        qasm2.cx(targets[-j - 1], targets[-j - 2])

@qasm2.extended
def pauli_basis_change(targets: tuple[qasm2.Qubit, ...], start: str, end: str):
    for i in range(len(targets)):
        qubit = targets[i]
        start_pauli = start[i]
        end_pauli = end[i]
        target = start_pauli + end_pauli
        if target == "ZX":
            qasm2.ry(qubit, math.pi / 2)
        elif target == "ZY":
            qasm2.rx(qubit, -math.pi / 2)
        elif target == "XY":
            qasm2.rz(qubit, math.pi / 2)
        elif target == "XZ":
            qasm2.ry(qubit, -math.pi / 2)
        elif target == "YX":
            qasm2.rz(qubit, -math.pi / 2)
        elif target == "YZ":
            qasm2.rx(qubit, math.pi / 2)

@qasm2.extended
def pauli_exponential(targets: tuple[qasm2.Qubit, ...], pauli: str, gamma: float):
    pauli_basis_change(targets=targets, start="Z" * len(targets), end=pauli)
    zzzz_gadget(targets=targets, gamma=gamma)
    pauli_basis_change(targets=targets, start=pauli, end="Z" * len(targets))

# ── Hamiltonian + pool (numpy, for expectation values) ────────
I2 = np.eye(2, dtype=complex)
X  = np.array([[0, 1], [1, 0]], dtype=complex)
Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z  = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = {'I': I2, 'X': X, 'Y': Y, 'Z': Z}

def kron_pauli(s):
    mat = np.array([[1.+0j]])
    for c in s:
        mat = np.kron(mat, PAULI[c])
    return mat

ham_terms = [
    ("II", -1.0523732), ("IZ",  0.3979374), ("ZI", -0.3979374),
    ("ZZ", -0.0112801), ("XX",  0.1809312),
]
H_mat = sum(c * kron_pauli(p) for p, c in ham_terms)

pool_terms = [
    [("XY", 1.0), ("YX", -1.0)],
    [("XX", 1.0), ("YY",  1.0)],
    [("ZX", 1.0), ("IY", -1.0)],
    [("ZY", 1.0), ("IX",  1.0)],
]
pool_mats = [sum(c * kron_pauli(p) for p, c in terms) for terms in pool_terms]

# ── Bloqade emulator ─────────────────────────────────────────
emulator = StackMemorySimulator(min_qubits=2)

# Conventions note:
# - Qiskit SparsePauliOp("XY"): leftmost char = highest qubit (X on q[1], Y on q[0])
# - Bloqade pauli_exponential(targets, "XY"): pauli[i] applies to targets[i]
#   → we REVERSE the Qiskit string when feeding Bloqade.
# - Qiskit PauliEvolutionGate(op, time=p) = exp(-i*p*op)
# - Bloqade pauli_exponential(..., gamma) = exp(-i*gamma/2 * P)
#   → gamma = 2*p*coeff for each Pauli term in the Lie-Trotter product.

def build_ansatz_factory(pauli_terms_qiskit):
    # All pool operators have exactly 2 Pauli terms — unroll manually
    # (Bloqade qasm2 doesn't support tuple unpacking inside the kernel)
    p0_rev = pauli_terms_qiskit[0][0][::-1]
    c0     = pauli_terms_qiskit[0][1]
    p1_rev = pauli_terms_qiskit[1][0][::-1]
    c1     = pauli_terms_qiskit[1][1]

    @qasm2.extended
    def ansatz(theta: float):
        register = qasm2.qreg(2)
        qasm2.x(register[0])  # reference state |01⟩ (HF for H2 parity)
        pauli_exponential((register[0], register[1]), p0_rev, 2.0 * theta * c0)
        pauli_exponential((register[0], register[1]), p1_rev, 2.0 * theta * c1)
        return register
    return ansatz

def get_statevector(ansatz_kernel, theta):
    task = emulator.task(ansatz_kernel, args=(theta,))
    results = task.run()
    state = emulator.quantum_state(results)
    return np.array(state.eigenvectors[:, 0], dtype=complex)

def expval(sv):
    return float(np.real(sv.conj() @ H_mat @ sv))

# ── Tensir ───────────────────────────────────────────────────
run = tensir.start(
    molecule="H2",
    basis_set="STO-3G",
    num_qubits=2,
    provider="quera",
    backend_name="bloqade_digital_emulator",
    exact_energy=-1.85727503,
)

# ── ADAPT: select pool operator with largest gradient ────────
ref_sv = np.zeros(4, dtype=complex)
ref_sv[1] = 1.0  # |01⟩ in Qiskit little-endian = state[1]

def compute_gradient(sv, op_mat):
    # [H, op] is anti-Hermitian → expectation value is purely imaginary
    # abs(complex(...)) gives the magnitude (the actual gradient)
    comm = H_mat @ op_mat - op_mat @ H_mat
    return abs(complex(sv.conj() @ comm @ sv))

grads = [compute_gradient(ref_sv, op) for op in pool_mats]
best_idx = int(np.argmax(grads))
best_grad = grads[best_idx]
selected_terms = pool_terms[best_idx]
print(f"best_grad={best_grad:.6f} (operator: {selected_terms})", flush=True)

# ── SPSA optimizer ───────────────────────────────────────────
ansatz_kernel = build_ansatz_factory(selected_terms)

def cost(p):
    return expval(get_statevector(ansatz_kernel, float(p[0])))

def spsa(cost_fn, x0, n_iter=15, a=0.3, c=0.1):
    x = np.array(x0, dtype=float)
    for k in range(1, n_iter + 1):
        ak = a / k**0.602
        ck = c / k**0.101
        delta = 2 * np.random.randint(0, 2, size=len(x)) - 1
        lp = cost_fn(x + ck * delta)
        lm = cost_fn(x - ck * delta)
        grad = (lp - lm) / (2 * ck * delta)
        x -= ak * grad
        print(f"  SPSA iter {k}: energy={lp:.6f}", flush=True)
    return x, cost_fn(x)

opt_params, result_energy = spsa(cost, x0=[0.0])
opt_params = list(opt_params)
print(f"Final energy={result_energy:.8f} Ha | Error={abs(result_energy-(-1.85727503))*1000:.4f} mHa")

# ── Log to tensir ────────────────────────────────────────────
op_str = f"SparsePauliOp({[p for p,_ in selected_terms]},\n              coeffs={[complex(c) for _,c in selected_terms]})"
run.log(1, best_grad, op_str, float(result_energy), opt_params)
run.done(float(result_energy), "examples/outputs/h2_quera_simulator.json")
print("JSON saved.")
