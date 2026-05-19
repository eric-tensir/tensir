import json
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class IterationRecord:
    iteration: int
    max_gradient: float
    selected_operator: str
    energy: float
    optimal_params: list


@dataclass
class RunRecord:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    molecule: str = ""
    basis_set: str = ""
    num_qubits: int = 0
    provider: str = ""
    backend_name: str = ""
    is_simulator: bool = True
    qubit_mapping: str = ""
    final_energy: float = 0.0
    exact_energy: float = 0.0
    error_mha: float = 0.0
    num_iterations: int = 0
    iterations: list = field(default_factory=list)
    tensir_version: str = "0.1.0"


class TensirLogger:
    def __init__(self, molecule, basis_set, num_qubits,
                 provider, backend_name, is_simulator=True,
                 qubit_mapping="parity", exact_energy=None):
        self.run = RunRecord(
            molecule=molecule,
            basis_set=basis_set,
            num_qubits=num_qubits,
            provider=provider,
            backend_name=backend_name,
            is_simulator=is_simulator,
            qubit_mapping=qubit_mapping,
            exact_energy=exact_energy or 0.0,
        )
        self.exact_energy = exact_energy

    def log_iteration(self, iteration, max_gradient, selected_operator, energy, optimal_params):
        record = IterationRecord(
            iteration=iteration,
            max_gradient=max_gradient,
            selected_operator=str(selected_operator),
            energy=energy,
            optimal_params=list(optimal_params),
        )
        self.run.iterations.append(asdict(record))

    def finalize(self, final_energy):
        self.run.final_energy = final_energy
        self.run.num_iterations = len(self.run.iterations)
        if self.exact_energy:
            self.run.error_mha = abs(final_energy - self.exact_energy) * 1000

    def save(self, path="tensir_run.json"):
        with open(path, "w") as f:
            json.dump(asdict(self.run), f, indent=2)
        print(f"Run saved → {path}")
        return path


    def log(self, iteration, max_gradient, selected_operator, energy, params):
        self.log_iteration(iteration, max_gradient, selected_operator, energy, params)

    def done(self, final_energy, path="tensir_run.json"):
        self.finalize(final_energy)
        return self.save(path)
