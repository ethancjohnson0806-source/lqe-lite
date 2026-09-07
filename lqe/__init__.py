"""LQE Lite — Pure NumPy Quantum Engine (Pyodide-compatible subset)."""
from .core.statevector import StatevectorSim
from .core.stabilizer import StabilizerSim
from .solvers.vqe_ground import VQE
from .solvers.qaoa import QAOA
from .solvers.orchestrator import auto_solve, tfi_hamiltonian
from .tensor_networks.mps_simulator import MPSStateAdaptive, mps_ghz_state
MPSSimulator = MPSStateAdaptive

__all__ = [
    "StatevectorSim", "StabilizerSim", "VQE", "QAOA",
    "auto_solve", "tfi_hamiltonian", "MPSSimulator",
    "MPSStateAdaptive", "mps_ghz_state",
]
