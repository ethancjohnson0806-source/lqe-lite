"""Pure-NumPy solver selection for the browser build."""
import numpy as np
from .vqe_ground import VQE, tfi_hamiltonian
from .qaoa import QAOA

def auto_solve(H, n_qubits, problem_type="generic", **kwargs):
    """Select a browser-safe solver and return its result."""
    if problem_type.lower() == "maxcut":
        edges = kwargs.get("edges", [])
        return QAOA(n_qubits, edges, p=kwargs.get("p", 1), max_iterations=kwargs.get("max_iterations", 80)).solve()
    return VQE(n_qubits, H, num_layers=kwargs.get("num_layers", 2), max_iterations=kwargs.get("max_iterations", 80)).solve()
