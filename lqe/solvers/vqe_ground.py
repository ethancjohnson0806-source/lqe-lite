import numpy as np
from ..core.statevector import StatevectorSim

def tfi_hamiltonian(n=4, J=1.0, h=0.5):
    """Transverse-field Ising model Hamiltonian."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    # ZZ interactions
    for i in range(n - 1):
        op = 1
        for k in range(n):
            op = np.kron(op, Z if k in (i, i + 1) else I2)
        H += -J * op
    # Transverse field X
    for i in range(n):
        op = 1
        for k in range(n):
            op = np.kron(op, X if k == i else I2)
        H += -h * op
    return H

def vqe_ansatz(params, n, layers=2):
    """Hardware-efficient ansatz: RY + CNOT chain."""
    sim = StatevectorSim(n)
    idx = 0
    for _ in range(layers):
        for q in range(n):
            sim.apply("RY", params[idx], q)
            idx += 1
        for q in range(n - 1):
            sim.apply("CNOT", q, q + 1)
    return sim

def vqe_energy(params, H, n, layers):
    sim = vqe_ansatz(params, n, layers)
    return float(np.real(np.vdot(sim.state, H @ sim.state)))

def spsa_vqe(H, n, layers, max_iter=300, verbose=False):
    d = n * layers
    theta = np.random.uniform(0, 2 * np.pi, size=d)
    a, c = 0.5, 0.1
    A, gamma_spsa, alpha = 2, 0.101, 0.602
    best_e = float("inf")
    best_theta = theta.copy()
    history = []
    no_improve = 0
    for k in range(max_iter):
        ak = a / (k + 1 + A) ** alpha
        ck = c / (k + 1) ** gamma_spsa
        delta = np.random.choice([-1, 1], size=d)
        e_plus = vqe_energy(theta + ck * delta, H, n, layers)
        e_minus = vqe_energy(theta - ck * delta, H, n, layers)
        grad = (e_plus - e_minus) / (2 * ck) * delta
        theta -= ak * grad
        theta = np.mod(theta, 2 * np.pi)
        e = vqe_energy(theta, H, n, layers)
        history.append(e)
        if e < best_e - 1e-6:
            best_e = e
            best_theta = theta.copy()
            no_improve = 0
        else:
            no_improve += 1
        if no_improve > 50 and k < max_iter - 50:
            theta = best_theta + np.random.normal(0, 0.3, size=d)
            theta = np.mod(theta, 2 * np.pi)
            no_improve = 0
        if verbose and k % 50 == 0:
            print(f"  iter {k:3d}: energy = {e:.6f}, best = {best_e:.6f}")
    return best_e, best_theta, history

class VQE:
    def __init__(self, num_qubits, hamiltonian, num_layers=2, max_iterations=300):
        self.n = num_qubits
        self.H = hamiltonian
        self.layers = num_layers
        self.max_iter = max_iterations
        self.exact = np.min(np.linalg.eigvalsh(self.H))

    def solve(self, verbose=False):
        if verbose:
            print(f"VQE: {self.n} qubits, {self.layers} layers")
            print(f"Exact ground energy: {self.exact:.6f}")
        e_vqe, theta, history = spsa_vqe(
            self.H, self.n, self.layers, max_iter=self.max_iter, verbose=verbose
        )
        error = abs(e_vqe - self.exact)
        rel_error = error / abs(self.exact) if self.exact != 0 else error
        return {
            "energy": e_vqe,
            "exact": self.exact,
            "error": error,
            "relative_error": rel_error,
            "parameters": theta,
            "history": history,
            "converged": rel_error < 0.15,
            "status": "PASS" if rel_error < 0.15 else "NEEDS_MORE_ITERATIONS"
        }
