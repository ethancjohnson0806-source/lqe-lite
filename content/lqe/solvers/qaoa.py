import numpy as np
from ..core.statevector import StatevectorSim

def maxcut_hamiltonian(n_nodes, edges):
    """Negated MaxCut Hamiltonian for minimization."""
    dim = 2 ** n_nodes
    H = np.zeros((dim, dim), dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    for (i, j) in edges:
        op = 1
        for k in range(n_nodes):
            op = np.kron(op, Z if k in (i, j) else I2)
        H += -0.5 * (np.eye(dim, dtype=complex) - op)
    return H

def qaoa_ansatz(params, n, edges, p):
    """Standard QAOA ansatz."""
    sim = StatevectorSim(n)
    for q in range(n):
        sim.apply("H", q)
    for layer in range(p):
        gamma = params[2 * layer]
        beta = params[2 * layer + 1]
        for (i, j) in edges:
            sim.apply("CNOT", i, j)
            sim.apply("RZ", -gamma, j)
            sim.apply("CNOT", i, j)
        for q in range(n):
            sim.apply("RX", 2 * beta, q)
    return sim

def qaoa_expectation(params, H, n, edges, p):
    sim = qaoa_ansatz(params, n, edges, p)
    return float(np.real(np.vdot(sim.state, H @ sim.state)))

def spsa_qaoa(H, n, edges, p, max_iter=300, verbose=False):
    d = 2 * p
    best_e = float("inf")
    best_theta = None
    for restart in range(5):
        theta = np.zeros(d)
        for k in range(p):
            theta[2 * k] = np.random.uniform(0.2, 0.8) * (k + 1) / p
            theta[2 * k + 1] = np.random.uniform(0.2, 0.6) * (1 - k / p)
        a, c = 0.8, 0.12
        A, gamma_spsa, alpha = 2, 0.101, 0.602
        local_best = float("inf")
        local_theta = theta.copy()
        no_improve = 0
        for k in range(max_iter):
            ak = a / (k + 1 + A) ** alpha
            ck = c / (k + 1) ** gamma_spsa
            delta = np.random.choice([-1, 1], size=d)
            e_plus = qaoa_expectation(theta + ck * delta, H, n, edges, p)
            e_minus = qaoa_expectation(theta - ck * delta, H, n, edges, p)
            grad = (e_plus - e_minus) / (2 * ck) * delta
            theta -= ak * grad
            theta[0::2] = np.clip(theta[0::2], 0, np.pi)
            theta[1::2] = np.clip(theta[1::2], 0, np.pi / 2)
            e = qaoa_expectation(theta, H, n, edges, p)
            if e < local_best - 1e-6:
                local_best = e
                local_theta = theta.copy()
                no_improve = 0
            else:
                no_improve += 1
            if no_improve > 60 and k < max_iter - 60:
                theta[0::2] = local_theta[0::2] + np.random.normal(0, 0.4, size=p)
                theta[1::2] = local_theta[1::2] + np.random.normal(0, 0.3, size=p)
                theta[0::2] = np.clip(theta[0::2], 0, np.pi)
                theta[1::2] = np.clip(theta[1::2], 0, np.pi / 2)
                no_improve = 0
        if local_best < best_e:
            best_e = local_best
            best_theta = local_theta.copy()
        if verbose:
            print(f"  restart {restart}: best = {local_best:.6f}")
    return best_e, best_theta, []

class QAOA:
    def __init__(self, n_nodes, edges, p=2, max_iterations=300):
        self.n = n_nodes
        self.edges = edges
        self.p = p
        self.max_iter = max_iterations
        self.H = maxcut_hamiltonian(n_nodes, edges)
        self.exact = np.min(np.linalg.eigvalsh(self.H))

    def solve(self, verbose=False):
        if verbose:
            print(f"QAOA: {self.n} nodes, {len(self.edges)} edges, p={self.p}")
            print(f"Exact ground energy: {self.exact:.6f}")
        e_qaoa, theta, history = spsa_qaoa(
            self.H, self.n, self.edges, self.p, max_iter=self.max_iter, verbose=verbose
        )
        error = abs(e_qaoa - self.exact)
        rel_error = error / abs(self.exact) if self.exact != 0 else error
        return {
            "energy": e_qaoa,
            "exact": self.exact,
            "error": error,
            "relative_error": rel_error,
            "parameters": theta,
            "history": history,
            "converged": rel_error < 0.15,
            "status": "PASS" if rel_error < 0.15 else "NEEDS_MORE_ITERATIONS"
        }
