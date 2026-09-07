import numpy as np

def apply_pauli_noise(state, n_qubits, error_prob=0.1, error_type="depolarizing"):
    """Apply stochastic Pauli noise to a statevector."""
    new_state = state.copy()
    for q in range(n_qubits):
        if np.random.random() < error_prob:
            if error_type == "depolarizing":
                pauli = np.random.choice(["X", "Y", "Z"])
            else:
                pauli = error_type
            shape = [2] * n_qubits
            psi = new_state.reshape(shape)
            psi = np.moveaxis(psi, q, 0)
            if pauli == "X":
                P = np.array([[0, 1], [1, 0]], dtype=complex)
            elif pauli == "Y":
                P = np.array([[0, -1j], [1j, 0]], dtype=complex)
            else:
                P = np.array([[1, 0], [0, -1]], dtype=complex)
            psi = np.tensordot(P, psi, axes=(1, 0))
            psi = np.moveaxis(psi, 0, q)
            new_state = psi.reshape(2 ** n_qubits)
    return new_state

def depolarizing_channel(rho, n_qubits, p):
    """Density-matrix depolarizing: (1-p)*rho + p*I/2^n."""
    dim = 2 ** n_qubits
    return (1 - p) * rho + p * np.eye(dim, dtype=complex) / dim
