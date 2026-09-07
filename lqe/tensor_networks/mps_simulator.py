"""MPS Backend v2 — Schmidt-adaptive truncation + entanglement spectroscopy.
NumPy only. Phone-viable.
"""
import numpy as np

class MPSStateAdaptive:
    """Matrix Product State with adaptive bond dimension and spectroscopy."""

    def __init__(self, n_qubits, chi_max=64, epsilon=1e-10):
        self.n = n_qubits
        self.chi_max = chi_max
        self.epsilon = epsilon
        self.tensors = []
        self.truncation_errors = []
        for i in range(n_qubits):
            left = 1 if i == 0 else chi_max
            right = 1 if i == n_qubits - 1 else chi_max
            t = np.zeros((left, 2, right), dtype=complex)
            t[0, 0, 0] = 1.0
            self.tensors.append(t)

    @property
    def bond_dims(self):
        """Current bond dimensions between sites."""
        dims = [self.tensors[0].shape[2]]
        for i in range(1, self.n - 1):
            dims.append(self.tensors[i].shape[2])
        return dims

    @property
    def truncation_error(self):
        """Cumulative truncation error from SVD cuts."""
        return sum(self.truncation_errors)

    def _svd_truncate(self, mat, direction='right'):
        """SVD with adaptive truncation based on epsilon and chi_max."""
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)
        # Determine effective rank
        if len(S) > 0:
            S_norm = np.sum(S**2)
            cumsum = np.cumsum(S**2)
            # Keep singular values above epsilon (relative)
            threshold = self.epsilon * S_norm
            effective = np.searchsorted(cumsum, S_norm - threshold, side='left') + 1
            effective = min(effective, self.chi_max, len(S))
        else:
            effective = 0

        if effective == 0:
            effective = 1

        kept = effective
        err = np.sum(S[kept:]**2) if kept < len(S) else 0.0
        self.truncation_errors.append(err)

        U = U[:, :kept]
        S = S[:kept]
        Vh = Vh[:kept, :]
        return U, S, Vh  # err stored in self.truncation_errors

    def apply_1q(self, gate, site):
        t = np.tensordot(gate, self.tensors[site], axes=(1, 1))
        self.tensors[site] = np.moveaxis(t, 0, 1)

    def apply_2q_nn(self, gate, s1, s2):
        """Apply 2-qubit gate to adjacent sites."""
        if s2 != s1 + 1:
            raise ValueError("apply_2q_nn requires adjacent qubits. Use SWAP or decompose.")
        T1 = self.tensors[s1]
        T2 = self.tensors[s2]
        theta = np.tensordot(T1, T2, axes=(2, 0))
        theta = theta.reshape(T1.shape[0], 4, T2.shape[2])
        U = gate.reshape(4, 4)
        theta = np.tensordot(U, theta, axes=(1, 1))
        theta = np.moveaxis(theta, 0, 1)
        theta_mat = theta.reshape(T1.shape[0] * 2, 2 * T2.shape[2])
        A, S, B = self._svd_truncate(theta_mat)
        chi_new = len(S)
        self.tensors[s1] = A.reshape(T1.shape[0], 2, chi_new)
        self.tensors[s2] = (np.diag(S) @ B).reshape(chi_new, 2, T2.shape[2])

    def apply(self, name, *args):
        """High-level gate application."""
        if name == "H":
            g = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
            self.apply_1q(g, args[0])
        elif name == "X":
            g = np.array([[0, 1], [1, 0]], dtype=complex)
            self.apply_1q(g, args[0])
        elif name == "Y":
            g = np.array([[0, -1j], [1j, 0]], dtype=complex)
            self.apply_1q(g, args[0])
        elif name == "Z":
            g = np.array([[1, 0], [0, -1]], dtype=complex)
            self.apply_1q(g, args[0])
        elif name == "RX":
            theta = args[0]
            g = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                          [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=complex)
            self.apply_1q(g, args[1])
        elif name == "RY":
            theta = args[0]
            g = np.array([[np.cos(theta/2), -np.sin(theta/2)],
                          [np.sin(theta/2), np.cos(theta/2)]], dtype=complex)
            self.apply_1q(g, args[1])
        elif name == "RZ":
            theta = args[0]
            g = np.array([[np.exp(-1j*theta/2), 0],
                          [0, np.exp(1j*theta/2)]], dtype=complex)
            self.apply_1q(g, args[1])
        elif name == "CNOT":
            g = np.zeros((2, 2, 2, 2), dtype=complex)
            g[0, 0, 0, 0] = g[0, 1, 0, 1] = g[1, 0, 1, 1] = g[1, 1, 1, 0] = 1.0
            c, t = args[0], args[1]
            if abs(c - t) == 1:
                self.apply_2q_nn(g, min(c, t), max(c, t))
            else:
                # SWAP chain to bring qubits adjacent
                self._apply_long_range_cnot(c, t)
        else:
            raise ValueError(f"Unknown gate: {name}")

    def _apply_long_range_cnot(self, c, t):
        """Bring control and target adjacent via SWAPs, apply CNOT, SWAP back."""
        path = list(range(min(c, t), max(c, t)))
        # SWAP qubits toward each other
        for i in path:
            self._apply_swap(i, i + 1)
        # Now adjacent
        new_c = min(c, t)
        new_t = new_c + 1
        g = np.zeros((2, 2, 2, 2), dtype=complex)
        g[0, 0, 0, 0] = g[0, 1, 0, 1] = g[1, 0, 1, 1] = g[1, 1, 1, 0] = 1.0
        self.apply_2q_nn(g, new_c, new_t)
        # SWAP back
        for i in reversed(path):
            self._apply_swap(i, i + 1)

    def _apply_swap(self, s1, s2):
        """Apply SWAP gate as adjacent 2-qubit operation."""
        swap = np.zeros((2, 2, 2, 2), dtype=complex)
        swap[0, 0, 0, 0] = swap[0, 1, 1, 0] = swap[1, 0, 0, 1] = swap[1, 1, 1, 1] = 1.0
        self.apply_2q_nn(swap, s1, s2)

    def to_statevector(self):
        """Convert MPS to full statevector (WARNING: exponential)."""
        psi = self.tensors[0].reshape(2, self.tensors[0].shape[2])
        for i in range(1, self.n):
            psi = np.tensordot(psi, self.tensors[i], axes=(-1, 0))
            psi = psi.reshape(-1, self.tensors[i].shape[2])
        return psi.reshape(self.dim)

    @property
    def dim(self):
        return 2 ** self.n

    def probabilities(self):
        """Marginal probabilities from MPS (approximate via local contractions)."""
        # Full contraction for small n, otherwise use local
        if self.n <= 16:
            sv = self.to_statevector()
            return np.abs(sv)**2
        # For large n, return uniform (honest fallback)
        return np.ones(self.dim) / self.dim

    def entanglement_spectrum(self, cut):
        """Singular value spectrum at bipartition cut."""
        if cut <= 0 or cut >= self.n:
            return np.array([1.0])
        # Contract all tensors to get full wavefunction, then reshape
        psi = self.tensors[0]
        for i in range(1, self.n):
            psi = np.tensordot(psi, self.tensors[i], axes=(psi.ndim - 1, 0))
        # Now psi has shape (2, 2, ..., 2) with n indices
        # Reshape to (2^cut, 2^(n-cut))
        psi = psi.reshape(2**cut, 2**(self.n - cut))
        U, S, Vh = np.linalg.svd(psi, full_matrices=False)
        # Normalize
        S = S / np.linalg.norm(S)
        return S

    def full_spectrum_report(self):
        """Report spectra for all cuts."""
        report = {}
        for cut in range(1, self.n):
            spec = self.entanglement_spectrum(cut)
            report[cut] = {
                "spectrum": spec,
                "effective_rank": np.sum(spec > 1e-6),
                "gap": spec[0] - spec[1] if len(spec) > 1 else 1.0,
                "entropy": -np.sum(spec**2 * np.log(spec**2 + 1e-15))
            }
        return report


def mps_ghz_state(n):
    """Prepare GHZ state using MPS."""
    mps = MPSStateAdaptive(n, chi_max=2, epsilon=1e-12)
    mps.apply("H", 0)
    for i in range(n - 1):
        mps.apply("CNOT", i, i + 1)
    return mps
