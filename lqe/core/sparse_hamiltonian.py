"""
sparse_hamiltonian.py — Legitimate Quantum Engine v4.0, Module 3

Sparse Hamiltonian builder and Lanczos eigensolver for n-qubit Pauli-string
Hamiltonians.  Pure NumPy, COO format with vectorized matvec.

Qubit convention: qubit 0 is the rightmost character in a Pauli string
(least-significant bit in basis-state integers).

Honest limitations:
    - n_qubits <= 16 is phone-comfortable (65536 basis states).
    - to_dense() is limited to n <= 12 to avoid memory blow-up.
    - Lanczos uses full reorthogonalization; cost scales as O(k^2 * N) where
      k = number of iterations.  For N = 65536, k = 50 is ~10 s.
"""

import numpy as np


class SparseHamiltonian:
    """
    Sparse n-qubit Hamiltonian stored in COO (coordinate) format.

    Built by accumulating Pauli-string terms.  Call finalize() before
    matvec() or to_dense().
    """

    def __init__(self, n_qubits: int):
        if not isinstance(n_qubits, int) or n_qubits < 1:
            raise ValueError("n_qubits must be a positive integer")
        self.n = n_qubits
        self.N = 1 << n_qubits          # 2**n
        self._entries = {}             # (row, col) -> complex
        self._finalized = False

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------
    def add_term(self, coefficient: complex, pauli_string: str):
        """
        Add ``coefficient * pauli_string`` to the Hamiltonian.

        Parameters
        ----------
        coefficient : complex
            Scalar prefactor.
        pauli_string : str
            Length-n string of 'I', 'X', 'Y', 'Z'.  Qubit 0 is the
            rightmost character (least-significant bit).
        """
        if len(pauli_string) != self.n:
            raise ValueError(
                f"Pauli string length {len(pauli_string)} != n_qubits {self.n}"
            )
        coeff = complex(coefficient)
        n, N = self.n, self.N

        for b in range(N):
            b_prime = b
            phase = 1.0 + 0.0j
            for q in range(n):
                bit = (b >> q) & 1
                p = pauli_string[n - 1 - q]
                if p == 'I':
                    continue
                elif p == 'X':
                    b_prime ^= (1 << q)
                elif p == 'Y':
                    b_prime ^= (1 << q)
                    phase *= (1j if bit == 0 else -1j)
                elif p == 'Z':
                    if bit == 1:
                        phase *= -1
                else:
                    raise ValueError(f"Invalid Pauli character '{p}'")
            key = (b_prime, b)
            self._entries[key] = self._entries.get(key, 0.0) + coeff * phase

    def finalize(self):
        """Convert the dictionary accumulator to NumPy arrays."""
        if self._finalized:
            return
        if not self._entries:
            self.rows = np.array([], dtype=np.int64)
            self.cols = np.array([], dtype=np.int64)
            self.data = np.array([], dtype=complex)
        else:
            items = list(self._entries.items())
            self.rows = np.array([k[0] for k, _ in items], dtype=np.int64)
            self.cols = np.array([k[1] for k, _ in items], dtype=np.int64)
            self.data = np.array([v for _, v in items], dtype=complex)
        self._entries = None
        self._finalized = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def nnz(self) -> int:
        """Number of stored non-zeros."""
        if self._finalized:
            return self.data.size
        return len(self._entries)

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------
    def matvec(self, v: np.ndarray) -> np.ndarray:
        """
        Compute ``H @ v``.

        Parameters
        ----------
        v : np.ndarray
            Input vector of length 2**n.

        Returns
        -------
        np.ndarray
            Output vector, dtype=complex.
        """
        if not self._finalized:
            raise RuntimeError("Call finalize() before matvec()")
        v = np.asarray(v, dtype=complex)
        out = np.zeros(self.N, dtype=complex)
        if self.data.size > 0:
            np.add.at(out, self.rows, self.data * v[self.cols])
        return out

    def expectation(self, psi: np.ndarray) -> float:
        """Return ``<psi| H |psi>`` (real part only)."""
        psi = np.asarray(psi, dtype=complex)
        return float(np.vdot(psi, self.matvec(psi)).real)

    def to_dense(self) -> np.ndarray:
        """
        Convert to a dense ``(2**n, 2**n)`` complex matrix.

        Honest limitation: raises for ``n > 12`` (4096x4096 ~ 256 MB).
        """
        if self.n > 12:
            raise ValueError(
                f"to_dense() limited to n <= 12, got n = {self.n}"
            )
        H = np.zeros((self.N, self.N), dtype=complex)
        for r, c, val in zip(self.rows, self.cols, self.data):
            H[r, c] += val
        return H


# =============================================================================
# Helpers
# =============================================================================

def build_sparse_heisenberg(n_qubits: int, J: float = 1.0, h: float = 0.0):
    """
    Build the open-boundary 1D Heisenberg Hamiltonian as a
    :class:`SparseHamiltonian`.

    ::
        H = J * sum_i (S_i^x S_{i+1}^x + S_i^y S_{i+1}^y + S_i^z S_{i+1}^z)
          + h * sum_i S_i^z
    """
    H = SparseHamiltonian(n_qubits)
    eye = ['I'] * n_qubits

    for i in range(n_qubits - 1):
        # XX
        p = eye.copy()
        p[i] = 'X'
        p[i + 1] = 'X'
        H.add_term(J / 4.0, ''.join(p))

        # YY
        p = eye.copy()
        p[i] = 'Y'
        p[i + 1] = 'Y'
        H.add_term(J / 4.0, ''.join(p))

        # ZZ
        p = eye.copy()
        p[i] = 'Z'
        p[i + 1] = 'Z'
        H.add_term(J / 4.0, ''.join(p))

    if h != 0.0:
        for i in range(n_qubits):
            p = eye.copy()
            p[i] = 'Z'
            H.add_term(h / 2.0, ''.join(p))

    H.finalize()
    return H


def sparse_lanczos(
    H: SparseHamiltonian,
    v0: np.ndarray | None = None,
    max_iter: int = 50,
    tol: float = 1e-10,
):
    """
    Lanczos iteration with full reorthogonalization.

    Parameters
    ----------
    H : SparseHamiltonian
        Hamiltonian to diagonalize.
    v0 : np.ndarray, optional
        Initial guess vector.  Random if None.
    max_iter : int
        Maximum Lanczos steps.
    tol : float
        Convergence tolerance on the off-diagonal beta.

    Returns
    -------
    energy : float
        Lowest Ritz value (ground-state energy estimate).
    psi : np.ndarray
        Ground-state vector estimate in the full Hilbert space.
    converged : bool
        Whether beta dropped below ``tol``.
    """
    N = H.N
    if v0 is None:
        rng = np.random.default_rng(42)
        v0 = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    v0 = np.asarray(v0, dtype=complex)
    norm = np.linalg.norm(v0)
    if norm == 0:
        raise ValueError("Initial vector cannot be zero")
    v0 /= norm

    V = [v0]          # Lanczos vectors
    alphas = []       # Diagonal of T
    betas = [0.0]     # Off-diagonal of T (betas[0] is dummy)

    for j in range(max_iter):
        w = H.matvec(V[j])
        if j > 0:
            w -= betas[j] * V[j - 1]

        alpha = float(np.vdot(V[j], w).real)
        alphas.append(alpha)
        w = w - alpha * V[j]

        # Full reorthogonalization (phone-runnable for N <= 65536)
        for i in range(j + 1):
            w -= np.vdot(V[i], w) * V[i]

        beta = float(np.linalg.norm(w))
        betas.append(beta)

        if beta < tol:
            break

        V.append(w / beta)

    # Diagonalize the tridiagonal Ritz matrix
    n = len(alphas)
    T = (
        np.diag(alphas)
        + np.diag(betas[1:n], 1)
        + np.diag(betas[1:n], -1)
    )
    eigs, vecs = np.linalg.eigh(T)
    energy = eigs[0]

    # Reconstruct ground-state vector in full space
    coeffs = vecs[:, 0]
    psi = np.zeros(N, dtype=complex)
    for i, c in enumerate(coeffs):
        psi += c * V[i]
    psi /= np.linalg.norm(psi)

    converged = betas[-1] < tol
    return energy, psi, converged


# =============================================================================
# Tests
# =============================================================================

def test_pauli_identity():
    """Identity Hamiltonian on 3 qubits."""
    print("\n=== Test: Pauli Identity ===")
    H = SparseHamiltonian(3)
    H.add_term(1.0, 'III')
    H.finalize()
    v = np.ones(8, dtype=complex) / np.sqrt(8)
    ev = H.expectation(v)
    assert abs(ev - 1.0) < 1e-10, f"Expected 1.0, got {ev}"
    print("  PASS")


def test_pauli_x():
    """Single X on qubit 0 (rightmost character)."""
    print("\n=== Test: Pauli X ===")
    H = SparseHamiltonian(2)
    H.add_term(1.0, 'IX')   # X on qubit 0, I on qubit 1
    H.finalize()
    v = np.array([1, 0, 0, 0], dtype=complex)
    out = H.matvec(v)
    assert abs(out[1] - 1.0) < 1e-10
    assert np.max(np.abs(out - np.array([0, 1, 0, 0], dtype=complex))) < 1e-10
    print("  PASS")


def test_pauli_y():
    """Single Y on qubit 0."""
    print("\n=== Test: Pauli Y ===")
    H = SparseHamiltonian(1)
    H.add_term(1.0, 'Y')
    H.finalize()
    v = np.array([1, 0], dtype=complex)
    out = H.matvec(v)
    assert abs(out[1] - 1j) < 1e-10
    assert abs(out[0]) < 1e-10
    print("  PASS")


def test_pauli_z():
    """Single Z on qubit 0."""
    print("\n=== Test: Pauli Z ===")
    H = SparseHamiltonian(1)
    H.add_term(1.0, 'Z')
    H.finalize()
    v = np.array([1, 1], dtype=complex) / np.sqrt(2)
    out = H.matvec(v)
    assert abs(out[0] - 1 / np.sqrt(2)) < 1e-10
    assert abs(out[1] + 1 / np.sqrt(2)) < 1e-10
    print("  PASS")


def test_matvec_vs_dense():
    """matvec() agrees with dense matrix-vector product."""
    print("\n=== Test: matvec vs dense ===")
    H = build_sparse_heisenberg(6, J=1.0, h=0.5)
    rng = np.random.default_rng(7)
    v = rng.standard_normal(H.N) + 1j * rng.standard_normal(H.N)
    v /= np.linalg.norm(v)

    sparse_out = H.matvec(v)
    dense_out = H.to_dense() @ v
    err = np.max(np.abs(sparse_out - dense_out))
    assert err < 1e-10, f"matvec mismatch: {err}"
    print("  PASS")


def test_heisenberg_lanczos():
    """Lanczos ground state matches exact diagonalization."""
    print("\n=== Test: Heisenberg Lanczos ===")
    n = 8
    H = build_sparse_heisenberg(n, J=1.0, h=0.0)
    energy, psi, conv = sparse_lanczos(H, max_iter=30)

    H_dense = H.to_dense()
    eigs = np.linalg.eigvalsh(H_dense)
    exact = eigs[0]

    print(f"  Lanczos: {energy:.6f}, Exact: {exact:.6f}, diff: {abs(energy - exact):.2e}")
    assert abs(energy - exact) < 1e-6, f"Lanczos {energy} != exact {exact}"
    print("  PASS")


def test_magnetic_field():
    """All-Z field: ground state is all spins down."""
    print("\n=== Test: Magnetic Field (Sparse) ===")
    n = 6
    H = build_sparse_heisenberg(n, J=0.0, h=2.0)
    energy, psi, conv = sparse_lanczos(H, max_iter=20)

    # E = -h * n/2 = -2 * 6 * 0.5 = -6
    print(f"  Lanczos: {energy:.6f}, Expected: -6.0")
    assert abs(energy - (-6.0)) < 1e-6
    print("  PASS")


def test_empty_hamiltonian():
    """Zero Hamiltonian gives zero energy."""
    print("\n=== Test: Empty Hamiltonian ===")
    H = SparseHamiltonian(4)
    H.finalize()
    v = np.ones(16, dtype=complex) / 4
    out = H.matvec(v)
    assert np.max(np.abs(out)) < 1e-10
    print("  PASS")


def run_all_tests():
    print("=" * 60)
    print("SPARSE HAMILTONIAN TEST SUITE")
    print("=" * 60)
    test_pauli_identity()
    test_pauli_x()
    test_pauli_y()
    test_pauli_z()
    test_matvec_vs_dense()
    test_heisenberg_lanczos()
    test_magnetic_field()
    test_empty_hamiltonian()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
