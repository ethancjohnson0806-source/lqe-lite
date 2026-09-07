import numpy as np

class QuantumGates:
    @staticmethod
    def Toffoli():
        U = np.eye(8, dtype=complex)
        U[6, 6] = 0
        U[6, 7] = 1
        U[7, 6] = 1
        U[7, 7] = 0
        return U

    @staticmethod
    def Fredkin():
        U = np.eye(8, dtype=complex)
        U[5, 5] = 0
        U[5, 6] = 1
        U[6, 5] = 1
        U[6, 6] = 0
        return U
