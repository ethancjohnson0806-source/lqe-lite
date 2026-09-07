def export_qasm(gates, n_qubits):
    """Export gate list to OpenQASM 2.0 string."""
    lines = [
        'OPENQASM 2.0;',
        'include "qelib1.inc";',
        f'qreg q[{n_qubits}];',
        f'creg c[{n_qubits}];'
    ]
    for g in gates:
        name = g[0]
        if name in ("H", "X", "Y", "Z"):
            lines.append(f"{name} q[{g[1]}];")
        elif name in ("RX", "RY", "RZ"):
            lines.append(f"{name}({g[1]}) q[{g[2]}];")
        elif name == "CNOT":
            lines.append(f"cx q[{g[1]}],q[{g[2]}];")
        elif name == "SWAP":
            lines.append(f"swap q[{g[1]}],q[{g[2]}];")
    return "\n".join(lines)
