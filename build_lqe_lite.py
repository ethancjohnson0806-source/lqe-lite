from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENGINE = Path(os.environ.get('LQE_ENGINE_PATH', str(ROOT.parent / 'Legitimate-Quantum-Engine'))) / 'legitimate_quantum_engine'
ECOSYSTEM = Path(os.environ.get('LQE_ECOSYSTEM_PATH', str(ROOT.parent / 'lqe-ecosystem')))

PURE_FILES = [
    'core/statevector.py',
    'core/sparse_hamiltonian.py',
    'core/stabilizer.py',
    'core/noise_models.py',
    'solvers/vqe_ground.py',
    'solvers/qaoa.py',
    'tensor_networks/mps_simulator.py',
    'circuit_tools/qasm_export.py',
    'utils/extended_gates.py',
]

PACKAGE_INITS = {
    'lqe/__init__.py': '''"""LQE Lite — Pure NumPy Quantum Engine (Pyodide-compatible subset)."""\nfrom .core.statevector import StatevectorSim\nfrom .core.stabilizer import StabilizerSim\nfrom .solvers.vqe_ground import VQE\nfrom .solvers.qaoa import QAOA\nfrom .solvers.orchestrator import auto_solve, tfi_hamiltonian\nfrom .tensor_networks.mps_simulator import MPSStateAdaptive, mps_ghz_state\nMPSSimulator = MPSStateAdaptive\n\n__all__ = [\n    "StatevectorSim", "StabilizerSim", "VQE", "QAOA",\n    "auto_solve", "tfi_hamiltonian", "MPSSimulator",\n    "MPSStateAdaptive", "mps_ghz_state",\n]\n''',
    'lqe/core/__init__.py': '''from .statevector import StatevectorSim\nfrom .sparse_hamiltonian import SparseHamiltonian, build_sparse_heisenberg, sparse_lanczos\nfrom .stabilizer import StabilizerSim\nfrom .noise_models import apply_pauli_noise, depolarizing_channel\n\n__all__ = [\n    "StatevectorSim", "SparseHamiltonian", "build_sparse_heisenberg",\n    "sparse_lanczos", "StabilizerSim", "apply_pauli_noise",\n    "depolarizing_channel",\n]\n''',
    'lqe/solvers/__init__.py': '''from .vqe_ground import VQE, tfi_hamiltonian, spsa_vqe, vqe_energy, vqe_ansatz\nfrom .qaoa import QAOA, maxcut_hamiltonian\nfrom .orchestrator import auto_solve\n\n__all__ = ["VQE", "tfi_hamiltonian", "spsa_vqe", "vqe_energy", "vqe_ansatz", "QAOA", "maxcut_hamiltonian", "auto_solve"]\n''',
    'lqe/tensor_networks/__init__.py': '''from .mps_simulator import MPSStateAdaptive, mps_ghz_state\nMPSSimulator = MPSStateAdaptive\n__all__ = ["MPSStateAdaptive", "MPSSimulator", "mps_ghz_state"]\n''',
    'lqe/circuit_tools/__init__.py': '''from .qasm_export import export_qasm\n__all__ = ["export_qasm"]\n''',
    'lqe/utils/__init__.py': '''from .extended_gates import QuantumGates\n__all__ = ["QuantumGates"]\n''',
}

ORCHESTRATOR = '''"""Pure-NumPy solver selection for the browser build."""\nimport numpy as np\nfrom .vqe_ground import VQE, tfi_hamiltonian\nfrom .qaoa import QAOA\n\ndef auto_solve(H, n_qubits, problem_type="generic", **kwargs):\n    """Select a browser-safe solver and return its result."""\n    if problem_type.lower() == "maxcut":\n        edges = kwargs.get("edges", [])\n        return QAOA(n_qubits, edges, p=kwargs.get("p", 1), max_iterations=kwargs.get("max_iterations", 80)).solve()\n    return VQE(n_qubits, H, num_layers=kwargs.get("num_layers", 2), max_iterations=kwargs.get("max_iterations", 80)).solve()\n'''

SPECIAL_NOTEBOOKS = {
    '09_Quantum_Error_Correction.ipynb': '''from lqe import StabilizerSim\nsim = StabilizerSim(5)\nprint("Browser-safe QEC demo: stabilizer backend initialized")\nprint("For full surface-code decoding, use the full LQE repository locally.")\n''',
    '10_Quantum_Chemistry.ipynb': '''import numpy as np\nfrom lqe import VQE, tfi_hamiltonian\nH = tfi_hamiltonian(2, J=1.0, h=0.5)\nout = VQE(2, H, num_layers=1, max_iterations=40).solve()\nprint(f"Toy molecular Hamiltonian energy: {out['energy']:.6f}")\nprint("Full OpenFermion/PySCF chemistry is intentionally not bundled in LQE Lite.")\n''',
    '11_Classical_Shadows.ipynb': '''import numpy as np\nfrom lqe import StatevectorSim\nsim = StatevectorSim(1)\nsim.apply("H", 0)\nprint("Browser-safe shadow-style measurement demo")\nprint("State probabilities:", np.round(sim.probabilities(), 4))\nprint("Full classical-shadow tomography is available in the full LQE repository.")\n''',
    '13_Solver_Orchestration.ipynb': '''import numpy as np\nfrom lqe import auto_solve, tfi_hamiltonian\nfor n in (2, 3, 4):\n    out = auto_solve(tfi_hamiltonian(n), n, max_iterations=40)\n    print(f"{n} qubits: energy={out['energy']:.6f}, status={out['status']}")\n''',
    '14_Surface_Codes.ipynb': '''from lqe import StabilizerSim\nsim = StabilizerSim(9)\nprint("Surface-code concept demo: 9-qubit stabilizer register initialized")\nprint("Full surface-code decoding is available in the full LQE repository.")\n''',
    '16_Quantum_Volume.ipynb': '''import numpy as np\nfrom lqe import StatevectorSim\nsim = StatevectorSim(3)\nfor q in range(3):\n    sim.apply("H", q)\nprint("Pure-NumPy quantum-volume-style smoke test")\nprint("Uniform probabilities:", np.round(sim.probabilities(), 4))\nprint("SciPy-based Haar random circuits are intentionally excluded from LQE Lite.")\n''',
    '17_Resource_Estimation.ipynb': '''from lqe import StatevectorSim\nn = 8\nsim = StatevectorSim(n)\nfor q in range(n):\n    sim.apply("H", q)\nprint({"qubits": n, "depth": 1, "gates": n, "backend": "pure NumPy statevector"})\n''',
    '18_Dynamic_Circuits.ipynb': '''from lqe import StatevectorSim\nsim = StatevectorSim(2)\nsim.apply("H", 0)\nsim.apply("CNOT", 0, 1)\nprint("Dynamic-circuit preparation demo:", sim.probabilities())\nprint("Interactive mid-circuit control is available in the full LQE repository.")\n''',
    '19_Quantum_Kernels.ipynb': '''import numpy as np\nfrom lqe import StatevectorSim\ndef fidelity(x, y):\n    a, b = StatevectorSim(2), StatevectorSim(2)\n    for q, angle in enumerate(x): a.apply("RY", float(angle), q)\n    for q, angle in enumerate(y): b.apply("RY", float(angle), q)\n    return float(abs(np.vdot(a.state, b.state)) ** 2)\nX = np.array([[0.1, 0.2], [0.8, 0.7]])\nprint(np.array([[fidelity(x, y) for y in X] for x in X]).round(4))\n''',
    '20_Full_Pipeline.ipynb': '''from lqe import VQE, tfi_hamiltonian\nfrom lqe.circuit_tools.qasm_export import export_qasm\nH = tfi_hamiltonian(4)\nout = VQE(4, H, num_layers=2, max_iterations=60).solve()\nprint(f"VQE energy={out['energy']:.6f}")\nprint(export_qasm([('h', [q]) for q in range(4)], 4))\n''',
}


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def notebook(code: str, title: str, note: str = '') -> dict:
    cells = [{"cell_type": "markdown", "metadata": {}, "source": [f"# {title}\n", "\n", note + "\n"]}]
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + '\n' for line in code.rstrip().splitlines()]})
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python (Pyodide)", "language": "python", "name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def transform_notebook(src: Path, dst: Path) -> None:
    data = json.loads(src.read_text(encoding='utf-8'))
    name = src.name
    if name in SPECIAL_NOTEBOOKS:
        title = next((''.join(c.get('source', [])) for c in data.get('cells', []) if c.get('cell_type') == 'markdown'), name)
        title = title.splitlines()[0].lstrip('# ').strip() or name
        out = notebook(SPECIAL_NOTEBOOKS[name], title, 'This lesson is adapted for the pure-NumPy browser build. See the full LQE repository for unavailable modules.')
    else:
        for cell in data.get('cells', []):
            if cell.get('cell_type') != 'code':
                continue
            lines = ''.join(cell.get('source', [] )).splitlines()
            cleaned = []
            for line in lines:
                stripped = line.strip()
                if (stripped in {'import os, sys', 'sys.path.insert(0, _p)'}
                        or line.startswith('_LQE = ') or line.startswith('_ECO = ')
                        or '_LQE' in line
                        or line.startswith('for _p in ') or stripped == 'if _p not in sys.path:'):
                    continue
                line = line.replace('legitimate_quantum_engine', 'lqe')
                cleaned.append(line)
            cell['source'] = [line + '\n' for line in cleaned]
        out = data
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')


def main() -> None:
    if ROOT.exists():
        for child in ROOT.iterdir():
            if child.name not in {'.git', 'build_lqe_lite.py', 'test_lqe_lite.py', '.gitignore'}:
                shutil.rmtree(child) if child.is_dir() else child.unlink()
    for rel in PURE_FILES:
        dest = ROOT / 'lqe' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ENGINE / rel, dest)
    for rel, text in PACKAGE_INITS.items():
        write(ROOT / rel, text)
    write(ROOT / 'lqe/solvers/orchestrator.py', ORCHESTRATOR)
    content = ROOT / 'content'
    content.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / 'lqe', content / 'lqe', dirs_exist_ok=True)
    for src in sorted((ECOSYSTEM / 'notebooks').glob('*.ipynb')):
        transform_notebook(src, content / src.name)
    welcome = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": [
        "# Welcome to LQE Lite\n", "\n",
        "**A quantum computing environment that runs in your browser.**\n", "\n",
        "No installation. No server. No account. Just quantum code.\n", "\n",
        "## First time?\n", "\n",
        "1. Wait for the environment to load.\n",
        "2. Open `01_What_Is_A_Qubit.ipynb` in the file browser.\n",
        "3. Run a code cell and edit it freely.\n", "\n",
        "## What works\n", "\n",
        "- Pure-NumPy statevector simulation\n", "- VQE and QAOA solvers\n",
        "- MPS tensor-network demonstrations\n", "- Stabilizer simulation\n",
        "- Twenty adapted educational notebooks\n", "\n",
        "## Honest limitations\n", "\n",
        "- No Numba, SciPy, OpenFermion, PySCF, networkx, or compiled extensions.\n",
        "- Some advanced lessons are simplified browser-safe demos.\n",
        "- Memory and speed depend on your phone or browser.\n",
        "- Work is saved in browser storage; there is no server account.\n", "\n",
        "**Ready?** Open `01_What_Is_A_Qubit.ipynb` and click Run.\n"
    ]}], "metadata": {"kernelspec": {"display_name": "Python (Pyodide)", "language": "python", "name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (content / 'welcome.ipynb').write_text(json.dumps(welcome, indent=1) + '\n', encoding='utf-8')
    write(ROOT / 'jupyter-lite.json', json.dumps({"jupyter-lite-schema-version": 0, "jupyter-config-data": {"appName": "LQE Lite", "appVersion": "1.0.0", "favicon32": "", "favicon16": "", "baseUrl": "./", "settingsUrl": "./", "themesUrl": "./", "terminalsAvailable": False, "disabledExtensions": ["@jupyterlab/extensionmanager-extension"]}}, indent=2) + '\n')
    write(ROOT / 'requirements.txt', 'numpy\n')
    write(ROOT / 'overrides.json', json.dumps({"@jupyterlab/apputils-extension:themes": {"theme": "JupyterLab Dark"}}, indent=2) + '\n')
    write(ROOT / '.github/workflows/deploy.yml', '''name: Deploy to GitHub Pages\non:\n  push:\n    branches: [main]\n  workflow_dispatch:\npermissions:\n  contents: read\n  pages: write\n  id-token: write\nconcurrency:\n  group: pages\n  cancel-in-progress: false\njobs:\n  build:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n      - uses: actions/setup-python@v5\n        with:\n          python-version: "3.11"\n      - name: Install JupyterLite\n        run: pip install jupyterlite-core jupyterlite-pyodide-kernel jupyter-server\n      - name: Build\n        run: jupyter lite build --contents content --output-dir dist\n      - uses: actions/upload-pages-artifact@v3\n        with:\n          path: dist\n  deploy:\n    environment:\n      name: github-pages\n      url: ${{ steps.deployment.outputs.page_url }}\n    runs-on: ubuntu-latest\n    needs: build\n    steps:\n      - name: Deploy\n        id: deployment\n        uses: actions/deploy-pages@v4\n''')
    write(ROOT / 'README.md', '''# LQE Lite\n\nLQE Lite is a static, phone-friendly JupyterLite site for learning quantum computing in the browser. It embeds a pure-NumPy subset of the [Legitimate Quantum Engine](https://github.com/ethancjohnson0806-source/Legitimate-Quantum-Engine) and twenty adapted educational notebooks.\n\n## Use it\n\nOpen the deployed GitHub Pages site, wait for the Pyodide kernel to load, open `welcome.ipynb`, and run the lessons directly in your browser. No account, server, or local installation is required.\n\n## Local build\n\n```bash\npip install jupyterlite-core jupyterlite-pyodide-kernel jupyter-server\npython build_lqe_lite.py\njupyter lite build --contents content --output-dir dist\n```\n\n## Scope and limitations\n\nThis build intentionally includes only pure-NumPy modules. Numba, SciPy, OpenFermion, PySCF, networkx, GPU support, and compiled extensions are not included. Lessons that depend on those packages contain simplified browser-safe demonstrations and point users to the full engine for local execution.\n\nThe site is static and uses browser storage. It does not collect accounts or run code on a server.\n''')

if __name__ == '__main__':
    main()
