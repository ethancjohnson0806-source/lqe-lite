# LQE Lite

LQE Lite is a static, phone-friendly JupyterLite site for learning quantum computing in the browser. It embeds a pure-NumPy subset of the [Legitimate Quantum Engine](https://github.com/ethancjohnson0806-source/Legitimate-Quantum-Engine) and twenty adapted educational notebooks.

## Use it

Open the deployed GitHub Pages site, wait for the Pyodide kernel to load, open `welcome.ipynb`, and run the lessons directly in your browser. No account, server, or local installation is required.

## Local build

```bash
pip install jupyterlite-core jupyterlite-pyodide-kernel jupyter-server
python build_lqe_lite.py
jupyter lite build --contents content --output-dir dist
```

## Scope and limitations

This build intentionally includes only pure-NumPy modules. Numba, SciPy, OpenFermion, PySCF, networkx, GPU support, and compiled extensions are not included. Lessons that depend on those packages contain simplified browser-safe demonstrations and point users to the full engine for local execution.

The site is static and uses browser storage. It does not collect accounts or run code on a server.
