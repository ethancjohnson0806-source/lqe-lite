# LQE Lite

## [Open LQE Lite in your browser](https://ethancjohnson0806-source.github.io/lqe-lite/)

**Click the link above to use the app.** Wait for the page to load, then double-click `welcome.ipynb` or `01_What_Is_A_Qubit.ipynb` in the file list on the left. Choose **Python (Pyodide)** when prompted, wait for it to finish loading, and use the **Run** button to execute a cell.

LQE Lite is a static, phone-friendly JupyterLite site for learning quantum computing in the browser. It embeds a pure-NumPy subset of the [Legitimate Quantum Engine](https://github.com/ethancjohnson0806-source/Legitimate-Quantum-Engine) and twenty adapted educational notebooks.

No account, server, or local installation is required. The repository contains the source; the link above is the actual app.

## Scope and limitations

This build intentionally includes only pure-NumPy modules. Numba, SciPy, OpenFermion, PySCF, networkx, GPU support, and compiled extensions are not included. Lessons that depend on those packages contain simplified browser-safe demonstrations and point users to the full engine for local execution.

The site is static and uses browser storage. It does not collect accounts or run code on a server.
