from __future__ import annotations
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
failed = []
for path in sorted((ROOT / "content").glob("*.ipynb")):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    ok = True
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            exec(compile(source, f"{path.name}:{index}", "exec"), {"__name__": "__main__"})
        except Exception as exc:
            ok = False
            failed.append((path.name, index, type(exc).__name__, str(exc)))
    print(f"{'OK' if ok else 'FAIL'}  {path.name}")
if failed:
    print("\nFailures:")
    for item in failed:
        print(" ", item)
    raise SystemExit(1)
print(f"\nALL {len(list((ROOT / 'content').glob('*.ipynb')))} NOTEBOOKS PASS")
