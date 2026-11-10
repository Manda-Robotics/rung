"""Put the checkout, ``src`` and ``deploy/robolab`` on the path so the suite runs uninstalled."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "deploy/robolab"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
