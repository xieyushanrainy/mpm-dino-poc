from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "v3" / "src"))
sys.path.insert(0, str(ROOT / "v2" / "src"))
