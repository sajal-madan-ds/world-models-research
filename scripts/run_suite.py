"""Run the teaching suite from a small explicit JSON recipe."""

import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
recipe = json.loads((root / "configs/suite.json").read_text())
command = [
    sys.executable,
    "-m",
    "world_models.lab",
    "--experiment",
    recipe["experiment"],
    "--device",
    recipe["device"],
    "--steps",
    str(recipe["steps"]),
    "--seeds",
    *map(str, recipe["seeds"]),
]
subprocess.run(command, cwd=root, check=True)
