#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

script = Path(__file__).with_name("04_rgb_resolution_ablation.py")

for factor in (2, 4, 8):
    print("=" * 80)
    print(f"Running RGB-resolution ablation x{factor}")
    print("=" * 80)
    subprocess.run(
        [sys.executable, str(script), "--factor", str(factor)],
        check=True,
    )
