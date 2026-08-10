"""Regenerate every figure and table in the paper.

    python scripts/make_all.py
"""

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    ("make_tables.py", "Tables 2, 3, 4"),
    ("make_figure3.py", "Figure 3  RQ1 baselines"),
    ("make_figure4.py", "Figure 4  RQ2 input representation"),
    ("make_figure5.py", "Figure 5  RQ3 batch size"),
    ("make_figure6.py", "Figure 6  RQ3 quantization"),
    ("make_figure7.py", "Figure 7  Pareto frontier"),
]

failed = []
for script, label in STEPS:
    print(f"\n=== {label} ===")
    try:
        runpy.run_path(str(HERE / script), run_name="__main__")
    except SystemExit as exc:            # the misfiled-run guard exits non-zero
        if exc.code:
            print(exc.code if isinstance(exc.code, str) else f"aborted ({exc.code})")
            failed.append(script)
    except Exception as exc:
        print(f"FAILED: {exc}")
        failed.append(script)

print("\n" + "=" * 60)
if failed:
    print("FAILED: " + ", ".join(failed))
    sys.exit(1)
print("All figures written to figures/ and tables to tables/.")
