"""Write k6 request bodies made of real test-subject windows.

bench/payloads/single.json  : 64 bodies of 1 window each
bench/payloads/batch32.json : 16 bodies of 32 windows each
"""

import json
from pathlib import Path

import numpy as np

from movement.data import load_cached, split_by_subject

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bench" / "payloads"


def main() -> None:
    _, _, test = split_by_subject(load_cached(ROOT / "data" / "windows.npz"))
    rng = np.random.default_rng(0)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, n_bodies, size in [("single", 64, 1), ("batch32", 16, 32)]:
        bodies = [json.dumps({"windows": test.X[rng.choice(len(test.X), size, replace=False)].tolist()})
                  for _ in range(n_bodies)]
        (OUT / f"{name}.json").write_text(json.dumps(bodies))
        print(f"{name}: {n_bodies} bodies, {len(bodies[0])} bytes each")


if __name__ == "__main__":
    main()
