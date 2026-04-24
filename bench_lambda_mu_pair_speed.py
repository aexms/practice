"""
Скорость пары: ZipTree (lambda_mu_product), Cannon (cannon_lambda_mu), NumPy (lambda_mu_dense_numpy).

Кейсы: 2D матричное произведение, 3D (1,1)-свернутое, 4D (2,1)-свернутое.

Запуск::
    python -m practice.bench_lambda_mu_pair_speed
    python -m practice.bench_lambda_mu_pair_speed --quick

Или из ``./practice``::
    python -m bench_lambda_mu_pair_speed
    python -m bench_lambda_mu_pair_speed --quick
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if (_ROOT / "practice" / "ziptree_tensor.py").is_file():
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
elif (_HERE / "ziptree_tensor.py").is_file() and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from practice.lambda_mu import lambda_mu_dense_numpy, lambda_mu_product
from practice.ziptree_tensor import ZipTreeTensor
from practice.cannon_tensor import cannon_lambda_mu, ziptree_to_dense


SPEC_2D_MATMUL = ([0], [], [1], [], [0], [1])
SPEC_3D_1_1 = ([0], [1], [2], [0], [1], [2])
SPEC_4D_2_1 = ([0], [1, 2], [3], [0, 1], [2], [3])


def _random_ziptree(dims: tuple[int, ...], density: float, rng: random.Random) -> ZipTreeTensor:
    zt = ZipTreeTensor(dims)
    n = int(np.prod(dims))
    for _ in range(max(1, int(n * density))):
        idx = tuple(rng.randint(0, dims[i] - 1) for i in range(len(dims)))
        zt.add_at(idx, rng.random())
    return zt


def _bench(fn, repeat: int) -> float:
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - t0) / repeat


def run_case(
    label: str,
    dims_a: tuple[int, ...],
    dims_b: tuple[int, ...],
    spec: tuple,
    density: float,
    rng: random.Random,
    repeat: int,
    fiber_s: int,
) -> None:
    za = _random_ziptree(dims_a, density, rng)
    zb = _random_ziptree(dims_b, density, rng)
    A = ziptree_to_dense(za)
    B = ziptree_to_dense(zb)

    t_np = _bench(lambda: lambda_mu_dense_numpy(A, B, *spec), repeat)
    t_zt = _bench(lambda: lambda_mu_product(za, zb, *spec), repeat)
    t_cn = _bench(lambda: cannon_lambda_mu(za, zb, *spec, fiber_s=fiber_s), repeat)

    ref = lambda_mu_dense_numpy(A, B, *spec)
    zc = lambda_mu_product(za, zb, *spec)
    cc = cannon_lambda_mu(za, zb, *spec, fiber_s=fiber_s)
    dz = float(np.max(np.abs(ziptree_to_dense(zc) - ref))) if ref.size else 0.0
    dc = float(np.max(np.abs(ziptree_to_dense(cc) - ref))) if ref.size else 0.0

    print(f"  {label}")
    print(f"    ZipTree {t_zt:.5f}s  Cannon {t_cn:.5f}s  NumPy {t_np:.5f}s  (rho={density})")
    print(f"    max|zt-np|={dz:.2e}  max|cannon-np|={dc:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    repeat = 2 if args.quick else 5
    fs = 4 if args.quick else 8

    print("=== Пары: ZipTree / Cannon / NumPy ===\n")

    run_case(
        "2D matmul",
        (96, 96),
        (96, 96),
        SPEC_2D_MATMUL,
        0.04,
        rng,
        repeat,
        fs,
    )
    run_case(
        "3D (1,1)",
        (24, 28, 32),
        (28, 32, 20),
        SPEC_3D_1_1,
        0.03,
        rng,
        repeat,
        fs,
    )
    run_case(
        "4D (2,1)",
        (16, 14, 12, 24),
        (14, 12, 24, 18),
        SPEC_4D_2_1,
        0.02,
        rng,
        repeat,
        fs,
    )


if __name__ == "__main__":
    main()
