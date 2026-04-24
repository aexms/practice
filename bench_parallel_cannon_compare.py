"""
Сравнение скорости: ZipTree (lambda_mu_product), NumPy (lambda_mu_dense_numpy),
Кэннон последовательный (cannon_lambda_mu), Кэннон параллельный (cannon_lambda_mu_parallel).

Кейсы: 2D matmul, 3D (1,1), 4D (2,1)

Запуск (нужен каталог ``practice`` внутри корня проекта на ``sys.path``)::

    python -m practice.bench_parallel_cannon_compare

Или из ``./practice``::

    python -m bench_parallel_cannon_compare

Флаги: ``--quick``, ``--workers N``, ``--heavy`` (дополнительные крупные кейсы),
``--heavy-only`.

Один общий ``ProcessPoolExecutor`` на все замеры параллельного Кэннона (меньше накладных расходов).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if (_ROOT / "practice" / "ziptree_tensor.py").is_file():
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
elif (_HERE / "ziptree_tensor.py").is_file() and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from practice.cannon_parallel import cannon_lambda_mu_parallel
from practice.cannon_tensor import cannon_lambda_mu, ziptree_to_dense
from practice.lambda_mu import lambda_mu_dense_numpy, lambda_mu_product
from practice.ziptree_tensor import ZipTreeTensor


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
    pool: ProcessPoolExecutor,
    workers: int,
    *,
    parallel_threshold: int = 2,
) -> None:
    za = _random_ziptree(dims_a, density, rng)
    zb = _random_ziptree(dims_b, density, rng)
    A = ziptree_to_dense(za)
    B = ziptree_to_dense(zb)
    ref = lambda_mu_dense_numpy(A, B, *spec)

    def warm() -> None:
        lambda_mu_product(za, zb, *spec)
        cannon_lambda_mu(za, zb, *spec, fiber_s=fiber_s)
        cannon_lambda_mu_parallel(
            za,
            zb,
            *spec,
            fiber_s=fiber_s,
            executor=pool,
            max_workers=workers,
            parallel_threshold=parallel_threshold,
        )

    warm()

    t_np = _bench(lambda: lambda_mu_dense_numpy(A, B, *spec), repeat)
    t_zt = _bench(lambda: lambda_mu_product(za, zb, *spec), repeat)
    t_cn = _bench(lambda: cannon_lambda_mu(za, zb, *spec, fiber_s=fiber_s), repeat)
    t_cnp = _bench(
        lambda: cannon_lambda_mu_parallel(
            za,
            zb,
            *spec,
            fiber_s=fiber_s,
            executor=pool,
            max_workers=workers,
            parallel_threshold=parallel_threshold,
        ),
        repeat,
    )

    zc = lambda_mu_product(za, zb, *spec)
    cc = cannon_lambda_mu(za, zb, *spec, fiber_s=fiber_s)
    ccp = cannon_lambda_mu_parallel(
        za,
        zb,
        *spec,
        fiber_s=fiber_s,
        executor=pool,
        max_workers=workers,
        parallel_threshold=parallel_threshold,
    )
    dz = float(np.max(np.abs(ziptree_to_dense(zc) - ref))) if ref.size else 0.0
    dcs = float(np.max(np.abs(ziptree_to_dense(cc) - ref))) if ref.size else 0.0
    dcp = float(np.max(np.abs(ziptree_to_dense(ccp) - ref))) if ref.size else 0.0

    print(f"  {label}")
    print(
        f"    NumPy {t_np:.5f}s  ZipTree {t_zt:.5f}s  "
        f"Cannon {t_cn:.5f}s  Cannon|| {t_cnp:.5f}s  (rho={density})"
    )
    print(
        f"    max|zt-np|={dz:.2e}  max|cannon-np|={dcs:.2e}  max|cannon||-np|={dcp:.2e}"
    )
    if t_cn > 1e-9 and t_cnp < t_cn * 0.98:
        pct = 100.0 * (1.0 - t_cnp / t_cn)
        print(f"    Cannon|| faster than Cannon by ~{pct:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="число процессов (по умолчанию min(8, CPU); в --heavy до 16)",
    )
    parser.add_argument(
        "--heavy",
        action="store_true",
        help="дополнительно прогнать крупные кейсы (больше работы на блочное умножение)",
    )
    parser.add_argument(
        "--heavy-only",
        action="store_true",
        help="только крупные кейсы (без малых по умолчанию)",
    )
    args = parser.parse_args()
    rng = random.Random(args.seed)
    repeat = 2 if args.quick else 4
    fs = 4 if args.quick else 8
    workers = args.workers if args.workers is not None else max(2, min(8, os.cpu_count() or 4))
    workers_heavy = (
        args.workers
        if args.workers is not None
        else max(2, min(16, os.cpu_count() or 4))
    )

    def run_default_block(pool: ProcessPoolExecutor, w: int) -> None:
        run_case(
            "2D matmul (uniform n, uniform-axis Cannon path)",
            (64, 64) if args.quick else (96, 96),
            (64, 64) if args.quick else (96, 96),
            SPEC_2D_MATMUL,
            0.04 if not args.quick else 0.08,
            rng,
            repeat,
            fs,
            pool,
            w,
        )
        run_case(
            "3D (1,1), uniform n",
            (16, 16, 16) if args.quick else (32, 32, 32),
            (16, 16, 16) if args.quick else (32, 32, 32),
            SPEC_3D_1_1,
            0.05 if args.quick else 0.03,
            rng,
            repeat,
            fs,
            pool,
            w,
        )
        run_case(
            "4D (2,1), uniform n",
            (10, 10, 10, 10) if args.quick else (16, 16, 16, 16),
            (10, 10, 10, 10) if args.quick else (16, 16, 16, 16),
            SPEC_4D_2_1,
            0.06 if args.quick else 0.025,
            rng,
            repeat,
            fs,
            pool,
            w,
        )

    def run_heavy_block(pool: ProcessPoolExecutor, w: int) -> None:
        """Крупные размерности и rho: больше nnz на блок, больше смысла от ProcessPool."""
        r_h = 2
        fs_h = 10 if args.quick else 12
        pt = 1
        if args.quick:
            run_case(
                "[HEAVY] 2D matmul",
                (144, 144),
                (144, 144),
                SPEC_2D_MATMUL,
                0.14,
                rng,
                r_h,
                fs_h,
                pool,
                w,
                parallel_threshold=pt,
            )
            run_case(
                "[HEAVY] 3D (1,1)",
                (40, 40, 40),
                (40, 40, 40),
                SPEC_3D_1_1,
                0.09,
                rng,
                r_h,
                fs_h,
                pool,
                w,
                parallel_threshold=pt,
            )
            run_case(
                "[HEAVY] 4D (2,1)",
                (22, 22, 22, 22),
                (22, 22, 22, 22),
                SPEC_4D_2_1,
                0.065,
                rng,
                r_h,
                fs_h,
                pool,
                w,
                parallel_threshold=pt,
            )
        else:
            run_case(
                "[HEAVY] 2D matmul",
                (288, 288),
                (288, 288),
                SPEC_2D_MATMUL,
                0.16,
                rng,
                r_h,
                fs_h,
                pool,
                w,
                parallel_threshold=pt,
            )
            run_case(
                "[HEAVY] 3D (1,1)",
                (72, 72, 72),
                (72, 72, 72),
                SPEC_3D_1_1,
                0.10,
                rng,
                r_h,
                fs_h,
                pool,
                w,
                parallel_threshold=pt,
            )
            run_case(
                "[HEAVY] 4D (2,1)",
                (30, 30, 30, 30),
                (30, 30, 30, 30),
                SPEC_4D_2_1,
                0.055,
                rng,
                r_h,
                fs_h,
                pool,
                w,
                parallel_threshold=pt,
            )

    print(
        "=== ZipTree / NumPy / Cannon (seq) / Cannon (parallel) ===\n"
        f"(ProcessPoolExecutor workers={workers})\n"
    )

    pool_workers = (
        max(workers, workers_heavy) if (args.heavy or args.heavy_only) else workers
    )
    with ProcessPoolExecutor(max_workers=pool_workers) as pool:
        if not args.heavy_only:
            run_default_block(pool, workers)
        if args.heavy or args.heavy_only:
            print(
                "\n=== HEAVY (larger tensors, higher rho, parallel_threshold=1) ===\n"
                f"(workers={workers_heavy})\n"
            )
            run_heavy_block(pool, workers_heavy)


if __name__ == "__main__":
    main()
