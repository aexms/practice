"""
Сравнение скорости: multiply vs multiply_with_reindex vs NumPy (пары и линейные цепочки).

Запуск::
    python -m practice.bench_multiply_speed
    python -m practice.bench_multiply_speed --quick

Или из ``/practice``::
    python -m bench_multiply_speed
    python -m bench_multiply_speed --quick
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

from practice.ziptree_tensor import ZipTreeTensor, _normalize_conv_axes
from practice.cannon_tensor import ziptree_to_dense


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


def bench_pair(
    dims_a: tuple[int, ...],
    dims_b: tuple[int, ...],
    axes,
    density: float,
    rng: random.Random,
    repeat: int,
) -> None:
    la, rb = _normalize_conv_axes(axes)
    za = _random_ziptree(dims_a, density, rng)
    zb = _random_ziptree(dims_b, density, rng)
    na = ziptree_to_dense(za)
    nb = ziptree_to_dense(zb)

    t_mul = _bench(lambda: za.multiply(zb, axes), repeat)
    t_rei = _bench(lambda: za.multiply_with_reindex(zb, la, rb), repeat)
    t_np = _bench(
        lambda: np.tensordot(na, nb, axes=(list(la), list(rb))),
        repeat,
    )
    print(
        f"  pair {dims_a} x {dims_b} axes={axes} rho={density}: "
        f"multiply {t_mul:.5f}s  reindex {t_rei:.5f}s  numpy {t_np:.5f}s"
    )


def bench_chain_matmul(
    dim: int,
    num: int,
    density: float,
    rng: random.Random,
    repeat: int,
) -> None:
    axes = (1, 0)
    la, rb = _normalize_conv_axes(axes)
    tensors = [_random_ziptree((dim, dim), density, rng) for _ in range(num)]
    arrs = [ziptree_to_dense(t) for t in tensors]

    def chain_mul():
        acc = tensors[0]
        for i in range(1, num):
            acc = acc.multiply(tensors[i], axes)
        return acc

    def chain_rei():
        acc = tensors[0]
        for i in range(1, num):
            acc = acc.multiply_with_reindex(tensors[i], la, rb)
        return acc

    def chain_np():
        acc = arrs[0]
        for i in range(1, num):
            acc = np.tensordot(acc, arrs[i], axes=(list(la), list(rb)))
        return acc

    t_mul = _bench(chain_mul, repeat)
    t_rei = _bench(chain_rei, repeat)
    t_np = _bench(chain_np, repeat)
    print(
        f"  chain n={num} matrices {dim}x{dim} rho={density}: "
        f"multiply {t_mul:.5f}s  reindex {t_rei:.5f}s  numpy {t_np:.5f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    repeat = 2 if args.quick else 5

    print("=== Пары: multiply / multiply_with_reindex / NumPy tensordot ===")
    bench_pair((64, 64), (64, 64), (1, 0), 0.05, rng, repeat)
    bench_pair((32, 48), (48, 40), (1, 0), 0.03, rng, repeat)

    print("\n=== Цепочки матриц (левая ассоциативность) ===")
    d = 24 if args.quick else 48
    bench_chain_matmul(d, 5, 0.04, rng, repeat)
    bench_chain_matmul(d, 8, 0.02, rng, max(1, repeat // 2))


if __name__ == "__main__":
    main()
