"""
Цепочки: ZipTree (линейно + DP), Cannon (линейно; для 2D matmul ещё и DP+Кэннон),
NumPy (линейно + DP по тому же дереву, что и ZipTree optimal).

Для 3D/4D с |λ|>0 оптимальное дерево из ``optimize_lambda_mu_tensor_chain`` использует
merge только по μ; это не обязано совпадать с цепочкой повторных ``lambda_mu_product``.

Запуск::
    python -m practice.bench_lambda_mu_chain_speed
    python -m practice.bench_lambda_mu_chain_speed --quick

Или из ``./practice``::
    python -m bench_lambda_mu_chain_speed
    python -m bench_lambda_mu_chain_speed --quick
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from typing import List
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if (_ROOT / "practice" / "ziptree_tensor.py").is_file():
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
elif (_HERE / "ziptree_tensor.py").is_file() and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from practice.chain_optimize import (
    _boundary_axis_indices,
    optimize_tensor_chain,
)
from practice.lambda_mu import (
    lambda_mu_dense_numpy,
    execute_lambda_mu_chain_linear,
    optimize_lambda_mu_tensor_chain,
    execute_lambda_mu_optimal_order,
)
from practice.ziptree_tensor import ZipTreeTensor
from practice.cannon_tensor import (
    cannon_lambda_mu,
    cannon_matrix_multiply_grid,
    ziptree_to_dense,
)


SPEC_2D = ([0], [], [1], [], [0], [1])
SPEC_3D = ([0], [1], [2], [0], [1], [2])
SPEC_4D = ([0], [1, 2], [3], [0, 1], [2], [3])


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


def _axes_pairs_from_edges(full_params_list):
    return [(tuple(p[2]), tuple(p[4])) for p in full_params_list]


def _execute_numpy_lambda_mu_chain(
    arrays: List[np.ndarray],
    edge_specs: list,
    splits,
    i: int,
    j: int,
    n_axes_list: List[int],
) -> np.ndarray:
    if i == j:
        return arrays[i]
    k = splits[i][j]
    left = _execute_numpy_lambda_mu_chain(arrays, edge_specs, splits, i, k, n_axes_list)
    right = _execute_numpy_lambda_mu_chain(arrays, edge_specs, splits, k + 1, j, n_axes_list)
    axes_pairs = _axes_pairs_from_edges(edge_specs)
    pos_l, pos_r = _boundary_axis_indices(i, j, k, splits, axes_pairs, n_axes_list)
    return np.tensordot(left, right, axes=(list(pos_l), list(pos_r)))


def _numpy_chain_linear(arrays: List[np.ndarray], edge_specs: list) -> np.ndarray:
    acc = arrays[0]
    for t in range(len(arrays) - 1):
        acc = lambda_mu_dense_numpy(acc, arrays[t + 1], *edge_specs[t])
    return acc


def _linear_splits(n: int):
    sp = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            sp[i][j] = j - 1
    return sp


def _execute_cannon_matmul_optimal(tensors, axes_pairs, splits, i, j):
    if i == j:
        return tensors[i]
    k = splits[i][j]
    L = _execute_cannon_matmul_optimal(tensors, axes_pairs, splits, i, k)
    R = _execute_cannon_matmul_optimal(tensors, axes_pairs, splits, k + 1, j)
    n_axes = [len(t.dims) for t in tensors]
    pos_l, pos_r = _boundary_axis_indices(i, j, k, splits, axes_pairs, n_axes)
    if len(L.dims) == 2 and len(R.dims) == 2 and pos_l == (1,) and pos_r == (0,):
        return cannon_matrix_multiply_grid(L, R, 1, 0, s=None)
    return L.multiply_with_reindex(R, pos_l, pos_r)


def _cannon_chain_linear_lambda_mu(tensors: List[ZipTreeTensor], edge_specs: list, fiber_s: int):
    acc = tensors[0]
    for t, spec in enumerate(edge_specs):
        acc = cannon_lambda_mu(acc, tensors[t + 1], *spec, fiber_s=fiber_s)
    return acc


def _max_abs_diff_sparse(a: ZipTreeTensor, b: ZipTreeTensor) -> float:
    da = dict(a.get_all_elements())
    db = dict(b.get_all_elements())
    diff = 0.0
    for k, v in da.items():
        diff = max(diff, abs(v - db.get(k, 0.0)))
    for k, v in db.items():
        if k not in da:
            diff = max(diff, abs(v))
    return float(diff)


def bench_block(
    title: str,
    dims_list: List[tuple[int, ...]],
    edge_specs: list,
    density: float,
    rng: random.Random,
    repeat: int,
    fiber_s: int,
    *,
    cannon_opt_matmul: bool,
) -> None:
    n = len(dims_list)
    tensors = [_random_ziptree(d, density, rng) for d in dims_list]
    n_axes_list = [len(t.dims) for t in tensors]
    lin_sp = _linear_splits(n)

    t_zt_lin = _bench(
        lambda: execute_lambda_mu_chain_linear(tensors, edge_specs),
        repeat,
    )

    cost, splits = optimize_lambda_mu_tensor_chain(tensors, edge_specs)
    t_zt_opt = _bench(
        lambda: execute_lambda_mu_optimal_order(tensors, edge_specs, splits, 0, n - 1),
        repeat,
    )

    t_cn_lin = _bench(lambda: _cannon_chain_linear_lambda_mu(tensors, edge_specs, fiber_s), repeat)

    t_cn_opt = float("nan")
    if cannon_opt_matmul:
        axes_pairs = _axes_pairs_from_edges(edge_specs)
        _, sp_mat = optimize_tensor_chain(tensors, axes_pairs)
        t_cn_opt = _bench(
            lambda: _execute_cannon_matmul_optimal(tensors, axes_pairs, sp_mat, 0, n - 1),
            repeat,
        )

    res_lin = execute_lambda_mu_chain_linear(tensors, edge_specs)
    res_opt = execute_lambda_mu_optimal_order(tensors, edge_specs, splits, 0, n - 1)
    diff_lin_opt = _max_abs_diff_sparse(res_lin, res_opt)

    arrs = [ziptree_to_dense(t) for t in tensors]
    t_np_lin = _bench(lambda: _numpy_chain_linear(arrs, edge_specs), repeat)
    t_np_opt = _bench(
        lambda: _execute_numpy_lambda_mu_chain(arrs, edge_specs, splits, 0, n - 1, n_axes_list),
        repeat,
    )

    print(f"  {title}  (DP cost ~ {cost:.3g})")
    print(
        f"    ZipTree:  linear {t_zt_lin:.5f}s   optimal {t_zt_opt:.5f}s"
    )
    print(
        f"    Cannon:   linear {t_cn_lin:.5f}s"
        + (f"   optimal(2D) {t_cn_opt:.5f}s" if cannon_opt_matmul else "   (optimal только для 2D matmul)")
    )
    print(f"    NumPy:    linear {t_np_lin:.5f}s   optimal {t_np_opt:.5f}s")
    print(f"    max diff ZipTree linear vs optimal = {diff_lin_opt:.2e}")
    if diff_lin_opt > 1e-6:
        print(
            "    (для |lambda|>0 линейная цепочка lambda_mu_product != merge только по mu в DP)"
        )

    nn = int(np.prod(res_opt.dims)) if res_opt.dims else 0
    if nn and nn <= 2_000_000:
        out_np = _execute_numpy_lambda_mu_chain(arrs, edge_specs, splits, 0, n - 1, n_axes_list)
        dzo = float(np.max(np.abs(ziptree_to_dense(res_opt) - out_np)))
        print(f"    max diff ZipTree optimal vs NumPy optimal = {dzo:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    repeat = 1 if args.quick else 3
    fs = 4 if args.quick else 8

    print("=== Цепочки: ZipTree / Cannon / NumPy ===\n")

    if args.quick:
        bench_block(
            "2D матрицы (4 шт., 32x32)",
            [(32, 32)] * 4,
            [SPEC_2D] * 3,
            0.05,
            rng,
            repeat,
            fs,
            cannon_opt_matmul=True,
        )
        bench_block(
            "3D гиперкуб (3 шт., 2x2x2)",
            [(2, 2, 2)] * 3,
            [SPEC_3D] * 2,
            0.4,
            rng,
            repeat,
            fs,
            cannon_opt_matmul=False,
        )
        bench_block(
            "4D гиперкуб (3 шт., 2^4)",
            [(2, 2, 2, 2)] * 3,
            [SPEC_4D] * 2,
            0.35,
            rng,
            repeat,
            fs,
            cannon_opt_matmul=False,
        )
    else:
        bench_block(
            "2D матрицы",
            [(120, 100), (100, 80), (80, 90), (90, 110)],
            [SPEC_2D] * 3,
            0.025,
            rng,
            repeat,
            fs,
            cannon_opt_matmul=True,
        )
        bench_block(
            "3D однородные размеры",
            [(12, 12, 12)] * 4,
            [SPEC_3D] * 3,
            0.02,
            rng,
            repeat,
            fs,
            cannon_opt_matmul=False,
        )
        bench_block(
            "4D однородные размеры",
            [(8, 8, 8, 8)] * 4,
            [SPEC_4D] * 3,
            0.015,
            rng,
            repeat,
            fs,
            cannon_opt_matmul=False,
        )


if __name__ == "__main__":
    main()
