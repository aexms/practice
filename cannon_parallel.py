"""
Параллельное расписание Кэннона: независимые блочные ``multiply_with_reindex``
внутри каждой фазы выполняются в ``ProcessPoolExecutor`` (отдельные процессы, обход GIL).

Поведение совпадает с ``cannon_tensor`` по математике; при малом числе задач или
``max_workers=1`` можно сводить к последовательному режиму без пула.

На Windows дочерние процессы импортируют этот модуль; воркер — функция верхнего уровня.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from functools import reduce
from itertools import product
from operator import mul

from .lambda_mu import lambda_mu_product, lambda_mu_result_shape
from .ziptree_tensor import ZipTreeTensor
from . import cannon_tensor as ct


def _par_worker_multiply(args: tuple) -> ZipTreeTensor | None:
    za, zb, la, rb = args
    if ct._is_empty(za) or ct._is_empty(zb):
        return None
    return za.multiply_with_reindex(zb, la, rb)


def _map_block_tasks(
    ex: ProcessPoolExecutor | None,
    tasks: list[tuple],
    *,
    sequential: bool,
) -> list[ZipTreeTensor | None]:
    if not tasks:
        return []
    if sequential or ex is None:
        return [_par_worker_multiply(t) for t in tasks]
    return list(ex.map(_par_worker_multiply, tasks))


def cannon_contract_product_grid_parallel(
    zt_a: ZipTreeTensor,
    zt_b: ZipTreeTensor,
    axes_spec,
    grid: tuple[int, ...],
    *,
    max_workers: int | None = None,
    executor: ProcessPoolExecutor | None = None,
    parallel_threshold: int = 2,
) -> ZipTreeTensor:
    """Аналог ``cannon_contract_product_grid``: фазы те же, пары блоков — параллельно."""
    la, rb = ct._normalize_conv_axes(axes_spec)
    k = len(la)
    if len(grid) != k:
        raise ValueError("len(grid) должен совпадать с числом пар свёртки")
    for u, v, g in zip(la, rb, grid):
        if zt_a.dims[u] != zt_b.dims[v]:
            raise ValueError(f"размеры осей {u} и {v} не совпадают")
        if zt_a.dims[u] % g or zt_b.dims[v] % g:
            raise ValueError(f"ось A{u} / B{v}: длина не делится на {g}")

    res_dims = ct._contract_dims_pair(zt_a.dims, zt_b.dims, la, rb)
    acc = ZipTreeTensor(res_dims)
    ranges = [range(g) for g in grid]

    blocks_a = ct._bucket_by_product_grid(zt_a, la, grid)
    blocks_b = ct._bucket_by_product_grid(zt_b, rb, grid)

    mw = max_workers if max_workers is not None else max(1, min(32, os.cpu_count() or 4))

    def one_phase(phase: tuple[int, ...], ex: ProcessPoolExecutor | None, seq: bool) -> None:
        tasks: list[tuple] = []
        for ka in product(*ranges):
            kb = tuple((ka[d] + phase[d]) % grid[d] for d in range(k))
            sub_a = blocks_a[ka]
            sub_b = blocks_b[kb]
            if ct._is_empty(sub_a) or ct._is_empty(sub_b):
                continue
            tasks.append((sub_a, sub_b, la, rb))
        use_seq = len(tasks) < parallel_threshold or mw < 2
        for part in _map_block_tasks(ex, tasks, sequential=use_seq or seq):
            if part is not None:
                ct._merge_add(acc, part)

    if executor is not None:
        for phase in product(*ranges):
            one_phase(phase, executor, seq=False)
    else:
        with ProcessPoolExecutor(max_workers=mw) as ex:
            for phase in product(*ranges):
                one_phase(phase, ex, seq=False)

    return acc


def cannon_contract_fiber_parallel(
    zt_a: ZipTreeTensor,
    zt_b: ZipTreeTensor,
    axes_spec,
    s_blocks: int,
    *,
    max_workers: int | None = None,
    executor: ProcessPoolExecutor | None = None,
    parallel_threshold: int = 2,
) -> ZipTreeTensor:
    """Аналог ``cannon_contract_fiber`` с параллельным перебором пар блоков по фазам."""
    la, rb = ct._normalize_conv_axes(axes_spec)
    if len(la) != len(rb):
        raise ValueError("|λ| != |μ|")
    for u, v in zip(la, rb):
        if zt_a.dims[u] != zt_b.dims[v]:
            raise ValueError(f"размеры осей {u} и {v} не совпадают")

    fiber_dims_a = ct._fiber_dims_from_dims(zt_a.dims, la)
    fiber_dims_b = ct._fiber_dims_from_dims(zt_b.dims, rb)
    if fiber_dims_a != fiber_dims_b:
        raise ValueError("волокна A и B по размерностям не совпадают")
    n_fiber = int(reduce(mul, fiber_dims_a, 1))
    if n_fiber % s_blocks != 0:
        raise ValueError(f"∏ dim(λ) = {n_fiber} не делится на S = {s_blocks}")

    intervals = ct._split_fiber_into_s_blocks(n_fiber, s_blocks)
    res_dims = ct._contract_dims_pair(zt_a.dims, zt_b.dims, la, rb)
    acc = ZipTreeTensor(res_dims)

    blocks_a = ct._bucket_by_fiber_intervals(zt_a, la, intervals)
    blocks_b = ct._bucket_by_fiber_intervals(zt_b, rb, intervals)

    mw = max_workers if max_workers is not None else max(1, min(32, os.cpu_count() or 4))

    def one_t(t: int, ex: ProcessPoolExecutor | None, seq: bool) -> None:
        tasks: list[tuple] = []
        for ka in range(s_blocks):
            kb = (ka + t) % s_blocks
            sub_a = blocks_a[ka]
            sub_b = blocks_b[kb]
            if ct._is_empty(sub_a) or ct._is_empty(sub_b):
                continue
            tasks.append((sub_a, sub_b, la, rb))
        use_seq = len(tasks) < parallel_threshold or mw < 2
        for part in _map_block_tasks(ex, tasks, sequential=use_seq or seq):
            if part is not None:
                ct._merge_add(acc, part)

    if executor is not None:
        for t in range(s_blocks):
            one_t(t, executor, seq=False)
    else:
        with ProcessPoolExecutor(max_workers=mw) as ex:
            for t in range(s_blocks):
                one_t(t, ex, seq=False)

    return acc


def cannon_matrix_multiply_grid_parallel(
    zt_a: ZipTreeTensor,
    zt_b: ZipTreeTensor,
    axis_contract_a: int = 1,
    axis_contract_b: int = 0,
    s: int | None = None,
    *,
    max_workers: int | None = None,
    executor: ProcessPoolExecutor | None = None,
    parallel_threshold: int = 2,
) -> ZipTreeTensor:
    """Аналог ``cannon_matrix_multiply_grid`` с параллелью по парам (i, j) на фазе."""
    if axis_contract_a != 1 or axis_contract_b != 0:
        raise ValueError(
            "cannon_matrix_multiply_grid_parallel: пока только A(m,n)*B(n,p), оси (1, 0)"
        )

    da, db = zt_a.dims, zt_b.dims
    if len(da) != 2 or len(db) != 2:
        raise ValueError("ожидаются 2D-тензоры (матрицы)")
    m, n = da[0], da[1]
    n2, p = db[0], db[1]
    if n != n2:
        raise ValueError("внутренние размеры матриц не совпадают")

    if s is None:
        s = ct._default_cannon_block_size(m, n, p)
    if s < 1 or m % s or n % s or p % s:
        raise ValueError(f"S={s}: требуется S|m, S|n, S|p")

    acc = ZipTreeTensor((m, p))
    blocks_a = ct._bucket_matrix_blocks(zt_a, m, n, s)
    blocks_b = ct._bucket_matrix_blocks(zt_b, n, p, s)

    mw = max_workers if max_workers is not None else max(1, min(32, os.cpu_count() or 4))

    def one_phase_t(t: int, ex: ProcessPoolExecutor | None, seq: bool) -> None:
        tasks: list[tuple] = []
        for i in range(s):
            for j in range(s):
                k_blk = (i + j + t) % s
                sub_a = blocks_a[i][k_blk]
                sub_b = blocks_b[k_blk][j]
                if ct._is_empty(sub_a) or ct._is_empty(sub_b):
                    continue
                tasks.append((sub_a, sub_b, (1,), (0,)))
        use_seq = len(tasks) < parallel_threshold or mw < 2
        for part in _map_block_tasks(ex, tasks, sequential=use_seq or seq):
            if part is not None:
                ct._merge_add(acc, part)

    if executor is not None:
        for t in range(s):
            one_phase_t(t, executor, seq=False)
    else:
        with ProcessPoolExecutor(max_workers=mw) as ex:
            for t in range(s):
                one_phase_t(t, ex, seq=False)

    return acc


def cannon_lambda_mu_parallel(
    zt_a: ZipTreeTensor,
    zt_b: ZipTreeTensor,
    l_axes,
    s_axes_A,
    c_axes_A,
    s_axes_B,
    c_axes_B,
    m_axes,
    *,
    fiber_s: int = 16,
    grid: tuple[int, ...] | None = None,
    block_edge: int | None = None,
    max_workers: int | None = None,
    executor: ProcessPoolExecutor | None = None,
    parallel_threshold: int = 2,
) -> ZipTreeTensor:
    """
    Тот же выбор веток, что у ``cannon_lambda_mu``; внутри grid/fiber — параллельные фазы.

    Если условия статьи не выполняются, возвращается ``lambda_mu_product`` (как в оригинале).
    """
    l_axes = list(l_axes)
    s_axes_A, c_axes_A = list(s_axes_A), list(c_axes_A)
    s_axes_B, c_axes_B = list(s_axes_B), list(c_axes_B)
    m_axes = list(m_axes)
    if len(s_axes_A) != len(s_axes_B) or len(c_axes_A) != len(c_axes_B):
        raise ValueError("cannon_lambda_mu_parallel: |λ| и |μ| должны совпадать")

    order_a = l_axes + s_axes_A + c_axes_A
    a_perm = zt_a.transpose_axes_to_order(order_a)
    order_b = s_axes_B + c_axes_B + m_axes
    b_perm = zt_b.transpose_axes_to_order(order_b)
    k_l = len(l_axes)
    k_s = len(s_axes_A)
    k_mu = len(c_axes_A)
    nu = len(m_axes)

    n_uni = ct._uniform_axis_size_n(a_perm, b_perm, k_l, k_s, k_mu, nu)
    kappa_eq_nu = k_l == nu

    kw = dict(
        max_workers=max_workers,
        executor=executor,
        parallel_threshold=parallel_threshold,
    )

    if n_uni is not None and kappa_eq_nu and n_uni >= 1:
        res_dims = lambda_mu_result_shape(
            zt_a.dims,
            zt_b.dims,
            l_axes,
            s_axes_A,
            c_axes_A,
            s_axes_B,
            c_axes_B,
            m_axes,
        )
        acc = ZipTreeTensor(res_dims)
        s_keys: set[tuple[int, ...]] = set()
        for idx_a, _ in a_perm.get_all_elements():
            s_keys.add(idx_a[k_l : k_l + k_s])
        for idx_b, _ in b_perm.get_all_elements():
            s_keys.add(idx_b[0:k_s])

        if grid is not None:
            if len(grid) != k_mu:
                raise ValueError("len(grid) должен совпадать с |μ|")
            grid_T = tuple(int(g) for g in grid)
        elif block_edge is not None:
            E = int(block_edge)
            if E < 1 or n_uni % E != 0:
                return lambda_mu_product(
                    zt_a, zt_b, l_axes, s_axes_A, c_axes_A, s_axes_B, c_axes_B, m_axes
                )
            T_axis = n_uni // E
            if T_axis < 1:
                return lambda_mu_product(
                    zt_a, zt_b, l_axes, s_axes_A, c_axes_A, s_axes_B, c_axes_B, m_axes
                )
            grid_T = (T_axis,) * k_mu
        else:
            T_axis = ct._block_count_along_axis(n_uni, fiber_s)
            grid_T = (T_axis,) * k_mu

        la_sec = tuple(range(k_l, k_l + k_mu))
        rb_sec = tuple(range(0, k_mu))

        for s_tuple in s_keys:
            a_sec = ct._extract_section_a_lc(a_perm, s_tuple, k_l, k_s, k_mu)
            b_sec = ct._extract_section_b_cm(b_perm, s_tuple, k_s, k_mu, nu)
            if ct._is_empty(a_sec) or ct._is_empty(b_sec):
                continue
            try:
                part = cannon_contract_product_grid_parallel(
                    a_sec, b_sec, (la_sec, rb_sec), grid_T, **kw
                )
            except ValueError:
                return lambda_mu_product(
                    zt_a, zt_b, l_axes, s_axes_A, c_axes_A, s_axes_B, c_axes_B, m_axes
                )
            ct._scatter_section_into_c(acc, s_tuple, k_l, k_s, part)
        return acc

    if len(s_axes_A) == 0:
        c_len = len(c_axes_A)
        la = tuple(
            range(len(l_axes) + len(s_axes_A), len(l_axes) + len(s_axes_A) + c_len)
        )
        rb = tuple(range(len(s_axes_B), len(s_axes_B) + c_len))
        axes_spec = (la, rb)
        if grid is not None:
            if len(grid) != c_len:
                raise ValueError("len(grid) должен совпадать с |μ|")
            return cannon_contract_product_grid_parallel(
                a_perm, b_perm, axes_spec, grid, **kw
            )
        n_fiber = int(reduce(mul, (a_perm.dims[i] for i in la), 1))
        s_use = fiber_s
        if s_use < 1:
            s_use = 1
        if n_fiber % s_use != 0:
            s_use = max(1, math.gcd(n_fiber, s_use))
        return cannon_contract_fiber_parallel(a_perm, b_perm, axes_spec, s_use, **kw)

    return lambda_mu_product(
        zt_a, zt_b, l_axes, s_axes_A, c_axes_A, s_axes_B, c_axes_B, m_axes
    )
