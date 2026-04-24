"""
Обобщение алгоритма Кэннона для ZipTreeTensor.

Структура по статье «Обобщение одного алгоритма параллельного умножения матриц…»:
  1) Перестановка к видам A_{l,s,c}, B_{s,c,m}.
  2) Внешний уровень: для каждого фиксированного набора скоттовых индексов s* сечения
     A|_{s=s*}, B|_{s=s*} дают (0, μ)-свёртку C|_{s=s*}.
  3) Внутри сечения: блочный Кэннон по кэлиевым осям c — ``cannon_contract_product_grid``
     с сеткой T×…×T (T = n / E), E — длина блока по каждой оси c; смешанно-радиксные
     фазы совпадают с обобщением сдвига Кэннона из статьи.

Вспомогательно: ``cannon_matrix_multiply_grid`` — классический Кэннон 2D;
``cannon_contract_fiber`` / ``cannon_contract_product_grid`` — низкоуровневые расписания.
"""

from __future__ import annotations

import math
from functools import reduce
from itertools import product
from operator import mul

from .ziptree_tensor import ZipTreeTensor, _contract_dims_pair, _normalize_conv_axes
from .lambda_mu import lambda_mu_product, lambda_mu_result_shape


def _mixed_radix_encode(components: tuple[int, ...], dims: tuple[int, ...]) -> int:
    """components[d] in 0..dims[d]-1 -> лексикографический индекс (последняя ось меняется быстрее)."""
    flat = 0
    m = 1
    for c, d in zip(components, dims):
        flat += c * m
        m *= d
    return flat


def _fiber_key(idx: tuple[int, ...], axes: tuple[int, ...], dims: tuple[int, ...]) -> int:
    comp = tuple(idx[a] for a in axes)
    return _mixed_radix_encode(comp, _fiber_dims_from_dims(dims, axes))


def _fiber_dims_from_dims(dims: tuple[int, ...], axes: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(dims[a] for a in axes)


def _split_fiber_into_s_blocks(fiber_size: int, s_blocks: int) -> list[tuple[int, int]]:
    """Интервалы [lo, hi) по плоскому индексу волокна, len == s_blocks, равномерно."""
    if fiber_size % s_blocks != 0:
        raise ValueError(f"fiber_size={fiber_size} не делится на S={s_blocks}")
    chunk = fiber_size // s_blocks
    return [(b * chunk, (b + 1) * chunk) for b in range(s_blocks)]


def _extract_by_fiber_block(
    zt: ZipTreeTensor,
    contract_axes: tuple[int, ...],
    fiber_lo: int,
    fiber_hi: int,
) -> ZipTreeTensor:
    """Подтензор: только элементы, у которых смешанно-радиксный ключ волокна в [fiber_lo, fiber_hi)."""
    out = ZipTreeTensor(zt.dims)
    for idx, val in zt.get_all_elements():
        fk = _fiber_key(idx, contract_axes, zt.dims)
        if fiber_lo <= fk < fiber_hi:
            out.insert_raw(idx, val)
    return out


def _extract_by_fiber_blocks_product(
    zt: ZipTreeTensor,
    contract_axes: tuple[int, ...],
    grid: tuple[int, ...],
    block_multi: tuple[int, ...],
) -> ZipTreeTensor:
    """
    Декартово произведение интервалов: ось contract_axes[d] делится на grid[d] равных частей,
    берётся блок с индексом block_multi[d].
    """
    if not (len(grid) == len(contract_axes) == len(block_multi)):
        raise ValueError("grid, block_multi и contract_axes должны совпадать по длине")
    lo_hi = []
    for ax, g, bid in zip(contract_axes, grid, block_multi):
        d = zt.dims[ax]
        if d % g != 0:
            raise ValueError(f"ось {ax}: размер {d} не делится на {g}")
        chunk = d // g
        lo = bid * chunk
        hi = (bid + 1) * chunk
        lo_hi.append((ax, lo, hi))
    out = ZipTreeTensor(zt.dims)
    for idx, val in zt.get_all_elements():
        ok = True
        for ax, lo, hi in lo_hi:
            if not (lo <= idx[ax] < hi):
                ok = False
                break
        if ok:
            out.insert_raw(idx, val)
    return out


def _merge_add(dst: ZipTreeTensor, src: ZipTreeTensor) -> None:
    for k, v in src.get_all_elements():
        dst.add_at(k, v)


def _uniform_axis_size_n(
    a_perm: ZipTreeTensor,
    b_perm: ZipTreeTensor,
    k_l: int,
    k_s: int,
    k_mu: int,
    nu: int,
) -> int | None:
    """
    Проверка предположения статьи: все индексы l,s,c,m принимают одно и то же число
    значений n (диапазон 0..n-1 на каждой оси после перестановки).
    """
    vals: set[int] = set()
    for i in range(k_l + k_s + k_mu):
        vals.add(a_perm.dims[i])
    for i in range(k_s + k_mu + nu):
        vals.add(b_perm.dims[i])
    if len(vals) != 1:
        return None
    n = next(iter(vals))
    return n if n >= 1 else None


def _block_count_along_axis(n: int, fiber_hint: int) -> int:
    """Число блоков T вдоль оси (n = T·E); fiber_hint — желаемый верхний предел для T."""
    if n < 2:
        return 1
    T = min(max(2, fiber_hint), n)
    while T > 1 and n % T != 0:
        T -= 1
    return max(1, T)


def _extract_section_a_lc(
    a_perm: ZipTreeTensor,
    s_tuple: tuple[int, ...],
    k_l: int,
    k_s: int,
    k_mu: int,
) -> ZipTreeTensor:
    """Сечение A|_{s=s*} с осями (l, c) — как в статье после фиксации скоттовых индексов."""
    dims_l = tuple(a_perm.dims[i] for i in range(k_l))
    dims_c = tuple(a_perm.dims[k_l + k_s + t] for t in range(k_mu))
    out = ZipTreeTensor(dims_l + dims_c)
    for idx, val in a_perm.get_all_elements():
        if idx[k_l : k_l + k_s] != s_tuple:
            continue
        new_k = idx[:k_l] + idx[k_l + k_s : k_l + k_s + k_mu]
        out.insert_raw(new_k, val)
    return out


def _extract_section_b_cm(
    b_perm: ZipTreeTensor,
    s_tuple: tuple[int, ...],
    k_s: int,
    k_mu: int,
    nu: int,
) -> ZipTreeTensor:
    """Сечение B|_{s=s*} с осями (c, m)."""
    dims_c = tuple(b_perm.dims[t] for t in range(k_s, k_s + k_mu))
    dims_m = tuple(b_perm.dims[k_s + k_mu + j] for j in range(nu))
    out = ZipTreeTensor(dims_c + dims_m)
    for idx, val in b_perm.get_all_elements():
        if idx[0:k_s] != s_tuple:
            continue
        new_k = idx[k_s : k_s + k_mu] + idx[k_s + k_mu :]
        out.insert_raw(new_k, val)
    return out


def _scatter_section_into_c(
    acc: ZipTreeTensor,
    s_tuple: tuple[int, ...],
    k_l: int,
    k_s: int,
    part: ZipTreeTensor,
) -> None:
    """Индексы part — (l, m); в C — (l, s*, m)."""
    for idx_lm, val in part.get_all_elements():
        out_k = idx_lm[:k_l] + s_tuple + idx_lm[k_l:]
        acc.add_at(out_k, val)


def _is_empty(zt: ZipTreeTensor) -> bool:
    return zt.root is None


def _bucket_by_fiber_intervals(
    zt: ZipTreeTensor, contract_axes: tuple[int, ...], intervals: list[tuple[int, int]]
) -> list[ZipTreeTensor]:
    """Однократное разбиение тензора по блокам плоского индекса волокна."""
    if not intervals:
        return []
    s_blocks = len(intervals)
    chunk = intervals[0][1] - intervals[0][0]
    out = [ZipTreeTensor(zt.dims) for _ in range(s_blocks)]
    for idx, val in zt.get_all_elements():
        fk = _fiber_key(idx, contract_axes, zt.dims)
        bid = fk // chunk
        if bid >= s_blocks:
            bid = s_blocks - 1
        out[bid].insert_raw(idx, val)
    return out


def _bucket_by_product_grid(
    zt: ZipTreeTensor, contract_axes: tuple[int, ...], grid: tuple[int, ...]
) -> dict[tuple[int, ...], ZipTreeTensor]:
    """Однократное разбиение тензора по декартовой сетке блоков вдоль осей свёртки."""
    chunks = [zt.dims[ax] // g for ax, g in zip(contract_axes, grid)]
    ranges = [range(g) for g in grid]
    out = {bm: ZipTreeTensor(zt.dims) for bm in product(*ranges)}
    for idx, val in zt.get_all_elements():
        bm = tuple(idx[ax] // ch for ax, ch in zip(contract_axes, chunks))
        out[bm].insert_raw(idx, val)
    return out


def _bucket_matrix_blocks(
    zt: ZipTreeTensor, rows: int, cols: int, s: int
) -> list[list[ZipTreeTensor]]:
    """Однократное разбиение 2D-матрицы на блоки s x s."""
    ch_r = rows // s
    ch_c = cols // s
    out = [[ZipTreeTensor(zt.dims) for _ in range(s)] for _ in range(s)]
    for idx, val in zt.get_all_elements():
        i, j = idx
        bi = i // ch_r
        bj = j // ch_c
        if bi >= s:
            bi = s - 1
        if bj >= s:
            bj = s - 1
        out[bi][bj].insert_raw(idx, val)
    return out


def cannon_contract_fiber(
    zt_a: ZipTreeTensor,
    zt_b: ZipTreeTensor,
    axes_spec,
    s_blocks: int,
) -> ZipTreeTensor:
    """
    (λ, μ)-свёртка по расписанию Кэннона вдоль **одномерного** волокна свёртки.

    axes_spec: (axis_a, axis_b) или (λ, μ) с |λ| = |μ| = k.
    Все dim(λ_i) должны совпадать с dim(μ_i). N = ∏ dim(λ_i), требуется N % s_blocks == 0.

    На фазе t ∈ {0,…,S-1} перемножаются блочные срезы волокна A и B в позициях,
    как в классическом Кэнноне: k_b = (k_a + t) mod S при переборе всех пар
    выходных блочных меток (здесь — один глобальный проход по k_a, k_b, t с
    условием согласования для накопления в полный результат).
    """
    la, rb = _normalize_conv_axes(axes_spec)
    if len(la) != len(rb):
        raise ValueError("|λ| != |μ|")
    for u, v in zip(la, rb):
        if zt_a.dims[u] != zt_b.dims[v]:
            raise ValueError(f"размеры осей {u} и {v} не совпадают")

    fiber_dims_a = _fiber_dims_from_dims(zt_a.dims, la)
    fiber_dims_b = _fiber_dims_from_dims(zt_b.dims, rb)
    if fiber_dims_a != fiber_dims_b:
        raise ValueError("волокна A и B по размерностям не совпадают")
    n_fiber = int(reduce(mul, fiber_dims_a, 1))
    if n_fiber % s_blocks != 0:
        raise ValueError(f"∏ dim(λ) = {n_fiber} не делится на S = {s_blocks}")

    intervals = _split_fiber_into_s_blocks(n_fiber, s_blocks)
    res_dims = _contract_dims_pair(zt_a.dims, zt_b.dims, la, rb)
    acc = ZipTreeTensor(res_dims)

    blocks_a = _bucket_by_fiber_intervals(zt_a, la, intervals)
    blocks_b = _bucket_by_fiber_intervals(zt_b, rb, intervals)

    for t in range(s_blocks):
        for ka in range(s_blocks):
            kb = (ka + t) % s_blocks
            sub_a = blocks_a[ka]
            sub_b = blocks_b[kb]
            if _is_empty(sub_a) or _is_empty(sub_b):
                continue
            part = sub_a.multiply_with_reindex(sub_b, la, rb)
            _merge_add(acc, part)
    return acc


def cannon_contract_product_grid(
    zt_a: ZipTreeTensor,
    zt_b: ZipTreeTensor,
    axes_spec,
    grid: tuple[int, ...],
) -> ZipTreeTensor:
    """
    Расписание «многомерного Кэннона»: тор S_0×…×S_{k-1} по осям свёртки.

    grid[d] делит dim(λ_d). Фазы — все кортежи phase ∈ ∏_d Z_{S_d}.
    На фазе phase для каждого ka ∈ ∏ Z_{S_d} берётся kb_d = (ka_d + phase_d) mod S_d
    (покоординатное обобщение сдвига Кэннона).

    Число фаз = ∏ S_d; внутри фазы — ∏ S_d пар блоков (полное покрытие декартова
    произведения блочных индексов при фиксированном phase).
    """
    la, rb = _normalize_conv_axes(axes_spec)
    k = len(la)
    if len(grid) != k:
        raise ValueError("len(grid) должен совпадать с числом пар свёртки")
    for u, v, g in zip(la, rb, grid):
        if zt_a.dims[u] != zt_b.dims[v]:
            raise ValueError(f"размеры осей {u} и {v} не совпадают")
        if zt_a.dims[u] % g or zt_b.dims[v] % g:
            raise ValueError(f"ось A{u} / B{v}: длина не делится на {g}")

    res_dims = _contract_dims_pair(zt_a.dims, zt_b.dims, la, rb)
    acc = ZipTreeTensor(res_dims)
    ranges = [range(g) for g in grid]

    blocks_a = _bucket_by_product_grid(zt_a, la, grid)
    blocks_b = _bucket_by_product_grid(zt_b, rb, grid)

    for phase in product(*ranges):
        for ka in product(*ranges):
            kb = tuple((ka[d] + phase[d]) % grid[d] for d in range(k))
            sub_a = blocks_a[ka]
            sub_b = blocks_b[kb]
            if _is_empty(sub_a) or _is_empty(sub_b):
                continue
            part = sub_a.multiply_with_reindex(sub_b, la, rb)
            _merge_add(acc, part)
    return acc


def _default_cannon_block_size(m: int, n: int, p: int) -> int:
    s = math.gcd(m, math.gcd(n, p))
    while s > 1 and (m % s or n % s or p % s):
        s -= 1
    return max(s, 1)


def cannon_matrix_multiply_grid(
    zt_a: ZipTreeTensor,
    zt_b: ZipTreeTensor,
    axis_contract_a: int = 1,
    axis_contract_b: int = 0,
    s: int | None = None,
) -> ZipTreeTensor:
    """
    Классический Кэннон на S×S блоках: C = A * B при свёртке (axis_contract_a, axis_contract_b).

    Поддерживается типичный случай A (m,n), B (n,p), оси (1, 0). Сетка: m, n, p делятся на S;
    на фазе t блоки A[i, k] и B[k, j] с k = (i + j + t) mod S перемножаются и суммируются в C[i, j].
    """
    if axis_contract_a != 1 or axis_contract_b != 0:
        raise ValueError("cannon_matrix_multiply_grid: пока поддерживается только A(m,n)*B(n,p) с осями (1, 0)")

    da, db = zt_a.dims, zt_b.dims
    if len(da) != 2 or len(db) != 2:
        raise ValueError("ожидаются 2D-тензоры (матрицы)")
    m, n = da[0], da[1]
    n2, p = db[0], db[1]
    if n != n2:
        raise ValueError("внутренние размеры матриц не совпадают")

    if s is None:
        s = _default_cannon_block_size(m, n, p)
    if s < 1 or m % s or n % s or p % s:
        raise ValueError(f"S={s}: требуется S|m, S|n, S|p")

    def bounds(dim: int, bid: int) -> tuple[int, int]:
        ch = dim // s
        return bid * ch, (bid + 1) * ch

    acc = ZipTreeTensor((m, p))

    blocks_a = _bucket_matrix_blocks(zt_a, m, n, s)
    blocks_b = _bucket_matrix_blocks(zt_b, n, p, s)

    for t in range(s):
        for i in range(s):
            for j in range(s):
                k_blk = (i + j + t) % s
                sub_a = blocks_a[i][k_blk]
                sub_b = blocks_b[k_blk][j]
                if _is_empty(sub_a) or _is_empty(sub_b):
                    continue
                part = sub_a.multiply_with_reindex(sub_b, (1,), (0,))
                _merge_add(acc, part)
    return acc


def cannon_lambda_mu(
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
) -> ZipTreeTensor:
    """
    (λ, μ)-свсвернутое умножение : сечения по s*, внутри — Кэннон по осям μ (сетка T).

    block_edge: если задано, длина блока E по каждой оси c (T = n / E). Иначе E = n / T,
    T подбирается из ``fiber_s`` (желаемое число блоков вдоль оси, с делимостью n).

    Если не выполнены условия статьи (общее n на l,s,c,m, κ=ν), возвращается
    ``lambda_mu_product`` — точное произведение без блочного расписания.
    """
    l_axes = list(l_axes)
    s_axes_A, c_axes_A = list(s_axes_A), list(c_axes_A)
    s_axes_B, c_axes_B = list(s_axes_B), list(c_axes_B)
    m_axes = list(m_axes)
    if len(s_axes_A) != len(s_axes_B) or len(c_axes_A) != len(c_axes_B):
        raise ValueError("cannon_lambda_mu: |λ| и |μ| должны совпадать")

    order_a = l_axes + s_axes_A + c_axes_A
    a_perm = zt_a.transpose_axes_to_order(order_a)
    order_b = s_axes_B + c_axes_B + m_axes
    b_perm = zt_b.transpose_axes_to_order(order_b)
    k_l = len(l_axes)
    k_s = len(s_axes_A)
    k_mu = len(c_axes_A)
    nu = len(m_axes)

    n_uni = _uniform_axis_size_n(a_perm, b_perm, k_l, k_s, k_mu, nu)
    kappa_eq_nu = k_l == nu

    if (
        n_uni is not None
        and kappa_eq_nu
        and n_uni >= 1
    ):
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
            T_axis = _block_count_along_axis(n_uni, fiber_s)
            grid_T = (T_axis,) * k_mu

        la_sec = tuple(range(k_l, k_l + k_mu))
        rb_sec = tuple(range(0, k_mu))

        for s_tuple in s_keys:
            a_sec = _extract_section_a_lc(a_perm, s_tuple, k_l, k_s, k_mu)
            b_sec = _extract_section_b_cm(b_perm, s_tuple, k_s, k_mu, nu)
            if _is_empty(a_sec) or _is_empty(b_sec):
                continue
            try:
                part = cannon_contract_product_grid(
                    a_sec, b_sec, (la_sec, rb_sec), grid_T
                )
            except ValueError:
                return lambda_mu_product(
                    zt_a, zt_b, l_axes, s_axes_A, c_axes_A, s_axes_B, c_axes_B, m_axes
                )
            _scatter_section_into_c(acc, s_tuple, k_l, k_s, part)
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
            return cannon_contract_product_grid(a_perm, b_perm, axes_spec, grid)
        n_fiber = int(reduce(mul, (a_perm.dims[i] for i in la), 1))
        s_use = fiber_s
        if s_use < 1:
            s_use = 1
        if n_fiber % s_use != 0:
            s_use = max(1, math.gcd(n_fiber, s_use))
        return cannon_contract_fiber(a_perm, b_perm, axes_spec, s_use)

    return lambda_mu_product(
        zt_a, zt_b, l_axes, s_axes_A, c_axes_A, s_axes_B, c_axes_B, m_axes
    )


# Опечатка в ранних версиях бенчмарков
cannon_lamba_mu = cannon_lambda_mu


def ziptree_to_dense(zt: ZipTreeTensor):
    import numpy as np

    a = np.zeros(zt.dims, dtype=np.float64)
    for k, v in zt.get_all_elements():
        a[k] = v
    return a
