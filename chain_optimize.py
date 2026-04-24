"""Оптимальный порядок скобок для цепочек свёрток (DP) и выполнение по дереву."""

from __future__ import annotations

from .ziptree_tensor import _normalize_conv_axes, _contract_dims_pair, _dense_merge_cost


def _interval_dims_linear(dims_list, axes_pairs, i, j):
    d = dims_list[i]
    for t in range(i, j):
        la, rb = _normalize_conv_axes(axes_pairs[t])
        d = _contract_dims_pair(d, dims_list[t + 1], la, rb)
    return d


def _tensor_layout(i, j, splits, axes_pairs, n_axes_list):
    if i == j:
        return [(i, a) for a in range(n_axes_list[i])]
    k_split = splits[i][j]
    left = _tensor_layout(i, k_split, splits, axes_pairs, n_axes_list)
    right = _tensor_layout(k_split + 1, j, splits, axes_pairs, n_axes_list)
    la, rb = _normalize_conv_axes(axes_pairs[k_split])
    remove_l = frozenset((k_split, ax) for ax in la)
    remove_r = frozenset((k_split + 1, ax) for ax in rb)
    return [m for m in left if m not in remove_l] + [m for m in right if m not in remove_r]


def _boundary_axis_indices(i, j, k_split, splits, axes_pairs, n_axes_list):
    la, rb = _normalize_conv_axes(axes_pairs[k_split])
    lay_l = _tensor_layout(i, k_split, splits, axes_pairs, n_axes_list)
    lay_r = _tensor_layout(k_split + 1, j, splits, axes_pairs, n_axes_list)
    pos_l = tuple(lay_l.index((k_split, ax)) for ax in la)
    pos_r = tuple(lay_r.index((k_split + 1, ax)) for ax in rb)
    return pos_l, pos_r


def optimize_tensor_chain(tensors, axes_pairs):
    """
    Оптимальный порядок скобок для цепочки парных свёрток (обобщение матричной цепочки).

    tensors: список ZipTreeTensor
    axes_pairs[t]: пары индексов осей (в T_t и T_{t+1}).
    """
    n = len(tensors)
    if n == 0:
        return 0, []
    if len(axes_pairs) != n - 1:
        raise ValueError("Число спецификаций осей должно быть len(tensors) - 1")

    dims_list = [t.dims for t in tensors]
    n_axes = [len(d) for d in dims_list]

    interval_dims = [[None] * n for _ in range(n)]
    for i in range(n):
        interval_dims[i][i] = dims_list[i]

    dp = [[0] * n for _ in range(n)]
    splits = [[0] * n for _ in range(n)]

    for span in range(2, n + 1):
        for i in range(n - span + 1):
            j = i + span - 1
            dp[i][j] = float("inf")
            for k in range(i, j):
                d_left = interval_dims[i][k]
                d_right = interval_dims[k + 1][j]
                merge_cost = _dense_merge_cost(d_left, d_right)
                cand = dp[i][k] + dp[k + 1][j] + merge_cost
                if cand < dp[i][j]:
                    dp[i][j] = cand
                    splits[i][j] = k
                    pos_l, pos_r = _boundary_axis_indices(
                        i, j, k, splits, axes_pairs, n_axes
                    )
                    interval_dims[i][j] = _contract_dims_pair(
                        d_left, d_right, pos_l, pos_r
                    )

    return dp[0][n - 1], splits


def construct_order(splits, i, j):
    if i == j:
        return f"T{i}"
    k = splits[i][j]
    left = construct_order(splits, i, k)
    right = construct_order(splits, k + 1, j)
    return f"({left} @ {right})"


def execute_optimal_order(tensors, axes_pairs, splits, i, j):
    if i == j:
        return tensors[i]
    k = splits[i][j]
    left_tensor = execute_optimal_order(tensors, axes_pairs, splits, i, k)
    right_tensor = execute_optimal_order(tensors, axes_pairs, splits, k + 1, j)
    n_axes = [len(t.dims) for t in tensors]
    pos_l, pos_r = _boundary_axis_indices(i, j, k, splits, axes_pairs, n_axes)
    return left_tensor.multiply_convoluted(right_tensor, pos_l, pos_r)


def execute_optimal_order_reind(tensors, axes_pairs, splits, i, j):
    if i == j:
        return tensors[i]
    k = splits[i][j]
    left_tensor = execute_optimal_order_reind(tensors, axes_pairs, splits, i, k)
    right_tensor = execute_optimal_order_reind(tensors, axes_pairs, splits, k + 1, j)
    n_axes = [len(t.dims) for t in tensors]
    pos_l, pos_r = _boundary_axis_indices(i, j, k, splits, axes_pairs, n_axes)
    return left_tensor.multiply_with_reindex(right_tensor, pos_l, pos_r)


def get_flat_index_order(splits, i, j):
    order = []

    def walk(i, j):
        if i == j:
            order.append(i)
            return
        k = splits[i][j]
        walk(i, k)
        walk(k + 1, j)

    walk(i, j)
    return order
