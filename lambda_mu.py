"""
(λ, μ)-свёрнутое произведение для ZipTreeTensor.

Реализация через перестановку в (l,s,c) / (s,c,m) и **волокна по совпадающим λ**:
на каждом фиксированном s вызывается ``multiply_with_reindex`` только по μ (оси c),
что согласуется с определением C_{l,s,m} = Σ_c A_{l,s,c} B_{s,c,m}.
"""

from __future__ import annotations

import numpy as np

from .ziptree_tensor import ZipTreeTensor
from .chain_optimize import optimize_tensor_chain, execute_optimal_order_reind


def lambda_mu_product(
    zt_a,
    zt_b,
    l_axes,
    s_axes_A,
    c_axes_A,
    s_axes_B,
    c_axes_B,
    m_axes,
):
    l_axes = list(l_axes)
    s_axes_A, c_axes_A = list(s_axes_A), list(c_axes_A)
    s_axes_B, c_axes_B = list(s_axes_B), list(c_axes_B)
    m_axes = list(m_axes)
    if len(l_axes) + len(s_axes_A) + len(c_axes_A) != len(zt_a.dims):
        raise ValueError("lambda_mu_product: некорректное разбиение осей A")
    if len(s_axes_B) + len(c_axes_B) + len(m_axes) != len(zt_b.dims):
        raise ValueError("lambda_mu_product: некорректное разбиение осей B")
    if len(s_axes_A) != len(s_axes_B) or len(c_axes_A) != len(c_axes_B):
        raise ValueError("lambda_mu_product: |λ| и |μ| должны совпадать на операндах")

    for i in range(len(s_axes_A)):
        if zt_a.dims[s_axes_A[i]] != zt_b.dims[s_axes_B[i]]:
            raise ValueError("lambda_mu_product: размеры скоттовых осей не совпадают")
    for i in range(len(c_axes_A)):
        if zt_a.dims[c_axes_A[i]] != zt_b.dims[c_axes_B[i]]:
            raise ValueError("lambda_mu_product: размеры кэлиевых осей не совпадают")

    order_a = l_axes + s_axes_A + c_axes_A
    a_perm = zt_a.transpose_axes_to_order(order_a)
    order_b = s_axes_B + c_axes_B + m_axes
    b_perm = zt_b.transpose_axes_to_order(order_b)

    k_l = len(l_axes)
    k_s = len(s_axes_A)
    k_mu = len(c_axes_A)
    nu = len(m_axes)

    res_dims = (
        tuple(a_perm.dims[i] for i in range(k_l))
        + tuple(a_perm.dims[k_l + i] for i in range(k_s))
        + tuple(b_perm.dims[k_s + k_mu + j] for j in range(nu))
    )
    acc = ZipTreeTensor(res_dims)

    dims_l = tuple(a_perm.dims[i] for i in range(k_l))
    dims_c = tuple(a_perm.dims[k_l + k_s + t] for t in range(k_mu))
    dims_m = tuple(b_perm.dims[k_s + k_mu + j] for j in range(nu))
    dims_a_sub = dims_l + dims_c
    dims_b_sub = dims_c + dims_m

    s_keys = set()
    for idx_a, _ in a_perm.get_all_elements():
        s_keys.add(idx_a[k_l : k_l + k_s])
    for idx_b, _ in b_perm.get_all_elements():
        s_keys.add(idx_b[0:k_s])

    la_sub = tuple(range(k_l, k_l + k_mu))
    rb_sub = tuple(range(0, k_mu))

    for s_tuple in s_keys:
        a_sub = ZipTreeTensor(dims_a_sub)
        for idx_a, va in a_perm.get_all_elements():
            if idx_a[k_l : k_l + k_s] != s_tuple:
                continue
            l_k = idx_a[:k_l]
            c_k = idx_a[k_l + k_s :]
            a_sub.add_at(l_k + c_k, va)

        b_sub = ZipTreeTensor(dims_b_sub)
        for idx_b, vb in b_perm.get_all_elements():
            if idx_b[0:k_s] != s_tuple:
                continue
            c_b = idx_b[k_s : k_s + k_mu]
            m_b = idx_b[k_s + k_mu :]
            b_sub.add_at(c_b + m_b, vb)

        if a_sub.root is None or b_sub.root is None:
            continue
        part = a_sub.multiply_with_reindex(b_sub, la_sub, rb_sub)
        for idx_p, vp in part.get_all_elements():
            l_k = idx_p[:k_l]
            m_k = idx_p[k_l:]
            out_k = l_k + s_tuple + m_k
            acc.add_at(out_k, vp)

    return acc


def lambda_mu_dense_numpy(
    A: np.ndarray,
    B: np.ndarray,
    l_axes,
    s_axes_A,
    c_axes_A,
    s_axes_B,
    c_axes_B,
    m_axes,
):
    l_axes = list(l_axes)
    s_axes_A, c_axes_A = list(s_axes_A), list(c_axes_A)
    s_axes_B, c_axes_B = list(s_axes_B), list(c_axes_B)
    m_axes = list(m_axes)
    A_p = np.transpose(A, l_axes + s_axes_A + c_axes_A)
    B_p = np.transpose(B, s_axes_B + c_axes_B + m_axes)
    k_l, k_s, k_mu = len(l_axes), len(s_axes_A), len(c_axes_A)
    nu = len(m_axes)
    subs_a = list(range(k_l)) + list(range(k_l, k_l + k_s)) + list(range(k_l + k_s, k_l + k_s + k_mu))
    subs_b = (
        list(range(k_l, k_l + k_s))
        + list(range(k_l + k_s, k_l + k_s + k_mu))
        + list(range(k_l + k_s + k_mu, k_l + k_s + k_mu + nu))
    )
    subs_out = list(range(k_l)) + list(range(k_l, k_l + k_s)) + list(
        range(k_l + k_s + k_mu, k_l + k_s + k_mu + nu)
    )
    return np.einsum(A_p, subs_a, B_p, subs_b, subs_out)


def lambda_mu_result_shape(
    dims_a,
    dims_b,
    l_axes,
    s_axes_A,
    c_axes_A,
    s_axes_B,
    c_axes_B,
    m_axes,
):
    da = tuple(dims_a)
    db = tuple(dims_b)
    l_axes = list(l_axes)
    s_axes_A, c_axes_A = list(s_axes_A), list(c_axes_A)
    s_axes_B, c_axes_B = list(s_axes_B), list(c_axes_B)
    m_axes = list(m_axes)
    for i in range(len(s_axes_A)):
        if da[s_axes_A[i]] != db[s_axes_B[i]]:
            raise ValueError("lambda_mu_result_shape: несовпадение размеров λ")
    for i in range(len(c_axes_A)):
        if da[c_axes_A[i]] != db[c_axes_B[i]]:
            raise ValueError("lambda_mu_result_shape: несовпадение размеров μ")
    return tuple(da[i] for i in l_axes) + tuple(da[i] for i in s_axes_A) + tuple(db[i] for i in m_axes)


def execute_lambda_mu_chain_linear(tensors, edge_specs_list):
    if not tensors:
        raise ValueError("пустая цепочка")
    if len(edge_specs_list) != len(tensors) - 1:
        raise ValueError("число спецификаций должно быть len(tensors)-1")
    acc = tensors[0]
    for t, spec in enumerate(edge_specs_list):
        acc = lambda_mu_product(acc, tensors[t + 1], *spec)
    return acc


def optimize_lambda_mu_tensor_chain(tensors, edge_specs):
    """
    DP по цепочке: на ребре учитываются только оси μ (c_axes_A, c_axes_B).
    Выполнение через execute_lambda_mu_optimal_order — только по μ через
    multiply_with_reindex; при |λ|>0 и >2 тензорах это может не совпадать с
    повторным lambda_mu_product (см. документацию в корневом ziptree.py).
    """
    axes_pairs = [(tuple(p[2]), tuple(p[4])) for p in edge_specs]
    return optimize_tensor_chain(tensors, axes_pairs)


def execute_lambda_mu_optimal_order(tensors, edge_specs, splits, i, j):
    axes_pairs = [(tuple(p[2]), tuple(p[4])) for p in edge_specs]
    return execute_optimal_order_reind(tensors, axes_pairs, splits, i, j)
