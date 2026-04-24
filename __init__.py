"""
Самодостаточный набор: ZipTreeTensor, lambda-mu (λ,μ), Кэннон, DP-порядок цепочек, бенчмарки.

Запуск из каталога ``diplom``:
    python -m practice.bench_multiply_speed
    python -m practice.bench_lambda_mu_pair_speed
    python -m practice.bench_lambda_mu_chain_speed
    python -m practice.bench_parallel_cannon_compare

Ноутбук ``evaluation.ipynb`` и текст ``REPORT.md`` — проверки и описание пакета.
"""

from .ziptree_tensor import (
    ZipTreeTensor,
    Node,
    _normalize_conv_axes,
    _contract_dims_pair,
    _dense_merge_cost,
)
from .lambda_mu import (
    lambda_mu_product,
    lambda_mu_dense_numpy,
    lambda_mu_result_shape,
    execute_lambda_mu_chain_linear,
    optimize_lambda_mu_tensor_chain,
    execute_lambda_mu_optimal_order,
)
from .chain_optimize import (
    optimize_tensor_chain,
    execute_optimal_order,
    execute_optimal_order_reind,
    construct_order,
    get_flat_index_order,
)
from .cannon_tensor import (
    cannon_matrix_multiply_grid,
    cannon_contract_fiber,
    cannon_contract_product_grid,
    cannon_lambda_mu,
    ziptree_to_dense,
)

__all__ = [
    "ZipTreeTensor",
    "Node",
    "_normalize_conv_axes",
    "_contract_dims_pair",
    "_dense_merge_cost",
    "lambda_mu_product",
    "lambda_mu_dense_numpy",
    "lambda_mu_result_shape",
    "execute_lambda_mu_chain_linear",
    "optimize_lambda_mu_tensor_chain",
    "execute_lambda_mu_optimal_order",
    "optimize_tensor_chain",
    "execute_optimal_order",
    "execute_optimal_order_reind",
    "construct_order",
    "get_flat_index_order",
    "cannon_matrix_multiply_grid",
    "cannon_contract_fiber",
    "cannon_contract_product_grid",
    "cannon_lambda_mu",
    "ziptree_to_dense",
]
