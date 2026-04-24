"""ZipTreeTensor и базовые операции свёртки (multiply / multiply_with_reindex). Без torch."""

from __future__ import annotations

import random


def _normalize_conv_axes(spec):
    if spec is None:
        raise ValueError("spec is None")
    if len(spec) == 2 and isinstance(spec[0], int) and isinstance(spec[1], int):
        return (spec[0],), (spec[1],)
    left, right = spec
    left = tuple(left)
    right = tuple(right)
    if len(left) != len(right):
        raise ValueError("(λ, μ): длины кортежей осей должны совпадать")
    return left, right


def _contract_dims_pair(dims_a, dims_b, left_axes, right_axes):
    la, rb = list(left_axes), list(right_axes)
    da, db = list(dims_a), list(dims_b)
    for u, v in zip(la, rb):
        if da[u] != db[v]:
            raise ValueError(
                f"Несовпадение размерностей при свёртке осей {u} и {v}: {da[u]} != {db[v]}"
            )
    drop_a = set(la)
    drop_b = set(rb)
    return tuple(d for i, d in enumerate(da) if i not in drop_a) + tuple(
        d for i, d in enumerate(db) if i not in drop_b
    )


def _dense_merge_cost(dims_a, dims_b):
    c = 1
    for d in dims_a:
        c *= d
    for d in dims_b:
        c *= d
    return c


class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.left = None
        self.right = None

        self.rank = 0
        while random.random() < 0.5:
            self.rank += 1

    def priority_less(self, other):
        if self.rank != other.rank:
            return self.rank < other.rank
        return self.key > other.key


class ZipTreeTensor:
    def __init__(self, dims):
        self.root = None
        self.dims = dims

    def _unzip(self, node, key):
        if not node:
            return None, None
        if node.key < key:
            node.right, right = self._unzip(node.right, key)
            return node, right
        else:
            left, node.left = self._unzip(node.left, key)
            return left, node

    def insert_raw(self, key, value):
        new_node = Node(key, value)
        parent, curr = None, self.root

        while curr and new_node.priority_less(curr):
            parent = curr
            curr = curr.left if key < curr.key else curr.right

        l, r = self._unzip(curr, key)
        new_node.left, new_node.right = l, r

        if not parent:
            self.root = new_node
        elif key < parent.key:
            parent.left = new_node
        else:
            parent.right = new_node

    def get_all_elements(self):
        res = []

        def _walk(n):
            if not n:
                return
            _walk(n.left)
            res.append((n.key, n.value))
            _walk(n.right)

        _walk(self.root)
        return res

    def transpose_for_axis(self, axis_to_lead):
        new_dims = (self.dims[axis_to_lead],) + tuple(
            d for i, d in enumerate(self.dims) if i != axis_to_lead
        )
        new_tensor = ZipTreeTensor(new_dims)

        for idx, val in self.get_all_elements():
            new_idx = (idx[axis_to_lead],) + tuple(
                x for i, x in enumerate(idx) if i != axis_to_lead
            )
            new_tensor.insert_raw(new_idx, val)
        return new_tensor

    def transpose_axes_to_order(self, old_axes_first):
        old_axes_first = tuple(old_axes_first)
        if sorted(old_axes_first) != list(range(len(self.dims))):
            raise ValueError("transpose_axes_to_order: ожидается перестановка всех осей")
        new_dims = tuple(self.dims[a] for a in old_axes_first)
        new_tensor = ZipTreeTensor(new_dims)
        for idx, val in self.get_all_elements():
            new_idx = tuple(idx[a] for a in old_axes_first)
            new_tensor.insert_raw(new_idx, val)
        return new_tensor

    def find_range(self, lead_value):
        matches = []

        def _search(n):
            if not n:
                return
            if n.key[0] == lead_value:
                matches.append((n.key, n.value))
                _search(n.left)
                _search(n.right)
            elif n.key[0] < lead_value:
                _search(n.right)
            else:
                _search(n.left)

        _search(self.root)
        return matches

    def find_range_prefix(self, prefix):
        r = len(prefix)
        matches = []

        def _search(n):
            if not n:
                return
            if n.key[:r] == prefix:
                matches.append((n.key, n.value))
                _search(n.left)
                _search(n.right)
            elif n.key[:r] < prefix:
                _search(n.right)
            else:
                _search(n.left)

        _search(self.root)
        return matches

    def multiply_convoluted(self, other, left_axes, right_axes):
        la = tuple(left_axes)
        rb = tuple(right_axes)
        if len(la) != len(rb):
            raise ValueError("multiply_convoluted: |λ| должно совпадать с |μ|")
        res_dims = _contract_dims_pair(self.dims, other.dims, la, rb)
        result = ZipTreeTensor(res_dims)
        la_set, rb_set = set(la), set(rb)
        for idx_a, val_a in self.get_all_elements():
            key = tuple(idx_a[t] for t in la)
            for idx_b, val_b in other.get_all_elements():
                if tuple(idx_b[t] for t in rb) != key:
                    continue
                new_idx = tuple(idx_a[i] for i in range(len(idx_a)) if i not in la_set) + tuple(
                    idx_b[j] for j in range(len(idx_b)) if j not in rb_set
                )
                result.add_at(new_idx, val_a * val_b)
        return result

    def multiply_with_reindex(self, tensor_b, left_axes, right_axes):
        if isinstance(left_axes, int) and isinstance(right_axes, int):
            la, rb = (left_axes,), (right_axes,)
        else:
            la = tuple(left_axes)
            rb = tuple(right_axes)
        ndb = len(tensor_b.dims)
        rest_b = [j for j in range(ndb) if j not in set(rb)]
        b_order = list(rb) + rest_b
        b_reindexed = tensor_b.transpose_axes_to_order(b_order)

        res_dims = _contract_dims_pair(self.dims, tensor_b.dims, la, rb)
        result = ZipTreeTensor(res_dims)
        la_set = set(la)
        r = len(rb)
        for idx_a, val_a in self.get_all_elements():
            prefix = tuple(idx_a[t] for t in la)
            for idx_b, val_b in b_reindexed.find_range_prefix(prefix):
                new_idx = tuple(x for i, x in enumerate(idx_a) if i not in la_set) + tuple(
                    idx_b[q] for q in range(r, len(idx_b))
                )
                result.add_at(new_idx, val_a * val_b)
        return result

    def multiply(self, other, axes):
        la, rb = _normalize_conv_axes(axes)
        return self.multiply_convoluted(other, la, rb)

    def find_by_index_value(self, axis, value):
        matches = []

        def _walk(node):
            if not node:
                return
            if node.key[axis] == value:
                matches.append((node.key, node.value))
            _walk(node.left)
            _walk(node.right)

        _walk(self.root)
        return matches

    def add_at(self, key, value):
        curr = self.root
        while curr:
            if curr.key == key:
                curr.value += value
                return
            curr = curr.left if key < curr.key else curr.right

        new_node = Node(key, value)
        parent, curr = None, self.root

        while curr and new_node.priority_less(curr):
            parent = curr
            curr = curr.left if key < curr.key else curr.right

        l, r = self._unzip(curr, key)
        new_node.left, new_node.right = l, r

        if not parent:
            self.root = new_node
        elif key < parent.key:
            parent.left = new_node
        else:
            parent.right = new_node
