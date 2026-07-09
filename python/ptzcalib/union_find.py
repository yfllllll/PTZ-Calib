"""Union-Find / Disjoint-Set data structure.

Python port of src/core/union_find.h.
"""

from __future__ import annotations

from typing import List


class UnionFind:
    """Union-Find with path compression and union by rank."""

    __slots__ = ("m_cc_parent", "m_cc_rank", "m_cc_size")

    def __init__(self):
        self.m_cc_parent: List[int] = []
        self.m_cc_rank: List[int] = []
        self.m_cc_size: List[int] = []

    def init_sets(self, num_cc: int) -> None:
        self.m_cc_size = [1] * num_cc
        self.m_cc_parent = list(range(num_cc))
        self.m_cc_rank = [0] * num_cc

    def get_num_nodes(self) -> int:
        return len(self.m_cc_size)

    def find(self, i: int) -> int:
        if i < 0 or i >= len(self.m_cc_parent):
            raise IndexError("Index out of range")
        # Iterative path compression to avoid Python recursion limits.
        root = i
        while self.m_cc_parent[root] != root:
            root = self.m_cc_parent[root]
        # Compress
        while self.m_cc_parent[i] != root:
            nxt = self.m_cc_parent[i]
            self.m_cc_parent[i] = root
            i = nxt
        return root

    def union(self, i: int, j: int) -> None:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i == root_j:
            return

        if self.m_cc_rank[root_i] < self.m_cc_rank[root_j]:
            self.m_cc_parent[root_i] = root_j
            self.m_cc_size[root_j] += self.m_cc_size[root_i]
        else:
            self.m_cc_parent[root_j] = root_i
            self.m_cc_size[root_i] += self.m_cc_size[root_j]
            if self.m_cc_rank[root_i] == self.m_cc_rank[root_j]:
                self.m_cc_rank[root_i] += 1

    def connected(self, i: int, j: int) -> bool:
        return self.find(i) == self.find(j)

    def size(self, i: int) -> int:
        return self.m_cc_size[self.find(i)]
