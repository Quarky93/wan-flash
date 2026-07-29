"""Wan2.1 attention shape registry + tile arithmetic.

The whole point of this repo: these are the ONLY shapes that matter, so trip
counts, tail sizes, and scheduler decisions can be specialized per shape
instead of carrying general-purpose machinery.
"""

from dataclasses import dataclass

HEAD_DIM = 128
S_480P = 32760   # 21 x 30 x 52
S_720P = 75600   # 21 x 45 x 80
TEXT_LEN = 512
CLIP_LEN = 257
SM_COUNT = 132   # H100 SXM


@dataclass(frozen=True)
class WanShape:
    kind: str      # "self" | "cross"
    b: int
    h: int
    sq: int
    skv: int
    d: int = HEAD_DIM

    @property
    def name(self):
        return f"{self.kind}_h{self.h}_sq{self.sq}_skv{self.skv}"

    def blocks(self, m_block: int, n_block: int):
        """(m_blocks, m_tail, n_blocks, n_tail) — tail sizes are compile-time
        constants per shape; only the LAST n-block needs key predication."""
        mb, mt = -(-self.sq // m_block), self.sq % m_block
        nb, nt = -(-self.skv // n_block), self.skv % n_block
        return mb, mt, nb, nt

    def fwd_ctas(self, m_block: int):
        return self.b * self.h * self.blocks(m_block, m_block)[0]


SELF_SHAPES = [WanShape("self", 1, h, s, s) for h in (12, 40) for s in (S_480P, S_720P)]
CROSS_SHAPES = [WanShape("cross", 1, h, s, TEXT_LEN) for h in (12, 40) for s in (S_480P, S_720P)]
ALL_SHAPES = SELF_SHAPES + CROSS_SHAPES
PRIMARY = [s for s in ALL_SHAPES if s.h == 12]  # 1.3B first
