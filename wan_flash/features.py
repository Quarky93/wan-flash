"""Feature-flag registry: every FA4-vs-FA3 technique choice is a measured
decision. Defaults hold the current per-feature verdicts (docs/FEATURES.md has
the numbers); bench --matrix sweeps them.

Flags (fwd):
  rescale_skip_threshold  FA4 sm100 trick ported to sm90: skip the O-rescale
                          multiply when the running max grew by less than
                          thr/log2(e). 0.0 = FA3 behavior (always rescale).
                          Mathematically exact either way (max-shift invariance);
                          only fp rounding differs.
  intra_wg_overlap        FA4 sm90: overlap softmax of block i with MMA of
                          block i+1 inside a warpgroup (two S buffers).
  mma_pv_is_rs            P kept in registers for the P@V matmul (rs) vs staged
                          through smem (ss).
  num_stages              KV smem pipeline depth.
  tile_m / tile_n         CTA tile sizes (per-shape override allowed).
  scheduler               "single" (FA4 sm90 default) | "lpt" | "persistent"
                          (FA3-style persistent worker grid).
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class FwdFeatures:
    # defaults = measured winners at Wan shapes (docs/FEATURES.md for numbers)
    rescale_skip_threshold: float = 8.0  # +2% self; exact math (max-shift inv.)
    intra_wg_overlap: bool = True        # +14% self (562 -> 641 TFLOP/s)
    mma_pv_is_rs: bool = True
    num_stages: int = 2                  # 3 was ~flat, no smem headroom left
    tile_m: int = 128
    tile_n: int = 128                    # 120/144/160/176/192 all measured slower
    scheduler: str = "persistent"        # +1.5% self@32760, +15% cross


_current = FwdFeatures()


def get() -> FwdFeatures:
    return _current


def set_overrides(**kw):
    global _current
    _current = replace(_current, **kw)
    return _current


def reset():
    global _current
    _current = FwdFeatures()
