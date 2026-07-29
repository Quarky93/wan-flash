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


@dataclass(frozen=True)
class BwdFeatures:
    """Backward flags (docs/FEATURES.md for measured verdicts).

    tile_m      Q-tile of the dK/dV-stationary main kernel. 80 = FA4/FA3
                non-causal hd128 config (dQ_swapAB derived as tile_m%64!=0);
                64 = the causal-config alternative (no dQ_swapAB).
    tile_n      KV tile; fixed 128 (= 64 * 2 MMA warpgroups under SdP_swapAB).
    num_stages  Q/dO/dS smem pipeline depth (2 = FA4 default).
    nsplit      Split-M for tiny-KV (cross-attn) shapes: each CTA handles a
                chunk of m_blocks; dK/dV go through fp32 gmem accumulation +
                a convert kernel (docs/SPECIALIZATION.md B5). 0 = auto (1 for
                self shapes; picked for >=95% wave efficiency when the base
                grid underfills the GPU). NOTE: nsplit > 1 makes dk/dv
                nondeterministic (atomic add order), like dq already is.
    """

    tile_m: int = 80
    tile_n: int = 128
    num_stages: int = 2
    nsplit: int = 0


_current = FwdFeatures()
_current_bwd = BwdFeatures()


def get() -> FwdFeatures:
    return _current


def get_bwd() -> BwdFeatures:
    return _current_bwd


def set_overrides(**kw):
    global _current
    _current = replace(_current, **kw)
    return _current


def set_bwd_overrides(**kw):
    global _current_bwd
    _current_bwd = replace(_current_bwd, **kw)
    return _current_bwd


def reset():
    global _current, _current_bwd
    _current = FwdFeatures()
    _current_bwd = BwdFeatures()
