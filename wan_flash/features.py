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
  cluster_mn              (2, 1): 2-CTA cluster, K/V TMA multicast across the
                          two m-blocks of a head pair (halves K/V L2/HBM
                          traffic; the C++ FA3 hopper trick FA4 sm90 leaves
                          unused). (1, 1) = no cluster.
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
    # "auto": persistent everywhere except long-chain large-KV shapes (units
    # per cluster slot >= 128 and n_blocks >= 16 -> "single"; only self
    # h40@75600 among Wan shapes), where the per-kernel cluster-pair coupling
    # of the persistent walk measures ~1.5% slower than retiring pairs per tile
    scheduler: str = "auto"
    cluster_mn: tuple = (2, 1)           # K/V TMA multicast pairs: +2-3% self


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
    cluster_n   2 = 2-CTA cluster over adjacent n-blocks of a head with Q/dO
                TMA multicast (halves the dominant L2 read stream of the
                dK/dV-stationary loop). Self-attn shapes only (forced to 1
                when the auto/explicit nsplit > 1).
    dq_cluster_reduce
                Cluster-pair dQ reduction over DSM (roadmap Phase 2): rank 1
                bulk-reduce-adds its sdQaccum chunk into rank 0's smem
                (cp.reduce.async.bulk.shared::cluster) and only rank 0
                flushes to gmem — halves the dQaccum L2-RMW stream. Only
                active when cluster_n == 2 and nsplit == 1.
    """

    tile_m: int = 80
    tile_n: int = 128
    num_stages: int = 2
    nsplit: int = 0
    cluster_n: int = 2   # Q/dO TMA multicast pairs: beats FA4 at all self shapes
    dq_cluster_reduce: bool = False  # Phase 2 candidate, under A/B


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
