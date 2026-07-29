# SPECIALIZATION

> Source study of FA4 CuTeDSL (installed flash-attn-4 4.0.0b23) against
> nvidia-cutlass-dsl 4.6.0, 2026-07-29. File:line refs are into the installed tree.

## Summary

FA4's sm90 hd128 tiles are fwd 128×128 (2 consumer WGs, RS-PV, intra-WG overlap, 2 KV stages, 160 KB smem, non-persistent grid `(m_blocks, h, b)`) and bwd 80×128 (2 WGs, SdP_swapAB+dQ_swapAB, dKV in registers, 224 KB smem, grid `(n_blocks, h, b)` with an inner loop over ALL m_blocks). For our shapes the self-attention cases are healthy (23–179 waves, ≥97% wave efficiency), but **cross-attention backward is catastrophic: S_kv=512 gives only 4 n-blocks, so h=12 launches 48 CTAs on 132 SMs (36% of one wave) and h=40 launches 160 (1.21 waves, 61% efficient)** — a 2.75× / 1.65× occupancy loss that a split-M schedule fixes for free because dQ already accumulates through `cp.reduce.async.bulk...add.f32` gmem atomics. Second-order specialization wins: FA4 marks every tensor's seqlen dim dynamic (`to_cute_tensor` → `mark_layout_dynamic`), so all trip counts, mask column limits and `ceil_div`s are runtime values; keeping S static makes them constexpr and turns the ragged tail into one compile-time-known peeled block. `tile_n=120` divides BOTH 32760 (273) and 75600 (630) exactly, eliminating the tail mask entirely. I give a 16-toggle A/B matrix with the FA3-side alternative for each, and the exact numerical contract (natural-log fp32 LSE (b,h,s), the internally-required log2 conversion for bwd, the ≤2×-FA3 rel_l2 gate at rel_l2 2.37e-3, and the SAC nondeterminism-band constraint that split-M dK/dV atomics would perturb).

## Details

# Deliverable 4 — Specialization + test/feature-matrix design

All paths relative to `/workspace/wan-attn/.venv/lib/python3.12/site-packages/`.

---

## 0. The FA4 sm90 hd128 baseline, exactly as installed

### 0.1 Forward tile config

`flash_attn/cute/interface.py:150-151`
```python
    elif head_dim <= 128:
        return FwdConfig(128, 128, True, True)
```
`FwdConfig` fields (`interface.py:117-122`): `m_block_size, n_block_size, mma_pv_is_rs, intra_wg_overlap` → **tile_m=128, tile_n=128, mma_pv_is_rs=True, intra_wg_overlap=True**. Note this branch is causal-agnostic: FA4 uses the same tile for causal and non-causal at hd128, i.e. it never tuned a non-causal-only hd128 tile.

Derived at `flash_fwd_sm90.py:206-224`:
```python
        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_qk.size          # 256
        self.num_wg_mma = self.num_mma_threads // 128     # 2
        self.num_threads = 128 * (self.num_wg_mma + 1)    # 384
        self.num_producer_threads = 32
        self.num_mma_regs, self.num_producer_regs = {1: (256, 56), 2: (240, 24), 3: (160, 32)}[self.num_wg_mma]
        ...
        self.use_scheduler_barrier = (
            (self.num_wg_mma >= 2 and self.tile_hdim <= 128) if const_expr(self.intra_wg_overlap) else (self.num_wg_mma == 2)
        )
```
`tiled_mma_qk` uses `atom_layout_mnk=(self.tile_m // 64, 1, 1)` (`flash_fwd_sm90.py:103`), so **tile_m is hard-locked to 64 × num_wg**. `rescale_O_before_gemm` is **off** for us: `flash_fwd_sm90.py:232` `self.rescale_O_before_gemm = self.tile_hdimv > 128 and self.intra_wg_overlap`.

Stages: hard-coded 2 at `interface.py:870-871` (`# num_stages=1` / `num_stages=2`). Threads default 384 (`interface.py:321`).

SMEM (verified against FA4's own model, `sm90_config_search.py:279-287`): `max(sQ,sO)=32768` + `sK=2·32768` + `sV=2·32768` + `sP=0` = **163 840 B = 160 KB** → 1 CTA/SM. Registers 64(S)+32(P)+64(O)=**160/216**. `sO` aliases `sQ` (`flash_fwd_sm90.py:535`: `sO = storage.sQ.get_tensor(...)`).

Scheduler — **non-persistent**, one CTA per (m_block, head, batch):
```python
# flash_fwd_sm90.py:316-322
        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = (SingleTileScheduler if const_expr(not self.is_causal or self.is_local) else SingleTileLPTScheduler)
# flash_fwd_sm90.py:343
            is_persistent=False,
```
`tile_scheduler.py:228-245`:
```python
    @staticmethod
    def get_grid_shape(params: Params, *, loc=None, ip=None) -> Tuple[Int32, Int32, Int32]:
        assert params.cluster_shape_mn[1] == 1, "Only cluster_shape_mn[1] == 1 is supported"
        ...
        return (grid_x, params.num_head * params.num_splits, params.num_batch)
```
Launch: `flash_fwd_sm90.py:394-399` `grid=grid_dim, block=[self.num_threads,1,1], stream=stream, min_blocks_per_mp=1` — no cluster, no PDL.

**No TMA multicast anywhere on sm90**: `flash_fwd_sm90.py:69` `self.cluster_shape_mn = (1, 1)`; `flash_fwd_sm90.py:293` and `:300` pass literal `1,  # No mcast for now` as the `num_multicast` arg of
```python
# cutlass/cute/nvgpu/cpasync/helpers.py:419-428
def make_tiled_tma_atom(op: TMAOp, gmem_tensor: Tensor, smem_layout_: Union[Layout, ComposedLayout],
                        cta_tiler: Tiler, num_multicast: int = 1, *, internal_type=None, loc=None, ip=None) -> TmaInfo:
```

### 0.2 Backward tile config

`interface.py:199-212`
```python
    elif head_dim <= 128:
        # C++ FA3: causal/local: 64, 128; non-causal: 80, 128 with dQ_swapAB
        is_causal_or_local = causal or local
        m_block_size = 64 if is_causal_or_local else 80
        ...
        return BwdConfig(
            m_block_size=m_block_size, n_block_size=128,
            num_stages_Q=2, num_stages_dO=2, num_stages_PdS=2,
            SdP_swapAB=True, dKV_swapAB=False,
            dQ_swapAB=m_block_size % 64 != 0,
            AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
        )
```
→ **tile_m=80, tile_n=128, Q/dO/PdS stages 2/2/2, SdP_swapAB=T, dKV_swapAB=F, dQ_swapAB=T, atom layouts (1,2,1), num_wg=2 → 384 threads.** `mma_dkv_is_rs = True` (`flash_bwd_sm90.py:107-112`), i.e. dK/dV GEMMs take P/dS straight from registers, no sP in smem.

I ran FA4's own feasibility model (`sm90_config_search.py`, read-only) for hd128; the chosen config is **rank 1 of 142** and the reg budget is the binding constraint:

| wg | tm | tn | SdP | dKV | dQ | (aSdP,adKV,adQ) | regs | smem | traffic/blk |
|---|---|---|---|---|---|---|---|---|---|
| 2 | **80** | **128** | T | F | T | (1,2,1) | **208/216** | 204K | **39.6** |
| 2 | 64 | 128 | T | F | F | (1,2,1) | 192/216 | 176K | 42.0 |
| 2 | 64 | 112 | F | T | F | (1,1,1) | 168/216 | 180K | 50.9 |
| 2 | 128 | 64 | F | F | F | (2,1,2) | 128/216 | 224K | 58.0 |

tile_m ∈ {96, 112} at tile_n=128 is **register-infeasible** (`max(2·regs_SdP, regs_dQ)+regs_dK+regs_dV` = 224 and 240 vs the 216 limit at `sm90_config_search.py:16` `REG_LIMITS = {2: 216, 3: 128}`). So 80 is genuinely the largest legal m tile — a real constraint we inherit.

Scheduler — **also non-persistent, and it parallelises over KV blocks**:
```python
# flash_bwd_sm90.py:518-546
        if const_expr(mCuSeqlensK is not None or mSeqUsedK is not None):
            TileScheduler = SingleTileVarlenScheduler
        elif const_expr(self.deterministic):
            TileScheduler = SingleTileLPTBwdScheduler
        else:
            TileScheduler = SingleTileScheduler
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mK.shape[0]), self.tile_n),   # num_block  <-- KV blocks
            cute.size(mQ.shape[2]),                                # num_head
            cute.size(mK.shape[3]) ...,                            # num_batch
            ...
            tile_shape_mn=(self.tile_n, self.tile_m),  # Swapping the role of Q & K
            is_persistent=False,
```
so **grid = (ceil(S_kv/128), h, b)** and each CTA loops over every m block:
```python
# flash_bwd_sm90.py:1329,1363  (mma warpgroups)
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
                    for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
# flash_bwd_sm90.py:936,980   (producer warp)
# flash_bwd_sm90.py:1820,1840 (dQaccum-store warp)
```
with (`block_info.py:58-71`) `m_block_max = cute.ceil_div(seqlen_info.seqlen_q, self.tile_m)`, `m_block_min = 0` when not causal/local.

dQ is reduced across n_block-CTAs by **fp32 gmem bulk atomics**:
```python
# flash_bwd_sm90.py:1879-1885
                            with cute.arch.elect_one():
                                copy_utils.cpasync_reduce_bulk_add_f32(
                                    sdQaccum[None, warp_group_idx].iterator,
                                    gdQaccum[(None, warp_group_idx), m_block_safe].iterator,
                                    self.tma_copy_bytes["dQ"],
                                )
# copy_utils.py:267-288
def cpasync_reduce_bulk_add_f32(smem_ptr: cute.Pointer, gmem_ptr: cute.Pointer, store_bytes: int | Int32, *, loc=None, ip=None):
    ...  "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [$0], [$1], $2;"
```
dK/dV, by contrast, are **exclusively owned** by their n_block CTA and TMA-stored directly (`flash_bwd_sm90.py:1654-1700`, `store_dK()`/`store_dV()` via `CopyBulkTensorTileS2GOp`). Only the GQA path (`qhead_per_kvhead > 1`) uses fp32 accumulators + a postprocess (`interface.py:1621-1655`, `dKV_postprocess`).

Backward launch uses PDL to overlap with the preprocess kernel: `flash_bwd_sm90.py:621-627` `..., min_blocks_per_mp=1, use_pdl=True)` and `flash_bwd_sm90.py:964` `cute.arch.griddepcontrol_wait()`.

---

## 1. Tile arithmetic for our exact shapes

### 1.1 Forward — grid (m_blocks, h, 1), tile 128×128, 1 CTA/SM, 132 SMs

| shape | m_blk | last m tile | n_blk | last n tile | CTAs | waves | tail CTAs | wave eff |
|---|---|---|---|---|---|---|---|---|
| self h12 S=32 760 | 256 | 120/128 | 256 | 120/128 | 3 072 | 23.27 | 36 | **0.970** |
| self h40 S=32 760 | 256 | 120/128 | 256 | 120/128 | 10 240 | 77.58 | 76 | 0.995 |
| self h12 S=75 600 | 591 | 80/128 | 591 | 80/128 | 7 092 | 53.73 | 96 | 0.995 |
| self h40 S=75 600 | 591 | 80/128 | 591 | 80/128 | 23 640 | 179.09 | 12 | 0.995 |
| cross h12 S=32 760 | 256 | 120/128 | **4** | 128/128 | 3 072 | 23.27 | 36 | 0.970 |
| cross h40 S=32 760 | 256 | 120/128 | **4** | 128/128 | 10 240 | 77.58 | 76 | 0.995 |
| cross h12 S=75 600 | 591 | 80/128 | **4** | 128/128 | 7 092 | 53.73 | 96 | 0.995 |
| cross h40 S=75 600 | 591 | 80/128 | **4** | 128/128 | 23 640 | 179.09 | 12 | 0.995 |

Forward occupancy is **not a problem at any shape**, including h=12. Worst case (h12/480p) loses 3.0% to wave quantization. Ragged-tail *compute* waste is trivial (8 of 128 rows in 1 of 256 m-tiles = 0.024%; 48/128 rows in 1 of 591 = 0.063%).

The forward problem for cross-attn is **loop depth, not occupancy**: 4 n-iterations with a 2-stage K/V pipeline means half the mainloop is pipeline warm-up, and the per-CTA fixed cost (TMA Q 32 KB in, TMA O 32 KB out, mbarrier init, `setmaxregister` reconfig, epilogue `finalize`+LSE) is amortised over 4 iterations instead of 256.

### 1.2 Backward — grid (n_blocks, h, 1), tile_m=80/tile_n=128

| shape | n_blk (=CTAs/head) | last n tile | m_blk (inner loop) | last m tile | CTAs | waves | wave eff |
|---|---|---|---|---|---|---|---|
| self h12 S=32 760 | 256 | 120/128 | 410 | 40/80 | 3 072 | 23.27 | 0.970 |
| self h40 S=32 760 | 256 | 120/128 | 410 | 40/80 | 10 240 | 77.58 | 0.995 |
| self h12 S=75 600 | 591 | 80/128 | **945** | **80/80 EXACT** | 7 092 | 53.73 | 0.995 |
| self h40 S=75 600 | 591 | 80/128 | 945 | 80/80 EXACT | 23 640 | 179.09 | 0.995 |
| **cross h12 S=32 760** | **4** | 128 EXACT | 410 | 40/80 | **48** | **0.36** | **0.364** |
| **cross h12 S=75 600** | **4** | 128 EXACT | 945 | 80 EXACT | **48** | **0.36** | **0.364** |
| **cross h40 S=32 760** | **4** | 128 EXACT | 410 | 40/80 | **160** | **1.21** | **0.606** |
| **cross h40 S=75 600** | **4** | 128 EXACT | 945 | 80 EXACT | **160** | **1.21** | **0.606** |

Nice accident: **75 600 = 945 × 80 exactly**, so 720p backward has zero m-tail.

### 1.3 Where specialization can win

**(a) Compile-time-fixed trip counts.** FA4 makes every seqlen dynamic:
```python
# cute_dsl_utils.py:62-84
def to_cute_tensor(t, assumed_align=16, leading_dim=-1, fully_dynamic=False, enable_tvm_ffi=True):
    ...
    if leading_dim == -1: leading_dim = t.ndim - 1
    return tensor.mark_layout_dynamic(leading_dim=leading_dim)
```
Consequences inside the kernel: `n_block_max = cute.ceil_div(seqlen_info.seqlen_k, self.tile_n)` (`block_info.py:31`) is a runtime `Int32`; every mainloop is `cutlass.range(n_block_max - n_block_min, unroll=1)` with a runtime bound; the tail mask recomputes `seqlenk_col_limit = self.seqlen_k - n_block*self.tile_n - thr_col_offset` per block (`mask.py:210`) and derives an r2p bitmask at runtime (`mask.py:221-222`). With `b=1` and S fixed we can pass fully-static layouts (plain `from_dlpack`, or `mark_compact_shape_dynamic` only on b/h) and get:
  * `n_block_max` as a Python `int` → `cutlass.range_constexpr` / statically-unrollable prologue+steady-state+epilogue; no loop-bound register, no trip-count computation per tile.
  * The ragged tail at a **known block index with a known column limit** → peel it as a separate constexpr code path with a *constant* bitmask, and drop `mask_fn` from all 255/590 full blocks. Today FA4 applies `mask_seqlen=True` on the first (rightmost) block unconditionally with the runtime-derived limit (`flash_fwd_sm90.py:1299`, comment at `:1297-1298` admits it is "redundant" but "applied anyway").
  * `check_inf` becomes provably unnecessary: dense non-causal ⇒ every row has ≥1 valid key ⇒ `row_max_cur == -inf` is impossible. Today it costs a compare+select per row per block (`softmax.py:164-165`, enabled at `flash_fwd_sm90.py:1024` `check_inf=True`).
  * Cost: one compiled kernel per (S, h, kind). 8 fwd + 8 bwd variants across our whole matrix — acceptable for an AOT-cached specialised library, but it means the correctness battery needs a parallel dynamic-S build (see §3.5).

**(b) No varlen / seqlen-info machinery.** `SeqlenInfoQK.create` (`seqlen_info.py:83-120`) does gmem loads of `cu_seqlens`/`seqused` and computes `padded_offset_q/k`; `SeqlenInfoCls` is threaded through load, mma and dQaccum-store paths and re-evaluated per work tile (`flash_fwd_sm90.py:1051`, `flash_bwd_sm90.py:1798`). With b=1 all of that collapses to constants: no `offset_batch_Q/K`, no `mCuSeqlensQ/K` / `mSeqUsedQ/K` kernel params (8 optional tensor arguments dropped from the fwd signature at `flash_fwd_sm90.py:409-413`), no `SingleTileVarlenScheduler` branch, no ragged-TMA tensor construction (`copy_utils.create_ragged_tensor_for_tma`, `flash_fwd_sm90.py:306`, `flash_bwd_sm90.py:494-501`). Also drops the `is_varlen_q` guard in the bwd `process_tile` predicate (`flash_bwd_sm90.py:940-943`, `:1331-1335`, `:1821-1826`) — with b=1 dense, `process_tile` is unconditionally true.

**(c) No causal branch.** Deletes: the `is_causal or is_local` blocks in `get_n_block_min_max`/`get_m_block_min_max` (`block_info.py:32-46`, `:61-70`), `get_n_block_min_causal_local_mask` / `get_n_block_before_local_mask` and their two extra mainloop segments in the fwd (`flash_fwd_sm90.py:1138-1180`), the LPT scheduler selection (`flash_fwd_sm90.py:320-321`, `tile_scheduler.py:393`), the deterministic-SPT lock-value computation (`flash_bwd_sm90.py:1859-1865`), and the whole causal/local half of `mask.apply_mask` (`mask.py:224+`). Practically: the fwd mainloop becomes **first-half-block → N−1 identical unmasked iterations → last-half-block**, with N a constant.

**(d) Scheduler simplification at h=12.** The forward at h=12 already has 3 072–7 092 CTAs, so a persistent scheduler buys *nothing on load balance* (3072 = 132·23 + 36, so 36 CTAs run 24 tiles and 96 run 23 — the same 24-wave makespan a non-persistent grid gets). What persistent buys is **prologue amortisation across ~23 tiles**: TMA-descriptor prefetch, mbarrier init, `setmaxregister_increase/decrease` (`flash_fwd_sm90.py:581`, `:604`), the `mma_init()` scheduler-barrier priming (`flash_fwd_sm90.py:1480-1487`), and — the big one — the ability to start loading the *next* tile's Q while the current tile's epilogue drains. Given O and Q share smem (`sO = storage.sQ...`, `flash_fwd_sm90.py:535`) a persistent variant needs either a separate sO buffer (+32 KB → 192 KB, still 1 CTA/SM) or an epilogue/Q-load barrier. `StaticPersistentTileScheduler` already exists (`tile_scheduler.py:287-390`) and is wired through `TileSchedulerArguments.is_persistent` (`tile_scheduler.py:162`); FA4 simply never turns it on for sm90. **This is the single highest-value forward A/B.** Note FA4's own comment that CLC/work-stealing regresses dense non-causal (`interface.py:627-632`: *"dense noncausal mostly just pays work-stealing overhead"*) — so test **static** persistence, not dynamic.

**(e) Cross-attention KV=512 — only 4 n-blocks.**

*Forward:* FA4 handles it correctly but inefficiently. `num_splits_heuristic` explicitly refuses to split (`interface.py:262-265`: *"If num_n_blocks is too small, use 1 split. For example, we never split for hdim = 128 and seqlen_k = 512"*), and split-KV is asserted off on sm90 anyway (`interface.py:859`). So each CTA runs a 4-iteration mainloop with 2-stage K/V. Specialisations worth measuring: (i) tile_n=256 → 2 iterations but 2 stages = full 512 KV resident (128 KB) + sQ/sO 32 KB = 160 KB, one prologue and zero steady state; (ii) persistent-over-m within a head so the 4 K/V blocks are loaded **once per CTA** instead of once per m-tile, and the LSE/epilogue cost is amortised; (iii) tile_m=192 (3 WG, noRS+noOL — the only register-feasible 192 config for hd128, 224 KB) to cut CTAs/head from 256→171 and deepen per-CTA reuse.

*Backward — this is the pathology.* `grid = (4, h, 1)`:
  * **h=12 → 48 CTAs on 132 SMs. 84 SMs idle for the entire kernel.** Wave efficiency 0.364, i.e. a **2.75× occupancy loss**.
  * **h=40 → 160 CTAs → 1.21 waves.** Second wave is 28/132 = 21% occupied; efficiency 0.606, a **1.65× loss**.
  * Each CTA then serially grinds 410 (480p) or 945 (720p) m-blocks. Order-of-magnitude: cross-bwd FLOPs at h12/S=32 760 are 2.5 · 4 · 12 · 32 760 · 512 · 128 = 258 GFLOP; at 48/132 SMs that's ≈1.18 ms vs ≈0.43 ms if fully occupied.
  * There is also **4× redundant Q/dO traffic**: the 4 n_block-CTAs of a head each stream the whole Q and dO for that head (per head 2·S·128·2 B = 16.8 MB at 480p ⇒ 805 MB total at h=12, plus an equal volume of dQ atomic traffic).

  **What a specialized small-KV schedule can do:**
  1. **Split-M.** Add a third grid axis `nsplit`; CTA `(n_block, head, split)` handles m ∈ [split·⌈M/nsplit⌉, …). dQ needs **no change at all** — it already goes through `cp.reduce.async.bulk...add.f32` (`flash_bwd_sm90.py:1879-1885`), which is commutative across arbitrarily many CTAs. Only dK/dV need reduction, and FA4 already ships that machinery for GQA: fp32 `dk_accum`/`dv_accum` + `_bwd_postprocess_convert` (`interface.py:1621-1655`, `:2037-2053`). Extra traffic is `nsplit · 4 · h · 2 · 128·128·4 B` = 69 MB at nsplit=11/h=12 → ~23 µs, against ~0.75 ms recovered.

| h | S | nsplit | CTAs | waves | wave eff | m_blocks/CTA |
|---|---|---|---|---|---|---|
| 12 | any | 1 (FA4) | 48 | 0.36 | 0.364 | 410 / 945 |
| 12 | any | 5 | 240 | 1.82 | 0.909 | 82 / 189 |
| 12 | any | **11** | **528** | **4.00** | **1.000** | 37.3 / 85.9 |
| 12 | any | 22 | 1 056 | 8.00 | 1.000 | 18.6 / 43.0 |
| 40 | any | 1 (FA4) | 160 | 1.21 | 0.606 | 410 / 945 |
| 40 | any | 3 | 480 | 3.64 | 0.909 | 136.7 / 315 |
| 40 | any | 11 | 1 760 | 13.33 | 0.952 | 37.3 / 85.9 |
| 40 | any | **33** | **5 280** | **40.00** | **1.000** | 12.4 / 28.6 |

  (4·12·11 = 528 = 4·132 and 4·40·33 = 5 280 = 40·132 are exact — pick nsplit as the smallest integer giving ≥4 waves and CTAs ≡ 0 mod 132.)
  2. **Cluster-4 over the n axis with TMA multicast of Q / dO / LSE / dPsum.** Since a head has exactly 4 n_blocks, `cluster_shape_mn = (4, 1)` covers a whole head; multicast collapses the 4× redundant Q/dO stream to 1× (805 MB → 201 MB at h12/480p). The DSL supports it (`make_tiled_tma_atom(..., num_multicast=4)`), `TileSchedulerArguments.cluster_shape_mn` and `SingleTileScheduler.get_grid_shape`'s `use_cluster_idx` path already exist (`tile_scheduler.py:157`, `:236-240`), and FA4's sm90 path just never uses them (`cluster_size = 1` at `interface.py:1416`). Composes cleanly with split-M: grid = 4·h·nsplit CTAs = h·nsplit clusters.
  3. **Do NOT shrink tile_n to 64 as the primary fix.** It doubles n_blocks to 8 (96 / 320 CTAs) but halves per-CTA dK/dV work and pushes the bwd off the rank-1 register config; use it only as a fallback A/B point.

**(f) Bonus, self-attention only: `tile_n = 120` divides BOTH sequences exactly.**

| tile_n | 32 760 | 75 600 | 512 |
|---|---|---|---|
| 112 | 293 blk, tail 56 | 675 blk **EXACT** | 5 blk, tail 64 |
| **120** | **273 blk EXACT** | **630 blk EXACT** | 5 blk, tail 32 |
| 128 (FA4) | 256 blk, tail 120 | 591 blk, tail 80 | 4 blk EXACT |
| 144 | 228 blk, tail 72 | 525 blk **EXACT** | 4 blk, tail 80 |

`tile_n=120` is a legal GMMA N (multiple of 8), feasible at 128×120 RS+OL: smem = 32 768 + 2·(120·128·2)·2 = 155 648 B (152 KB), regs 60+30+64 = 154/216. It removes the masked iteration entirely from the self-attention forward (and any tail-mask codegen). Cost: +6.6% / +6.6% loop iterations at −6.25% work each (same total MACs) plus more mbarrier round-trips. Genuinely uncertain → A/B it. Same trick does **not** transfer to the backward: with `SdP_swapAB=True` the swapped M is tile_n, and `_check_mma` requires `M % (atom_layout_m·64) == 0` (`sm90_config_search.py:39`), so bwd tile_n must stay a multiple of 64.

---

## 2. Feature A/B matrix

Each row is an independent toggle with a defined FA3-side alternative, so "if FA3's variant wins, use FA3's variant" is a per-feature measured decision. Sweep on the 8 production shapes; report per-shape, never averaged.

### 2.1 Forward

| # | Toggle | Our default | Alternatives | FA3-side alternative technique | Expected mechanism |
|---|---|---|---|---|---|
| F1 | Scheduler | non-persistent `(m_blocks,h,1)` (FA4 sm90, `flash_fwd_sm90.py:343`) | `StaticPersistentTileScheduler` (`tile_scheduler.py:287`) with 132·k CTAs | FA3 C++ persistent scheduler + `sm_margin` knob (`flash_attn_interface.py:94`) to leave SMs for other work | amortise prologue over ~23 tiles; needs separate sO or an epilogue/Q barrier (Q and O alias smem today) |
| F2 | Consumer warpgroups / pingpong | 2 WG (tile_m=128), `use_scheduler_barrier=True` (`flash_fwd_sm90.py:220-224`, barriers at `:1524-1544`) | 1 WG (tile_m=64, no pingpong); 3 WG (tile_m=192, must be noRS+noOL — only feasible 192 hd128 config, 224 KB, 128/128 regs) | FA3 C++ ping-pong vs. cooperative WG schedule (same named-barrier mechanism) | 3 WG = better softmax/MMA interleave but no RS-PV and no intra-WG overlap at hd128 |
| F3 | KV pipeline stages | 2 (`interface.py:871`) | 3 (32 768 + 3·32 768 + 3·32 768 = 229 376 B + mbars ≈ 225 KB, fits the 227 KB dynamic-smem cap); 4 does not fit | FA3 C++ `kStages` template param | deeper prefetch vs. losing the last ~2 KB of headroom |
| F4 | `mma_pv_is_rs` | True (P from registers) | False (P staged through sP, +32 KB smem, +14.0 vs 10.0 traffic/blk per FA4's own model) | FA3 C++ `Mma_PV_is_RS` | RS avoids the sP round-trip but pins 32 regs |
| F5 | `intra_wg_overlap` | True (`mma_one_n_block_intrawg_overlap`, `flash_fwd_sm90.py:1411`) | False (`mma_one_n_block`, `:1349`) | FA3 C++ non-overlapped mainloop | overlap needs the extra P register set (160 vs 128 regs) |
| F6 | tile_n | 128 | **120 (tail-free, §1.3f)**, 144, 160, 176, 192 (all feasible per `sm90_config_search.py`; 192 = 224 KB / 208 regs) | FA3 hd≤96 uses 144/192 tiles — the same "wider N" idea, never tried at hd128 | fewer iterations & fewer mbarrier waits vs. larger S accumulator |
| F7 | tile_m | 128 (=64·num_wg, forced) | 64 (1 WG), 192 (3 WG) | FA3 C++ `kBlockM` | ties to F2 |
| F8 | Tail handling | FA4: runtime `seqlenk_col_limit` mask on the first block always (`mask.py:210`, `flash_fwd_sm90.py:1299`) | (i) constexpr peeled tail block with a compile-time bitmask; (ii) tail-free tile_n (F6=120); (iii) separate smaller wgmma N for the last block | FA3 C++ `Is_even_MN` template specialisation — same idea, compile-time | removes a masked iteration and all runtime index math from 100% of CTAs |
| F9 | `check_inf` | True (`flash_fwd_sm90.py:1024`, `softmax.py:164-165`) | **False** (provably safe: dense non-causal) | FA3 C++ `Check_inf` template param, set false for non-causal | one fcmp+select per row per block × 256–591 blocks |
| F10 | O-rescale placement | `rescale_O_before_gemm=False` at hd128 (`flash_fwd_sm90.py:232`) | True (rescale in the QK-GEMM shadow, `:1436-1437`) | FA3 C++ `RescaleOBeforeGemm` template param | hides 64 FFMA/thread under the wgmma |
| F11 | **Rescale-skip (vote)** | absent | skip `rescale_O` when the running max is unchanged, using `cute.arch.vote_any_sync(pred: Boolean, mask: Int = FULL_MASK) -> Boolean` | FA4 *tried and disabled this*: a `warp_vote_any_lt` helper is commented out at `utils.py:507-524` (`vote.sync.any.pred`) | at 256–591 blocks the max stops moving early (P(update) ≈ 1/k); expected to pay off far more at our S than at FA's usual 4–8 K benchmarks — **re-measure, don't inherit their verdict** |
| F12 | TMA multicast / cluster | none (`cluster_shape_mn=(1,1)`, `num_multicast=1`) | `(2,1)` cluster multicasting K/V across 2 m-tiles | FA3 C++ hopper uses a ClusterShape with K/V multicast | halves K/V HBM→SM traffic; interacts with F1 tile ordering |
| F13 | Static vs dynamic shapes | FA4: all dynamic (`cute_dsl_utils.py:80-84`) | fully static layouts (§1.3a) | n/a (FA3 is C++-templated on `Is_even_MN` only) | constexpr trip counts, no bound registers, statically-peeled prologue/epilogue |
| F14 | Softmax-scale folding | `acc_S·scale_log2 − max·scale_log2` per element (`softmax.py:168-177`) | fold `scale·log2 e` into Q at load; exp2 becomes a plain subtract | FA3 C++ does the same per-element multiply | saves 1 FMUL per S element; **must be numerics-gated** (bf16 Q pre-scaling changes rounding) |

### 2.2 Backward

| # | Toggle | Our default | Alternatives | FA3-side alternative | Expected mechanism |
|---|---|---|---|---|---|
| B1 | tile_m | 80 (`interface.py:202`) | 64 (the FA3 causal config, 176 KB, 192/216 regs); 96/112 are register-infeasible | FA3 C++ uses 64 for causal, 80 for non-causal — identical split | 80 gives 39.6 vs 42.0 traffic/block but only 8 spare registers |
| B2 | PdS stages | 2 (`interface.py:207`) | 1 (frees 20 KB smem → room for dO_stage or a bigger tile) | FA3 C++ `kStages_dS` | smem is the binding constraint at 224 KB |
| B3 | dKV in registers | `mma_dkv_is_rs=True` via `AtomLayoutNdKV=2` (`flash_bwd_sm90.py:107-112`) | `AtomLayoutNdKV=1` + `dKV_swapAB=True` (rank-3 config, 45.6 traffic/blk) | FA3 C++ `AtomLayoutNdKV` | RS removes the sP store entirely |
| B4 | `dQ_single_wg` | False at hd128 (True only at hd≤96, `interface.py:197`) | True: WG0 does the whole dQ GEMM, WG1 skips (`flash_bwd_sm90.py:839-851`, credited to Ben Spector) | n/a — FA4-only | frees WG1 for SdP; changes `num_threads_post_dQ` (`interface.py:2023`) |
| B5 | **Cross-attn split-M** | absent (nsplit=1) | nsplit ∈ {5, 11, 22} (h12), {3, 11, 33} (h40) + fp32 dK/dV accum + `_bwd_postprocess_convert` | n/a — FA3 has the same 48-CTA pathology | **2.75× (h12) / 1.65× (h40) occupancy recovery**; costs ~23 µs of extra dK/dV traffic and makes dk/dv nondeterministic |
| B6 | **Cross-attn cluster-4 multicast** | absent | `cluster_shape_mn=(4,1)` multicasting Q/dO/LSE/dPsum across the 4 n_blocks of a head | n/a | 4× cut in Q/dO HBM traffic (805 MB → 201 MB at h12/480p) |
| B7 | Determinism | non-deterministic dQ atomics; `SingleTileLPTBwdScheduler` + `dQ_semaphore` when `deterministic=True` (`flash_bwd_sm90.py:520-521`, `interface.py:1660-1661`) | deterministic on/off; and for split-M, dK/dV semaphores (`interface.py:1665-1667`) vs atomics | FA3 `deterministic: bool = False` (`flash_attn_interface.py:280`) — already exercised by our `WAN_ATTN_BWD=fa3_det` path (`wan_attn/dispatch/wan_attention.py:81-82`) | determinism cost must be measured, not assumed |
| B8 | PDL preprocess→main | on (`flash_bwd_sm90.py:626` `use_pdl=True`, `:964` `griddepcontrol_wait()`) | off; or fuse dPsum/LSE-log2 into the main kernel's prologue and drop the preprocess launch | FA3 C++ has a separate preprocess kernel, no PDL | one fewer launch + one fewer full pass over O/dO |
| B9 | Static vs dynamic shapes | dynamic | static (constexpr `m_block_max`, no `SeqlenInfoQK`) | n/a | inner loop over 410/945 m-blocks becomes a constant-trip loop |
| B10 | dQ postprocess | separate `_bwd_postprocess_convert` launch (`interface.py:2029-2035`) | fuse the fp32→bf16 dQ conversion into the tail of the main kernel (only legal once every n_block has landed → needs a grid-wide fence or a persistent last-CTA) | FA3 C++ also uses a separate convert kernel | removes one full read+write of the S·d fp32 dq_accum (32 760·128·4·12 = 201 MB at h12/480p) |

### 2.3 Sweep protocol

* Vary **one toggle at a time** from the FA4-equivalent base; then a second pass over the top-3 winners' pairwise combinations (F1×F3, F1×F6, F6×F9, B5×B6).
* Reuse `wan_attn/timing.py` verbatim — `do_bench` with `WARMUP_MS=200`, `REP_MS=1000`, L2 flush, one untimed JIT call, SM clocks logged (`timing.py:1-9`). Do not lower the warmup; the module documents a ~30% under-measurement at the 25 ms default.
* fwd / bwd / fwd+bwd timed separately; the FA3 and FA4 columns of `RESULTS.md` L1 are the reference (fa3 fwd 9.26 ms / bwd 26.88 ms at h12 S=32 760; 712 TFLOP/s forward = env healthy).
* Gate each accepted change through `python -m wan_attn.perf_gate --note "..."` (1.5% regression threshold, `perf_gate.py:31`).
* **Every perf variant must pass §3's numerical battery before it is allowed into the ledger** — a rescale-skip or scale-folding change can be fast and wrong.

---

## 3. Numerical-parity requirements

### 3.1 The LSE contract (load-bearing for SAC)

* **Public LSE = natural log, fp32, shape `(b, h, s)`.** Set by `interface.py:474` (`lse_shape = (batch_size, num_head, seqlen_q)`, `torch.float32`) and produced by `softmax.finalize`:
```python
# softmax.py:220-226
            row_sum_cur = row_sum[r]
            LN2 = math.log(2.0)
            row_sum[r] = (
                (row_max[r] * scale_log2 + cute.math.log2(row_sum_cur, fastmath=True)) * LN2
                if not acc_O_mn_row_is_zero_or_nan else -Float32.inf
            )
```
  The `* LN2` converts the internal log2-domain accumulator back to natural log. Our oracle is `lse = torch.logsumexp(scores, dim=-1)` (`wan_attn/numerics.py:34`), matching FA's own reference `lse = torch.logsumexp(scores, dim=-1)  # [b, h, t]` (`testing.py:429`). **Emitting log2-scaled LSE would silently break SAC** — `wan_attn/dispatch/wan_attention.py:67` saves `lse` for the backward and `register_fake` declares `(b, h, sq)` fp32 (`:61`).
* **Internally the backward needs log2 LSE.** FA4 computes it in the preprocess kernel (`flash_bwd_preprocess.py:261-262`: `LOG2_E = math.log2(math.e); softmax_scale_log2 = softmax_scale * LOG2_E`) into a separate `lse_log2` buffer of shape `(b, h, seqlen_q_rounded)` fp32 (`interface.py:1600-1602`), and the bwd kernel receives **that**, not the public LSE (`interface.py:1827` `lse_log2_tensor, dpsum_tensor = [...]`). Our kernel must reproduce this conversion internally and must **not** leak it to the caller. `seqlen_q_rounded = ceil(S/tile_m)·tile_m` — with tile_m=80, 32 760 → 32 800 (40 pad rows) and 75 600 → 75 600 (none).
* `-inf` for an all-masked row (`softmax.py:225`) cannot occur for dense non-causal; if the guard is removed as a specialization, add an assertion test on the tail block.

### 3.2 Precision contract

* **bf16 in/out, fp32 accumulate.** Enforced at `flash_bwd_sm90.py:183-198`: q/k/v/dO same dtype ∈ {fp16, bf16}; `mLSE`, `mdPsum`, `mdQaccum` must be `Float32`; `mdK`/`mdV` must match `mQ`'s dtype when `qhead_per_kvhead == 1`. QK and PV accumulators are `Float32` (`flash_fwd_sm90.py:102`, `:111`).
* **If we adopt split-M for cross-attn bwd (B5), dK/dV switch to fp32 accumulators + a bf16 convert postprocess** — exactly the shape of the existing GQA path (`interface.py:1621-1655`). That path is already numerically validated upstream, so it is the safe implementation choice.

### 3.3 Scale application — exactly once per path

* Forward: `softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(softmax_scale, self.score_mod)` (`flash_fwd_sm90.py:348-350`), folded into `exp2(acc_S·scale_log2 − max·scale_log2)` (`softmax.py:168-177`). `softmax_scale = d^-0.5 = 128^-0.5`.
* Backward: `softmax_scale_log2 = softmax_scale * LOG2_E` (`flash_bwd_sm90.py:551-555`); **dK is scaled at the dKV epilogue** — `flash_bwd_sm90.py:1396-1397`
```python
                if const_expr(self.qhead_per_kvhead == 1):
                    acc_dK.store(acc_dK.load() * softmax_scale)
```
  while **dQ is scaled in the postprocess** (`interface.py:2029-2030`, `_bwd_postprocess_convert(dq_accum, dq, softmax_scale, ...)`) and **dV carries scale 1.0** (`interface.py:2048`). Any restructuring (especially B10 "fuse the dQ convert") must preserve exactly-one application per tensor. This is the classic silent-wrongness site.

### 3.4 The gates our kernel must clear

From `tests/test_attention_exactness.py`:
1. Per-tensor, per-shape rel_l2 vs the chunked-fp32 oracle, gated at 2× FA3 on identical inputs (`:62-64`):
```python
        assert e_c["rel_l2"] <= 2.0 * e_b["rel_l2"] + 1e-9, (
            f"{name} at S={s}: hybrid {e_c['rel_l2']:.3e} > 2x FA3 {e_b['rel_l2']:.3e}")
```
   applied **separately** to `o`, `dq`, `dk`, `dv` — never to an aggregate (`wan_attn/numerics.py:7-9` documents why: dq/dk carry the softmax Jacobian). Current FA3/hybrid level is **rel_l2 2.37e-3** (`RESULTS.md:44`).
2. LSE tightness: `assert (lse_h.float().reshape(ref_lse.shape) - ref_lse).abs().max() < 1e-4` (`:66`).
3. Finiteness on every tensor (`:60`).
4. FA3↔FA4 LSE interchangeability, max |Δ| < 1e-5 and |Δo| < 4e-3 (`:73-79`) — our kernel must join this equivalence class, since the production op mixes an FA3-shaped LSE into an FA4-shaped backward.
5. Edge-shape battery over `EDGE_SEQS = [63, 64, 65, 127, 128, 129, 191, 192, 193, 504, 528]` (`wan_attn/shapes.py:66`), which brackets every tile size in play — extend it with **119/120/121** and **79/80/81** if F6 (tile_n=120) or the bwd tile_m=80 tail path is touched.
6. Cross-attention shape (kv=512, dense **unmasked** — `shapes.py:6-7` "never inject a mask").
7. Non-contiguous / transposed dO.
8. Full true-shape oracle check at S=32 760 behind `WAN_ATTN_SLOW_TESTS=1`.

### 3.5 Test-matrix consequences of static specialization

* If we compile per-(S, h, kind), the edge battery cannot exercise the production kernels. **Ship two builds of the same source**: a *dynamic-S* build for the edge/small-shape battery, and *static* builds for the 8 production shapes. Add a test asserting `static(S) == dynamic(S)` **bitwise** at each production S (same inputs, same stream) — that is the only thing that transfers edge-shape confidence onto the specialized binaries.
* Add a compile-key/shape-guard test: dispatching a static kernel on a mismatched S must raise, not silently read out of bounds.

### 3.6 Determinism budget

* dQ is already nondeterministic (fp32 gmem atomics) unless `deterministic=True`; the SAC gate is written against a *measured* run-to-run band, not against bitwise equality:
```python
# tests/test_sac.py
    band = maxdiff(g_full, g_full2)   # full-vs-full rerun band
    assert d_sac <= max(band, 1e-6) * 4 + 1e-7, "SAC grads outside nondeterminism band"
```
* **Adopting B5 (split-M) makes dk/dv nondeterministic too.** That will widen `band`, and because the assertion is relative to `band` it will still pass — but the *absolute* gradient noise floor rises. Requirement: re-run `test_sac.py` and record the new band in `RESULTS.md`; if it grows by more than ~2×, gate split-M behind a `deterministic` flag using the dK/dV semaphore machinery (`interface.py:1665-1667`, `barrier.wait_eq` / `barrier.arrive_inc` as in `flash_bwd_sm90.py:1866-1895`) and keep `WAN_ATTN_BWD=fa3_det` as the deterministic escape hatch.

### 3.7 Recommended CI shape × feature grid

| axis | values |
|---|---|
| kind | self, cross |
| S_q | 32 760, 75 600, + edge set |
| h | 12 (primary), 40 (14B extension) |
| dtype | bf16 only (fp16 not a target; assert-reject fp32 in) |
| dO layout | contiguous, transposed, non-contiguous slice |
| toggles | each F1–F14 / B1–B10 variant that is *kept* must re-run the full battery, not just a smoke test |
| determinism | default (atomics) and `deterministic=True` |

Sanity FLOP/time anchors for reading the numbers (`wan_attn/shapes.py:39-46`: `fwd_flops = 4·b·h·s_q·s_kv·d`, `bwd_flops = 2.5·fwd_flops`): self h12/S=32 760 = 6.59 TFLOP fwd; self h40/S=75 600 = 117 TFLOP fwd; cross h12/S=32 760 = 103 GFLOP fwd / 258 GFLOP bwd.
