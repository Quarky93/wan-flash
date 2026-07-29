# BWD_STUDY

> Source study of FA4 CuTeDSL (installed flash-attn-4 4.0.0b23) against
> nvidia-cutlass-dsl 4.6.0, 2026-07-29. File:line refs are into the installed tree.

## Summary

FA4's SM90 backward is a **3-kernel chain** driven from `interface.py:_flash_attn_bwd` (line 1308): (1) `flash_bwd_preprocess.py` computes `D = rowsum(O*dO)` (fp32, seqlen-padded), converts `LSE → LSE*log2(e)` (fp32, seqlen-padded), and **zeroes the fp32 `dQaccum` buffer**; (2) `flash_bwd_sm90.py` is a warp-specialized 384-thread kernel, **dK/dV-stationary** (one CTA owns one `n_block` = KV tile, loops over all `m_block`s), 2 MMA warpgroups × 5 WGMMAs + 1 producer warpgroup split into `warp0 = TMA loads` / `warp1 = dQ accumulate-store`; dQ is accumulated into gmem with `cp.reduce.async.bulk...add.f32` (a hardware fp32 reduction, hence nondeterministic unless a semaphore chain is enabled); (3) `flash_bwd_postprocess.py` reads `dQaccum` fp32, multiplies by `softmax_scale`, converts to bf16, writes `dQ`. For hd128 non-causal bf16 the config is **tile_m=80, tile_n=128, 2 MMA WGs, Q/dO/PdS stages 2/2/2, SdP_swapAB=True, dQ_swapAB=True, AtomLayoutNdKV=2 → `mma_dkv_is_rs=True`** (P stays in registers, no `sP` buffer), ≈226 KiB smem, 1 CTA/SM. For our fixed-shape non-causal MHA case roughly a third of the machinery (varlen, causal/local block ranges, GQA dK/dV fp32 accum + 2 extra postprocess kernels, block-sparsity, score_mod/mask_mod, LPT scheduler, semaphores) is compile-time dead. The one real *architectural* risk for Wan2.1 is **cross-attention S_kv=512 → only 4 n_blocks → 4×nheads CTAs (48 CTAs at nheads=12) on 132 SMs**; the dK/dV-stationary decomposition under-fills the GPU there.

## Details

All paths below are under `/workspace/wan-attn/.venv/lib/python3.12/site-packages/`.

---

# 1. The 3-phase structure

## Host chaining — `flash_attn/cute/interface.py`

`_flash_attn_bwd` (interface.py:1308) is the only entry; `FlashAttnFunc.backward` (interface.py:2549) calls it with the *natural-log* `lse` saved by forward (interface.py:2576-2596).

### Intermediate buffers allocated (interface.py:1583-1670)

```python
head_dim_rounded = (head_dim + 32 - 1) // 32 * 32          # interface.py:1583  → 128 for hd128

if cu_seqlens_q is None:
    dq_accum = ... torch.empty(
        batch_size, num_head,
        seqlen_q_rounded * head_dim_rounded,                # interface.py:1589-1595
        dtype=torch.float32, device=device)
    dpsum    = torch.empty(batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, ...)  # 1597
    lse_log2 = torch.empty(batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, ...)  # 1600
```

- `seqlen_q_rounded = ceil(seqlen_q / m_block_size) * m_block_size` (interface.py:1484) — note this uses **`m_block_size` = 80** for hd128 non-causal, not 128.
- `dq_accum` is **flat** `seqlen_q_rounded * head_dim_rounded` fp32 per (batch, head) — *not* a `(s, d)` matrix; see §3, its element order is MMA-fragment order.
- `dQ_semaphore` only when `deterministic` (interface.py:1660-1663): `torch.zeros(batch, nhead, seqlen_q_rounded // m_block_size, cluster_size, dtype=int32)`.
- `dk_accum`/`dv_accum` fp32 exist **only for GQA** (`dKV_postprocess = qhead_per_kvhead > 1`, interface.py:1621-1655). For MHA, dK/dV are written directly as bf16 by TMA from the main kernel.

### Launch order (interface.py:1674 → 1984 → 2029)

```python
_bwd_preprocess(out, dout, dpsum, lse, lse_log2, dq_accum,
                cu_seqlens_q, seqused_q, dlse,
                dtype, head_dim, head_dim_v, m_block_size)      # interface.py:1674-1678
...
_flash_attn_bwd.compile_cache[compile_key](q, k, v, dout,
    lse_log2, dpsum, dq_accum, dk, dv, softmax_scale, ...)      # interface.py:1984-2017
...
_bwd_postprocess_convert(dq_accum, dq, softmax_scale,
    cu_seqlens_q, seqused_q,
    arch, dtype, head_dim, m_block_size, num_threads_post_dQ,
    AtomLayoutMdQ, dQ_swapAB, ...)                              # interface.py:2029-2035
```

Note `lse_log2` is passed **in the `mLSE` slot** of the main kernel — the main kernel never sees natural-log LSE.

### Phase A — `flash_bwd_preprocess.py` (465 lines)

Class `FlashAttentionBackwardPreprocess`, `tile_m = m_block_size` (=80 here), `num_threads=256` (preprocess.py:45-46), `head_dim_padded = ceil(hd/32)*32` (preprocess.py:68-70).

Computes, per `(m_block, head, batch)` tile:

1. **D = rowsum(O ⊙ dO)** in fp32:
```python
pdpsum = (tOrO.load().to(Float32) * tOrdO.load().to(Float32)).reduce(
    cute.ReductionOp.ADD, init_val=0.0, reduction_profile=(0, None, 1))     # preprocess.py:388-390
threads_per_row = gmem_tiled_copy_O.layout_src_tv_tiled[0].shape[0]
pdpsum = utils.warp_reduce(pdpsum, operator.add, width=threads_per_row)     # preprocess.py:393
```
   Written with an OOB guard, and optionally the `dLSE` correction `D' = D − dLSE`:
```python
if tOcO[0, 0, 0][1] == 0:
    for m in cutlass.range(cute.size(PdP_sum), unroll_full=True):
        row = tOcO[0, m, 0][0]
        PdPsum_val = 0.0
        if row < seqlen_limit:
            PdPsum_val = PdP_sum[m]
            if const_expr(mdLSE is not None):
                PdPsum_val -= gdLSE[row]
        gPdPsum[row] = PdPsum_val                                           # preprocess.py:406-414
```
   The dLSE algebra is documented at preprocess.py:5-13 (`dS = P*(dP − (D − dLSE))`).

2. **Zero `dQaccum`** (required, since the main kernel does `+=` into gmem):
```python
blkdQaccum_shape = (self.tile_m * self.head_dim_padded,)
gdQaccum = cute.local_tile(mdQaccum_cur, blkdQaccum_shape, (m_block,))
zero = cute.make_rmem_tensor_like(tdQgdQaccum); zero.fill(0.0)
cute.copy(gmem_tiled_copy_dQaccum, zero, tdQgdQaccum)                       # preprocess.py:425-431
```

3. **LSE → log2 base** with a +inf sentinel for padded rows (see §4):
```python
lse = Float32.inf
if tidx < seqlen_limit:
    lse = gLSE[tidx]                                                        # preprocess.py:351-353
...
LOG2_E = math.log2(math.e)
lse_log2 = lse * LOG2_E if lse != -Float32.inf else 0.0                     # preprocess.py:433-434
if tidx < seqlen_q_rounded - m_block * self.tile_m:
    gLSElog2[tidx] = lse_log2                                               # preprocess.py:441-442
```

PDL handshake (this is load-bearing and easy to get wrong):
```python
if const_expr(self.use_pdl):
    cute.arch.griddepcontrol_wait()      # preprocess.py:320-321, before touching O/dO/LSE
...
# after O and dO are in registers:
if const_expr(self.use_pdl):
    cute.arch.griddepcontrol_launch_dependents()   # preprocess.py:385-386
```
with the comment at preprocess.py:383-384 explaining that correctness of the *outputs* is instead guaranteed by a second `griddepcontrol_wait()` inside the main kernel (flash_bwd_sm90.py:964).

**Buffer dtypes/layouts out of phase A:** `dpsum` fp32 `(b, h, s_rounded)`; `lse_log2` fp32 `(b, h, s_rounded)`; `dq_accum` fp32 `(b, h, s_rounded*hd_rounded)` zeroed.

### Phase B — `flash_bwd_sm90.py` (see §2)

### Phase C — `flash_bwd_postprocess.py` (587 lines)

`FlashAttentionBackwardPostprocess`, sm90 branch. It **must be constructed with the same `AtomLayoutMdQ` and `dQ_swapAB` and the same warpgroup count** as the main kernel, because `dQaccum`'s gmem element order is the MMA accumulator fragment order:

```python
elif const_expr(self.arch // 10 == 9):
    num_wg_mma = self.num_threads // 128
    atom_layout_dQ = (self.AtomLayoutMdQ, num_wg_mma // self.AtomLayoutMdQ)
    tiler_mn_dQ = (self.tile_m // atom_layout_dQ[0], self.tile_hdim // atom_layout_dQ[1])
    tiled_mma = sm90_utils_basic.make_trivial_tiled_mma(
        self.dtype, self.dtype,
        warpgroup.OperandMajorMode.K,   # These don't matter, we only care about the accum
        warpgroup.OperandMajorMode.K, Float32,
        atom_layout_mnk=(atom_layout_dQ if not self.dQ_swapAB else atom_layout_dQ[::-1]) + (1,),
        tiler_mn=tiler_mn_dQ if not self.dQ_swapAB else tiler_mn_dQ[::-1])   # postprocess.py:104-117
```
and
```python
self.s2r_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
    cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128),
    cute.make_layout((num_threads_per_warp_group, num_wg_mma)),  # thr_layout
    cute.make_layout(128 // Float32.width))                       # val_layout
self.sdQaccum_layout = cute.make_layout(
    (self.tile_m * self.tile_hdim // num_wg_mma, num_wg_mma))     # postprocess.py:160-167
```

Body (postprocess.py:491-587): G→S `cp.async` of the flat fp32 tile → S→R `autovec_copy` reinterpreted as the accumulator fragment → scale + convert → R→S via `StMatrix` (`utils.get_smem_store_atom(arch, dtype, transpose=self.dQ_swapAB)`, postprocess.py:537-540) → S→R → predicated gmem store:

```python
tdQrdQaccum = cute.make_tensor(acc.iterator, cute.make_layout(tdQsdQaccum.shape))
cute.autovec_copy(tdQsdQaccum, tdQrdQaccum)
rdQ = cute.make_fragment_like(acc, self.dtype)
rdQ.store((acc.load() * scale).to(self.dtype))                    # postprocess.py:528-532
```
This is the **only place the `softmax_scale` is applied to dQ** (interface.py:2030 passes `softmax_scale`; for `dv` it passes `1.0`, interface.py:2048).

`self.tile_hdim` here is `ceil(hd/32)*32` (postprocess.py:59-61), and `num_threads_post_dQ = 128 if dQ_single_wg else cfg.num_wg * 128` (interface.py:2023).

---

# 2. Main bwd kernel — `flash_bwd_sm90.py`

## 2.1 Parallel decomposition: dK/dV stationary

Grid comes from the tile scheduler with **Q and K roles swapped**:

```python
tile_sched_args = TileSchedulerArguments(
    cute.ceil_div(cute.size(mK.shape[0]), self.tile_n),   # num_block = n_blocks
    cute.size(mQ.shape[2]),                               # num_head
    cute.size(mK.shape[3]),                               # num_batch
    1,                                                    # num_splits
    cute.size(mQ.shape[0]),        # pass seqlen_q or total_q for seqlen_k
    mQ.shape[1], mV.shape[1],
    ...
    tile_shape_mn=(self.tile_n, self.tile_m),  # Swapping the role of Q & K   # sm90.py:525-546
```
`SingleTileScheduler.get_grid_shape` → `(num_block, num_head*num_splits, num_batch)` (tile_scheduler.py:241-245). So **grid.x = number of KV tiles**, each CTA owns exactly one `n_block` and iterates over `m_block`s:

```python
m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
...
dKV_accumulate = False
for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
    consumer_state_Q, consumer_state_dO = mma_one_m_block_all(
        m_block, consumer_state_Q, consumer_state_dO,
        mask_fn=mask_fn, score_mod_fn=..., score_mod_bwd_fn=...,
        dKV_accumulate=dKV_accumulate)
    dKV_accumulate = True                                                    # sm90.py:1362-1373
```
`acc_dK`/`acc_dV` live in registers across the entire m loop (`zero_init = not dKV_accumulate` on the first iteration), and are flushed once in `epilogue_dKV` (sm90.py:1398-1417). `get_m_block_min_max` for non-causal/non-local returns `(0, ceil_div(seqlen_q, tile_m))` (block_info.py:58-71).

With `deterministic=True` the scheduler becomes `SingleTileLPTBwdScheduler` (sm90.py:520-521), a flattened 1-D grid `(total_blocks, 1, 1)` with L2 head-swizzle and largest-processing-time-first ordering (tile_scheduler.py:648-761).

## 2.2 Warp specialization (sm90.py:762-851)

`num_threads = (cfg.num_wg + 1) * 128 = 384` (interface.py:1414). Threads 0-127 = producer WG, 128-383 = 2 MMA WGs.

```python
if warp_idx < 4:
    cute.arch.setmaxregister_decrease(self.num_producer_regs)          # 24 regs, sm90.py:437
    if warp_idx == 0:
        self.load(...)          # TMA: K, V (once), then Q, LSE, dO, dPsum per m_block
    if warp_idx == 1:
        self.dQaccum_store(...) # gmem reduction of dQ + optional semaphore
else:
    ...
    if const_expr(self.num_wg_dQ == self.num_wg_mma):
        cute.arch.setmaxregister_increase(self.num_mma_regs_wg0)       # 240
        self.mma(*mma_args, is_dQ_wg=True)
    else:
        warp_idx_in_mma = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - 4
        if warp_idx_in_mma < 4:
            cute.arch.setmaxregister_increase(self.num_mma_regs_wg0)   # 256
            self.mma(*mma_args, is_dQ_wg=True)
        else:
            cute.arch.setmaxregister_increase(self.num_mma_regs_wg1)   # 224
            self.mma(*mma_args, is_dQ_wg=False)                        # sm90.py:762-851
```
Warps 2 and 3 of the producer WG are idle. Register budget (sm90.py:428-446): 2 WG → `(240, 240, 24)`; with `dQ_single_wg` → `(256, 224, 24)`; `REG_LIMIT = 504` for 2 WGs.

Both MMA warpgroups run **all five GEMMs cooperatively** (they are partitioned by the WGMMA atom layout, not by role). The only role split is optional `dQ_single_wg` (a "Credit: Ben Spector" trick, sm90.py:138-143): WG0 does the entire dQ GEMM and WG1 skips it (sm90.py:1610-1621).

## 2.3 The five matmuls, in `mma_one_m_block` (sm90.py:1467-1625)

| # | Code label | Math | TiledMMA | Accum lifetime |
|---|---|---|---|---|
| 1 | `(1) [GEMM 1]` sm90.py:1500-1502 | `S = Q @ Kᵀ` | `tiled_mma_SdP` | per-m_block, `gemm_zero_init` |
| 2 | `(2) [GEMM 2]` sm90.py:1505-1509 | `dP = dO @ Vᵀ` | `tiled_mma_SdP` | per-m_block, `gemm_zero_init` |
| 3 | `(5) [GEMM 3]` sm90.py:1567-1573 | `dV += Pᵀ @ dO` | `tiled_mma_dV` | **whole n_block** |
| 4 | `(6) [GEMM 4]` sm90.py:1579-1581 | `dQ = dS @ K` | `tiled_mma_dQ` | per-m_block → gmem reduce |
| 5 | `(7) [GEMM 5]` sm90.py:1584-1590 | `dK += dSᵀ @ Q` | `tiled_mma_dK` | **whole n_block** |

Interleaved pointwise:
- `(3) [Pointwise 1] P = exp(S − LSE)` sm90.py:1518-1528
- `(4) [Pointwise 2] dS = P*(dP−dPsum)` sm90.py:1540-1546

Exact software-pipeline sequence with WGMMA wait depths:

```python
pipeline_Q.consumer_wait(consumer_state_Q, pipeline_Q.consumer_try_wait(consumer_state_Q))
acc_S = mma_qk_fn(A_idx=smem_idx_Q, wg_wait=-1)                      # sm90.py:1501-1502
tLSErLSE = copy_utils.load_s2r(tLSEsLSE[None, smem_idx_Q])           # sm90.py:1504
pipeline_dO.consumer_wait(consumer_state_dO_cur, pipeline_dO.consumer_try_wait(...))
acc_dP = mma_dov_fn(A_idx=smem_idx_Q, wg_wait=1)                     # sm90.py:1506-1509
...  # mask, then exp2
tLSErdPsum = copy_utils.load_s2r(tLSEsdPsum[None, smem_idx_dO])      # sm90.py:1529
tdVrP = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_S), self.dtype)   # sm90.py:1532
if const_expr(not self.mma_dkv_is_rs):
    if const_expr(self.PdS_stage == 1): PdS_barrier.arrive_and_wait()
    copy_P_r2s(tdVrP, dst_idx=smem_idx_PdS)                          # sm90.py:1534-1538
warpgroup.wait_group(0)                                              # sm90.py:1541  (dP ready)
... dS = P*(dP - D) ...
tdKrdS = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_dP), self.dtype)  # sm90.py:1552
if const_expr(not self.mma_dkv_is_rs or (self.PdS_stage == 1 and self.mma_dkv_is_rs)):
    cute.arch.fence_view_async_shared(); PdS_barrier.arrive_and_wait()        # sm90.py:1560-1562
copy_dS_r2s(tdKrdS, dst_idx=smem_idx_PdS)                            # sm90.py:1565
mma_pdo_fn(tCrA=tdVrP, B_idx=smem_idx_dO, zero_init=not dKV_accumulate, wg_wait=-1)  # sm90.py:1573
cute.arch.fence_view_async_shared(); PdS_barrier.arrive_and_wait()   # sm90.py:1576-1577
acc_dQ = mma_dsk_fn(A_idx=smem_idx_PdS, wg_wait=1)                   # sm90.py:1581
pipeline_dO.consumer_release(consumer_state_dO_cur)                  # sm90.py:1582
mma_dsq_fn(tCrA=tdKrdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=1)   # sm90.py:1590
cute.arch.barrier(barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
                  number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE)
tdQrdQaccum_flat = cute.make_tensor(acc_dQ.iterator, cute.make_layout(tdQsdQaccum.shape))
cute.autovec_copy(tdQrdQaccum_flat, tdQsdQaccum)
cute.arch.fence_view_async_shared()
cute.arch.barrier_arrive(barrier_id=int(NamedBarrierBwd.dQFullWG0) + warp_group_idx,
                         number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE)
warpgroup.wait_group(0)
pipeline_Q.consumer_release(consumer_state_Q)                        # sm90.py:1594-1609
```

`mma_dkv_is_rs` decides whether P/dS go through smem or stay in registers as the WGMMA A-operand:
```python
self.mma_dkv_is_rs = (AtomLayoutMSdP == 1 and AtomLayoutNdKV == self.num_wg_mma
                      and SdP_swapAB and not dKV_swapAB)             # sm90.py:107-112
```
For hd128 non-causal this is **True**, so `sP` is not allocated (`cosize_sP = 0`, sm90.py:319) and `mma_pdo_fn` / `mma_dsq_fn` take `tCrA=` from registers (sm90.py:1187, 1200).

## 2.4 Producer: TMA load loop (sm90.py:853-1019)

K and V are loaded **once per CTA**, piggy-backed on the first Q/dO pipeline stage via `extra_tx_count`:

```python
pipeline_Q.producer_acquire(producer_state_Q, extra_tx_count=self.tma_copy_bytes["K"])
load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q))
load_Q(first_m_block, producer_state=producer_state_Q)
# Wait for bwd preprocess to finish writing LSE and dPsum
cute.arch.griddepcontrol_wait()                                       # sm90.py:958-964
load_LSE(first_m_block, producer_state=producer_state_Q)
producer_state_dO_cur = (producer_state_dO if const_expr(self.Q_stage != self.dO_stage)
                         else producer_state_Q)
pipeline_dO.producer_acquire(producer_state_dO_cur, extra_tx_count=self.tma_copy_bytes["V"])
load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO_cur))
load_dO(first_m_block, producer_state=producer_state_dO_cur)
load_dPsum(first_m_block, producer_state=producer_state_dO_cur)       # sm90.py:965-978
```
Two TMA pipelines with fused transaction counts:
```python
pipeline_Q = pipeline.PipelineTmaAsync.create(
    barrier_storage=storage.mbar_ptr_Q.data_ptr(), num_stages=self.Q_stage,
    producer_group=pipeline_producer_group, consumer_group=pipeline_consumer_group,
    tx_count=self.tma_copy_bytes["Q"] + self.tma_copy_bytes["LSE"], defer_sync=True)
pipeline_dO = pipeline.PipelineTmaAsync.create(..., num_stages=self.dO_stage,
    tx_count=self.tma_copy_bytes["dO"] + self.tma_copy_bytes["dPsum"], defer_sync=False)  # sm90.py:692-707
```
Q/dO/dK/dV use `cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), ...)` (sm90.py:468-514); LSE/dPsum use `copy_utils.cpasync_bulk_get_copy_fn` (sm90.py:931-934) — a plain `cp.async.bulk` of a 1-D fp32 vector, not a TMA tensor copy. `extra_tx_count` is a FA4-local extension to `PipelineTmaAsync.producer_acquire` (pipeline.py:301-330).

## 2.5 Tile sizes / stages for hd128 (interface.py:199-212)

```python
elif head_dim <= 128:
    # C++ FA3: causal/local: 64, 128; non-causal: 80, 128 with dQ_swapAB
    is_causal_or_local = causal or local
    m_block_size = 64 if is_causal_or_local else 80
    if sparse_block_size_q is not None and sparse_block_size_q % m_block_size != 0:
        m_block_size = 64
    return BwdConfig(
        m_block_size=m_block_size,
        n_block_size=128,
        num_stages_Q=2, num_stages_dO=2, num_stages_PdS=2,
        SdP_swapAB=True, dKV_swapAB=False,
        dQ_swapAB=m_block_size % 64 != 0,
        AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
    )
```
So **our exact case (bf16, hd128, non-causal) → `tile_m=80, tile_n=128, num_wg=2 (384 threads), Q/dO/PdS stages = 2/2/2, SdP_swapAB=True, dKV_swapAB=False, dQ_swapAB=True, AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1, dQ_single_wg=False`**.

`tile_m=80` is not a multiple of 64, hence `dQ_swapAB=True` so the dQ WGMMA computes `dQᵀ` (M=128 hd, N=80). `shuffle_LSE/shuffle_dPsum` are False here (`SdP_swapAB and tile_hdim <= 64`, sm90.py:123-124), so each thread privately holds `tile_m/4 = 20` LSE and 20 dPsum floats.

SMEM (from `_get_shared_storage_cls`, sm90.py:306-341, all `Align[..., 1024]`):

| buffer | expression | bytes |
|---|---|---|
| `sQ` | 80·128·2·`Q_stage`2 | 40960 |
| `sK` | 128·128·2 | 32768 |
| `sV` | 128·128·2 | 32768 |
| `sdO` | 80·128·2·`dO_stage`2 | 40960 |
| `sP` | 0 (`mma_dkv_is_rs`) | 0 |
| `sdS` | 80·128·2·`PdS_stage`2 | 40960 |
| `sdQaccum` | 80·128·4 | 40960 |
| `sLSE` | `round_up(80,64)`·2·4 | 1024 |
| `sdPsum` | `round_up(80,64)`·2·4 | 1024 |
| mbarriers | (2·2+2·2)·8 | 64 |
| **total** | | **≈231,488 B = 226.1 KiB** |

That is 1 CTA/SM on H100 (227 KiB opt-in limit); the launch confirms `min_blocks_per_mp=1, use_pdl=True` (sm90.py:621-627). The independent cost model in `sm90_config_search.py:16` uses `SMEM_LIMIT = 224*1024` and `REG_LIMITS = {2: 216, 3: 128}`, with peak accumulator regs `max(2*regs_SdP, regs_dQ) + regs_dK + regs_dV` (sm90_config_search.py:95-98) = `max(80,40)+64+64 = 208` for this config.

SMEM layouts are built to hold both a matrix and its transpose (needed because Q is the B-operand of `dK = dSᵀQ` in MN-major):
```python
# We need to accommodate both Q and Q^T (and dO and dO^T) in shared memory.
# Q & dO are used in the SdP Mma and Q^T and dO^T are used in the dKV Mma.
# The M dimension (tile_m) doesn't matter for the layout, only the K dimension
wg_d_dKV = self.num_wg_mma // self.AtomLayoutNdKV
self.sQ_layout, self.sdO_layout = [
    sm90_utils.make_smem_layout(self.dtype, LayoutEnum.ROW_MAJOR, shape, stage,
                                major_mode_size=mms)
    for shape, stage, mms in [
        ((self.tile_m, self.tile_hdim),  self.Q_stage,  self.tile_hdim // wg_d_dKV),
        ((self.tile_m, self.tile_hdimv), self.dO_stage, self.tile_hdim // wg_d_dKV)]]  # sm90.py:202-219
```
and the P/dS layout must satisfy **both** consumers:
```python
wg_n_SdP = self.num_wg_mma // self.AtomLayoutMSdP
wg_n_dKV = self.AtomLayoutNdKV
self.sPdS_layout = sm90_utils.make_smem_layout(
    self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_n), stage=self.PdS_stage,
    major_mode_size=math.gcd(self.tile_n // wg_n_SdP, self.tile_n // wg_n_dKV))          # sm90.py:233-242
```
Transposed views are free composition: `layout_utils.transpose_view` (quack/layout_utils.py:10-14) used for `sPt`, `sdOt`, `sdSt`, `sQt`, `sKt` (sm90.py:1176-1202).

## 2.6 dK/dV epilogue for MHA (sm90.py:1654-1700)

dK/dV are converted to bf16 in registers, staged **through the now-dead `sK`/`sV` buffers**, and TMA'd out — no fp32 dK/dV buffer and no dK/dV postprocess kernel at all:

```python
cute.arch.cp_async_bulk_wait_group(1, read=True)
epi_barrier.arrive_and_wait()
copy_dV_r2s(acc_dV, dst_idx=None)
cute.arch.fence_view_async_shared()
epi_barrier.arrive_and_wait()
if warp_idx == 4:
    store_dV(); cute.arch.cp_async_bulk_commit_group()
cute.arch.cp_async_bulk_wait_group(1, read=True)
epi_barrier.arrive_and_wait()
copy_dK_r2s(acc_dK, dst_idx=None)
cute.arch.fence_view_async_shared()
epi_barrier.arrive_and_wait()
if warp_idx == 4:
    store_dK(); cute.arch.cp_async_bulk_commit_group()              # sm90.py:1685-1700
```
`softmax_scale` is applied to dK just before the epilogue (`acc_dK.store(acc_dK.load() * softmax_scale)`, sm90.py:1396-1397); dV needs no scale.

---

# 3. How dQ is accumulated

**Not `atomicAdd`, and not a plain store.** Each CTA's `acc_dQ` (fp32 registers) is dumped to smem `sdQaccum`, then warp 1 issues a **bulk async fp32 reduction to global memory**:

```python
for warp_group_idx in cutlass.range_constexpr(num_dQ_chunks):
    cute.arch.barrier(barrier_id=int(NamedBarrierBwd.dQFullWG0) + warp_group_idx,
                      number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE)
    with cute.arch.elect_one():
        copy_utils.cpasync_reduce_bulk_add_f32(
            sdQaccum[None, warp_group_idx].iterator,
            gdQaccum[(None, warp_group_idx), m_block_safe].iterator,
            self.tma_copy_bytes["dQ"])
    cute.arch.cp_async_bulk_commit_group()                          # sm90.py:1873-1885
```
which is one PTX instruction (quack/copy_utils.py:685-703):
```python
cute.arch.inline_ptx(
    "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [{$r0}], [{$r1}], {$r2};",
    read_only_args=[gmem_ptr.llvm_ptr, smem_ptr_i32, Int32(store_bytes)], ...)
```
`tma_copy_bytes["dQ"] = tile_m * tile_hdim * 4 / num_wg_dQ` (sm90.py:462-464) = 20480 B per WG chunk here.

Key consequences:
- The gmem `dQaccum` **element order is the WGMMA accumulator fragment order**, because both the R→S copy and the S→G reduce are flat reinterpretations:
  `tdQrdQaccum_flat = cute.make_tensor(acc_dQ.iterator, cute.make_layout(tdQsdQaccum.shape))` (sm90.py:1598-1600) with `sdQaccum_layout = cute.make_layout((tile_m*tile_hdim//num_wg_dQ, num_wg_dQ))` (sm90.py:243-245) and `r2s_tiled_copy_dQaccum` thr_layout `(128, num_wg_dQ)`, val 4×f32 (sm90.py:247-252). The postprocess kernel undoes exactly this (postprocess.py:157-167, 505-529) — which is why `AtomLayoutMdQ`, `dQ_swapAB`, and the WG count must be threaded from the main kernel to the postprocess (interface.py:2022-2034).
- Ping-pong handshake between MMA WGs and the store warp uses **two named barriers per WG**: `dQFullWG{0,1,2}` and `dQEmptyWG{0,1,2}` (named_barrier.py:34-39), each with `128 + 32` participants (one MMA WG + the store warp).
- Multi-buffering across chunks: the store warp waits `cp_async_bulk_wait_group(num_dQ_chunks - 1 - warp_group_idx, read=read_flag)` before releasing each `dQEmpty` (sm90.py:1846-1855).

**Determinism.** With `deterministic=False` (default) the order in which different `n_block` CTAs reduce into the same `m_block` is arbitrary → **run-to-run bitwise nondeterminism in dQ** (dK/dV are exact, they're register-accumulated within one CTA). With `deterministic=True`:

```python
if const_expr(self.deterministic):
    if const_expr(self.spt):
        _, n_block_max_for_m_block = block_info.get_n_block_min_max(seqlen, m_block_safe)
        lock_value = n_block_max_for_m_block - 1 - n_block
    else:
        lock_value = n_block
    barrier.wait_eq(mdQ_semaphore_cur[(m_block_safe, None)].iterator,
                    warp_local_tidx, 0, lock_value)                          # sm90.py:1857-1871
...
if const_expr(self.deterministic):
    cute.arch.cp_async_bulk_wait_group(0, read=read_flag)   # read_flag == False here
    barrier.arrive_inc(mdQ_semaphore_cur[(m_block_safe, None)].iterator,
                       warp_local_tidx, 0, 1)                                # sm90.py:1887-1895
```
so n_blocks add in strictly ascending index order, making the fp32 summation order fixed. `read_flag = const_expr(not self.deterministic)` (sm90.py:1792) — deterministic mode waits for *full* completion of the gmem reduce, not just source-read completion, before releasing the lock. The semaphore primitives are hand-rolled PTX (barrier.py):
```python
"ld.global.acquire.gpu.b32 $0, [$1];"      # barrier.py:15  (ld_acquire, spin)
"red.release.gpu.global.add.s32 [$0], $1;" # barrier.py:47  (red_release, arrive_inc)
```
`wait_eq` spins only on thread 0 of the warp (barrier.py:56-61) — every other thread relies on warp lockstep, no `__syncwarp`.

There is also a deadlock-avoidance path for `local` + deterministic where an n_block "signals" the m_blocks it will never visit (sm90.py:1914-1926), and `assert not self.deterministic` for block-sparse (sm90.py:1896-1899).

---

# 4. LSE handling

**Base-2 throughout, with a `+inf` sentinel for padding and a `0.0` sentinel for fully-masked rows.**

1. Forward saves `lse` in **natural log**; preprocess writes `lse_log2 = lse * log2(e)` into the padded `(b,h,s_rounded)` fp32 buffer (preprocess.py:433-442, quoted in §1). Rows `>= seqlen_q` get `lse = Float32.inf` (preprocess.py:351-353) so their `lse_log2 = +inf`; rows whose LSE is `-inf` (fully masked) are clamped to `0.0` to avoid `inf - inf = NaN`.
2. Main kernel folds `log2(e)` into the softmax scale on the host side:
```python
LOG2_E = math.log2(math.e)
if const_expr(self.score_mod is None):
    softmax_scale_log2 = softmax_scale * LOG2_E
else:
    softmax_scale_log2 = LOG2_E                                       # sm90.py:551-555
```
3. The recompute is a single fused `exp2` on the **unscaled** `S = Q@Kᵀ`:
```python
acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=self.SdP_swapAB)
lane_idx = cute.arch.lane_idx()
for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
    lse_val = self._get_stat(tLSErLSE, r, lane_idx, shuffle=self.shuffle_LSE)
    for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
        acc_S_mn[r, c] = cute.math.exp2(
            acc_S_mn[r, c] * softmax_scale_log2 - lse_val, fastmath=True)   # sm90.py:1521-1528
```
Because `lse_log2` already contains `log2(e)`, `exp2(S*s*log2e − lse*log2e) = exp(S*s − lse) = P`. Padded Q rows get `exp2(x − inf) = 0`, so **the m-tail needs no mask** — this is the trick that lets the main kernel apply only the K-direction mask.

4. `dS = P * (dP − D)` with no scale (sm90.py:1543-1546); the `softmax_scale` factor is deferred to the dK epilogue (sm90.py:1397) and the dQ postprocess (interface.py:2030). dV gets scale `1.0` (interface.py:2048).

5. LSE/dPsum broadcast into the accumulator layout is a **zero-stride tensor trick**, not a shuffle:
```python
tLSEsLSE = layout_utils.mma_partition_C_vec(sLSE, thr_mma_SdP,
    expand_shape=self.tile_n, is_colvec=not self.SdP_swapAB)          # sm90.py:1241-1246
```
`mma_partition_C_vec` (quack/layout_utils.py:280-294) builds a rank-3 view with stride 0 in the broadcast dimension and then partitions it with the MMA. The optional `shuffle_LSE` path (`SdP_swapAB and tile_hdim <= 64`, sm90.py:123-124, `_get_stat` at sm90.py:1451-1465) spreads rows over 8 lane-quads and `shuffle_sync`s them back — a register-pressure relief valve that is **off for hd128**.

---

# 5. SM90 backward limitations (evidence)

| Limitation | Evidence |
|---|---|
| **fp16/bf16 only** | `if dtype not in [cutlass.Float16, cutlass.BFloat16]: return False` — `can_implement`, sm90.py:156-157; type check at sm90.py:183-199 |
| **No FP8 backward at all** | `raise NotImplementedError("FA4 CuTe FP8 backward is not supported yet (forward-only).")` — interface.py:468 |
| **head_dim 8..256, multiple of 8 (bf16)** | `is_sm90_range = 8 <= head_dim <= 256 and 8 <= head_dim_v <= 256`, `_validate_head_dims`, interface.py:104-109; `head_dim % 8 != 0 → False`, sm90.py:158-161 |
| **No MLA / sparse-MLA backward on sm90** | `_flash_attn_bwd_sparse_mla`: `assert arch // 10 in [10, 11], "Unsupported compute capability. Supported: 10.x, 11.x"` — interface.py:2092; also `assert nheads_kv == 1 and qhead_per_kvhead == 128` interface.py:2100. The MLA bwd kernels are `flash_bwd_mla_*_sm100.py` only. |
| **No SplitKV in backward** | `BlockInfo(..., False,  # is_split_kv` — sm90.py:731-740; scheduler args pass `1,  # num_splits` — sm90.py:531. There is no `flash_bwd_combine`. |
| **No pack_gqa in backward** | `# pack_gqa backward not yet supported in bwd` / `pack_gqa = False` — interface.py:1548-1549 |
| **GQA requires equal hd and exactly 2 WGs, and forces fp32 dK/dV + 2 extra postprocess launches** | sm90.py:115-117 (`assert self.same_hdim_kv`, `assert self.num_wg_mma == 2`); dtype rule sm90.py:193-198; `dKV_postprocess = qhead_per_kvhead > 1 and not use_dedicated_hd256_kernel` interface.py:1621; extra launches interface.py:2037-2053 |
| **Deterministic ⊕ block-sparse** | `assert not self.deterministic, "Deterministic not implemented for block-sparse backward"` — sm90.py:1896-1899 |
| **`V_in_regs` is dead on sm90 bwd** | accepted at sm90.py:70, stored at sm90.py:113, and **never read anywhere else** in the file |
| **softcap is not native** | implemented by synthesizing a `score_mod` pair — interface.py:1551-1556 |
| **`dQ_single_wg` only for 2 WGs** | `assert self.num_wg_mma == 2, "dQ_single_wg only supports 2 warp groups"` — sm90.py:141-142 |
| **hd 256 falls off a cliff on sm90** | `BwdConfig(m_block_size=64, n_block_size=64, num_stages_Q=1, num_stages_dO=1, num_stages_PdS=1, ... AtomLayoutNdKV=1 ...)` — interface.py:231-238 |

---

# 6. Simplification opportunities for fixed-shape non-causal MHA hd128

Everything below is `const_expr`-gated in FA4 and would simply not be written in a greenfield kernel:

**Drops out entirely**
1. **Varlen**: `SeqlenInfoQK` (`cu_seqlens`, `seqused`, `padded_offset_q/k`, `offset_batch_*`), `SingleTileVarlenScheduler`, `copy_utils.create_ragged_tensor_for_tma` (sm90.py:494-502), and the ragged-TMA dK/dV path. `seqlen_q`, `seqlen_k`, `m_block_max` all become compile-time constants → the m loop becomes a fixed trip count and `cutlass.range(..., unroll=1)` can be tuned/unrolled.
2. **Causal/local**: `BlockInfo.get_m_block_min_max` collapses to `(0, ceil_div(seqlen_q, tile_m))` (block_info.py:58-71); `window_size_left/right`, `AttentionMask` causal/local branches (mask.py:177-222+), the `spt` LPT scheduler (sm90.py:520-524, tile_scheduler.py:648-761), and the "signal remaining m_blocks" deadlock fix (sm90.py:1914-1926).
3. **The `process_tile == False` branch** that writes zero dK/dV (sm90.py:1418-1442) — guarded by `use_block_sparsity or is_local or is_varlen_q`, all false.
4. **GQA**: `qhead_per_kvhead_divmod`, the entire fp32 `dKaccum/dVaccum` epilogue (sm90.py:1701-1776, ~75 lines), `dK/dV_semaphore`, `dk_accum/dv_accum` allocation (interface.py:1621-1655), and **two of the three postprocess launches** (interface.py:2037-2053). MHA keeps only the direct bf16 TMA store (sm90.py:1654-1700).
5. **Block sparsity**: all of `block_sparse_utils` (`get_total_q_block_count_bwd`, `produce_block_sparse_q_loads_bwd_sm90`, `consume_block_sparse_mma_bwd_sm90`, `dQaccum_store_block_sparse_bwd_sm90`), `q_subtile_factor`, `BlockSparseTensors`.
6. **score_mod / mask_mod / softcap / aux tensors / dLSE**: `apply_score_mod` (sm90.py:1021-1062), `apply_score_mod_bwd` (sm90.py:1064-1106), `AuxData`, `FastDivmodDivisor` for seqlen, `acc_S_pre` copy (sm90.py:1511-1513, which costs a full extra 40-register accumulator copy). Also removes the `dlse` branch from preprocess.
7. **`shuffle_LSE`/`shuffle_dPsum`** (`_get_stat`, sm90.py:1451-1465) — false for hd128.
8. **`V_in_regs`** — already dead.
9. **Multi-arch dispatch** in postprocess (`arch // 10 in [8, 9, 10, 11, 12]` branches, postprocess.py:91-208, 376-490) — keep ~40 of 587 lines.

**Stays / must be handled**
- **Tail masking is still required.** `S_q = S_kv = 32760` is *not* a multiple of `tile_n=128` (`32760 = 255·128 + 120`) nor of `tile_m=80` (`409·80 + 40`); `75600` *is* a multiple of 80 but not of 128 (`590·128 + 80`). The m-tail is free (LSE `+inf` sentinel, §4), but the **K-direction tail needs the `mask_seqlen` path**. FA4 applies it on **every** m_block iteration (`mask_fn` bound once at sm90.py:1349-1361, invoked at sm90.py:1519-1520). Since one CTA == one `n_block`, you can hoist it: emit two specializations of the m loop and pick `n_block == n_blocks-1` at CTA start, dropping the per-iteration compare from ~99.6% of iterations. Cross-attn `S_kv=512` is exactly `4·128`, so **no K-mask at all** there.
- Consider `tile_m = 84` (75600 = 900·84, 32760 = 390·84) or just accept `tile_m=80` (75600 = 945·80 exactly, 32760 = 409.5·80). `tile_m=80` gives an exact fit for 75600.
- **The dQaccum fragment-order coupling** between main kernel and postprocess is a real hazard; with fixed shapes you could instead make `sdQaccum → gmem` write a row-major tile and use a trivial postprocess (or fold `*softmax_scale` + bf16 convert into a fused epilogue), at the cost of an in-smem transpose.

**The one thing that does *not* simplify away — cross-attention occupancy**
Grid.x = number of KV tiles (sm90.py:526, tile_scheduler.py:241-245). For cross-attn `S_kv = 512, tile_n = 128 → 4 n_blocks`:
- nheads=40, batch=1 → **160 CTAs**; nheads=12 → **48 CTAs**. H100 has 132 SMs and this kernel is 1 CTA/SM (226 KiB smem).
- At nheads=12 that is **36% of the machine idle for the entire cross-attention backward**, with each of the 48 CTAs serially looping 410 (or 945) m_blocks.
- Mitigations to design in from the start: (a) shrink `tile_n` for the cross-attn variant (e.g. `tile_n=64 → 8 n_blocks`, 96/384 CTAs) — note `can_implement` only requires `tile_n % 16 == 0` (sm90.py:162-163); (b) split the m loop across CTAs and use the same `cp.reduce.async.bulk.add.f32` for dK/dV that GQA already uses (sm90.py:1746-1773) — the machinery exists, it just needs a zeroed fp32 dK/dV buffer + the dK/dV postprocess; (c) fuse across heads. Self-attention is fine: `32760 → 256 n_blocks × 40 heads = 10240 CTAs`.

---

# 7. Atomicity / determinism / barrier tricks — summary

1. **gmem accumulation is `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32`** (quack/copy_utils.py:695-703), issued by a *single elected lane* of the dedicated store warp, 20 KiB per issue. This is a bulk-async hardware reduction, not per-element `atom.add` — it is far cheaper than atomics but has arbitrary inter-CTA ordering.
2. **Determinism** is opt-in and implemented as a **strict n_block ordering lock** per (batch, head, m_block) with hand-written `ld.global.acquire` spin / `red.release.global.add` release (barrier.py:9-71; use sites sm90.py:1857-1895). Deterministic mode also changes the scheduler to LPT with head swizzling (sm90.py:520-521) and switches `cp_async_bulk_wait_group(..., read=False)` so the release happens only after the gmem write is complete (sm90.py:1792, 1889).
3. **Named barriers** (named_barrier.py:28-39) do the cross-warp choreography:
   - `NamedBarrierBwd.PdS` (`num_mma_threads` = 256) sequences the P/dS smem buffer between the two MMA WGs (sm90.py:1263-1265, arrive_and_wait at 1537, 1562, 1577).
   - `dQFullWG{i}` / `dQEmptyWG{i}` with **`128 + 32` participants** (one MMA WG + the store warp) form a per-WG producer/consumer ring on `sdQaccum` (sm90.py:1594-1606 producer side, sm90.py:1851-1878 consumer side).
   - `NamedBarrierBwd.Epilogue` (256 threads) serializes the smem reuse of `sK`/`sV` for the dK/dV staging (sm90.py:1649-1651, 1686-1697).
4. **`fence_view_async_shared()` before every named barrier that publishes smem** (sm90.py:1561, 1576, 1602, 1688, 1696, 1744, 1763) — the async-proxy fence, mandatory before WGMMA or TMA reads smem written by ordinary STS/STMATRIX.
5. **PDL (programmatic dependent launch)** chains preprocess → main: preprocess `griddepcontrol_wait()` before reading O/dO (preprocess.py:320-321), `griddepcontrol_launch_dependents()` as soon as O/dO are in registers (preprocess.py:385-386), and the main kernel's loader `griddepcontrol_wait()` placed **precisely between `load_Q` and `load_LSE`** so the Q TMA overlaps the tail of preprocess (sm90.py:962-965). Both are launched with `use_pdl=True` (preprocess.py:281-286, sm90.py:621-627).
6. **K/V transaction-count piggybacking**: K and V are folded into the first Q and dO mbarrier via the FA4-local `producer_acquire(..., extra_tx_count=...)` extension (pipeline.py:304-327; use at sm90.py:958-974), avoiding two extra mbarriers.
7. **`defer_sync=True` on `pipeline_Q`, `False` on `pipeline_dO`** (sm90.py:698, 706) — only the last-created pipeline emits the barrier-init `syncthreads`.
8. Final drain: `if warp_idx == 4: cute.arch.cp_async_bulk_wait_group(0, read=True)` (sm90.py:1447-1449) and, in the store warp, `if const_expr(not self.deterministic): cute.arch.cp_async_bulk_wait_group(0, read=True)` (sm90.py:1931-1932).

## Caveats

- I read flash_bwd_sm90.py (all 1932 lines), flash_bwd_preprocess.py (465), flash_bwd_postprocess.py (587), and the _flash_attn_bwd path in interface.py (1171-2058) in full. Supporting APIs (quack/sm90_utils.py, quack/copy_utils.py, quack/layout_utils.py, barrier.py, block_info.py, named_barrier.py, tile_scheduler.py, pipeline.py) were read only at the specific definitions cited.
- The SMEM byte table in §2.5 is my arithmetic from the layout expressions at sm90.py:206-252 and 306-341; I did not execute the kernel to confirm the final allocator total. Alignment padding (buffer_align_bytes=1024) happens to be a no-op for every buffer at this config since all cosizes are already 1024-multiples.
- The register-count numbers (acc_S/dP = 40, dK/dV = 64, dQ = 40) come from applying sm90_config_search.py's model (lines 22-40, 95-98) to the hd128 non-causal config; they exclude the ~40 registers per thread for the privately-held LSE/dPsum vectors (shuffle disabled at hd128), so real pressure is higher than the 208/216 the search model reports.
- The cross-attention occupancy claim assumes 1 CTA/SM (implied by 226 KiB smem and min_blocks_per_mp=1 at sm90.py:625) and batch=1 as you specified; I did not benchmark it.
- I did not read mask.py's apply_mask in full — only the non-causal/non-local seqlen-masking branch (mask.py:177-222). The exact per-thread index arithmetic under SdP_swapAB=True (the ROW/COL swap and the r2p bitmask path) is more subtle than my one-line summary and is worth re-reading before you write your own tail mask.
