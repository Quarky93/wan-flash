# FWD_STUDY

> Source study of FA4 CuTeDSL (installed flash-attn-4 4.0.0b23) against
> nvidia-cutlass-dsl 4.6.0, 2026-07-29. File:line refs are into the installed tree.

## Summary

Read `flash_fwd_sm90.py` (1545 lines) end-to-end plus every import it depends on (`flash_fwd.py`, `softmax.py`, `block_info.py`, `tile_scheduler.py`, `named_barrier.py`, `pipeline.py`, `utils.py`, `quack/sm90_utils.py`, `quack/copy_utils.py`, `quack/layout_utils.py`) and cross-checked every API signature against the installed nvidia-cutlass-dsl 4.6.0 source.

Structure for hdim 128 bf16 on H100: **3 warpgroups** (1 producer warpgroup of which only warp 0 issues TMA + 2 MMA consumer warpgroups), 384 threads, `setmaxregister` 240/24 split, tile 128×128, 2-stage K and 2-stage V TMA pipelines + a separate 1-stage Q pipeline, **no multicast, no cluster** (`cluster_shape_mn=(1,1)`, both K/V atoms built with `num_multicast=1` and a literal `# No mcast for now` comment). `mma_pv_is_rs=True` (P stays in registers, no sP smem round-trip) and `intra_wg_overlap=True` (the QK GEMM of block *i+1* and the PV GEMM of block *i* are both in flight while the exp2 for block *i+1* is computed). Two consumer warpgroups ping-pong on named barriers (`WarpSchedulerWG1/WG2`) so their WGMMA-issue phases interleave.

Key negative finding for your spec: **the FA4 "rescale less often" trick (`rescale_threshold`) does NOT exist on sm90** — it is `SoftmaxSm100`-only (`softmax.py:243-331`, consumed at `flash_fwd_sm100.py:2045` and `flash_fwd_mla_sm100.py:2631`); `grep -c rescale_threshold flash_fwd_sm90.py` = 0. sm90 has a *different* deferral trick, `rescale_O_before_gemm`, which is gated off for hdim_v ≤ 128, i.e. off for Wan.

Key caveat for your fixed shapes: 32760 mod 128 = 120 and 75600 mod 128 = 80, so **neither S_q nor S_kv is a multiple of any legal tile_m**, and S_kv is not a multiple of any conventional tile_n. You cannot delete last-block seqlen-K masking nor Q-row predication in the epilogue/LSE store. (tile_n = 120 divides both 32760 and 75600 exactly and is GMMA-legal — a real specialization option worth benchmarking.)

## Details

# FA4 sm90 FORWARD — technique inventory from source

Base: `/workspace/wan-attn/.venv/lib/python3.12/site-packages/`
Primary file: `flash_attn/cute/flash_fwd_sm90.py` (1545 lines), class `FlashAttentionForwardSm90(FlashAttentionForwardBase)`.
Base class `FlashAttentionForwardBase` lives in `flash_attn/cute/flash_fwd.py:40`.

Concrete config assumed throughout for Wan: `dtype=bf16, head_dim=head_dim_v=128, tile_m=128, tile_n=128, num_stages=2, mma_pv_is_rs=True, intra_wg_overlap=True, is_causal=False, is_local=False, pack_gqa=False, qhead_per_kvhead=1`.

---

## 1. Warp specialization structure

### 1.1 Warpgroup count / thread count — derived from the MMA tiler, not hardcoded

`flash_fwd_sm90.py:206-217`:
```python
tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
self.num_mma_threads = tiled_mma_qk.size
self.num_threads_per_warp_group = 128
self.num_wg_mma = self.num_mma_threads // self.num_threads_per_warp_group
assert self.num_wg_mma in [1, 2, 3]
self.num_threads = self.num_threads_per_warp_group * (self.num_wg_mma + 1)
self.num_producer_threads = 32
self.num_Q_load_threads = self.num_threads_per_warp_group  # If not TMA_Q
self.num_epilogue_threads = self.num_mma_threads
self.num_mma_regs, self.num_producer_regs = {1: (256, 56), 2: (240, 24), 3: (160, 32)}[
    self.num_wg_mma
]
```
For `tile_m=128`: `atom_layout_mnk=(128//64,1,1)=(2,1,1)` ⇒ `num_wg_mma=2` ⇒ **`num_threads = 384`** (1 producer WG + 2 MMA WGs), **`num_mma_regs=240`, `num_producer_regs=24`**.

There is a register-budget override when the producer must run cp.async instead of TMA (`flash_fwd_sm90.py:229-231`):
```python
# Producer needs more registers when doing cp.async Q or KV loads
if const_expr(self.num_wg_mma == 2 and (not self.use_tma_Q or not self.use_tma_KV)):
    self.num_mma_regs, self.num_producer_regs = 224, 40
```
Irrelevant for us (always TMA), so we get the 240/24 split.

### 1.2 Producer/consumer split + `setmaxreg`

`flash_fwd_sm90.py:580-610`:
```python
if warp_idx < 4:  # Producer
    cute.arch.setmaxregister_decrease(self.num_producer_regs)
    self.load(
        mQ, mK, mV, sQ, sK, sV,
        tma_atom_Q, tma_atom_K, tma_atom_V,
        pipeline_k, pipeline_v, pipeline_q,
        gmem_tiled_copy_Q, mPageTable, blocksparse_tensors,
        block_info, SeqlenInfoCls, TileSchedulerCls,
    )
else:  # Consumer
    cute.arch.setmaxregister_increase(self.num_mma_regs)
    tidx, _, _ = cute.arch.thread_idx()
    tidx = tidx - 128
```
Note the consumer rebases `tidx` by −128 so consumer-local tidx ∈ [0, 256).

Exact installed signatures (`cutlass/cute/arch/nvvm_wrappers.py:1186,1198`):
```python
def setmaxregister_increase(reg_count: int, *, loc=None, ip=None) -> None
def setmaxregister_decrease(reg_count: int, *, loc=None, ip=None) -> None
```
(`warpgroup_reg_alloc` still exists but is `@deprecated("API is deprecated, use setmaxregister_increase instead")` at `nvvm_wrappers.py:1211`.)

Inside `load()`, the producer WG further specializes — only warp 0 of the producer WG issues TMA (`flash_fwd_sm90.py:660-667`):
```python
warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
tidx, _, _ = cute.arch.thread_idx()
# TMA: only warp 0 loads. cp_async: all warps load.
# When not use_tma_Q, all 128 producer threads participate in Q loading.
is_load_warp = warp_idx_in_wg == 0 or const_expr(not self.use_tma_KV or not self.use_tma_Q)
# KV loading restricted to warp 0 for TMA, all warps for non-TMA KV
is_kv_load_warp = warp_idx_in_wg == 0 or const_expr(not self.use_tma_KV)
```
So with TMA the effective producer is **one warp**; the other 3 producer warps fall straight through to exit.

### 1.3 Named barriers

`flash_attn/cute/named_barrier.py:6-12` (barrier 0 reserved for `__syncthreads`):
```python
class NamedBarrierFwd(enum.IntEnum):
    Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
    WarpSchedulerWG1 = enum.auto()
    WarpSchedulerWG2 = enum.auto()
    WarpSchedulerWG3 = enum.auto()
    PFull = enum.auto()
    PEmpty = enum.auto()
```
`PFull`/`PEmpty` are declared but unused in the sm90 fwd path (grep: no references in `flash_fwd_sm90.py`).

### 1.4 The "warp scheduler barrier" (consumer-WG ping-pong)

Enable predicate, `flash_fwd_sm90.py:220-224`:
```python
self.use_scheduler_barrier = (
    (self.num_wg_mma >= 2 and self.tile_hdim <= 128)
    if const_expr(self.intra_wg_overlap)
    else (self.num_wg_mma == 2)
)
```
For hdim 128 + 2 WGs + overlap ⇒ **True**.

Bootstrap (`flash_fwd_sm90.py:1479-1487`):
```python
@cute.jit
def mma_init(self):
    warp_group_idx = utils.canonical_warp_group_idx(sync=False)
    if const_expr(self.use_scheduler_barrier):
        if warp_group_idx == 1:
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1),
                number_of_threads=2 * self.num_threads_per_warp_group,
            )
```
Wait / signal (`flash_fwd_sm90.py:1524-1545`):
```python
def warp_scheduler_barrier_sync(self):
    if const_expr(self.use_scheduler_barrier):
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1)
            - 1
            + utils.canonical_warp_group_idx(sync=False),
            number_of_threads=2 * self.num_threads_per_warp_group,
        )

def warp_scheduler_barrier_arrive(self):
    if const_expr(self.use_scheduler_barrier):
        assert self.num_wg_mma in [2, 3]
        cur_wg = utils.canonical_warp_group_idx(sync=False) - 1
        if const_expr(self.num_wg_mma == 2):
            next_wg = 1 - cur_wg
        else:
            t = cur_wg + 1
            next_wg = t % self.num_wg_mma
        cute.arch.barrier_arrive(
            barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + next_wg,
            number_of_threads=2 * self.num_threads_per_warp_group,
        )
```
`canonical_warp_group_idx` is CTA-global (`utils.py:499-503`: `cute.arch.thread_idx()[0] // 128`), so producer=WG0, consumers=WG1/WG2. WG1 waits on barrier `WarpSchedulerWG1`, WG2 on `WarpSchedulerWG2`; each arrives on the *other* one. This forces the two consumer warpgroups to alternate their WGMMA-issue windows (classic FA3 "ping-pong", retained here) so that one WG's softmax/exp2 overlaps the other WG's tensor-core work.

Installed signatures (`cutlass/cute/arch/nvvm_wrappers.py:683,710`):
```python
def barrier(*, barrier_id=None, number_of_threads=None, loc=None, ip=None) -> None
def barrier_arrive(*, barrier_id=None, number_of_threads=None, aligned=True, loc=None, ip=None) -> None
```
Both are keyword-only. `aligned=False` emits `barrier.cta.arrive` for divergent subsets — worth knowing, FA4 does not use it here.

---

## 2. TMA usage

### 2.1 Copy ops and atom construction

`flash_fwd_sm90.py:260-301`:
```python
# TMA
gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
gmem_tiled_copy_KV = cpasync.CopyBulkTensorTileG2SOp()  # Might multicast
gmem_tiled_copy_O = cpasync.CopyBulkTensorTileS2GOp()
self.tma_copy_bytes = {
    name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
    for name, mX, layout in [
        ("Q", mQ, self.sQ_layout),
        ("K", mK, self.sK_layout),
        ("V", mV, self.sV_layout),
    ]
}
```
```python
tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
    gmem_tiled_copy_KV,
    mK,
    cute.select(self.sK_layout, mode=[0, 1]),
    (self.tile_n, self.tile_hdim),
    1,  # No mcast for now
)
tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
    gmem_tiled_copy_KV,
    mV,
    cute.select(self.sV_layout, mode=[0, 1]),
    (self.tile_n, self.tile_hdimv),
    1,  # No mcast for now
)
```
Q uses no `num_multicast` argument at all (`flash_fwd_sm90.py:279-284`):
```python
tma_atom_Q, tma_tensor_Q = make_tiled_tma_atom_fn(
    gmem_tiled_copy_Q,
    mQ_og if const_expr(self.pack_gqa) else mQ,
    self.sQ_layout,
    (self.tile_m, self.tile_hdim),  # No mcast
)
```
Note the **stage mode is stripped for K/V** (`cute.select(layout, mode=[0,1])`) but kept for Q (single-stage, so `sQ_layout` is already 2-D).

Exact installed signature (`cutlass/cute/nvgpu/cpasync/helpers.py:419-429`):
```python
def make_tiled_tma_atom(
    op: TMAOp,
    gmem_tensor: Tensor,
    smem_layout_: Union[Layout, ComposedLayout],
    cta_tiler: Tiler,
    num_multicast: int = 1,
    *,
    internal_type: Optional[Type[Numeric]] = None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> TmaInfo:
```
Returns a `TmaInfo` that unpacks as `(atom, tma_tensor)` — the kernel does exactly `tma_atom_K, tma_tensor_K = ...` (2-tuple unpack works even though the docstring mentions a third `smem_layout` field).

Multicast op exists but is unused here: `CopyBulkTensorTileG2SMulticastOp` at `cutlass/cute/nvgpu/cpasync/copy.py:461`. `self.cluster_shape_mn = (1, 1)` at `flash_fwd_sm90.py:69`. **There is no cluster and no multicast in the sm90 forward.** For Wan (batch 1, 12/40 heads, huge S) a 2-CTA cluster with K/V multicast is an obvious unexploited win the reference kernel leaves on the table.

### 2.2 Descriptor prefetch

`flash_fwd_sm90.py:441-446` — first thing in the kernel, before smem allocation:
```python
warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
# Prefetch tma descriptor
if warp_idx == 0:
    for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
        if const_expr(tma_atom is not None):
            cpasync.prefetch_descriptor(tma_atom)
```
Signature (`cutlass/cute/nvgpu/cpasync/helpers.py:729`): `def prefetch_descriptor(tma_atom: atom.CopyAtom, *, loc=None, ip=None) -> None`.

### 2.3 Issuing copies — the closure pattern

The kernel never calls `cute.copy` on TMA directly; it builds partial-applied closures once per work tile via `quack.copy_utils.tma_get_copy_fn` (`quack/copy_utils.py:868-913`) and wraps them with the pipeline barrier via `tma_producer_copy_fn` (`quack/copy_utils.py:1071-1080`):
```python
def tma_producer_copy_fn(copy: Callable, pipeline: cutlass.pipeline.PipelineAsync):
    def copy_fn(src_idx, producer_state: cutlass.pipeline.PipelineState, **new_kwargs):
        copy(
            src_idx=src_idx,
            dst_idx=producer_state.index,
            tma_bar_ptr=pipeline.producer_get_barrier(producer_state),
            **new_kwargs,
        )
    return copy_fn
```
Call site (`flash_fwd_sm90.py:711-721`):
```python
gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (None, 0))
gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (None, 0))
# TODO: mcast
tma_load_K_fn, _, _ = copy_utils.tma_get_copy_fn(
    tma_atom_K, 0, cute.make_layout(1), gK, sK
)
tma_load_K_fn = copy_utils.tma_producer_copy_fn(tma_load_K_fn, pipeline_k)
```
`cta_coord=0`, `cta_layout=cute.make_layout(1)` — i.e. degenerate single-CTA TMA partition. Q uses `single_stage=True` (`flash_fwd_sm90.py:687-690`) which returns a nullary `copy_tma_single_stage` closure.

Actual issue point (`flash_fwd_sm90.py:916-934`):
```python
def load_KV(self, tma_load_fn, paged_kv_manager, sX, block, pipeline_kv, producer_state, K_or_V, page_idx=None):
    if const_expr(self.use_tma_KV):
        src_idx = block if const_expr(page_idx is None) else page_idx
        tma_load_fn(src_idx=src_idx, producer_state=producer_state)
    else:
        paged_kv_manager.load_KV(block, sX[None, None, producer_state.index], K_or_V)
        cute.arch.cp_async_commit_group()
    pipeline_kv.producer_commit(producer_state)
```

Q is issued with an explicit barrier pointer because its pipeline is manually phase-tracked (`flash_fwd_sm90.py:790-794`):
```python
if const_expr(self.use_tma_Q):
    if warp_idx_in_wg == 0:
        pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
        load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
        q_producer_phase ^= 1
```

### 2.4 Pipelines and stages

`flash_fwd_sm90.py:452-513`. Three pipelines, all with `defer_sync=True`:
```python
ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
tma_warp = ThreadCooperativeGroup(1)
load_threads = ThreadCooperativeGroup(self.num_threads_per_warp_group)
mma_warps = ThreadCooperativeGroup(self.num_mma_threads // cute.arch.WARP_SIZE)
if const_expr(self.use_tma_Q):
    pipeline_q = pipeline_custom.PipelineTmaAsync.create(
        barrier_storage=mbar_ptr_Q,
        num_stages=1,
        producer_group=tma_warp,
        consumer_group=mma_warps,
        tx_count=self.tma_copy_bytes["Q"],
        defer_sync=True,
    )
```
```python
pipeline_k = pipeline_custom.PipelineTmaAsync.create(
    barrier_storage=storage.mbar_ptr_K.data_ptr(),
    num_stages=self.num_stages,
    producer_group=tma_warp,
    consumer_group=mma_warps,
    tx_count=self.tma_copy_bytes["K"],
    defer_sync=True,
)
```
(V identical with `mbar_ptr_V` / `tma_copy_bytes["V"]`.)

**`num_stages` = 2 for both K and V** — set at the call site `interface.py:870-871`:
```python
# num_stages=1,
num_stages=2,
```
So the prefetch distance is 2 K-tiles and 2 V-tiles = 4×(128×128×2 B) = 128 KB of KV smem. Plus 32 KB Q. Total ≈ 160 KB, launched with `min_blocks_per_mp=1` (`flash_fwd_sm90.py:398`) ⇒ **1 CTA/SM**.

Barrier storage is explicitly sized in the shared-storage struct (`flash_fwd_sm90.py:131-134`):
```python
# 1 stage * 2 for Q pipeline (full + empty), self.num_stages*2 for K, self.num_stages*2 for V,
mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, 1 * 2]
mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
```

Init/sync fences: `pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)` at `flash_fwd_sm90.py:516` (right after mbarrier init), and `pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)` at `:578` (just before the producer/consumer branch). With `cluster=(1,1)` these degrade to `mbarrier_init_fence()` and a plain CTA `__syncthreads` (`cutlass/pipeline/helpers.py:933-975`).

The `PipelineTmaAsync` used is FA4's own subclass (`flash_attn/cute/pipeline.py:300-330`) that adds `extra_tx_count` to `producer_acquire` and the `_w_index_phase` mixin (`pipeline.py:118-156`) letting you drive a pipeline by explicit `(index, phase)` instead of a `PipelineState` — that's what makes the single-stage Q pipeline work across scheduler iterations.

Also note FA4's cheaper pipeline state, `PipelineStateSimple` (`flash_attn/cute/pipeline.py:38-95`), which packs index+phase into one `Int32` and uses `divmod` (bit twiddling for power-of-2 stages):
```python
@property
def index(self) -> Int32:
    if const_expr(self._stages == 1):
        return Int32(0)
    else:
        return self._phase_index % self._stages
```
(The sm90 fwd main loop actually uses the stock `pipeline.make_pipeline_state` at `:671` and `:997`, but `PipelineStateSimple` is available and is what the sm100 paths use.)

### 2.5 Producer main loop with intra-WG overlap (K runs one block ahead of V)

`flash_fwd_sm90.py:833-871` — the `intra_wg_overlap and use_tma_KV` branch. K for block *n* and V for block *n+1* are acquired/issued in the same iteration, and V for `n_block_min` is issued in a tail:
```python
for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
    n_block_prev = n_block_max - i - 1
    n_block = n_block_prev - 1
    ...
    kv_producer_state_prev = kv_producer_state.clone()
    kv_producer_state.advance()
    pipeline_k.producer_acquire(kv_producer_state)
    load_K(block=n_block, producer_state=kv_producer_state, page_idx=page_idx)
    pipeline_v.producer_acquire(kv_producer_state_prev)
    load_V(block=n_block_prev, producer_state=kv_producer_state_prev, page_idx=page_idx_prev)
n_block = n_block_min
...
pipeline_v.producer_acquire(kv_producer_state)
load_V(block=n_block, producer_state=kv_producer_state, page_idx=page_idx)
kv_producer_state.advance()
```
Note the loop iterates blocks **descending** (`n_block_max-1` down to `n_block_min`).

Producer tail (`flash_fwd_sm90.py:910-914`):
```python
# Producer tail is only useful for cluster to avoid early exit of blocks.
# We only need producer_tail on V since that's the last that's loaded, we don't
# need it for Q (no cluster) and K.
if is_kv_load_warp:
    pipeline_v.producer_tail(kv_producer_state)
```

---

## 3. Online-softmax pipeline

### 3.1 Scale folding into log2(e) — done on the host

`flash_attn/cute/utils.py:185-197`:
```python
def compute_softmax_scale_log2(softmax_scale, score_mod):
    """...When score_mod is None, fold the log2(e) factor into softmax_scale_log2 and set softmax_scale
    to None. When score_mod is present, keep softmax_scale separate ..."""
    if const_expr(score_mod is None):
        return softmax_scale * LOG2_E, None
    else:
        return LOG2_E, softmax_scale
```
Called at `flash_fwd_sm90.py:348-350`. With no `score_mod` (our case) the kernel only ever sees `softmax_scale_log2 = 1/sqrt(d) * log2(e)` and **`softmax_scale is None`** — the QK accumulator is never multiplied by the raw scale anywhere; the scale rides entirely inside the `exp2` argument.

### 3.2 The per-n_block online softmax

`flash_attn/cute/softmax.py:126-190` — this is the whole sm90 hot path:
```python
acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
row_scale = cute.make_fragment_like(self.row_max, Float32)
...
for r in cutlass.range(cute.size(row_max), unroll_full=True):
    acc_S_row = acc_S_mn[r, None].load()  # (n_block_size)

    row_max_cur = utils.fmax_reduce(
        acc_S_row,
        init_val=row_max[r] if cutlass.const_expr(not is_first) else None,
        arch=arch,
    )

    row_max_cur = cute.arch.warp_reduction_max(row_max_cur, threads_in_group=4)
    # Update row_max before changing row_max_cur to safe value for -inf
    row_max_prev = row_max[r]
    row_max[r] = row_max_cur

    if cutlass.const_expr(check_inf):
        row_max_cur = 0.0 if row_max_cur == -Float32.inf else row_max_cur

    if cutlass.const_expr(is_first):
        row_max_cur_scaled = row_max_cur * scale_log2
        acc_S_row_exp = cute.math.exp2(
            acc_S_row * scale_log2 - row_max_cur_scaled, fastmath=True
        )
        acc_S_row_sum = utils.fadd_reduce(acc_S_row_exp, init_val=None, arch=arch)
        row_scale[r] = 1.0
    else:
        row_max_cur_scaled = row_max_cur * scale_log2
        acc_S_row_exp = cute.math.exp2(
            acc_S_row * scale_log2 - row_max_cur_scaled, fastmath=True
        )
        # row_scale[r] = cute.math.exp2(row_max_prev * self.scale_log2 - row_max_cur_scaled)
        row_scale[r] = cute.math.exp2(
            (row_max_prev - row_max_cur) * scale_log2, fastmath=True
        )
        acc_S_row_sum = utils.fadd_reduce(
            acc_S_row_exp, init_val=row_sum[r] * row_scale[r], arch=arch
        )

    row_sum[r] = acc_S_row_sum
    acc_S_mn[r, None].store(acc_S_row_exp)
```
Points that matter:
- **`num_rows` is 2** for our shape (`flash_fwd_sm90.py:1005`: `num_rows=acc_O.shape[0][0] * acc_O.shape[1]`; WGMMA C layout is `((2,2,N/8), MMA_M, MMA_N)` and `convert_layout_acc_mn` at `quack/layout_utils.py:168-196` maps M to `(2, MMA_M)`; for a 64×128 per-WG tile MMA_M=1). So the `unroll_full` loop is 2 iterations.
- **Row-max is quad-reduced every n_block** (`threads_in_group=4`, because 4 lanes own the 128 columns of one row), but **row-sum is NOT** — the sum's cross-lane reduction is deferred to `finalize()` exactly once per tile (`softmax.py:203-204`). This is a real win: one `warp_reduce` of width 4 per tile instead of per n_block.
- The rescale is computed as `exp2((max_prev − max_cur) * scale_log2)`, not `exp2(max_prev*s − max_cur_scaled)` — the commented-out line right above it (`softmax.py:179`) shows they deliberately switched to save one multiply.
- `check_inf` handles fully-masked rows by clamping `row_max_cur` to 0 **after** storing the true `-inf` into `row_max[r]`, so `finalize` can still emit `-inf` LSE.
- `cute.math.exp2(..., fastmath=True)` on a whole `TensorSSA` — vectorized, one `ex2.approx.ftz.f32` per element.

Reduction helpers: `utils.fmax_reduce` (`utils.py:367-391`) hand-unrolls a 4-way max tree for `arch < 100`; `utils.fadd_reduce` (`utils.py:418-424`) just uses `x.reduce(cute.ReductionOp.ADD, init_val, 0)` on sm90. The packed-f32x2 variants (`cute.arch.add_packed_f32x2`) are gated to `arch >= 100`.

### 3.3 "Rescale less often" — **NOT PRESENT ON SM90**

`grep -c rescale_threshold flash_attn/cute/flash_fwd_sm90.py` → **0**.

It exists only in `SoftmaxSm100` (`flash_attn/cute/softmax.py:243-331`):
```python
@dataclass
class SoftmaxSm100(Softmax):
    rescale_threshold: cutlass.Constexpr[float] = 0.0
    max_offset: cutlass.Constexpr[int] = 0
```
```python
# softmax.py:319-331
row_max_old = self.row_max[0]
row_max_new = self._compute_row_max(acc_S_row, init_val=row_max_old)
row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
acc_scale_ = (row_max_old - row_max_safe) * self.scale_log2
acc_scale = cute.math.exp2(acc_scale_, fastmath=True)
if cutlass.const_expr(self.rescale_threshold > 0.0):
    if acc_scale_ >= -self.rescale_threshold:
        row_max_new = row_max_old
        row_max_safe = row_max_old
        acc_scale = 1.0
self.row_max[0] = row_max_new
```
i.e. *if the new max only grew by less than `rescale_threshold` in log2 units, keep the old max and skip the O rescale entirely*. Consumers: `flash_fwd_sm100.py:2045-2052` and `flash_fwd_mla_sm100.py:2631` (`rescale_threshold=8.0` for 16-bit). The safety comment is at `flash_fwd_sm100.py:2028-2030`:
```python
# P is scaled by 2^max_offset before the FP8 conversion. With rescale_threshold > 0
# the row max can be stale by up to rescale_threshold (in log2 units), so P can reach
# 2^(max_offset + rescale_threshold). max_offset + rescale_threshold must stay within
```
**Implication for us:** this is a Blackwell trick because there the rescale is a TMEM-side cost. On sm90 the rescale is 64 FMAs/thread inside the consumer WG. Porting it to sm90 is *possible* (it's pure register math) and is a plausible original optimization for Wan — for bf16 with `rescale_threshold ≈ 8`, you'd skip most `rescale_O` calls at the cost of P values up to 2^8 (still far inside bf16 range, since bf16 max ≈ 3.4e38). Worth prototyping; it is genuinely not in the sm90 reference.

### 3.4 The sm90 analogue: `rescale_O_before_gemm` (off for hdim 128)

`flash_fwd_sm90.py:232`:
```python
self.rescale_O_before_gemm = self.tile_hdimv > 128 and self.intra_wg_overlap
```
For `tile_hdimv == 128` this is **False**. When on, it defers `rescale_O` by one iteration so it can be hidden under the QK WGMMA (`flash_fwd_sm90.py:1435-1437`):
```python
acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
# RescaleOBeforeGemm: rescale O while QK GEMM is in flight, before PV GEMM
if const_expr(self.rescale_O_before_gemm):
    softmax.rescale_O(acc_O, scores_scale)
```
and stores this iteration's scale for the next (`:1469-1472`):
```python
if const_expr(not self.rescale_O_before_gemm):
    softmax.rescale_O(acc_O, row_scale)
if const_expr(self.rescale_O_before_gemm):
    scores_scale.store(row_scale.load())
```
For Wan this whole branch compiles away.

### 3.5 Where exponentials overlap MMA — the intra-warpgroup overlap loop

`flash_fwd_sm90.py:1410-1477`, `mma_one_n_block_intrawg_overlap`. This is the single most important structure to copy:
```python
smem_pipe_read_v = smem_pipe_read.clone()
smem_pipe_read.advance()
pipeline_k.consumer_wait(smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read))
self.warp_scheduler_barrier_sync()
# S = Q @ K.T
acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
if const_expr(self.rescale_O_before_gemm):
    softmax.rescale_O(acc_O, scores_scale)
pipeline_v.consumer_wait(smem_pipe_read_v, pipeline_v.consumer_try_wait(smem_pipe_read_v))
# O += P @ V
mma_pv_fn(B_idx=smem_pipe_read_v.index, wg_wait=-1)
self.warp_scheduler_barrier_arrive()
warpgroup.wait_group(1)
pipeline_k.consumer_release(smem_pipe_read)

# handle score mods and masking
if const_expr(score_mod_fn is not None):
    score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
if const_expr(mask_fn is not None):
    mask_fn(acc_S=acc_S, n_block=n_block)

row_scale = softmax.online_softmax(acc_S, check_inf=check_inf)
warpgroup.wait_group(0)
pipeline_v.consumer_release(smem_pipe_read_v)
tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
tOrP_cur = (
    tOrP
    if const_expr(self.mma_pv_is_rs)
    else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
)
# tOrP_cur.store(tOrP_acc.load().to(self.dtype))
# the "to(self.dtype)" conversion fails to vectorize for block sizes other
# than 128 x 128, i.e. it calls convert on 1 fp32 element at a time instead of
# 2 elements. So we just call ptx directly.
utils.cvt_f16(tOrP_acc, tOrP_cur)
```
The critical ordering:
1. issue **QK GEMM for block *i*** with `wg_wait=-1` (do not wait);
2. issue **PV GEMM for block *i−1*'s P** with `wg_wait=-1`;
3. `warpgroup.wait_group(1)` — waits for the *older* group (QK) only, **leaving PV in flight**;
4. mask + `online_softmax` (all the `exp2`) **run while the PV WGMMA is executing**;
5. `warpgroup.wait_group(0)` — now drain PV;
6. convert P to bf16, rescale O.

So on sm90 the exponentials are hidden behind the *previous* block's PV GEMM, and the two consumer WGs additionally ping-pong so one WG's math hides the other's issue stalls.

`utils.cvt_f16` (`utils.py:645-674`) emits `cvt.rn.bf16x2.f32` two elements at a time via inline PTX (`utils.py:620-634`) precisely because the DSL's `.to(dtype)` fails to vectorize. **Copy this verbatim** — it's a free 2× on the P conversion.

### 3.6 Prologue and epilogue half-blocks

Because the loop is software-pipelined, the first and last iterations are peeled.

`first_half_block_overlap` (`flash_fwd_sm90.py:1269-1323`) — QK only, `wg_wait=0`, mask with `mask_seqlen=True` always, `is_first=True` softmax, no PV:
```python
pipeline_k.consumer_wait(kv_consumer_state, pipeline_k.consumer_try_wait(kv_consumer_state))
acc_S = mma_qk_fn(B_idx=kv_consumer_state.index, wg_wait=0)
pipeline_k.consumer_release(kv_consumer_state)
...
# Apply mask; mask_seqlen always True for first block
# Caveat: if full block further right than mask block, seqlen masking is redundant;
# however, masking is being applied anyway, so essentially no perf hit
mask_fn(acc_S, n_block=n_block, mask_seqlen=True)
row_scale = softmax.online_softmax(acc_S, is_first=is_first_block)
```
`last_half_block_overlap` (`flash_fwd_sm90.py:1325-1346`) — the trailing PV only:
```python
pipeline_v.consumer_wait(kv_consumer_state, pipeline_v.consumer_try_wait(kv_consumer_state))
mma_pv_fn(B_idx=kv_consumer_state.index, zero_init=zero_init, wg_wait=0)
pipeline_v.consumer_release(kv_consumer_state)
kv_consumer_state.advance()
```

### 3.7 Finalize / LSE

`softmax.py:192-227`:
```python
# quad reduction for row_sum as we didn't do it during each iteration of online softmax
row_sum.store(utils.warp_reduce(row_sum.load(), operator.add, width=4))
row_scale = cute.make_fragment_like(row_max, Float32)

for r in cutlass.range(cute.size(row_sum), unroll_full=True):
    ...
    # if row_sum is zero or nan, set acc_O_mn_row to 1.0
    acc_O_mn_row_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
    row_scale[r] = (
        cute.arch.rcp_approx(row_sum[r] if not acc_O_mn_row_is_zero_or_nan else 1.0)
    ) * final_scale
    row_sum_cur = row_sum[r]
    LN2 = math.log(2.0)
    row_sum[r] = (
        (row_max[r] * scale_log2 + cute.math.log2(row_sum_cur, fastmath=True)) * LN2
        if not acc_O_mn_row_is_zero_or_nan
        else -Float32.inf
    )
return row_scale
```
`row_sum` is destructively overwritten with the **natural-log LSE** and that same tensor is handed to the epilogue as `lse` (`flash_fwd_sm90.py:1244-1252`). `rcp_approx` is the fast reciprocal, not an IEEE divide.

---

## 4. WGMMA issue pattern

### 4.1 The two tiled MMAs

`flash_fwd_sm90.py:96-118`:
```python
tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(
    self.dtype,
    self.dtype,
    warpgroup.OperandMajorMode.K,
    warpgroup.OperandMajorMode.K,
    Float32,
    atom_layout_mnk=(self.tile_m // 64, 1, 1),
    tiler_mn=(64, self.tile_n),
)
tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(
    self.dtype,
    self.dtype,
    warpgroup.OperandMajorMode.K,
    warpgroup.OperandMajorMode.MN,
    Float32,
    atom_layout_mnk=(self.tile_m // 64, 1, 1),  # Might need (1, 2, 1) for hdim 512
    tiler_mn=(64, self.tile_hdimv),
    a_source=warpgroup.OperandSource.RMEM
    if self.mma_pv_is_rs
    else warpgroup.OperandSource.SMEM,
)
```
- **QK**: A = Q (K-major, SMEM), B = K (K-major, SMEM) ⇒ **SS** GMMA producing S = Q·Kᵀ. `atom_layout_mnk=(2,1,1)` splits the 128 M-rows across the 2 consumer WGs (64 rows each); N tiler = full `tile_n`.
- **PV**: A = P (K-major, **RMEM** when `mma_pv_is_rs=True`), B = Vᵀ (**MN-major**, SMEM) ⇒ **RS** GMMA producing O += P·V.

Exact installed signature (`cutlass/utils/hopper_helpers.py:92-104`):
```python
def make_trivial_tiled_mma(
    a_dtype: Type[Numeric],
    b_dtype: Type[Numeric],
    a_leading_mode: OperandMajorMode,
    b_leading_mode: OperandMajorMode,
    acc_dtype: Type[Numeric],
    atom_layout_mnk: Tuple[int, int, int],
    tiler_mn: Tuple[int, int],
    a_source: OperandSource = OperandSource.SMEM,
    *, loc=None, ip=None,
) -> cute.TiledMma:
```
(`quack/sm90_utils.py:45-74` wraps it as `make_tiled_mma(a_dtype, a_major, b_major, tiler_n, source="SS"|"RS", ...)` — either is fine.)

### 4.2 The V transpose is a *view*, not a copy

`flash_fwd_sm90.py:529-530`:
```python
# Transpose view of V to tensor with layout (head_dim_v, tile_n) for tiled mma
sVt = layout_utils.transpose_view(sV)
```
`quack/layout_utils.py:10-14`:
```python
def transpose_view(a: cute.Tensor) -> cute.Tensor:
    """Transpose the first two dimensions of a tensor on smem."""
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))
```
V is stored row-major `(tile_n, hdim_v)` and simply *read* as `(hdim_v, tile_n)` MN-major by the PV GMMA. **No smem transpose pass, no LDSM.T.** This is why `sV_layout_atom` is built with `LayoutEnum.ROW_MAJOR` at `flash_fwd_sm90.py:78-83` and `b_leading_mode=OperandMajorMode.MN` on the PV MMA.

### 4.3 Fragment partitioning + the gemm wrappers

`flash_fwd_sm90.py:966-982`:
```python
warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
warp_group_thread_layout = cute.make_layout(
    self.num_wg_mma, stride=self.num_threads_per_warp_group
)
thr_mma_qk = tiled_mma_qk.get_slice(tidx)
wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx))
wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx))
_, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
    wg_mma_qk, (self.tile_m, self.tile_n, self.tile_hdim), sQ, sK
)
mma_qk_fn = partial(
    sm90_utils.gemm_zero_init, tiled_mma_qk, (self.tile_m, self.tile_n), tSrQ, tSrK
)
acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(
    wg_mma_pv, (self.tile_m, self.tile_hdimv, self.tile_n), sP, sVt
)
mma_pv_fn = partial(sm90_utils.gemm_w_idx, tiled_mma_pv, acc_O, tOrP, tOrVt)
```
Note `thr_mma_qk` (per-thread slice, used for `partition_C` in masking) vs `wg_mma_*` (per-**warpgroup** slice, used for operand fragments) — two different slicings of the same tiled MMA.

`partition_fragment_ABC` (`quack/sm90_utils.py:165-193`) picks RMEM vs SMEM A automatically:
```python
is_rs = thr_mma.op.a_src == warpgroup.OperandSource.RMEM
if const_expr(not swap_AB):
    acc = cute.make_rmem_tensor(thr_mma.partition_shape_C(shape_mnk[:2]), Float32)
    if const_expr(not is_rs):
        assert sA is not None
        tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA))
    else:
        tCrA = thr_mma.make_fragment_A(thr_mma.partition_shape_A((shape_mnk[0], shape_mnk[2])))
    assert sB is not None
    tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB))
```
For PV with `mma_pv_is_rs=True`, `sP` is `None` and `tOrP` is a pure register fragment.

The GEMM itself (`quack/sm90_utils.py:97-121`):
```python
warpgroup.fence()
# We make a new mma_atom since we'll be modifying its attribute (accumulate).
# Otherwise the compiler complains "operand #0 does not dominate this use"
mma_atom = cute.make_mma_atom(tiled_mma.op)
mma_atom.set(warpgroup.Field.ACCUMULATE, not zero_init)
for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
    cute.gemm(mma_atom, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
    mma_atom.set(warpgroup.Field.ACCUMULATE, True)
warpgroup.commit_group()
if const_expr(wg_wait >= 0):
    warpgroup.wait_group(wg_wait)
```
`wg_wait=-1` means "issue and return without waiting" — that's the whole overlap mechanism. `gemm_zero_init` (`sm90_utils.py:124-143`) allocates a fresh `acc_S` per call with `zero_init=True`; `gemm_w_idx` (`:146-162`) accumulates into the persistent `acc_O` with a runtime `zero_init: Boolean` (used for the first n_block).

### 4.4 S → P handoff

`flash_fwd_sm90.py:1455`: `tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)` — a pure **layout reinterpretation** of the QK accumulator into GMMA A-operand register layout (`quack/layout_utils.py:207-251`):
```python
# For Sm90, FP16/BF16, convert acc_layout from ((2, 2, N / 8), MMA_M, MMA_N) to ((2, 2, 2), MMA_M, (N / 16, MMA_N))
```
Then `utils.cvt_f16(tOrP_acc, tOrP_cur)` writes bf16 straight into `tOrP` (the PV A-fragment). **Zero data movement between the two GEMMs** — this is the payoff of `mma_pv_is_rs=True`.

The `mma_pv_is_rs=False` alternative (dead code for us) round-trips P through smem with StMatrix (`flash_fwd_sm90.py:1466-1476`, `986-992`):
```python
smem_copy_atom_P = utils.get_smem_store_atom(self.arch.major * 10 + self.arch.minor, self.dtype)
smem_thr_copy_P = cute.make_tiled_copy_C(smem_copy_atom_P, tiled_mma_qk).get_slice(tidx)
tPsP = smem_thr_copy_P.partition_D(sP) if const_expr(sP is not None) else None
```
```python
if const_expr(not self.mma_pv_is_rs):
    # Fence and barrier to make sure smem store is visible to WGMMA
    cute.arch.fence_view_async_shared()
    cute.arch.sync_warp()  # Only need syncwarp since each warp is using its own P values for MmaPV
```

### 4.5 Non-overlap variant (for reference)

`flash_fwd_sm90.py:1348-1408`, `mma_one_n_block`: strictly QK → wait(0) → softmax → convert → rescale_O → wait V → PV → wait(0). Slower; `intra_wg_overlap=True` is the default for every sm90 config.

---

## 5. Tile sizes for head_dim 128 on sm90

Host heuristic (`flash_attn/cute/interface.py:117-157`):
```python
@dataclass(frozen=True)
class FwdConfig:
    m_block_size: int
    n_block_size: int
    mma_pv_is_rs: bool
    intra_wg_overlap: bool


def _tile_size_fwd_sm90(head_dim, head_dim_v, is_causal, is_local, sparse_block_size_q=None):
    """Return FwdConfig for SM90 forward.

    Tile sizes and flags based on tile_size_fwd_sm90 in hopper/tile_size.h, adjusted
    for the Python kernel's different register/smem tradeoffs (benchmarked on H100 SXM).
    ...
    """
    if head_dim <= 64:
        # C++: 192×192 non-causal, 192×128 causal/local.
        # Python: 192×128 RS+OL is consistently best across seqlens.
        if sparse_block_size_q is not None and sparse_block_size_q % 192 != 0:
            return FwdConfig(128, 128, True, True)
        return FwdConfig(192, 128, True, True)
    elif head_dim <= 96:
        # C++: 192×144 noRS+OL for all cases.
        # Python: RS is catastrophic with 192× tiles (~300 vs ~600 TFLOPS).
        # noRS+OL is always required. Causal: 192×128 slightly better short seqlen.
        ...
    elif head_dim <= 128:
        return FwdConfig(128, 128, True, True)
```
**⇒ hdim 128: `tile_m=128, tile_n=128, mma_pv_is_rs=True, intra_wg_overlap=True`.**

Kernel instantiation (`interface.py:858-881`):
```python
elif arch // 10 == 9:
    assert not is_split_kv, "SplitKV not supported on SM 9.0"
    fa_fwd = FlashAttentionForwardSm90(
        dtype, head_dim, head_dim_v, qhead_per_kvhead,
        is_causal=causal, is_local=local, pack_gqa=pack_gqa,
        tile_m=tile_m, tile_n=tile_n,
        # num_stages=1,
        num_stages=2,
        num_threads=num_threads,
        Q_in_regs=False,
        intra_wg_overlap=intra_wg_overlap,
        mma_pv_is_rs=mma_pv_is_rs,
        mask_mod=mask_mod, score_mod=score_mod,
        has_aux_tensors=aux_tensors is not None,
        q_subtile_factor=q_subtile_factor,
        paged_kv_non_tma=page_size not in [None, tile_n],
    )
```
Note `num_threads` passed in is **ignored** for sm90 — it's recomputed from the MMA at `flash_fwd_sm90.py:211`.

### Budget model (from the repo's own search tool)

`flash_attn/cute/sm90_config_search.py:15-18`:
```python
# H100 hardware limits
SMEM_LIMIT = 224 * 1024  # 228 KB minus ~3 KB for LSE, dPsum, mbarriers
REG_LIMITS = {2: 216, 3: 128}  # per-WG budget: 2WG=240-24, 3WG=160-32
THREADS_PER_WG = 128
```
`sm90_config_search.py:255-285`:
```python
def _check_fwd_config(hdim, hdimv, tile_n, num_wg, pv_is_rs, overlap_wg):
    reg_limit = REG_LIMITS[num_wg]
    tile_m = num_wg * 64
    ...
    regs_S = _acc_regs(tile_m, tile_n, num_wg)
    regs_O = _acc_regs(tile_m, hdimv, num_wg)
    regs_P = regs_S // 2  # bf16 = half of f32
    if overlap_wg:
        total_regs = regs_S + regs_P + regs_O
    else:
        total_regs = regs_S + regs_O
    if total_regs > reg_limit:
        return None
    # SMEM: 1 stage Q, 2 stages K/V, O overlaps Q, sP if not RS
    sQ = tile_m * hdim * 2
    sK = tile_n * hdim * 2 * 2
    sV = tile_n * hdimv * 2 * 2
    sO = tile_m * hdimv * 2
    sP = tile_m * tile_n * 2 if not pv_is_rs else 0
    smem = max(sQ, sO) + sK + sV + sP
```
Plugging in our numbers: `regs_S=64, regs_O=64, regs_P=32 ⇒ 160 / 216` registers per thread of accumulator state, and `smem = 32K + 64K + 64K = 160 KB` of 224 KB. **We have ~56 registers/thread and ~64 KB smem of headroom** — enough for `num_stages=3` on K/V (192 KB + 32 KB = 224 KB, exactly at the limit) if we want a deeper prefetch for the S=75600 case. That is a concrete tuning knob the reference kernel leaves fixed at 2.

Smem layouts (`flash_fwd_sm90.py:235-248`):
```python
self.sQ_layout, self.sK_layout, self.sV_layout, self.sO_layout = [
    sm90_utils.make_smem_layout(mX.element_type, LayoutEnum.ROW_MAJOR, shape, stage)
    for mX, shape, stage in [
        (mQ, (self.tile_m, self.tile_hdim), None),
        (mK, (self.tile_n, self.tile_hdim), self.num_stages),
        (mV, (self.tile_n, self.tile_hdimv), self.num_stages),
        (mO, (self.tile_m, self.tile_hdimv), None),
    ]
]
```
All ROW_MAJOR ⇒ `get_smem_layout_atom` picks `K_SW128` (128 elems × 16 bits = 2048 bits, divisible by 1024) per `cutlass/utils/hopper_helpers.py:207-208`. Buffers are 1024-B aligned (`flash_fwd_sm90.py:64`: `self.buffer_align_bytes = 1024`).

Shared storage struct (`flash_fwd_sm90.py:136-144`) — note the **deliberate field order** (V before Q before K):
```python
@cute.struct
class SharedStorageQKV:
    mbar_ptr_Q: mbar_ptr_Q_struct
    mbar_ptr_K: mbar_ptr_K_struct
    mbar_ptr_V: mbar_ptr_V_struct
    sV: sV_struct
    sQ: sQ_struct
    sK: sK_struct
    sP: sP_struct
```
and **sO aliases sQ** (`flash_fwd_sm90.py:534-535`):
```python
# reuse sQ's data iterator
sO = storage.sQ.get_tensor(sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype)
```
`.launch()` passes **no `smem=`** (`flash_fwd_sm90.py:394-399`) — the DSL infers dynamic smem from `SmemAllocator`; `LaunchConfig.smem` is `int | None = None` (`cutlass/base_dsl/dsl.py:1332`).

---

## 6. Tile scheduler

Selection (`flash_fwd_sm90.py:315-322`):
```python
if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
    TileScheduler = SingleTileVarlenScheduler
else:
    TileScheduler = (
        SingleTileScheduler
        if const_expr(not self.is_causal or self.is_local)
        else SingleTileLPTScheduler
    )
```
Non-varlen + non-causal ⇒ **`SingleTileScheduler`**.

Arguments (`flash_fwd_sm90.py:323-346`), note `is_persistent=False`:
```python
tile_sched_args = TileSchedulerArguments(
    cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
    cute.size(mQ.shape[2]),
    cute.size(mQ.shape[3]) if const_expr(mCuSeqlensQ is None) else cute.size(mCuSeqlensQ.shape[0] - 1),
    1,  # num_splits
    ...
    tile_shape_mn=(self.tile_m, self.tile_n),
    ...
    is_persistent=False,
    lpt=self.is_causal or self.is_local,
)
tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
```
`SingleTileScheduler` (`flash_attn/cute/tile_scheduler.py:169-269`) is **purely static — it is the CUDA grid**:
```python
@staticmethod
def get_grid_shape(params: Params, *, loc=None, ip=None) -> Tuple[Int32, Int32, Int32]:
    # TODO: this hard-codes the fact that we only use cluster = (1, 1) or (2, 1)
    assert params.cluster_shape_mn[1] == 1, "Only cluster_shape_mn[1] == 1 is supported"
    if const_expr(params.use_cluster_idx):
        grid_x = params.num_block * params.cluster_shape_mn[0]
    else:
        grid_x = cute.round_up(params.num_block, params.cluster_shape_mn[0])
    return (grid_x, params.num_head * params.num_splits, params.num_batch)
```
```python
def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
    block_idx, head_idx, batch_idx = self._blk_coord
    ...
    return WorkTileInfo((block_idx, head_idx, batch_idx, split_idx), self._is_first_block)

def initial_work_tile_info(self, *, loc=None, ip=None):
    return self.get_current_work(loc=loc, ip=ip)

def prefetch_next_work(self, *, loc=None, ip=None):
    pass

def advance_to_next_work(self, *, loc=None, ip=None):
    self._is_first_block = False
    return self.get_current_work()
```
Both `load()` and `mma()` wrap their bodies in `while work_tile.is_valid_tile:` (`flash_fwd_sm90.py:676`, `:1046`), but for this scheduler `is_valid_tile` goes false after one pass, so the loop executes exactly once. The scheduler machinery costs one `_is_first_block` flag and nothing else.

### What simplifies for Wan
- `batch=1` ⇒ `gridDim.z = 1`.
- non-causal ⇒ no LPT, no `SingleTileLPTScheduler` (which does a hilbert-ish remap over 1650 lines of `tile_scheduler.py`).
- **Grid becomes `(ceil(S_q/128), nheads, 1)`.** Self-attn 32760/12: `(256, 12, 1)` = 3072 CTAs on 132 SMs → 23.3 waves. 75600/40: `(591, 40, 1)` = 23640 CTAs → 179 waves. Quantization loss is negligible at these wave counts, so a static scheduler is right and **you can delete the entire `tile_scheduler.py` dependency**, hardcoding `m_block = blockIdx.x, head_idx = blockIdx.y`.
- Cross-attn `S_q × 512`: `(256, 12, 1)` with only 4 n_blocks each — here the *KV* pipeline depth matters far more than scheduling, and a shorter `num_stages` may be better since you can't amortize the prologue.
- The one thing worth keeping/adding: **head-swizzle or block-swizzle for L2**. `TileSchedulerArguments` has a `head_swizzle: cutlass.Constexpr[bool] = False` field (`tile_scheduler.py:165`) that the sm90 fwd never sets. With batch=1 and 12–40 heads, consecutive `blockIdx.x` CTAs share the same K/V head → they already hit L2 well; but ordering `(head, m_block)` vs `(m_block, head)` in the grid changes K/V reuse substantially. Cheap to A/B.

---

## 7. Epilogue

Shared by all archs: `flash_attn/cute/flash_fwd.py:330-449`. Entered from `flash_fwd_sm90.py:1243-1264`:
```python
# normalize acc_O by row_sum and calculate the lse
row_scale = softmax.finalize(sink_val=sink_val)
softmax.rescale_O(acc_O, row_scale)
self.epilogue(acc_O, softmax.row_sum, mO, mLSE, sO, seqlen,
              gmem_tiled_copy_O, tma_atom_O, tiled_mma_pv, tidx, m_block, head_idx, batch_idx)
```

### 7.1 O: rmem → smem via StMatrix → gmem via TMA

`flash_fwd.py:347-360`:
```python
# store acc_O
rO = cute.make_fragment_like(acc_O, self.dtype)
rO.store(acc_O.load().to(self.dtype))
# Make sure all threads have finished reading V
cute.arch.barrier(
    barrier_id=int(NamedBarrierFwd.Epilogue), number_of_threads=self.num_epilogue_threads
)
smem_copy_atom_O = utils.get_smem_store_atom(self.arch.major * 10 + self.arch.minor, self.dtype)
smem_thr_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma).get_slice(tidx)
taccOrO = smem_thr_copy_O.retile(rO)
taccOsO = smem_thr_copy_O.partition_D(sO)
# copy acc O from rmem to smem with the smem copy atom
cute.copy(smem_copy_atom_O, taccOrO, taccOsO)
```
The `Epilogue` barrier here is because **sO aliases sQ**, and Q must be fully consumed by the WGMMA before it's overwritten.

`get_smem_store_atom` (`utils.py:302-315`) selects StMatrix for 16-bit on sm90+:
```python
def get_smem_store_atom(arch, element_type, transpose=False) -> cute.CopyAtom:
    if const_expr(arch < 90 or element_type.width != 16):
        return cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), element_type,
                                   num_bits_per_copy=2 * element_type.width)
    else:
        return cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=transpose, num_matrices=4),
            element_type,
        )
```

TMA store (`flash_fwd.py:398-417`) — note the split-count barrier trick and that the **store is issued by consumer warp 4**, not the producer:
```python
if const_expr(self.use_tma_O):
    # ensure smem writes are visible to TMA
    cute.arch.fence_view_async_shared()
    cute.arch.barrier_arrive(
        barrier_id=int(NamedBarrierFwd.Epilogue),
        number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE,
    )
    gO = cute.local_tile(mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
    store_O, _, _ = copy_utils.tma_get_copy_fn(
        tma_atom_O, 0, cute.make_layout(1), sO, gO, single_stage=True
    )
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx == 4:
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue),
            number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE,
        )
        store_O()
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0, read=True)
```
Arrival count = 256 + 32: all 256 consumer threads `barrier_arrive` once, then warp 4 (32 threads, already counted once) arrives a second time when it enters `barrier`, so only warp 4 blocks and the other 224 threads flow on. `cp_async_bulk_wait_group(0, read=True)` waits for the TMA to finish *reading* smem so sO/sQ can be reused.

### 7.2 LSE writeback

`flash_fwd.py:367-390`:
```python
if const_expr(mLSE is not None):
    mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2)[None, head_idx]
    if const_expr(not self.pack_gqa):
        gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
        gLSE_expanded_layout = cute.append(
            gLSE.layout, cute.make_layout((self.tile_hdimv,), stride=(0,))
        )
        gLSE_expanded = cute.make_tensor(gLSE.iterator, gLSE_expanded_layout)
        thr_mma = tiled_mma.get_slice(tidx)
        taccOgLSE = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(gLSE_expanded))
        assert cute.size(taccOgLSE, mode=[0]) == cute.size(lse)
        taccOcO = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cO))
        t0accOcO = layout_utils.reshape_acc_to_mn(thr_mma.get_slice(0).partition_C(cO))
        # Only the thread corresponding to column 0 writes out the lse to gmem
        if taccOcO[0][1] == 0:
            for m in cutlass.range(cute.size(taccOgLSE.shape[1]), unroll_full=True):
                if (
                    t0accOcO[m, 0][0]
                    < seqlen.seqlen_q - m_block * self.tile_m - taccOcO[0][0]
                ):
                    taccOgLSE[m, 0] = lse[m]
```
Elegant trick: the LSE gmem tensor is given a **stride-0 fake second mode** of length `tile_hdimv` so it can be partitioned by the *same* `thr_mma.partition_C` as O; then only lanes owning column 0 store. The row predicate uses `t0accOcO` (thread-0's coords, compile-time constant) with the limit shifted by this thread's row offset — same trick as in `load_Q` (`flash_fwd.py:470-473`):
```python
# Instead of using tQcQ, we using t0QcQ and subtract the offset from the limit
# (seqlen - block * kBlockM). This is because the entries of t0QcQ are known at compile time.
```

**Wan needs this predicate**: `32760 mod 128 = 120`, `75600 mod 128 = 80` — the last m_block is always partial.

---

## 8. New-in-FA4 vs classic FA3 (identified from structure/comments)

1. **RS PV GEMM is now the default at hdim ≤ 128** (`mma_pv_is_rs=True`, `interface.py:151`). Classic FA3 kept P in smem for the PV GMMA on many configs; here P never leaves registers, killing the StMatrix store + `fence_view_async_shared` + `sync_warp` per n_block. Note the counter-datapoint they recorded (`interface.py:141-143`): *"Python: RS is catastrophic with 192× tiles (~300 vs ~600 TFLOPS)"* — RS is a register-pressure knife-edge.
2. **Explicit `intra_wg_overlap` as a first-class flag** (`flash_fwd_sm90.py:56, 1014-1017`), selecting between two entirely separate mainloop bodies (`mma_one_n_block` vs `mma_one_n_block_intrawg_overlap`) plus peeled prologue/epilogue half-blocks. FA3's C++ had this as a template bool too, but here the peeled `first_half_block_overlap`/`last_half_block_overlap` are cleanly factored and reusable.
3. **Deferred row-sum warp reduction** (`softmax.py:203-204` comment: *"quad reduction for row_sum as we didn't do it during each iteration of online softmax"*) — the width-4 shuffle for the sum happens once per tile instead of once per n_block.
4. **Hand-written packed bf16 conversion** because the DSL mis-vectorizes (`flash_fwd_sm90.py:1461-1465` / `utils.py:620-674`). Direct `cvt.rn.bf16x2.f32` PTX.
5. **`rescale_O_before_gemm`** (`flash_fwd_sm90.py:232`) — hoists the O rescale one iteration earlier so it lands in the WGMMA shadow. New relative to FA3, but gated to hdim_v > 128.
6. **`PipelineStateSimple`** (`flash_attn/cute/pipeline.py:38-95`) — single-Int32 index+phase.
7. **`_w_index_phase` pipeline mixin** (`flash_attn/cute/pipeline.py:118-156`) letting a 1-stage pipeline be driven by an explicit XOR'd phase across scheduler iterations — that's how Q gets its own tiny pipeline separate from KV.
8. **`elect_one_release` / `syncwarp_before_release` on pipelines** (`pipeline.py:211-231`) with the comment: *"Set syncwarp to False when threads are already converged (e.g. after wgmma wait_group)"* — saves a `bar.warp.sync` per release.
9. **`defer_sync=True` on all pipeline creates** + a single `pipeline_init_arrive` / `pipeline_init_wait` pair (`flash_fwd_sm90.py:516, 578`) instead of a fence per pipeline.
10. **FlexAttention `score_mod` / `mask_mod`** (`softmax.py:466-591`, `mask.py:224-280`) with per-callable `__vec_size__` and `FastDivmodDivisor`-based index wrapping. Entirely new vs FA3.
11. **Block sparsity** (`block_sparse_utils.py`, 1947 lines) with `produce_block_sparse_loads` / `consume_block_sparse_loads` hooks in both `load()` and `mma()` (`flash_fwd_sm90.py:872-903`, `1193-1218`).
12. **Learnable attention sinks** (`flash_fwd_sm90.py:1230-1244`, `softmax.py:208-213`) — an extra `exp2(sink*log2e − row_max*scale_log2)` added to `row_sum` at finalize.
13. **`assume_tensor_aligned`** (`cute_dsl_utils.py:55-59`) applied to Q/K/V/O before anything else — rebuilds the layout with 128-bit-aligned stride assumptions so the compiler can vectorize.
14. **`layout_utils.select` transposes** (`flash_fwd_sm90.py:195-204`) — the (b, s, h, d) torch layout is permuted to (s, d, h, b) *logically* on the host so all kernel indexing is `[seq, hdim, head, batch]`. Free.
15. **`min_blocks_per_mp=1`** on launch (`flash_fwd_sm90.py:398`) — emits `nvvm.minctasm`, forcing the 160 KB smem allocation to be legal.
16. **Descriptor prefetch for all four atoms including O** before smem allocation (`flash_fwd_sm90.py:443-446`).

---

## 9. Deletion candidates — our specialization surface

Everything below is reachable in `flash_fwd_sm90.py` and can be deleted for Wan (bf16, hdim 128, MHA, non-causal, dense, batch 1, no mask/dropout/softcap):

| Feature | Where | Notes |
|---|---|---|
| **varlen** (`mCuSeqlensQ/K`, `mSeqUsedQ/K`) | `:166-169, 192, 195-204, 305-308, 315-316, 547-566`; `SeqlenInfoQK` (`seqlen_info.py:66-304`); `SingleTileVarlenScheduler` (`tile_scheduler.py:788-1090`); `copy_utils.create_ragged_tensor_for_tma` | `SeqlenInfoQK` collapses to two constants. |
| **causal / local / sliding window** | `:171-172, 351-352, 537-546, 567-574, 1138-1180`; all of `BlockInfo` (`block_info.py`); `SingleTileLPTScheduler` (`tile_scheduler.py:393-660`) | `n_block_min=0`, `n_block_max=ceil(S_kv/tile_n)` become constants. Kills 3 of the 4 mainloop phases. |
| **GQA / pack_gqa** | `:226-227, 253-258, 272-276, 681-683, 759-763, 796-799, 1232-1241`; `pack_gqa.py` (263 lines); `make_packgqa_tiled_tma_atom` | `qhead_per_kvhead=1` ⇒ `head_idx_kv = head_idx`. |
| **paged KV** | `:170, 65-68, 287, 692-740, 778-782, 787-788, 813-819, 837-846, 928-932`; `paged_kv.py` (247 lines); `PipelineCpAsync` branches at `:496-513` | Deletes the entire non-TMA producer path. |
| **softcap** | `interface.py:609-611` (`utils.create_softcap_scoremod`) → becomes a `score_mod` | Deleting `score_mod` deletes softcap. |
| **`score_mod` / `mask_mod` (FlexAttention)** | `:1083-1094, 1489-1522`; `softmax.py:19-89, 454-711`; `mask.py:224-330`; `aux_data`, `fastdiv_mods` (`:353-355, 1053-1069`) | Also lets `compute_softmax_scale_log2` return `(scale*LOG2_E, None)` unconditionally. |
| **block sparsity** | `:30-34, 174, 218, 872-903, 1110, 1193-1218`, `q_subtile_factor`; `block_sparse_utils.py` (1947), `block_sparsity.py` (726), `compute_block_sparsity.py` (551) | ~3200 lines. |
| **learnable sink** | `:173, 620, 1230-1244`; `softmax.py:194, 197-198, 207-213` | |
| **split-KV / appends / newK** | `:342 (num_splits=1)`, `interface.py:859` `assert not is_split_kv`; `block_info.py:73-102` (`get_n_block_k_new_min_max`); `flash_fwd_combine.py` (698) | Already unsupported on sm90 fwd; just don't write it. |
| **fp8 / descale** | `interface.py:808-822`; `benchmark_flash_attention_fp8.py` | |
| **`Q_in_regs`** | `:99, 120-155 (SharedStorageSharedQV), 523-528` | Always False on sm90. |
| **`mma_pv_is_rs=False` path** | `:85-93, 129-130, 244-248, 531-533, 986-992, 1311-1316, 1394-1401, 1466-1476` | sP smem buffer + StMatrix P store + fence/syncwarp. |
| **`intra_wg_overlap=False` path** | `:1348-1408` (`mma_one_n_block`), `:1124-1134`, `:1190-1191` | |
| **`rescale_O_before_gemm` path** | `:232, 1010-1012, 1318-1321, 1338-1340, 1436-1437, 1469-1472` | Dead at hdim_v=128. |
| **hdim OOB predication** | `flash_fwd.py:88-89` (`check_hdim_oob`, `check_hdim_v_oob`); `utils.predicate_k`; `:433, 445, 469-479, 517, 525, 552-576` | 128 % 16 == 0 ⇒ both already False, but the code paths still exist. |
| **non-TMA O epilogue** | `flash_fwd.py:418-449` | `use_tma_O` always True on sm90. |
| **`num_wg_mma ∈ {1,3}`** | `:215-217, 220-224, 1533-1545` | Only 2 for hdim 128. `warp_scheduler_barrier_arrive` collapses to `next_wg = 1 - cur_wg`. |
| **dropout** | not present anywhere in the CuTeDSL kernels | Nothing to delete. |
| **tile scheduler abstraction** | `tile_scheduler.py` (1650 lines) | Replace with `m_block, head_idx = blockIdx.x, blockIdx.y`. |

**Net:** for the fwd you keep roughly `_get_tiled_mma` + smem layouts + 3 TMA atoms + 3 pipelines + `mma_one_n_block_intrawg_overlap` + `first/last_half_block_overlap` + `Softmax.online_softmax/finalize/rescale_O` + the epilogue. Call it 400–500 lines vs the current 1545 + ~6000 lines of dependencies.

---

## 10. Caveats and things to double-check before you commit to a design

1. **Neither Wan seqlen is tile-aligned.** `32760 = 255·128 + 120`, `75600 = 590·128 + 80`. So:
   - You **cannot** delete last-n_block seqlen-K masking. Keep the `not mask_causal and not mask_local and mask_mod is None` branch of `AttentionMask.apply_mask` (`mask.py:211-222`) — it's ~10 lines and hoists the `oob` test out of the row loop.
   - You **cannot** delete Q-row predication in the epilogue O store and LSE store (`flash_fwd.py:382-388`).
   - You *can* delete every other masking branch.
   - **`tile_n = 120` divides both 32760 and 75600 exactly** and satisfies the GMMA N%8==0 constraint. That would make every n_block full, deleting masking entirely and removing the ragged-tail bubble on a 255-iteration loop. Costs 6.25% of the N tile vs 128. Worth measuring; `_tile_size_fwd_sm90` already uses non-power-of-2 tile_n elsewhere (`144`, `112`, `80`).
   - Cross-attention `S_kv = 512 = 4·128` is exact, so the cross-attn kernel variant genuinely needs no K masking.
2. **`num_stages=2` is not tuned for your seqlens.** At 160 KB you have room for `num_stages=3` (224 KB, exactly at `SMEM_LIMIT`) which would drop to 1 CTA/SM anyway (already the case). With 255–590 n_block iterations the steady state dominates, so deeper prefetch is plausibly a win — measure.
3. **No cluster / no multicast is a real gap.** `cluster_shape_mn=(1,1)` (`:69`), `num_multicast=1` with `# No mcast for now` (`:293, 300`) and `# TODO: mcast` (`:713`). With batch=1 and 12–40 heads, a 2-CTA cluster over adjacent m_blocks sharing identical K/V would halve K/V DRAM traffic. `CopyBulkTensorTileG2SMulticastOp` (`cutlass/cute/nvgpu/cpasync/copy.py:461`) and `PipelineTmaAsync.create(..., cta_layout_vmnk=..., mcast_mode_mn=..., enable_multicast_signaling=...)` (`cutlass/pipeline/sm90.py:529-542`) are all present in this DSL version. This is probably the single highest-leverage original optimization available for Wan self-attention.
4. **Port `rescale_threshold` from `SoftmaxSm100` to sm90 yourself.** It's absent from the sm90 reference (§3.3) but is pure register math with no arch dependency. At bf16 with 255–590 iterations, skipping most `rescale_O` calls (2 rows × 64 f32 FMAs each per block) is measurable.
5. **API gotchas confirmed in this exact DSL build:**
   - `cute.arch.barrier` / `barrier_arrive` are **keyword-only** (`barrier_id=`, `number_of_threads=`).
   - `cute.arch.warp_reduction_max(val, threads_in_group=4)` — `threads_in_group` is keyword-only (`nvvm_wrappers.py:640-647`); `warp_reduction_max` is a `functools.partial` binding `op` (`:675-678`).
   - `cutlass.pipeline.PipelineTmaAsync.create` is **keyword-only** except nothing; all args including `num_stages`, `producer_group`, `consumer_group`, `tx_count` must be passed by keyword (`cutlass/pipeline/sm90.py:529-542`).
   - `PipelineCpAsync.create` is **positional-first** (`barrier_storage, num_stages, producer_group, consumer_group, ...`) — inconsistent with the TMA one (`cutlass/pipeline/sm90.py:396-405`).
   - `make_tiled_tma_atom` returns a `TmaInfo` that unpacks as a 2-tuple in practice.
   - Use `cute.make_rmem_tensor(layout_or_shape, dtype)` / `cute.make_rmem_tensor_like(src, dtype=None)` (`cutlass/cute/tensor.py:863, 927`); `make_fragment_like` still exists (`:1017-1041`) and both are used interchangeably in FA4.
   - `.launch()` takes `grid=`, `block=`, `stream=`, optional `smem=` (int|None, auto-inferred), `min_blocks_per_mp=` (`cutlass/base_dsl/dsl.py:1325-1346`).
   - `warpgroup.fence()`, `warpgroup.commit_group()`, `warpgroup.wait_group(n)` at `cutlass/cute/nvgpu/warpgroup/helpers.py:94, 104, 114`.
6. **Two kernels, not one.** Self-attn (S×S, 255–590 n_blocks) and cross-attn (S×512, 4 n_blocks) have opposite prologue-amortization characteristics. The cross-attn case is prologue/epilogue-bound: with only 4 n_blocks, `num_stages=2` means half the loop is pipeline fill. Consider `num_stages=4` (S_kv=512 fully resident: 4×128×128×2×2 = 256 KB — too much; but `tile_n=128, stages=4` for K only + V 2-stage, or `tile_n=256`) or just accept a separate tuning. Do not assume one config serves both.
7. **`num_threads` passed to `FlashAttentionForwardSm90.__init__` is dead** — it's overwritten at `:211` from the MMA tiler. Don't replicate that footgun.
