"""Greenfield FlashAttention FORWARD for Hopper sm90 in CuTeDSL, specialized
for Wan2.1 (bf16, head_dim 128, MHA, non-causal, dense, fixed shapes).

Architecture (see docs/FWD_STUDY.md):
  - 384 threads: 1 producer warpgroup (only warp 0 issues TMA, setmaxregister 24)
    + 2 consumer MMA warpgroups (setmaxregister 240).
  - K/V TMA pipelines with num_stages stages; separate 1-stage Q pipeline.
  - Mainloop iterates KV blocks DESCENDING so the ragged tail block (the only
    one needing seqlen-K masking) is processed first, merged with the
    is_first online-softmax step; the remaining blocks are clean.
  - Online softmax in exp2 domain (softmax scale folded into scale_log2),
    fp32 accumulators, rescale-O per block (FA3 behavior) by default.
  - Feature `rescale_skip_threshold` (port of FA4's sm100 trick to sm90):
    when the running max moves by less than threshold (in log2 units), keep
    the old max so the O-rescale becomes *= 1.0, and skip the FMA loop when
    the whole warp agrees (vote.all). Exact w.r.t. max-shift invariance.
  - Feature `tile_n=120`: divides both Wan self-attn S exactly (no masked
    block at all). PV contraction (K = tile_n) is padded to a multiple of 16
    with zeroed smem V rows + zeroed P register slots.
  - Epilogue: O through smem (StMatrix) + TMA store; LSE (natural log, fp32)
    stored with m-tail predication.

Everything static: shapes are baked per compile (see interface.py cache).
"""

import math
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, BFloat16, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from quack import sm90_utils, copy_utils, layout_utils

LOG2_E = math.log2(math.e)
LN_2 = math.log(2.0)


@dsl_user_op
def _cvt_f32x2_bf16x2(a, b, *, loc=None, ip=None) -> cutlass.Int32:
    """Pack two fp32 into one bf16x2 register (cvt.rn.bf16x2.f32 d, hi, lo)."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _cvt_bf16_frag(src: cute.Tensor, dst: cute.Tensor):
    """fp32 fragment -> bf16 fragment via packed converts (the FA3/FA4 idiom;
    TensorSSA .to() can emit one scalar cvt per element)."""
    dst_i32 = cute.recast_tensor(dst, cutlass.Int32)
    for i in cutlass.range_constexpr(cute.size(dst_i32)):
        dst_i32[i] = _cvt_f32x2_bf16x2(src[2 * i], src[2 * i + 1])

# named barriers (0 is reserved for sync_threads)
BAR_EPILOGUE = 1
BAR_WG_SCHED1 = 2  # consumer-warpgroup ping-pong (intra_wg_overlap builds)
BAR_WG_SCHED2 = 3
BAR_O_FREE = 4  # persistent scheduler: sO drained by TMA, safe to overwrite
SM_COUNT = int(__import__("os").environ.get("WAN_FLASH_SM_COUNT", "132"))  # H100 SXM


class WanFlashFwdSm90:
    def __init__(
        self,
        head_dim: int = 128,
        tile_m: int = 128,
        tile_n: int = 128,
        num_stages: int = 2,
        rescale_skip_threshold: float = 0.0,
        intra_wg_overlap: bool = True,
        mma_pv_is_rs: bool = True,
        scheduler: str = "single",
        cluster_mn: tuple = (1, 1),
    ):
        assert scheduler in ("single", "persistent")
        assert head_dim % 16 == 0 and head_dim <= 256
        assert tile_m % 64 == 0
        assert tile_n % 8 == 0 and tile_n <= 256, "GMMA N must be mult of 8, <= 256"
        assert mma_pv_is_rs, "only the RS PV path is implemented"
        assert tuple(cluster_mn) in ((1, 1), (2, 1)), "cluster: (1,1) or (2,1) only"
        self.cluster_m = int(cluster_mn[0])
        # cluster pairs are m-blocks of the same head sharing the K/V multicast.
        # The persistent scheduler is cluster-aware (phantom-tile tail); the
        # single scheduler needs m_blocks % cluster_m == 0 (checked at launch,
        # grid x must divide by the cluster).
        self.head_dim = head_dim
        self.tile_m = tile_m
        self.tile_n = tile_n
        # PV contraction dim (= tile_n) must tile by the k16 GMMA atom; pad V
        # rows / P slots with zeros when it does not (tile_n=120 case).
        self.tile_n_pad = ((tile_n + 15) // 16) * 16
        self.num_stages = num_stages
        self.rescale_skip_threshold = float(rescale_skip_threshold)
        self.intra_wg_overlap = intra_wg_overlap
        self.dtype = BFloat16
        self.num_wg_mma = tile_m // 64  # 2 for tile_m=128
        self.num_mma_threads = 128 * self.num_wg_mma
        self.num_threads = 128 * (self.num_wg_mma + 1)
        self.num_mma_regs, self.num_producer_regs = {1: (256, 56), 2: (240, 24), 3: (160, 32)}[
            self.num_wg_mma
        ]
        self.scale_log2 = (1.0 / math.sqrt(head_dim)) * LOG2_E
        self.use_scheduler_barrier = self.num_wg_mma == 2 and self.intra_wg_overlap
        self.persistent = scheduler == "persistent"

    # ------------------------------------------------------------- helpers
    def _make_storage(self, sQ_layout, sK_layout, sV_layout, sO_layout):
        dtype = self.dtype
        stages = self.num_stages
        Aligned = lambda l: cute.struct.Align[
            cute.struct.MemRange[dtype, cute.cosize(l)], 1024
        ]

        if self.persistent:
            # separate sO: next tile's Q load must not race the O TMA drain
            @cute.struct
            class SharedStorageP:
                mbar_Q: cute.struct.MemRange[cutlass.Int64, 1 * 2]
                mbar_K: cute.struct.MemRange[cutlass.Int64, stages * 2]
                mbar_V: cute.struct.MemRange[cutlass.Int64, stages * 2]
                sV: Aligned(sV_layout)
                sQ: Aligned(sQ_layout)
                sK: Aligned(sK_layout)
                sO: Aligned(sO_layout)

            return SharedStorageP

        @cute.struct
        class SharedStorage:
            mbar_Q: cute.struct.MemRange[cutlass.Int64, 1 * 2]
            mbar_K: cute.struct.MemRange[cutlass.Int64, stages * 2]
            mbar_V: cute.struct.MemRange[cutlass.Int64, stages * 2]
            sV: Aligned(sV_layout)
            sQ: Aligned(sQ_layout)
            sK: Aligned(sK_layout)

        return SharedStorage

    def _sV_layout_padded(self):
        """Staged smem layout for V with tile_n_pad rows per stage.

        The TMA box is (tile_n, head_dim); the extra (tile_n_pad - tile_n)
        rows sit at the tail of each stage and are zero-filled once at kernel
        start. Row padding keeps whole swizzle atoms (8 rows x 128B), so the
        pad region is a contiguous address range per stage.
        """
        atom = warpgroup.make_smem_layout_atom(
            cutlass.utils.hopper_helpers.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR, self.dtype, self.head_dim
            ),
            self.dtype,
        )
        box = cute.tile_to_shape(atom, (self.tile_n, self.head_dim), order=(0, 1))
        padded = cute.tile_to_shape(
            atom, (self.tile_n_pad, self.head_dim), order=(0, 1)
        )
        stage_elems = cute.cosize(padded)
        staged = cute.make_composed_layout(
            box.inner,
            0,
            cute.append(box.outer, cute.make_layout(self.num_stages, stride=stage_elems)),
        )
        return staged, stage_elems

    # ---------------------------------------------------------------- host
    @cute.jit
    def __call__(self, mQ, mK, mV, mO, mLSE, stream: cuda.CUstream):
        # (b, s, h, d) -> (s, d, h, b): all kernel indexing is [seq, hdim, head, batch]
        mQ, mK, mV, mO = [layout_utils.select(t, [1, 3, 2, 0]) for t in (mQ, mK, mV, mO)]
        mLSE = layout_utils.select(mLSE, [2, 1, 0])  # (b, h, s) -> (s, h, b)

        sQ_layout = sm90_utils.make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_m, self.head_dim), None
        )
        sK_layout = sm90_utils.make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.head_dim), self.num_stages
        )
        if const_expr(self.tile_n_pad == self.tile_n):
            sV_layout = sm90_utils.make_smem_layout(
                self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.head_dim), self.num_stages
            )
            sV_stage_elems = 0
        else:
            sV_layout, sV_stage_elems = self._sV_layout_padded()
        sO_layout = sm90_utils.make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_m, self.head_dim), None
        )

        # WGMMAs: S = Q @ K^T (SS, both K-major); O += P @ V (RS, B is MN-major V^T view)
        tiled_mma_qk = cutlass.utils.hopper_helpers.make_trivial_tiled_mma(
            self.dtype, self.dtype,
            warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.num_wg_mma, 1, 1),
            tiler_mn=(64, self.tile_n),
        )
        tiled_mma_pv = cutlass.utils.hopper_helpers.make_trivial_tiled_mma(
            self.dtype, self.dtype,
            warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(self.num_wg_mma, 1, 1),
            tiler_mn=(64, self.head_dim),
            a_source=warpgroup.OperandSource.RMEM,
        )

        tma_copy_bytes_Q = cute.size_in_bytes(self.dtype, cute.select(sQ_layout, mode=[0, 1]))
        tma_copy_bytes_K = cute.size_in_bytes(
            self.dtype, cute.select(sK_layout, mode=[0, 1])
        )
        # V tx counts the (tile_n, head_dim) box actually copied, not the pad
        tma_copy_bytes_V = self.tile_n * self.head_dim * self.dtype.width // 8

        op_g2s = cpasync.CopyBulkTensorTileG2SOp()
        # K/V multicast across the cluster-m CTAs (each loads 1/cluster_m of
        # the box, hardware fans it out to every CTA in the mcast mask)
        op_g2s_kv = (
            cpasync.CopyBulkTensorTileG2SMulticastOp()
            if const_expr(self.cluster_m > 1)
            else op_g2s
        )
        op_s2g = cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            op_g2s, mQ, sQ_layout, (self.tile_m, self.head_dim)
        )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            op_g2s_kv, mK, cute.select(sK_layout, mode=[0, 1]),
            (self.tile_n, self.head_dim), self.cluster_m,
        )
        # V TMA uses the unpadded (tile_n, head_dim) box layout
        sV_box_layout = cute.select(sV_layout, mode=[0, 1])
        if const_expr(self.tile_n_pad != self.tile_n):
            atomV = warpgroup.make_smem_layout_atom(
                cutlass.utils.hopper_helpers.get_smem_layout_atom(
                    LayoutEnum.ROW_MAJOR, self.dtype, self.head_dim
                ),
                self.dtype,
            )
            sV_box_layout = cute.tile_to_shape(
                atomV, (self.tile_n, self.head_dim), order=(0, 1)
            )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            op_g2s_kv, mV, sV_box_layout, (self.tile_n, self.head_dim), self.cluster_m
        )
        tma_atom_O, tma_tensor_O = cpasync.make_tiled_tma_atom(
            op_s2g, mO, sO_layout, (self.tile_m, self.head_dim)
        )

        SharedStorage = self._make_storage(sQ_layout, sK_layout, sV_layout, sO_layout)

        m_blocks = cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m)
        if const_expr(self.persistent):
            # cluster-aware: the schedule unit is a cluster_m-pack of adjacent
            # m-blocks of the same head (all CTAs of a cluster share K/V)
            m_units = cute.ceil_div(m_blocks, self.cluster_m)
            total_units = m_units * cute.size(mQ.shape[2]) * cute.size(mQ.shape[3])
            num_clusters = min(total_units, SM_COUNT // self.cluster_m)
            grid = (num_clusters * self.cluster_m, 1, 1)
        else:
            # x-adjacent CTA pairs form the clusters; grid padded up to whole
            # clusters (the pad CTA recomputes its peer's m-block, skips stores)
            m_units = cute.ceil_div(m_blocks, self.cluster_m)
            grid = (
                m_units * self.cluster_m,
                cute.size(mQ.shape[2]),
                cute.size(mQ.shape[3]),
            )
        kernel = self.kernel(
            tma_tensor_Q, tma_tensor_K, tma_tensor_V, tma_tensor_O, mLSE,
            tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O,
            sQ_layout, sK_layout, sV_layout, sO_layout,
            tiled_mma_qk, tiled_mma_pv,
            tma_copy_bytes_Q, tma_copy_bytes_K, tma_copy_bytes_V,
            sV_stage_elems,
            SharedStorage,
        )
        if const_expr(self.cluster_m > 1):
            kernel.launch(
                grid=grid,
                block=[self.num_threads, 1, 1],
                cluster=(self.cluster_m, 1, 1),
                stream=stream,
                min_blocks_per_mp=1,
            )
        else:
            kernel.launch(
                grid=grid,
                block=[self.num_threads, 1, 1],
                stream=stream,
                min_blocks_per_mp=1,
            )

    # -------------------------------------------------------------- device
    @cute.jit
    def _fmax_tree(self, x, init_val=None):
        """4-wide parallel max tree over a TensorSSA row (FA3/FA4 idiom:
        x.reduce(MAX) emits a serial chain; this gives ILP 4)."""
        res = cute.make_rmem_tensor(x.shape, Float32)
        res.store(x)
        m0, m1, m2, m3 = res[0], res[1], res[2], res[3]
        for i in cutlass.range_constexpr(4, cute.size(x.shape), 4):
            m0 = cute.arch.fmax(m0, res[i + 0])
            m1 = cute.arch.fmax(m1, res[i + 1])
            m2 = cute.arch.fmax(m2, res[i + 2])
            m3 = cute.arch.fmax(m3, res[i + 3])
        m0 = cute.arch.fmax(m0, m1)
        m2 = cute.arch.fmax(m2, m3)
        m0 = cute.arch.fmax(m0, m2)
        if const_expr(init_val is not None):
            m0 = cute.arch.fmax(m0, init_val)
        return m0

    @cute.jit
    def softmax_step(
        self,
        acc_S: cute.Tensor,
        row_max: cute.Tensor,
        row_sum: cute.Tensor,
        row_scale: cute.Tensor,
        is_first: cutlass.Constexpr[bool],
    ):
        """Online softmax over one S tile. acc_S is overwritten with
        exp2(S*scale_log2 - rowmax*scale_log2); row_scale gets the O-rescale
        factor for this block (1.0 on the first block).
        row_sum stays a per-lane partial (quad reduction deferred to finalize).
        """
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        scale_log2 = self.scale_log2
        for r in cutlass.range_constexpr(cute.size(row_max)):
            acc_S_row = acc_S_mn[r, None].load()
            if const_expr(is_first):
                row_max_cur = self._fmax_tree(acc_S_row)
                row_max_cur = cute.arch.warp_reduction_max(row_max_cur, threads_in_group=4)
                row_max[r] = row_max_cur
                row_max_scaled = row_max_cur * scale_log2
                acc_S_row_exp = cute.math.exp2(
                    acc_S_row * scale_log2 - row_max_scaled, fastmath=True
                )
                row_sum[r] = acc_S_row_exp.reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
                row_scale[r] = 1.0
            else:
                row_max_prev = row_max[r]
                row_max_cur = self._fmax_tree(acc_S_row, init_val=row_max_prev)
                row_max_cur = cute.arch.warp_reduction_max(row_max_cur, threads_in_group=4)
                acc_scale_log2 = (row_max_prev - row_max_cur) * scale_log2  # <= 0
                acc_scale = cute.math.exp2(acc_scale_log2, fastmath=True)
                if const_expr(self.rescale_skip_threshold > 0.0):
                    # sm100 trick ported: if the max moved by < threshold in
                    # log2 units, keep the stale max; O-rescale becomes 1.0.
                    if acc_scale_log2 >= -self.rescale_skip_threshold:
                        row_max_cur = row_max_prev
                        acc_scale = Float32(1.0)
                row_max[r] = row_max_cur
                row_max_scaled = row_max_cur * scale_log2
                acc_S_row_exp = cute.math.exp2(
                    acc_S_row * scale_log2 - row_max_scaled, fastmath=True
                )
                row_sum[r] = acc_S_row_exp.reduce(
                    cute.ReductionOp.ADD, row_sum[r] * acc_scale, 0
                )
                row_scale[r] = acc_scale
            acc_S_mn[r, None].store(acc_S_row_exp)

    @cute.jit
    def rescale_O(self, acc_O: cute.Tensor, row_scale: cute.Tensor):
        acc_O_mn = layout_utils.reshape_acc_to_mn(acc_O)
        if const_expr(self.rescale_skip_threshold > 0.0):
            # scales are exactly 1.0 when skipped; skip the FMA loop when the
            # whole warp agrees (uniform branch, exact numerics either way).
            need = Float32(row_scale[0]) < 1.0
            for r in cutlass.range_constexpr(1, cute.size(row_scale)):
                need = need | (Float32(row_scale[r]) < 1.0)
            if cute.arch.vote_any_sync(need):
                for r in cutlass.range_constexpr(cute.size(row_scale)):
                    acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])
        else:
            for r in cutlass.range_constexpr(cute.size(row_scale)):
                acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])

    @cute.jit
    def apply_seqlenk_mask(
        self,
        acc_S: cute.Tensor,
        tiled_mma_qk,
        thr_mma_qk,
        seqlenk_col_start: Int32,
    ):
        """Set scores whose global key index >= seqlen_k to -inf.
        seqlenk_col_start = seqlen_k - n_block*tile_n (columns >= it are OOB).
        """
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma_qk.partition_C(cS))
        t0ScS_mn = layout_utils.reshape_acc_to_mn(
            tiled_mma_qk.get_slice(0).partition_C(cS)
        )
        # thread-0's coords are compile-time; shift the limit by this thread's
        # column offset instead of comparing per-element runtime coords.
        limit = seqlenk_col_start - tScS_mn[0][1]
        for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
            if t0ScS_mn[0, c][1] >= limit:
                for r in cutlass.range_constexpr(cute.size(tScS_mn.shape[0])):
                    acc_S_mn[r, c] = -Float32.inf

    @cute.jit
    def convert_P(self, acc_S: cute.Tensor, tOrP: cute.Tensor):
        """fp32 exp2 scores -> bf16 A-operand registers for the PV GMMA.
        Element order of the S accumulator and the A fragment coincide
        linearly; with tile_n_pad > tile_n the fragment tail slots stay 0.
        Packed cvt.rn.bf16x2.f32 (acc_S size is always even).
        """
        tOrP_view = cute.make_tensor(tOrP.iterator, cute.make_layout(acc_S.shape))
        _cvt_bf16_frag(acc_S, tOrP_view)

    # ---- consumer-warpgroup ping-pong (FA3-style warp scheduler barrier) ----
    @cute.jit
    def mma_init(self, wg_idx):
        if const_expr(self.use_scheduler_barrier):
            if wg_idx == 0:
                cute.arch.barrier_arrive(
                    barrier_id=BAR_WG_SCHED1, number_of_threads=2 * 128
                )

    @cute.jit
    def warp_scheduler_barrier_sync(self, wg_idx):
        if const_expr(self.use_scheduler_barrier):
            cute.arch.barrier(
                barrier_id=BAR_WG_SCHED1 + wg_idx, number_of_threads=2 * 128
            )

    @cute.jit
    def warp_scheduler_barrier_arrive(self, wg_idx):
        if const_expr(self.use_scheduler_barrier):
            cute.arch.barrier_arrive(
                barrier_id=BAR_WG_SCHED1 + (1 - wg_idx), number_of_threads=2 * 128
            )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tma_copy_bytes_Q: cutlass.Constexpr[int],
        tma_copy_bytes_K: cutlass.Constexpr[int],
        tma_copy_bytes_V: cutlass.Constexpr[int],
        sV_stage_elems: cutlass.Constexpr[int],
        SharedStorage: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_V)
            cpasync.prefetch_descriptor(tma_atom_O)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        Group = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        tma_warp = Group(1)
        mma_warps = Group(self.num_mma_threads // cute.arch.WARP_SIZE)
        # multicast K/V: each consumer warp signals the empty barrier of every
        # CTA in the mcast group => arrive count is warps * cluster_m
        mma_warps_kv = Group(self.cluster_m * self.num_mma_threads // cute.arch.WARP_SIZE)
        cta_layout_vmnk = cute.make_layout((1, self.cluster_m, 1, 1))
        pipeline_q = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_Q.data_ptr(),
            num_stages=1,
            producer_group=tma_warp,
            consumer_group=mma_warps,
            tx_count=tma_copy_bytes_Q,
            defer_sync=True,
        )
        pipeline_k = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_K.data_ptr(),
            num_stages=self.num_stages,
            producer_group=tma_warp,
            consumer_group=mma_warps_kv,
            tx_count=tma_copy_bytes_K,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_v = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_V.data_ptr(),
            num_stages=self.num_stages,
            producer_group=tma_warp,
            consumer_group=mma_warps_kv,
            tx_count=tma_copy_bytes_V,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=(self.cluster_m, 1), is_relaxed=True)

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sVt = layout_utils.transpose_view(sV)
        if const_expr(self.persistent):
            # dedicated sO: next tile's Q load overlaps this tile's O drain
            sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
        else:
            # O reuses sQ's smem for the epilogue
            sO = storage.sQ.get_tensor(
                sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype
            )

        # zero-fill V pad rows (tile_n_pad > tile_n only): the pad rows of
        # each stage are the trailing whole swizzle atoms => a contiguous
        # address range never touched by the TMA box.
        if const_expr(self.tile_n_pad != self.tile_n):
            pad_elems = (self.tile_n_pad - self.tile_n) * self.head_dim
            box_elems = sV_stage_elems - pad_elems
            tidx_all, _, _ = cute.arch.thread_idx()
            for stage in cutlass.range_constexpr(self.num_stages):
                pad = cute.make_tensor(
                    storage.sV.data_ptr() + stage * sV_stage_elems + box_elems,
                    cute.make_layout(pad_elems),
                )
                for i in cutlass.range(
                    tidx_all, pad_elems, self.num_threads, unroll=1
                ):
                    pad[i] = self.dtype(0.0)
            cute.arch.fence_view_async_shared()

        seqlen_q = cute.size(mQ.shape[0])
        seqlen_k = cute.size(mK.shape[0])
        n_blocks = cute.ceil_div(seqlen_k, self.tile_n)
        n_tail = seqlen_k - (n_blocks - 1) * self.tile_n  # in (0, tile_n]
        m_blocks = cute.ceil_div(seqlen_q, self.tile_m)
        num_heads = cute.size(mQ.shape[2])

        # per-CTA tile schedule: "single" = one tile from blockIdx;
        # "persistent" = strided walk over (batch, head, m_unit) units, where a
        # unit is a cluster_m-pack of adjacent m-blocks (this CTA takes
        # m_unit * cluster_m + cluster_rank). cluster_m == 1 degenerates to the
        # plain per-tile walk. All CTAs of a cluster see the same unit count,
        # so the K/V multicast streams stay in lockstep; a phantom tail tile
        # (m_blocks % cluster_m != 0) recomputes its peer's m-block and only
        # skips the O/LSE stores.
        m_units = cute.ceil_div(m_blocks, self.cluster_m)
        bidx, _, _ = cute.arch.block_idx()
        cluster_rank = bidx % self.cluster_m  # 0 when cluster_m == 1
        if const_expr(self.persistent):
            cluster_id = bidx // self.cluster_m
            total_units = m_units * num_heads * cute.size(mQ.shape[3])
            grid_units = min(total_units, SM_COUNT // self.cluster_m)
            num_my_tiles = (total_units - cluster_id + grid_units - 1) // grid_units
        else:
            num_my_tiles = 1

        pipeline_init_wait(cluster_shape_mn=(self.cluster_m, 1))

        if warp_idx < 4:
            # ======================= PRODUCER =======================
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            if warp_idx == 0:
                if const_expr(self.cluster_m > 1):
                    cluster_layout_mnk = cute.make_layout((self.cluster_m, 1, 1))
                    kv_mcast_mask = cute.make_layout_image_mask(
                        cluster_layout_mnk,
                        cluster_layout_mnk.get_flat_coord(Int32(cluster_rank)),
                        mode=0,
                    )
                q_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, 1
                )
                kv_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_stages
                )
                for tile in cutlass.range(num_my_tiles, unroll=1):
                    if const_expr(self.persistent):
                        unit_id = cluster_id + tile * grid_units
                        batch = unit_id // (m_units * num_heads)
                        rest = unit_id % (m_units * num_heads)
                        head = rest // m_units
                        m_block = rest % m_units * self.cluster_m + cluster_rank
                        if const_expr(m_blocks % self.cluster_m != 0):
                            # phantom tail pair: recompute the peer's m-block
                            m_block = cutlass.min(m_block, Int32(m_blocks - 1))
                    else:
                        m_block, head, batch = cute.arch.block_idx()
                        if const_expr(m_blocks % self.cluster_m != 0):
                            m_block = cutlass.min(m_block, Int32(m_blocks - 1))
                    mQ_cur = mQ[None, None, head, batch]
                    mK_cur = mK[None, None, head, batch]
                    mV_cur = mV[None, None, head, batch]
                    gQ = cute.local_tile(mQ_cur, (self.tile_m, self.head_dim), (m_block, 0))
                    gK = cute.local_tile(mK_cur, (self.tile_n, self.head_dim), (None, 0))
                    gV = cute.local_tile(mV_cur, (self.tile_n, self.head_dim), (None, 0))
                    load_Q, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True
                    )
                    if const_expr(self.cluster_m > 1):
                        load_K, _, _ = copy_utils.tma_get_copy_fn(
                            tma_atom_K, cluster_rank,
                            cute.make_layout(self.cluster_m), gK, sK,
                            mcast_mask=kv_mcast_mask,
                        )
                        load_V, _, _ = copy_utils.tma_get_copy_fn(
                            tma_atom_V, cluster_rank,
                            cute.make_layout(self.cluster_m), gV, sV,
                            mcast_mask=kv_mcast_mask,
                        )
                    else:
                        load_K, _, _ = copy_utils.tma_get_copy_fn(
                            tma_atom_K, 0, cute.make_layout(1), gK, sK
                        )
                        load_V, _, _ = copy_utils.tma_get_copy_fn(
                            tma_atom_V, 0, cute.make_layout(1), gV, sV
                        )
                    load_K = copy_utils.tma_producer_copy_fn(load_K, pipeline_k)
                    load_V = copy_utils.tma_producer_copy_fn(load_V, pipeline_v)

                    pipeline_q.producer_acquire(q_state)
                    load_Q(tma_bar_ptr=pipeline_q.producer_get_barrier(q_state))
                    q_state.advance()

                    if const_expr(self.intra_wg_overlap):
                        # K runs one block ahead of V, matching the consumer's
                        # QK(i+1)-before-PV(i) order.
                        pipeline_k.producer_acquire(kv_state)
                        load_K(src_idx=n_blocks - 1, producer_state=kv_state)
                        pipeline_k.producer_commit(kv_state)
                        for i in cutlass.range(n_blocks - 1, unroll=1):
                            n_block_prev = n_blocks - 1 - i
                            kv_state_prev = kv_state.clone()
                            kv_state.advance()
                            pipeline_k.producer_acquire(kv_state)
                            load_K(src_idx=n_block_prev - 1, producer_state=kv_state)
                            pipeline_k.producer_commit(kv_state)
                            pipeline_v.producer_acquire(kv_state_prev)
                            load_V(src_idx=n_block_prev, producer_state=kv_state_prev)
                            pipeline_v.producer_commit(kv_state_prev)
                        pipeline_v.producer_acquire(kv_state)
                        load_V(src_idx=0, producer_state=kv_state)
                        pipeline_v.producer_commit(kv_state)
                        kv_state.advance()
                    else:
                        for i in cutlass.range(n_blocks, unroll=1):
                            n_block = n_blocks - 1 - i
                            pipeline_k.producer_acquire(kv_state)
                            load_K(src_idx=n_block, producer_state=kv_state)
                            pipeline_k.producer_commit(kv_state)
                            pipeline_v.producer_acquire(kv_state)
                            load_V(src_idx=n_block, producer_state=kv_state)
                            pipeline_v.producer_commit(kv_state)
                            kv_state.advance()
                if const_expr(self.cluster_m > 1):
                    # peer consumers arrive on OUR K empty barriers too; wait
                    # for those arrives before this CTA may exit (V tail alone
                    # covers V; K's last arrives are a separate transaction)
                    k_state = kv_state.clone()
                    pipeline_k.producer_tail(k_state)
                pipeline_v.producer_tail(kv_state)
        else:
            # ======================= CONSUMER =======================
            cute.arch.setmaxregister_increase(self.num_mma_regs)
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            wg_idx = cute.arch.make_warp_uniform(tidx // 128)
            wg_layout = cute.make_layout(self.num_wg_mma, stride=128)
            thr_mma_qk = tiled_mma_qk.get_slice(tidx)
            thr_mma_pv = tiled_mma_pv.get_slice(tidx)
            wg_mma_qk = tiled_mma_qk.get_slice(wg_layout(wg_idx))
            wg_mma_pv = tiled_mma_pv.get_slice(wg_layout(wg_idx))

            _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
                wg_mma_qk, (self.tile_m, self.tile_n, self.head_dim), sQ, sK
            )
            acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(
                wg_mma_pv, (self.tile_m, self.head_dim, self.tile_n_pad), None, sVt
            )
            if const_expr(self.tile_n_pad != self.tile_n):
                tOrP.fill(self.dtype(0.0))

            num_rows = acc_O.shape[0][0] * acc_O.shape[1]
            row_max = cute.make_rmem_tensor(num_rows, Float32)
            row_sum = cute.make_rmem_tensor(num_rows, Float32)
            row_scale = cute.make_rmem_tensor(num_rows, Float32)

            mma_qk_fn = partial(
                sm90_utils.gemm_zero_init,
                tiled_mma_qk, (self.tile_m, self.tile_n), tSrQ, tSrK,
            )
            mma_pv_fn = partial(sm90_utils.gemm_w_idx, tiled_mma_pv, acc_O, tOrP, tOrVt)

            st_atom_O = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4),
                self.dtype,
            )
            thr_copy_O = cute.make_tiled_copy_C(st_atom_O, tiled_mma_pv).get_slice(tidx)
            cO = cute.make_identity_tensor((self.tile_m, self.head_dim))
            taccOcO = layout_utils.reshape_acc_to_mn(thr_mma_pv.partition_C(cO))
            t0accOcO = layout_utils.reshape_acc_to_mn(
                tiled_mma_pv.get_slice(0).partition_C(cO)
            )

            q_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            kv_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_stages
            )
            self.mma_init(wg_idx)
            warp_idx_c = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            if const_expr(self.persistent):
                # prime "sO free": there is no O TMA in flight initially
                if warp_idx_c == 4:
                    cute.arch.barrier_arrive(
                        barrier_id=BAR_O_FREE,
                        number_of_threads=self.num_mma_threads + cute.arch.WARP_SIZE,
                    )

            for tile in cutlass.range(num_my_tiles, unroll=1):
                tile_valid = cutlass.Boolean(True)
                if const_expr(self.persistent):
                    unit_id = cluster_id + tile * grid_units
                    batch = unit_id // (m_units * num_heads)
                    rest = unit_id % (m_units * num_heads)
                    head = rest // m_units
                    m_block = rest % m_units * self.cluster_m + cluster_rank
                    if const_expr(m_blocks % self.cluster_m != 0):
                        # phantom tail pair: full compute on the peer's
                        # m-block (keeps the cluster lockstep), stores skipped
                        tile_valid = m_block < m_blocks
                        m_block = cutlass.min(m_block, Int32(m_blocks - 1))
                else:
                    m_block, head, batch = cute.arch.block_idx()
                    if const_expr(m_blocks % self.cluster_m != 0):
                        tile_valid = m_block < m_blocks
                        m_block = cutlass.min(m_block, Int32(m_blocks - 1))
                q_state, kv_state = self._consumer_tile(
                    mO, mLSE, sO, sQ, tiled_mma_qk, tiled_mma_pv, thr_mma_qk,
                    thr_mma_pv, mma_qk_fn, mma_pv_fn, acc_O, tOrP,
                    row_max, row_sum, row_scale, num_rows,
                    pipeline_q, pipeline_k, pipeline_v, q_state, kv_state,
                    tma_atom_O, st_atom_O, thr_copy_O, taccOcO, t0accOcO,
                    tidx, wg_idx, warp_idx_c, m_block, head, batch, tile_valid,
                    seqlen_q, seqlen_k, n_blocks, n_tail,
                )

    @cute.jit
    def _consumer_tile(
        self, mO, mLSE, sO, sQ, tiled_mma_qk, tiled_mma_pv, thr_mma_qk,
        thr_mma_pv, mma_qk_fn, mma_pv_fn, acc_O, tOrP,
        row_max, row_sum, row_scale, num_rows: cutlass.Constexpr[int],
        pipeline_q, pipeline_k, pipeline_v, q_state, kv_state,
        tma_atom_O, st_atom_O, thr_copy_O, taccOcO, t0accOcO,
        tidx, wg_idx, warp_idx_c, m_block, head, batch, tile_valid,
        seqlen_q: cutlass.Constexpr[int], seqlen_k: cutlass.Constexpr[int],
        n_blocks: cutlass.Constexpr[int], n_tail: cutlass.Constexpr[int],
    ):
            # phantom tiles (cluster tail) exist only when cluster_m doesn't
            # divide m_blocks; everywhere else tile_valid folds to constant True
            has_phantom = const_expr(
                self.cluster_m > 1
                and cute.ceil_div(seqlen_q, self.tile_m) % self.cluster_m != 0
            )
            pipeline_q.consumer_wait(q_state, pipeline_q.consumer_try_wait(q_state))

            if const_expr(self.intra_wg_overlap):
                # =========== FA4-style intra-warpgroup overlap ===========
                # QK of block i+1 and PV of block i are both in flight while
                # the softmax exp2 of block i+1 runs; the two consumer WGs
                # ping-pong their WGMMA-issue windows via named barriers.
                acc_O.fill(0.0)
                # peeled first (masked, is_first) block: QK + softmax only
                pipeline_k.consumer_wait(kv_state, pipeline_k.consumer_try_wait(kv_state))
                acc_S = mma_qk_fn(B_idx=kv_state.index, wg_wait=0)
                pipeline_k.consumer_release(kv_state)
                if const_expr(seqlen_k % self.tile_n != 0):
                    self.apply_seqlenk_mask(acc_S, tiled_mma_qk, thr_mma_qk, Int32(n_tail))
                self.softmax_step(acc_S, row_max, row_sum, row_scale, is_first=True)
                self.convert_P(acc_S, tOrP)

                for i in cutlass.range(n_blocks - 1, unroll=1):
                    kv_state_v = kv_state.clone()
                    kv_state.advance()
                    pipeline_k.consumer_wait(
                        kv_state, pipeline_k.consumer_try_wait(kv_state)
                    )
                    self.warp_scheduler_barrier_sync(wg_idx)
                    acc_S = mma_qk_fn(B_idx=kv_state.index, wg_wait=-1)
                    pipeline_v.consumer_wait(
                        kv_state_v, pipeline_v.consumer_try_wait(kv_state_v)
                    )
                    mma_pv_fn(zero_init=False, B_idx=kv_state_v.index, wg_wait=-1)
                    self.warp_scheduler_barrier_arrive(wg_idx)
                    warpgroup.wait_group(1)  # QK done; PV still in flight
                    pipeline_k.consumer_release(kv_state)
                    self.softmax_step(acc_S, row_max, row_sum, row_scale, is_first=False)
                    warpgroup.wait_group(0)  # drain PV
                    pipeline_v.consumer_release(kv_state_v)
                    self.convert_P(acc_S, tOrP)
                    self.rescale_O(acc_O, row_scale)

                # all QK gemms done (wait_group above): release Q for the
                # producer's next-tile load
                if const_expr(self.persistent):
                    pipeline_q.consumer_release(q_state)
                    q_state.advance()
                # trailing PV of the last consumed block
                pipeline_v.consumer_wait(kv_state, pipeline_v.consumer_try_wait(kv_state))
                mma_pv_fn(zero_init=False, B_idx=kv_state.index, wg_wait=0)
                pipeline_v.consumer_release(kv_state)
                kv_state.advance()
            else:
                # ================= simple non-overlap loop =================
                # ---- peeled first block: the ragged KV tail (masked), is_first
                pipeline_k.consumer_wait(kv_state, pipeline_k.consumer_try_wait(kv_state))
                acc_S = mma_qk_fn(B_idx=kv_state.index, wg_wait=0)
                pipeline_k.consumer_release(kv_state)
                if const_expr(seqlen_k % self.tile_n != 0):
                    self.apply_seqlenk_mask(acc_S, tiled_mma_qk, thr_mma_qk, Int32(n_tail))
                self.softmax_step(acc_S, row_max, row_sum, row_scale, is_first=True)
                self.convert_P(acc_S, tOrP)
                pipeline_v.consumer_wait(kv_state, pipeline_v.consumer_try_wait(kv_state))
                mma_pv_fn(zero_init=True, B_idx=kv_state.index, wg_wait=0)
                pipeline_v.consumer_release(kv_state)
                kv_state.advance()

                # ---- clean unmasked mainloop (descending blocks n_blocks-2 .. 0)
                for i in cutlass.range(n_blocks - 1, unroll=1):
                    pipeline_k.consumer_wait(kv_state, pipeline_k.consumer_try_wait(kv_state))
                    acc_S = mma_qk_fn(B_idx=kv_state.index, wg_wait=0)
                    pipeline_k.consumer_release(kv_state)
                    self.softmax_step(acc_S, row_max, row_sum, row_scale, is_first=False)
                    self.rescale_O(acc_O, row_scale)
                    self.convert_P(acc_S, tOrP)
                    pipeline_v.consumer_wait(kv_state, pipeline_v.consumer_try_wait(kv_state))
                    mma_pv_fn(zero_init=False, B_idx=kv_state.index, wg_wait=0)
                    pipeline_v.consumer_release(kv_state)
                    kv_state.advance()
                if const_expr(self.persistent):
                    pipeline_q.consumer_release(q_state)
                    q_state.advance()

            # ---- finalize: quad-reduce row_sum, compute LSE + 1/sum scale
            for r in cutlass.range_constexpr(num_rows):
                row_sum[r] = cute.arch.warp_reduction_sum(row_sum[r], threads_in_group=4)
                sum_r = row_sum[r]
                is_bad = (sum_r == 0.0) or (sum_r != sum_r)
                row_scale[r] = cute.arch.rcp_approx(1.0 if is_bad else sum_r)
                lse_r = (
                    row_max[r] * self.scale_log2 + cute.math.log2(sum_r, fastmath=True)
                ) * LN_2
                row_sum[r] = -Float32.inf if is_bad else lse_r
            self.rescale_O(acc_O, row_scale)

            # ---- epilogue: O -> smem (StMatrix) -> gmem (TMA); LSE direct
            rO = cute.make_fragment_like(acc_O, self.dtype)
            _cvt_bf16_frag(acc_O, rO)
            if const_expr(self.persistent):
                # wait until the previous tile's O TMA drained sO (also joins
                # the two consumer WGs, keeping the tile step in lock-step)
                cute.arch.barrier(
                    barrier_id=BAR_O_FREE,
                    number_of_threads=self.num_mma_threads + cute.arch.WARP_SIZE,
                )
            else:
                # both consumer WGs must be done reading sQ (sO aliases it)
                cute.arch.barrier(
                    barrier_id=BAR_EPILOGUE, number_of_threads=self.num_mma_threads
                )
            cute.copy(st_atom_O, thr_copy_O.retile(rO), thr_copy_O.partition_D(sO))

            # LSE store with m-tail predication (natural log, fp32)
            mLSE_cur = mLSE[None, head, batch]
            gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
            gLSE_expanded = cute.make_tensor(
                gLSE.iterator,
                cute.append(gLSE.layout, cute.make_layout(self.head_dim, stride=0)),
            )
            taccOgLSE = layout_utils.reshape_acc_to_mn(thr_mma_pv.partition_C(gLSE_expanded))
            lse_lane = taccOcO[0][1] == 0  # lanes owning column 0
            if const_expr(has_phantom):
                lse_lane = lse_lane & tile_valid
            if lse_lane:
                row_limit = seqlen_q - m_block * self.tile_m - taccOcO[0][0]
                for r in cutlass.range_constexpr(num_rows):
                    if t0accOcO[r, 0][0] < row_limit:
                        taccOgLSE[r, 0] = row_sum[r]

            cute.arch.fence_view_async_shared()
            cute.arch.barrier_arrive(
                barrier_id=BAR_EPILOGUE,
                number_of_threads=self.num_mma_threads + cute.arch.WARP_SIZE,
            )
            mO_cur = mO[None, None, head, batch]
            gO = cute.local_tile(mO_cur, (self.tile_m, self.head_dim), (m_block, 0))
            store_O, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_O, 0, cute.make_layout(1), sO, gO, single_stage=True
            )
            if warp_idx_c == 4:
                cute.arch.barrier(
                    barrier_id=BAR_EPILOGUE,
                    number_of_threads=self.num_mma_threads + cute.arch.WARP_SIZE,
                )
                if const_expr(has_phantom):
                    if tile_valid:
                        store_O()
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                else:
                    store_O()
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                if const_expr(self.persistent):
                    cute.arch.barrier_arrive(
                        barrier_id=BAR_O_FREE,
                        number_of_threads=self.num_mma_threads + cute.arch.WARP_SIZE,
                    )
            return q_state, kv_state
