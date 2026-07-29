"""Greenfield FlashAttention BACKWARD for Hopper sm90 in CuTeDSL, specialized
for Wan2.1 (bf16, head_dim 128, MHA, non-causal, dense, fixed shapes).

Three-kernel chain (see docs/BWD_STUDY.md for the FA4 architecture this
follows):
  1. WanFlashBwdPreprocessSm90: D = rowsum(O * dO) fp32 into a seqlen-padded
     (b, h, s_rounded) buffer; LSE (natural log) -> LSE*log2(e) with a +inf
     sentinel for pad rows (makes the m-tail self-masking in the main kernel);
     zeroes the fp32 dQaccum buffer.
  2. WanFlashBwdSm90: warp-specialized 384-thread main kernel, dK/dV-stationary
     (one CTA per (n_block, head, batch), loops over ALL m_blocks).
     Producer warp 0 = TMA loads (K/V once, then Q+LSE / dO+dPsum pipelines);
     warp 1 = dQ gmem accumulation (cp.reduce.async.bulk.add.f32, the FA3/FA4
     nondeterministic default); warps 4-11 = 2 MMA warpgroups running the five
     WGMMAs (S=QK^T, dP=dO V^T, dV+=P^T dO, dQ=dS K, dK+=dS^T Q).
     Config (FA4 hd128 non-causal): tile_m=80, tile_n=128, stages 2/2/2,
     SdP_swapAB=True, dKV_swapAB=False, dQ_swapAB=(tile_m%64!=0),
     AtomLayout(MSdP,NdKV,MdQ)=(1,2,1) => mma_dkv_is_rs=True (P/dS feed the
     dK/dV WGMMAs straight from registers; only dS round-trips smem for dQ).
     dK is scaled by softmax_scale in the epilogue; dV unscaled.
  3. WanFlashBwdPostprocessSm90: dQaccum fp32 (MMA-fragment element order)
     -> *softmax_scale -> bf16 dq, via smem (StMatrix) for coalesced stores.

Scale is applied exactly once per tensor: dK in the main epilogue, dQ in the
postprocess, dV never. exp2 recompute: P = exp2(S*scale*log2e - lse*log2e).

Everything static: shapes are baked per compile (see interface.py cache).
"""

import math
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Boolean, BFloat16, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from quack import sm90_utils, copy_utils, layout_utils
from quack.sm90_utils import gemm_zero_init, gemm_w_idx

LOG2_E = math.log2(math.e)


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
def _cvt_bf16_frag(src: cute.Tensor) -> cute.Tensor:
    """fp32 fragment -> bf16 fragment via packed converts. TensorSSA .to()
    emits one scalar cvt per element at our (non-128x128) accumulator shapes;
    this is the FA3/FA4 packed idiom."""
    dst = cute.make_fragment_like(src, BFloat16)
    dst_i32 = cute.recast_tensor(dst, cutlass.Int32)
    for i in cutlass.range_constexpr(cute.size(dst_i32)):
        dst_i32[i] = _cvt_f32x2_bf16x2(src[2 * i], src[2 * i + 1])
    return dst

# named barriers (0 is reserved for sync_threads)
BAR_PDS = 1          # sequences the sdS buffer between the two MMA WGs
BAR_DQ_EMPTY0 = 2    # +wg_idx: store warp released sdQaccum chunk wg
BAR_DQ_FULL0 = 4     # +wg_idx: MMA WG wg published its sdQaccum chunk
BAR_EPI = 6          # dK/dV staging through sK/sV


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


# =====================================================================
# Phase A: preprocess  (D = rowsum(O*dO), lse -> lse*log2e, zero dQaccum)
# =====================================================================
class WanFlashBwdPreprocessSm90:
    def __init__(self, head_dim: int = 128, tile_m: int = 80, num_threads: int = 256):
        assert head_dim % 32 == 0 and head_dim <= 256
        assert num_threads % 32 == 0 and num_threads >= tile_m
        self.head_dim = head_dim
        self.tile_m = tile_m
        self.num_threads = num_threads
        self.dtype = BFloat16

    @cute.jit
    def __call__(
        self,
        mO: cute.Tensor,       # (b, s, h, d) bf16
        mdO: cute.Tensor,      # (b, s, h, d) bf16
        mLSE: cute.Tensor,     # (b, h, s) fp32, natural log
        mdPsum: cute.Tensor,   # (b, h, s_rounded) fp32 out
        mLSElog2: cute.Tensor, # (b, h, s_rounded) fp32 out
        mdQaccum: cute.Tensor, # (b, h, s_rounded * head_dim) fp32 out (zeroed)
        stream: cuda.CUstream,
    ):
        # (b, s, h, d) -> (s, d, h, b); (b, h, X) -> (X, h, b)
        mO, mdO = [layout_utils.select(t, [1, 3, 2, 0]) for t in (mO, mdO)]
        mLSE, mdPsum, mLSElog2, mdQaccum = [
            layout_utils.select(t, [2, 1, 0]) for t in (mLSE, mdPsum, mLSElog2, mdQaccum)
        ]

        # 128-bit row-chunk loads of O/dO: 8 bf16 per thread, 16 threads/row
        num_copy_elems = 128 // self.dtype.width
        threads_per_row = self.head_dim // num_copy_elems
        gmem_tiled_copy_O = copy_utils.tiled_copy_2d(
            self.dtype, threads_per_row, self.num_threads, num_copy_elems
        )
        assert (self.tile_m * self.head_dim // 4) % self.num_threads == 0
        gmem_tiled_copy_dQaccum = copy_utils.tiled_copy_1d(Float32, self.num_threads, 4)

        m_blocks = cute.ceil_div(cute.size(mO.shape[0]), self.tile_m)
        grid = (m_blocks, cute.size(mO.shape[2]), cute.size(mO.shape[3]))
        self.kernel(
            mO, mdO, mLSE, mdPsum, mLSElog2, mdQaccum,
            gmem_tiled_copy_O, gmem_tiled_copy_dQaccum,
        ).launch(grid=grid, block=[self.num_threads, 1, 1], stream=stream,
                 use_pdl=True)

    @cute.kernel
    def kernel(
        self,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mLSElog2: cute.Tensor,
        mdQaccum: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        gmem_tiled_copy_dQaccum: cute.TiledCopy,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, head, batch = cute.arch.block_idx()
        seqlen_q = cute.size(mO.shape[0])
        seqlen_limit = seqlen_q - m_block * self.tile_m  # rows >= it are OOB

        # PDL: O/dO/LSE come from the upstream kernel; wait before reading
        cute.arch.griddepcontrol_wait()

        mO_cur = mO[None, None, head, batch]
        mdO_cur = mdO[None, None, head, batch]
        gO = cute.local_tile(mO_cur, (self.tile_m, self.head_dim), (m_block, 0))
        gdO = cute.local_tile(mdO_cur, (self.tile_m, self.head_dim), (m_block, 0))

        # natural-log LSE -> lse_log2 with +inf sentinel for pad rows
        mLSE_cur = mLSE[None, head, batch]
        gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
        lse = Float32.inf
        if tidx < seqlen_limit:
            lse = gLSE[tidx]

        thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOgO = thr_copy_O.partition_S(gO)
        tOgdO = thr_copy_O.partition_S(gdO)
        cO = cute.make_identity_tensor((self.tile_m, self.head_dim))
        tOcO = thr_copy_O.partition_S(cO)
        t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)

        tOrO = cute.make_rmem_tensor_like(tOgO)
        tOrdO = cute.make_rmem_tensor_like(tOgdO)
        tOrO.fill(0.0)
        tOrdO.fill(0.0)
        for m in cutlass.range(cute.size(tOrO.shape[1]), unroll_full=True):
            # t0OcO coords are compile-time; shift the limit by this thread's row
            if t0OcO[0, m, 0][0] < seqlen_limit - tOcO[0][0]:
                copy_utils.copy(tOgO[None, m, None], tOrO[None, m, None])
                copy_utils.copy(tOgdO[None, m, None], tOrdO[None, m, None])

        # O/dO/LSE are in registers: let the main kernel start its prologue
        # (its griddepcontrol_wait before load_LSE orders our stores)
        cute.arch.griddepcontrol_launch_dependents()

        # D = rowsum(O*dO) fp32: intra-thread reduce over the k chunk, then
        # butterfly across the threads_per_row lanes sharing the row
        pdpsum = (tOrO.load().to(Float32) * tOrdO.load().to(Float32)).reduce(
            cute.ReductionOp.ADD, init_val=0.0, reduction_profile=(0, None, 1)
        )
        threads_per_row = gmem_tiled_copy_O.layout_src_tv_tiled[0].shape[0]
        PdP_sum = cute.make_rmem_tensor(cute.size(tOrO, mode=[1]), Float32)
        PdP_sum.store(pdpsum)
        for m in cutlass.range(cute.size(PdP_sum), unroll_full=True):
            PdP_sum[m] = cute.arch.warp_reduction_sum(
                PdP_sum[m], threads_in_group=threads_per_row
            )

        # write D into the padded buffer (0.0 for pad rows)
        mdPsum_cur = mdPsum[None, head, batch]
        gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (m_block,))
        if tOcO[0, 0, 0][1] == 0:  # lanes owning column 0
            for m in cutlass.range(cute.size(PdP_sum), unroll_full=True):
                row = tOcO[0, m, 0][0]
                val = Float32(0.0)
                if row < seqlen_limit:
                    val = PdP_sum[m]
                gdPsum[row] = val

        # zero this (m_block, head, batch) chunk of dQaccum
        mdQaccum_cur = mdQaccum[None, head, batch]
        gdQaccum = cute.local_tile(
            mdQaccum_cur, (self.tile_m * self.head_dim,), (m_block,)
        )
        thr_copy_dq = gmem_tiled_copy_dQaccum.get_slice(tidx)
        tdQgdQ = thr_copy_dq.partition_S(gdQaccum)
        zero = cute.make_rmem_tensor_like(tdQgdQ)
        zero.fill(0.0)
        cute.copy(gmem_tiled_copy_dQaccum, zero, tdQgdQ)

        # lse_log2 (padded rows get +inf; -inf rows clamp to 0 to avoid NaN)
        mLSElog2_cur = mLSElog2[None, head, batch]
        gLSElog2 = cute.local_tile(mLSElog2_cur, (self.tile_m,), (m_block,))
        lse_log2 = lse * LOG2_E if lse != -Float32.inf else 0.0
        if tidx < self.tile_m:
            gLSElog2[tidx] = lse_log2


# =====================================================================
# Phase B: main kernel (dK/dV-stationary, warp-specialized)
# =====================================================================
class WanFlashBwdSm90:
    def __init__(
        self,
        head_dim: int = 128,
        tile_m: int = 80,
        tile_n: int = 128,
        num_stages: int = 2,
        nsplit: int = 1,
    ):
        self.num_wg_mma = 2
        assert head_dim % 16 == 0 and head_dim <= 128
        assert tile_n == 64 * self.num_wg_mma, "SdP_swapAB M = tile_n = 64*num_wg"
        assert tile_m % 16 == 0, "dK/dV contraction (K = tile_m) tiles by k16"
        assert num_stages in (1, 2)
        assert nsplit >= 1
        self.head_dim = head_dim
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.nsplit = nsplit  # >1: split the m loop; dK/dV via fp32 gmem accum
        self.Q_stage = num_stages
        self.dO_stage = num_stages
        self.PdS_stage = num_stages
        self.dtype = BFloat16
        # FA4 hd128 non-causal config, hard-wired:
        self.SdP_swapAB = True
        self.dKV_swapAB = False
        self.dQ_swapAB = tile_m % 64 != 0
        self.AtomLayoutMSdP = 1
        self.AtomLayoutNdKV = self.num_wg_mma
        self.AtomLayoutMdQ = 1
        self.num_wg_dQ = self.num_wg_mma
        self.mma_dkv_is_rs = True  # (MSdP==1 and NdKV==num_wg and SdP_swap and not dKV_swap)
        self.num_mma_threads = 128 * self.num_wg_mma
        self.num_threads = 128 * (self.num_wg_mma + 1)
        self.num_mma_regs = 240
        self.num_producer_regs = 24
        self.softmax_scale = 1.0 / math.sqrt(head_dim)
        self.scale_log2 = self.softmax_scale * LOG2_E

    # ------------------------------------------------------------- layouts
    def _make_layouts(self):
        wg_d_dKV = self.num_wg_mma // self.AtomLayoutNdKV  # 1
        # sQ/sdO must serve both Q (K-major B of SdP) and Q^T (MN-major B of dK)
        sQ_layout, sdO_layout = [
            sm90_utils.make_smem_layout(
                self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_m, self.head_dim),
                stage, major_mode_size=self.head_dim // wg_d_dKV,
            )
            for stage in (self.Q_stage, self.dO_stage)
        ]
        wg_d_dQ = self.num_wg_dQ // self.AtomLayoutMdQ  # 2
        # sK serves K (SdP) and K^T (dQ)
        sK_layout = sm90_utils.make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.head_dim),
            None, major_mode_size=self.head_dim // wg_d_dQ,
        )
        sV_layout = sm90_utils.make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.head_dim), None
        )
        # sdS serves dS (B of dQ) and dS^T (StMatrix-transpose write target)
        wg_n_SdP = self.num_wg_mma // self.AtomLayoutMSdP
        wg_n_dKV = self.AtomLayoutNdKV
        sdS_layout = sm90_utils.make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_n),
            stage=self.PdS_stage,
            major_mode_size=math.gcd(self.tile_n // wg_n_SdP, self.tile_n // wg_n_dKV),
        )
        sdQaccum_layout = cute.make_layout(
            (self.tile_m * self.head_dim // self.num_wg_dQ, self.num_wg_dQ)
        )
        return sQ_layout, sdO_layout, sK_layout, sV_layout, sdS_layout, sdQaccum_layout

    def _make_tiled_mmas(self):
        swap_mn = lambda a, swap: (a[1], a[0], *a[2:]) if swap else a
        # S = Q@K^T / dP = dO@V^T, computed transposed (SdP_swapAB)
        atom_SdP = (self.AtomLayoutMSdP, self.num_wg_mma // self.AtomLayoutMSdP, 1)
        tiled_mma_SdP = cutlass.utils.hopper_helpers.make_trivial_tiled_mma(
            self.dtype, self.dtype,
            warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=swap_mn(atom_SdP, self.SdP_swapAB),
            tiler_mn=(64, self.tile_m // atom_SdP[0]),
        )
        # dV += P^T @ dO ; dK += dS^T @ Q  (A from registers: mma_dkv_is_rs)
        atom_dKV = (self.AtomLayoutNdKV, self.num_wg_mma // self.AtomLayoutNdKV, 1)
        tiled_mma_dKV = cutlass.utils.hopper_helpers.make_trivial_tiled_mma(
            self.dtype, self.dtype,
            warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=atom_dKV,
            tiler_mn=(64, self.head_dim // atom_dKV[1]),
            a_source=warpgroup.OperandSource.RMEM,
        )
        # dQ = dS @ K (computed transposed when dQ_swapAB)
        atom_dQ = (self.AtomLayoutMdQ, self.num_wg_dQ // self.AtomLayoutMdQ, 1)
        tiler_dQ = (self.tile_m // atom_dQ[0], self.head_dim // atom_dQ[1])
        tiled_mma_dQ = cutlass.utils.hopper_helpers.make_trivial_tiled_mma(
            self.dtype, self.dtype,
            warpgroup.OperandMajorMode.K if not self.dQ_swapAB else warpgroup.OperandMajorMode.MN,
            warpgroup.OperandMajorMode.MN if not self.dQ_swapAB else warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=swap_mn(atom_dQ, self.dQ_swapAB),
            tiler_mn=(64, tiler_dQ[1] if not self.dQ_swapAB else tiler_dQ[0]),
        )
        return tiled_mma_SdP, tiled_mma_dKV, tiled_mma_dQ

    def _make_storage(self, sQ_l, sdO_l, sK_l, sV_l, sdS_l, sdQaccum_l):
        dtype = self.dtype
        Aligned = lambda l, t: cute.struct.Align[
            cute.struct.MemRange[t, cute.cosize(l)], 1024
        ]
        stat_elems = _round_up(self.tile_m, 64)

        @cute.struct
        class SharedStorage:
            mbar_Q: cute.struct.MemRange[cutlass.Int64, self.Q_stage * 2]
            mbar_dO: cute.struct.MemRange[cutlass.Int64, self.dO_stage * 2]
            sLSE: cute.struct.Align[
                cute.struct.MemRange[Float32, stat_elems * self.Q_stage], 128
            ]
            sdPsum: cute.struct.Align[
                cute.struct.MemRange[Float32, stat_elems * self.dO_stage], 128
            ]
            sQ: Aligned(sQ_l, dtype)
            sV: Aligned(sV_l, dtype)
            sK: Aligned(sK_l, dtype)
            sdO: Aligned(sdO_l, dtype)
            sdS: Aligned(sdS_l, dtype)
            sdQaccum: Aligned(sdQaccum_l, Float32)

        return SharedStorage

    # ---------------------------------------------------------------- host
    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,        # (b, s_q, h, d) bf16
        mK: cute.Tensor,        # (b, s_kv, h, d) bf16
        mV: cute.Tensor,        # (b, s_kv, h, d) bf16
        mdO: cute.Tensor,       # (b, s_q, h, d) bf16
        mLSElog2: cute.Tensor,  # (b, h, s_q_rounded) fp32 (lse * log2e, +inf pad)
        mdPsum: cute.Tensor,    # (b, h, s_q_rounded) fp32
        mdQaccum: cute.Tensor,  # (b, h, s_q_rounded * d) fp32, pre-zeroed
        mdK: cute.Tensor,       # nsplit=1: (b, s_kv, h, d) bf16 out
                                # nsplit>1: (b, h, s_kv_rounded * d) fp32 accum
        mdV: cute.Tensor,       # same
        stream: cuda.CUstream,
    ):
        # (b, s, h, d) -> (s, d, h, b); (b, h, X) -> (X, h, b)
        mQ, mK, mV, mdO = [
            layout_utils.select(t, [1, 3, 2, 0]) for t in (mQ, mK, mV, mdO)
        ]
        if const_expr(self.nsplit == 1):
            mdK, mdV = [layout_utils.select(t, [1, 3, 2, 0]) for t in (mdK, mdV)]
        else:
            mdK, mdV = [layout_utils.select(t, [2, 1, 0]) for t in (mdK, mdV)]
        mLSElog2, mdPsum, mdQaccum = [
            layout_utils.select(t, [2, 1, 0]) for t in (mLSElog2, mdPsum, mdQaccum)
        ]

        # static shapes: baked per compile (K-tail mask codegen depends on it)
        self.seqlen_k_static = cute.size(mK.shape[0])

        sQ_l, sdO_l, sK_l, sV_l, sdS_l, sdQaccum_l = self._make_layouts()
        tiled_mma_SdP, tiled_mma_dKV, tiled_mma_dQ = self._make_tiled_mmas()

        tma_bytes = {
            name: cute.size_in_bytes(self.dtype, cute.select(l, mode=[0, 1]))
            for name, l in [("Q", sQ_l), ("K", sK_l), ("V", sV_l), ("dO", sdO_l)]
        }
        tma_bytes_stat = self.tile_m * Float32.width // 8
        tma_bytes_dQ = self.tile_m * self.head_dim * Float32.width // 8 // self.num_wg_dQ

        op_g2s = cpasync.CopyBulkTensorTileG2SOp()
        op_s2g = cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            op_g2s, mQ, cute.select(sQ_l, mode=[0, 1]), (self.tile_m, self.head_dim)
        )
        tma_atom_dO, tma_tensor_dO = cpasync.make_tiled_tma_atom(
            op_g2s, mdO, cute.select(sdO_l, mode=[0, 1]), (self.tile_m, self.head_dim)
        )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            op_g2s, mK, cute.select(sK_l, mode=[0, 1]), (self.tile_n, self.head_dim)
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            op_g2s, mV, cute.select(sV_l, mode=[0, 1]), (self.tile_n, self.head_dim)
        )
        if const_expr(self.nsplit == 1):
            tma_atom_dK, tma_tensor_dK = cpasync.make_tiled_tma_atom(
                op_s2g, mdK, cute.select(sK_l, mode=[0, 1]), (self.tile_n, self.head_dim)
            )
            tma_atom_dV, tma_tensor_dV = cpasync.make_tiled_tma_atom(
                op_s2g, mdV, cute.select(sV_l, mode=[0, 1]), (self.tile_n, self.head_dim)
            )
        else:
            tma_atom_dK = tma_atom_dV = None
            tma_tensor_dK, tma_tensor_dV = mdK, mdV

        # dQaccum r2s: flat 128-bit chunks, thread t of WG w -> (t, w) slot
        r2s_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128),
            cute.make_layout((128, self.num_wg_dQ)),
            cute.make_layout(128 // Float32.width),
        )
        # dK/dV accum r2s (nsplit>1): per-WG flat chunk, thread-local 4-elem slots
        r2s_tiled_copy_dKVaccum = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128),
            cute.make_layout(128),
            cute.make_layout(128 // Float32.width),
        )

        SharedStorage = self._make_storage(sQ_l, sdO_l, sK_l, sV_l, sdS_l, sdQaccum_l)

        n_blocks = cute.ceil_div(cute.size(mK.shape[0]), self.tile_n)
        grid = (n_blocks * self.nsplit, cute.size(mQ.shape[2]), cute.size(mQ.shape[3]))
        self.kernel(
            tma_tensor_Q, tma_tensor_K, tma_tensor_V, tma_tensor_dO,
            tma_tensor_dK, tma_tensor_dV,
            mLSElog2, mdPsum, mdQaccum,
            tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_dO, tma_atom_dK, tma_atom_dV,
            sQ_l, sdO_l, sK_l, sV_l, sdS_l, sdQaccum_l,
            r2s_tiled_copy_dQaccum, r2s_tiled_copy_dKVaccum,
            tiled_mma_SdP, tiled_mma_dKV, tiled_mma_dQ,
            tma_bytes["Q"], tma_bytes["K"], tma_bytes["V"], tma_bytes["dO"],
            tma_bytes_stat, tma_bytes_dQ,
            SharedStorage,
        ).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=True,
        )

    # -------------------------------------------------------------- device
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mLSElog2: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        sQ_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sdS_layout: cute.ComposedLayout,
        sdQaccum_layout: cute.Layout,
        r2s_tiled_copy_dQaccum: cute.TiledCopy,
        r2s_tiled_copy_dKVaccum: cute.TiledCopy,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dKV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        tma_bytes_Q: cutlass.Constexpr[int],
        tma_bytes_K: cutlass.Constexpr[int],
        tma_bytes_V: cutlass.Constexpr[int],
        tma_bytes_dO: cutlass.Constexpr[int],
        tma_bytes_stat: cutlass.Constexpr[int],
        tma_bytes_dQ: cutlass.Constexpr[int],
        SharedStorage: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            for atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_dO,
                         tma_atom_dK, tma_atom_dV):
                if const_expr(atom is not None):
                    cpasync.prefetch_descriptor(atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        Group = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        tma_warp = Group(1)
        mma_warps = Group(self.num_mma_threads // cute.arch.WARP_SIZE)
        pipeline_q = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_Q.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=tma_warp,
            consumer_group=mma_warps,
            tx_count=tma_bytes_Q + tma_bytes_stat,
            defer_sync=True,
        )
        pipeline_do = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_dO.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=tma_warp,
            consumer_group=mma_warps,
            tx_count=tma_bytes_dO + tma_bytes_stat,
            defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sdS = storage.sdS.get_tensor(sdS_layout.outer, swizzle=sdS_layout.inner)
        stat_stride = _round_up(self.tile_m, 64)
        sLSE = storage.sLSE.get_tensor(
            cute.make_layout((self.tile_m, self.Q_stage), stride=(1, stat_stride))
        )
        sdPsum = storage.sdPsum.get_tensor(
            cute.make_layout((self.tile_m, self.dO_stage), stride=(1, stat_stride))
        )
        sdQaccum = storage.sdQaccum.get_tensor(sdQaccum_layout)

        seqlen_q = cute.size(mQ.shape[0])
        seqlen_k = cute.size(mK.shape[0])
        m_blocks = cute.ceil_div(seqlen_q, self.tile_m)
        bx, head, batch = cute.arch.block_idx()
        if const_expr(self.nsplit == 1):
            n_block = bx
            m_lo, m_hi = 0, m_blocks
        else:
            n_block = bx // self.nsplit
            split = bx % self.nsplit
            chunk_m = cute.ceil_div(m_blocks, self.nsplit)
            m_lo = split * chunk_m
            m_hi = cutlass.min(Int32(m_blocks), m_lo + chunk_m)

        pipeline_init_wait(cluster_shape_mn=(1, 1))

        if warp_idx < 4:
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            if warp_idx == 0:
                self._load(
                    mQ, mK, mV, mdO, mLSElog2, mdPsum,
                    sQ, sK, sV, sdO, sLSE, sdPsum,
                    tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_dO,
                    pipeline_q, pipeline_do,
                    n_block, head, batch, m_lo, m_hi,
                    tma_bytes_K, tma_bytes_V,
                )
            if warp_idx == 1:
                self._dq_store(mdQaccum, sdQaccum, head, batch, m_lo, m_hi,
                               tma_bytes_dQ)
        else:
            cute.arch.setmaxregister_increase(self.num_mma_regs)
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            self._mma(
                mdK, mdV, sQ, sK, sV, sdO, sdS, sLSE, sdPsum, sdQaccum,
                tma_atom_dK, tma_atom_dV,
                r2s_tiled_copy_dQaccum, r2s_tiled_copy_dKVaccum,
                tiled_mma_SdP, tiled_mma_dKV, tiled_mma_dQ,
                pipeline_q, pipeline_do,
                tidx, n_block, head, batch, m_lo, m_hi, seqlen_k,
            )

    # ---------------------------------------------------- producer (warp 0)
    @cute.jit
    def _load(
        self, mQ, mK, mV, mdO, mLSElog2, mdPsum,
        sQ, sK, sV, sdO, sLSE, sdPsum,
        tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_dO,
        pipeline_q, pipeline_do,
        n_block, head, batch, m_lo, m_hi,
        tma_bytes_K: cutlass.Constexpr[int], tma_bytes_V: cutlass.Constexpr[int],
    ):
        mK_cur = mK[None, None, head, batch]
        mV_cur = mV[None, None, head, batch]
        gK = cute.local_tile(mK_cur, (self.tile_n, self.head_dim), (n_block, 0))
        gV = cute.local_tile(mV_cur, (self.tile_n, self.head_dim), (n_block, 0))
        mQ_cur = mQ[None, None, head, batch]
        mdO_cur = mdO[None, None, head, batch]
        gQ = cute.local_tile(mQ_cur, (self.tile_m, self.head_dim), (None, 0))
        gdO = cute.local_tile(mdO_cur, (self.tile_m, self.head_dim), (None, 0))
        gLSE = cute.local_tile(mLSElog2[None, head, batch], (self.tile_m,), (None,))
        gdPsum = cute.local_tile(mdPsum[None, head, batch], (self.tile_m,), (None,))

        load_K, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_K, 0, cute.make_layout(1), gK, sK, single_stage=True
        )
        load_V, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_V, 0, cute.make_layout(1), gV, sV, single_stage=True
        )
        load_Q, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_Q, 0, cute.make_layout(1), gQ, sQ
        )
        load_Q = copy_utils.tma_producer_copy_fn(load_Q, pipeline_q)
        load_dO, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_dO, 0, cute.make_layout(1), gdO, sdO
        )
        load_dO = copy_utils.tma_producer_copy_fn(load_dO, pipeline_do)
        load_LSE = copy_utils.cpasync_bulk_get_copy_fn(gLSE, sLSE)
        load_LSE = copy_utils.tma_producer_copy_fn(load_LSE, pipeline_q)
        load_dPsum = copy_utils.cpasync_bulk_get_copy_fn(gdPsum, sdPsum)
        load_dPsum = copy_utils.tma_producer_copy_fn(load_dPsum, pipeline_do)

        # K/V are loaded once, piggybacked on the FIRST Q/dO stage barriers via
        # extra transaction counts (FA4 idiom): the first S GEMM waits only
        # Q(0)+K; V completes with dO(0). First iteration peeled: K/Q(0) TMA
        # may fly during the preprocess tail (PDL); LSE/dPsum/dQaccum are
        # preprocess outputs, so wait before the first stat load.
        q_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.Q_stage
        )
        if const_expr(self.nsplit == 1):
            q_state = self._load_stream(
                q_state, pipeline_q, pipeline_do,
                load_K, load_V, load_Q, load_dO, load_LSE, load_dPsum,
                m_lo, m_hi, tma_bytes_K, tma_bytes_V,
            )
        else:
            if m_lo < m_hi:  # a split's range can be empty
                q_state = self._load_stream(
                    q_state, pipeline_q, pipeline_do,
                    load_K, load_V, load_Q, load_dO, load_LSE, load_dPsum,
                    m_lo, m_hi, tma_bytes_K, tma_bytes_V,
                )

    @cute.jit
    def _load_stream(
        self, q_state, pipeline_q, pipeline_do,
        load_K, load_V, load_Q, load_dO, load_LSE, load_dPsum,
        m_lo, m_hi, tma_bytes_K: cutlass.Constexpr[int],
        tma_bytes_V: cutlass.Constexpr[int],
    ):
        pipeline_q.producer_acquire(q_state)
        with cute.arch.elect_one():
            cute.arch.mbarrier_expect_tx(
                pipeline_q.producer_get_barrier(q_state), tma_bytes_K
            )
        load_K(tma_bar_ptr=pipeline_q.producer_get_barrier(q_state))
        load_Q(m_lo, producer_state=q_state)
        cute.arch.griddepcontrol_wait()
        load_LSE(m_lo, producer_state=q_state)
        pipeline_do.producer_acquire(q_state)
        with cute.arch.elect_one():
            cute.arch.mbarrier_expect_tx(
                pipeline_do.producer_get_barrier(q_state), tma_bytes_V
            )
        load_V(tma_bar_ptr=pipeline_do.producer_get_barrier(q_state))
        load_dO(m_lo, producer_state=q_state)
        load_dPsum(m_lo, producer_state=q_state)
        q_state.advance()
        for m_block in cutlass.range(m_lo + 1, m_hi, unroll=1):
            pipeline_q.producer_acquire(q_state)
            load_Q(m_block, producer_state=q_state)
            load_LSE(m_block, producer_state=q_state)
            pipeline_do.producer_acquire(q_state)
            load_dO(m_block, producer_state=q_state)
            load_dPsum(m_block, producer_state=q_state)
            q_state.advance()
        return q_state

    # ------------------------------------------------- dQ store loop (warp 1)
    @cute.jit
    def _dq_store(self, mdQaccum, sdQaccum, head, batch, m_lo, m_hi, tma_bytes_dQ):
        mdQaccum_cur = mdQaccum[None, head, batch]
        # ((tile_m*d/num_wg, num_wg), m_blocks) view of the flat padded buffer
        gdQaccum = cute.local_tile(
            mdQaccum_cur,
            (cute.make_layout((self.tile_m * self.head_dim // self.num_wg_dQ,
                               self.num_wg_dQ)),),
            (None,),
        )
        for i in cutlass.range(m_lo, m_hi, unroll=1):
            for wg in cutlass.range_constexpr(self.num_wg_dQ):
                # previous bulk-add from chunk wg must have read sdQaccum out
                cute.arch.cp_async_bulk_wait_group(
                    self.num_wg_dQ - 1 - wg, read=True
                )
                cute.arch.barrier_arrive(
                    barrier_id=BAR_DQ_EMPTY0 + wg,
                    number_of_threads=128 + cute.arch.WARP_SIZE,
                )
            for wg in cutlass.range_constexpr(self.num_wg_dQ):
                cute.arch.barrier(
                    barrier_id=BAR_DQ_FULL0 + wg,
                    number_of_threads=128 + cute.arch.WARP_SIZE,
                )
                with cute.arch.elect_one():
                    copy_utils.cpasync_reduce_bulk_add_f32(
                        sdQaccum[None, wg].iterator,
                        gdQaccum[(None, wg), i].iterator,
                        tma_bytes_dQ,
                    )
                cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0, read=True)
        if const_expr(self.nsplit > 1):
            # release sdQaccum to the MMA WGs for the dK/dV accum epilogue
            for wg in cutlass.range_constexpr(self.num_wg_dQ):
                cute.arch.barrier_arrive(
                    barrier_id=BAR_DQ_EMPTY0 + wg,
                    number_of_threads=128 + cute.arch.WARP_SIZE,
                )

    # -------------------------------------------------- consumers (2 MMA WGs)
    @cute.jit
    def _mma(
        self, mdK, mdV, sQ, sK, sV, sdO, sdS, sLSE, sdPsum, sdQaccum,
        tma_atom_dK, tma_atom_dV,
        r2s_tiled_copy_dQaccum, r2s_tiled_copy_dKVaccum,
        tiled_mma_SdP, tiled_mma_dKV, tiled_mma_dQ,
        pipeline_q, pipeline_do,
        tidx, n_block, head, batch, m_lo, m_hi, seqlen_k,
    ):
        wg_idx = cute.arch.make_warp_uniform(tidx // 128)
        wg_layout = cute.make_layout(self.num_wg_mma, stride=128)
        thr_mma_SdP = tiled_mma_SdP.get_slice(tidx)
        wg_mma_SdP = tiled_mma_SdP.get_slice(wg_layout(wg_idx))
        wg_mma_dKV = tiled_mma_dKV.get_slice(wg_layout(wg_idx))
        wg_mma_dQ = tiled_mma_dQ.get_slice(wg_layout(wg_idx))

        # S^T = K @ Q^T (swapped); Q via B_idx (staged)
        shape_S = (self.tile_m, self.tile_n, self.head_dim)
        _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
            wg_mma_SdP, shape_S, sQ, sK, swap_AB=True
        )
        mma_qk_fn = partial(
            gemm_zero_init, tiled_mma_SdP, shape_S[:2], tSrQ, tSrK, swap_AB=True
        )
        # dP^T = V @ dO^T (swapped)
        _, tdPrdO, tdPrV = sm90_utils.partition_fragment_ABC(
            wg_mma_SdP, shape_S, sdO, sV, swap_AB=True
        )
        mma_dov_fn = partial(
            gemm_zero_init, tiled_mma_SdP, shape_S[:2], tdPrdO, tdPrV, swap_AB=True
        )
        # dV += P^T @ dO: A = P^T straight from registers, B = dO^T (MN-major)
        sdOt = layout_utils.transpose_view(sdO)
        shape_dV = (self.tile_n, self.head_dim, self.tile_m)
        acc_dV, _, tdVrdOt = sm90_utils.partition_fragment_ABC(
            wg_mma_dKV, shape_dV, None, sdOt, swap_AB=False
        )
        mma_pdo_fn = partial(gemm_w_idx, tiled_mma_dKV, acc_dV, tCrB=tdVrdOt)
        # dK += dS^T @ Q: A = dS^T from registers, B = Q^T
        sQt = layout_utils.transpose_view(sQ)
        acc_dK, _, tdKrQt = sm90_utils.partition_fragment_ABC(
            wg_mma_dKV, shape_dV, None, sQt, swap_AB=False
        )
        mma_dsq_fn = partial(gemm_w_idx, tiled_mma_dKV, acc_dK, tCrB=tdKrQt)
        # dQ = dS @ K (transposed when dQ_swapAB): A = K^T, B = dS (staged)
        sKt = layout_utils.transpose_view(sK)
        shape_dQ = (self.tile_m, self.head_dim, self.tile_n)
        _, tdQrdS, tdQrKt = sm90_utils.partition_fragment_ABC(
            wg_mma_dQ, shape_dQ, sdS, sKt, swap_AB=self.dQ_swapAB
        )
        mma_dsk_fn = partial(
            gemm_zero_init, tiled_mma_dQ, shape_dQ[:2], tdQrdS, tdQrKt,
            swap_AB=self.dQ_swapAB,
        )

        # dS r2s: StMatrix-transpose into the dS^T view (SdP accum is dS^T)
        sdSt = layout_utils.transpose_view(sdS)
        mms_PdS = self.tile_n // (self.num_wg_mma // self.AtomLayoutMSdP)
        copy_dS_r2s, _, _ = copy_utils.get_smem_store_C(
            tiled_mma_SdP, sdSt, tidx,
            transpose=True, position_independent=True, major_mode_size=mms_PdS,
        )

        # LSE/dPsum broadcast views: stride-0 expand over the KV direction
        tLSEsLSE = layout_utils.mma_partition_C_vec(
            sLSE, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=False
        )
        tLSEsdPsum = layout_utils.mma_partition_C_vec(
            sdPsum, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=False
        )

        smem_thr_copy_dQaccum = r2s_tiled_copy_dQaccum.get_slice(tidx)
        tdQsdQaccum = smem_thr_copy_dQaccum.partition_D(sdQaccum)

        PdS_barrier = pipeline.NamedBarrier(
            barrier_id=BAR_PDS, num_threads=self.num_mma_threads
        )

        # K-direction ragged tail: mask columns >= seqlen_k - n_block*tile_n.
        # Only the last n_block CTA has a tail (uniform runtime skip; compiled
        # out entirely when seqlen_k divides tile_n).
        seqlenk_col_start = seqlen_k - n_block * self.tile_n
        need_k_mask = Boolean(seqlenk_col_start < self.tile_n)

        if const_expr(self.nsplit > 1):
            # a split's range can be empty; its accumulators must still be
            # defined (they contribute 0 to the gmem accumulation)
            acc_dV.fill(0.0)
            acc_dK.fill(0.0)

        c_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        dKV_accumulate = Boolean(False)
        for m_block in cutlass.range(m_lo, m_hi, unroll=1):
            c_state = self._mma_m_block(
                c_state, wg_idx,
                mma_qk_fn, mma_dov_fn, mma_pdo_fn, mma_dsq_fn, mma_dsk_fn,
                copy_dS_r2s, pipeline_q, pipeline_do,
                tLSEsLSE, tLSEsdPsum, tdQsdQaccum,
                thr_mma_SdP, tiled_mma_SdP, seqlenk_col_start, need_k_mask,
                PdS_barrier, dKV_accumulate,
            )
            dKV_accumulate = Boolean(True)

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        acc_dK.store(acc_dK.load() * self.softmax_scale)
        if const_expr(self.nsplit == 1):
            # ---------- epilogue: dK/dV -> bf16 -> smem (sK/sV) -> TMA store
            self._epilogue_dKV(
                acc_dV, mdV, sV, acc_dK, mdK, sK,
                tma_atom_dK, tma_atom_dV, tiled_mma_dKV,
                tidx, n_block, head, batch,
            )
            if warp_idx == 4:
                cute.arch.cp_async_bulk_wait_group(0, read=True)
        else:
            # ---------- epilogue: fp32 gmem accumulation (split-M)
            self._epilogue_dKV_accum(
                acc_dV, mdV, acc_dK, mdK, sdQaccum, r2s_tiled_copy_dKVaccum,
                tidx, wg_idx, warp_idx, n_block, head, batch,
            )

    @cute.jit
    def _mma_m_block(
        self, c_state, wg_idx,
        mma_qk_fn, mma_dov_fn, mma_pdo_fn, mma_dsq_fn, mma_dsk_fn,
        copy_dS_r2s, pipeline_q, pipeline_do,
        tLSEsLSE, tLSEsdPsum, tdQsdQaccum,
        thr_mma_SdP, tiled_mma_SdP, seqlenk_col_start, need_k_mask: Boolean,
        PdS_barrier, dKV_accumulate: Boolean,
    ):
        smem_idx = c_state.index
        smem_idx_PdS = smem_idx if const_expr(self.PdS_stage > 1) else 0

        # [GEMM 1] S^T = K @ Q^T
        pipeline_q.consumer_wait(c_state, pipeline_q.consumer_try_wait(c_state))
        acc_S = mma_qk_fn(A_idx=smem_idx, wg_wait=-1)
        tLSErLSE = copy_utils.load_s2r(tLSEsLSE[None, smem_idx])
        # [GEMM 2] dP^T = V @ dO^T
        pipeline_do.consumer_wait(c_state, pipeline_do.consumer_try_wait(c_state))
        acc_dP = mma_dov_fn(A_idx=smem_idx, wg_wait=1)  # waits GEMM 1

        # K-tail mask (m-tail is masked for free by the +inf LSE sentinel
        # written in preprocess). Only the last n_block CTA runs the selects;
        # compiled out entirely when seqlen_k % tile_n == 0.
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=True)
        if const_expr(self.seqlen_k_static % self.tile_n != 0):
            if need_k_mask:
                cS = cute.make_identity_tensor((self.tile_n, self.tile_m))
                tScS_mn = layout_utils.reshape_acc_to_mn(
                    thr_mma_SdP.partition_C(cS), transpose=True
                )
                t0ScS_mn = layout_utils.reshape_acc_to_mn(
                    tiled_mma_SdP.get_slice(0).partition_C(cS), transpose=True
                )
                limit = seqlenk_col_start - tScS_mn[0][0]
                for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                    oob = t0ScS_mn[0, c][0] >= limit
                    for r in cutlass.range_constexpr(cute.size(tScS_mn.shape[0])):
                        acc_S_mn[r, c] = -Float32.inf if oob else acc_S_mn[r, c]

        # [Pointwise 1] P = exp2(S*scale_log2 - lse_log2)
        for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
            lse_val = tLSErLSE[r]
            for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
                acc_S_mn[r, c] = cute.math.exp2(
                    acc_S_mn[r, c] * self.scale_log2 - lse_val, fastmath=True
                )
        tLSErdPsum = copy_utils.load_s2r(tLSEsdPsum[None, smem_idx])

        # P -> bf16 A-fragments for the dV WGMMA (packed cvt)
        tdVrP = _cvt_bf16_frag(layout_utils.reshape_acc_to_frgA(acc_S))

        # [Pointwise 2] dS = P * (dP - D)
        warpgroup.wait_group(0)  # GEMM 2 done
        acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP, transpose=True)
        for r in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[0])):
            dpsum_val = tLSErdPsum[r]
            for c in cutlass.range(cute.size(acc_dP_mn, mode=[1]), unroll_full=True):
                acc_dP_mn[r, c] = acc_S_mn[r, c] * (acc_dP_mn[r, c] - dpsum_val)

        # dS -> bf16 fragments (A of the dK WGMMA) and smem (B of the dQ WGMMA)
        tdKrdS = _cvt_bf16_frag(layout_utils.reshape_acc_to_frgA(acc_dP))
        if const_expr(self.PdS_stage == 1):
            # single-buffer dS: previous iteration's dQ GEMM must be done reading
            cute.arch.fence_view_async_shared()
            PdS_barrier.arrive_and_wait()
        copy_dS_r2s(tdKrdS, dst_idx=smem_idx_PdS)

        # [GEMM 3] dV += P^T @ dO
        mma_pdo_fn(tCrA=tdVrP, B_idx=smem_idx, zero_init=~dKV_accumulate, wg_wait=-1)

        # publish sdS to the other WG before anyone's dQ GEMM reads it
        cute.arch.fence_view_async_shared()
        PdS_barrier.arrive_and_wait()

        # [GEMM 4] dQ = dS @ K
        acc_dQ = mma_dsk_fn(A_idx=smem_idx_PdS, wg_wait=1)  # waits GEMM 3
        pipeline_do.consumer_release(c_state)
        # [GEMM 5] dK += dS^T @ Q
        mma_dsq_fn(tCrA=tdKrdS, B_idx=smem_idx, zero_init=~dKV_accumulate, wg_wait=1)  # waits GEMM 4

        # acc_dQ -> sdQaccum chunk (ping-pong with the store warp)
        cute.arch.barrier(
            barrier_id=BAR_DQ_EMPTY0 + wg_idx,
            number_of_threads=128 + cute.arch.WARP_SIZE,
        )
        tdQrdQaccum_flat = cute.make_tensor(
            acc_dQ.iterator, cute.make_layout(tdQsdQaccum.shape)
        )
        cute.autovec_copy(tdQrdQaccum_flat, tdQsdQaccum)
        cute.arch.fence_view_async_shared()
        cute.arch.barrier_arrive(
            barrier_id=BAR_DQ_FULL0 + wg_idx,
            number_of_threads=128 + cute.arch.WARP_SIZE,
        )

        warpgroup.wait_group(0)  # GEMM 5 done (sQ free)
        pipeline_q.consumer_release(c_state)
        c_state.advance()
        return c_state

    @cute.jit
    def _epilogue_dKV(
        self, acc_dV, mdV, sV, acc_dK, mdK, sK,
        tma_atom_dK, tma_atom_dV, tiled_mma_dKV,
        tidx, n_block, head, batch,
    ):
        epi_barrier = pipeline.NamedBarrier(
            barrier_id=BAR_EPI, num_threads=self.num_mma_threads
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        mdK_cur = mdK[None, None, head, batch]
        mdV_cur = mdV[None, None, head, batch]
        gdK = cute.local_tile(mdK_cur, (self.tile_n, self.head_dim), (n_block, 0))
        gdV = cute.local_tile(mdV_cur, (self.tile_n, self.head_dim), (n_block, 0))
        store_dK, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_dK, 0, cute.make_layout(1), sK, gdK, single_stage=True
        )
        store_dV, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_dV, 0, cute.make_layout(1), sV, gdV, single_stage=True
        )
        copy_dV_r2s, _, _ = copy_utils.get_smem_store_C(
            tiled_mma_dKV, sV, tidx, transpose=False, position_independent=True
        )
        copy_dK_r2s, _, _ = copy_utils.get_smem_store_C(
            tiled_mma_dKV, sK, tidx, transpose=False, position_independent=True
        )
        # both WGs past their last-iteration wait_group(0): sK/sV are dead
        epi_barrier.arrive_and_wait()
        copy_dV_r2s(acc_dV, dst_idx=None)
        cute.arch.fence_view_async_shared()
        epi_barrier.arrive_and_wait()
        if warp_idx == 4:
            store_dV()
            cute.arch.cp_async_bulk_commit_group()
        epi_barrier.arrive_and_wait()
        copy_dK_r2s(acc_dK, dst_idx=None)
        cute.arch.fence_view_async_shared()
        epi_barrier.arrive_and_wait()
        if warp_idx == 4:
            store_dK()
            cute.arch.cp_async_bulk_commit_group()

    @cute.jit
    def _epilogue_dKV_accum(
        self, acc_dV, mdVaccum, acc_dK, mdKaccum, sdQaccum,
        r2s_tiled_copy_dKVaccum, tidx, wg_idx, warp_idx, n_block, head, batch,
    ):
        """Split-M epilogue: bulk-reduce-add fp32 dK/dV fragments into the flat
        gmem accum buffers, one WG chunk at a time through the (now idle)
        sdQaccum smem. Element order = dKV accumulator fragment order; the
        dKV postprocess kernel undoes it."""
        epi_barrier = pipeline.NamedBarrier(
            barrier_id=BAR_EPI, num_threads=self.num_mma_threads
        )
        chunk = self.tile_n * self.head_dim // self.num_wg_mma
        s_chunk = cute.make_tensor(sdQaccum.iterator, cute.make_layout(chunk))
        thr_copy = r2s_tiled_copy_dKVaccum.get_slice(tidx % 128)
        t_s = thr_copy.partition_D(s_chunk)
        gdVaccum = cute.local_tile(mdVaccum[None, head, batch], (chunk,), (None,))
        gdKaccum = cute.local_tile(mdKaccum[None, head, batch], (chunk,), (None,))

        # the store warp's last dQ bulk-adds must be done reading sdQaccum
        cute.arch.barrier(
            barrier_id=BAR_DQ_EMPTY0 + wg_idx,
            number_of_threads=128 + cute.arch.WARP_SIZE,
        )
        for acc, g in ((acc_dV, gdVaccum), (acc_dK, gdKaccum)):
            for wg in cutlass.range_constexpr(self.num_wg_mma):
                if wg_idx == wg:
                    t_r = cute.make_tensor(acc.iterator, cute.make_layout(t_s.shape))
                    cute.autovec_copy(t_r, t_s)
                    cute.arch.fence_view_async_shared()
                epi_barrier.arrive_and_wait()
                if warp_idx == 4:
                    with cute.arch.elect_one():
                        copy_utils.cpasync_reduce_bulk_add_f32(
                            s_chunk.iterator,
                            g[None, n_block * self.num_wg_mma + wg].iterator,
                            chunk * Float32.width // 8,
                        )
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                epi_barrier.arrive_and_wait()


# =====================================================================
# Phase C: postprocess (fp32 accum in fragment order -> *scale -> bf16)
# =====================================================================
class WanFlashBwdPostprocessSm90:
    """Converts a flat fp32 accumulation buffer whose element order is a
    WGMMA accumulator fragment order into a bf16 (b, s, h, d) tensor.
    Two instantiations, matching the main kernel's MMAs:
      dQ:    tile_rows=tile_m, atom_m=AtomLayoutMdQ=1, swapAB=dQ_swapAB,
             scale=softmax_scale
      dK/dV: tile_rows=tile_n, atom_m=AtomLayoutNdKV=2, swapAB=False,
             scale=1.0 (dK was scaled in the main epilogue)"""

    def __init__(
        self,
        head_dim: int = 128,
        tile_rows: int = 80,
        num_wg: int = 2,
        atom_m: int = 1,
        swapAB: bool = False,
        scale: float = 1.0,
    ):
        assert num_wg % atom_m == 0
        self.head_dim = head_dim
        self.tile_rows = tile_rows
        self.num_wg = num_wg
        self.atom_m = atom_m
        self.swapAB = swapAB
        self.scale = scale
        self.num_threads = 128 * num_wg
        self.dtype = BFloat16

    def _make_tiled_mma(self):
        atom = (self.atom_m, self.num_wg // self.atom_m)
        tiler = (self.tile_rows // atom[0], self.head_dim // atom[1])
        return cutlass.utils.hopper_helpers.make_trivial_tiled_mma(
            self.dtype, self.dtype,
            warpgroup.OperandMajorMode.K,  # majorness irrelevant: accum only
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(atom if not self.swapAB else atom[::-1]) + (1,),
            tiler_mn=tiler if not self.swapAB else tiler[::-1],
        )

    @cute.jit
    def __call__(
        self,
        mdQaccum: cute.Tensor,  # (b, h, s_rounded * d) fp32
        mdQ: cute.Tensor,       # (b, s, h, d) bf16 out
        stream: cuda.CUstream,
    ):
        mdQ = layout_utils.select(mdQ, [1, 3, 2, 0])          # (s, d, h, b)
        mdQaccum = layout_utils.select(mdQaccum, [2, 1, 0])   # (flat, h, b)

        tiled_mma = self._make_tiled_mma()

        # G2S: flat 128-bit cp.async chunks
        assert (self.tile_rows * self.head_dim // 4) % self.num_threads == 0
        g2s_tiled_copy = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                Float32, num_bits_per_copy=128,
            ),
            cute.make_layout(self.num_threads),
            cute.make_layout(4),
        )
        # S2R: the same (thread, chunk) mapping the main kernel used to dump
        s2r_tiled_copy = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32,
                                num_bits_per_copy=128),
            cute.make_layout((128, self.num_wg)),
            cute.make_layout(4),
        )
        sdQaccum_layout = cute.make_layout(
            (self.tile_rows * self.head_dim // self.num_wg, self.num_wg)
        )
        sdQ_layout = sm90_utils.make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_rows, self.head_dim),
            None, major_mode_size=self.head_dim // (self.num_wg // self.atom_m),
        )
        num_copy_elems = 128 // self.dtype.width
        threads_per_row = math.gcd(128, self.head_dim) // num_copy_elems
        gmem_tiled_copy_dQ = copy_utils.tiled_copy_2d(
            self.dtype, threads_per_row, self.num_threads, num_copy_elems
        )

        m_blocks = cute.ceil_div(cute.size(mdQ.shape[0]), self.tile_rows)
        grid = (m_blocks, cute.size(mdQ.shape[2]), cute.size(mdQ.shape[3]))
        self.kernel(
            mdQaccum, mdQ, tiled_mma, sdQaccum_layout, sdQ_layout,
            g2s_tiled_copy, s2r_tiled_copy, gmem_tiled_copy_dQ,
        ).launch(grid=grid, block=[self.num_threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        mdQaccum: cute.Tensor,
        mdQ: cute.Tensor,
        tiled_mma: cute.TiledMma,
        sdQaccum_layout: cute.Layout,
        sdQ_layout: cute.ComposedLayout,
        g2s_tiled_copy: cute.TiledCopy,
        s2r_tiled_copy: cute.TiledCopy,
        gmem_tiled_copy_dQ: cute.TiledCopy,
    ):
        smem = cutlass.utils.SmemAllocator()
        sdQaccum = smem.allocate_tensor(Float32, sdQaccum_layout, byte_alignment=1024)
        sdQaccum_flat = cute.make_tensor(
            sdQaccum.iterator, cute.make_layout(cute.size(sdQaccum))
        )
        sdQ = cute.make_tensor(
            cute.recast_ptr(sdQaccum.iterator, dtype=self.dtype), sdQ_layout
        )
        sdQt = layout_utils.transpose_view(sdQ)

        tidx, _, _ = cute.arch.thread_idx()
        m_block, head, batch = cute.arch.block_idx()
        seqlen_q = cute.size(mdQ.shape[0])

        mdQaccum_cur = mdQaccum[None, head, batch]
        gdQaccum = cute.local_tile(
            mdQaccum_cur, (self.tile_rows * self.head_dim,), (m_block,)
        )
        mdQ_cur = mdQ[None, None, head, batch]
        gdQ = cute.local_tile(mdQ_cur, (self.tile_rows, self.head_dim), (m_block, 0))

        # 1) G -> S (cp.async, coalesced)
        g2s_thr = g2s_tiled_copy.get_slice(tidx)
        tdQgdQaccum = g2s_thr.partition_S(gdQaccum)
        tdQsdQaccum_g2s = g2s_thr.partition_D(sdQaccum_flat)
        cute.copy(g2s_tiled_copy, tdQgdQaccum, tdQsdQaccum_g2s)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        # 2) S -> R: reinterpret this thread's flat slots as its accum fragment
        s2r_thr = s2r_tiled_copy.get_slice(tidx)
        tdQsdQaccum = s2r_thr.partition_S(sdQaccum)
        tile_shape = (self.tile_rows, self.head_dim)
        acc_shape = tiled_mma.partition_shape_C(
            tile_shape if const_expr(not self.swapAB) else tile_shape[::-1]
        )
        acc = cute.make_rmem_tensor(acc_shape, Float32)
        tdQrdQaccum = cute.make_tensor(
            acc.iterator, cute.make_layout(tdQsdQaccum.shape)
        )
        cute.autovec_copy(tdQsdQaccum, tdQrdQaccum)
        acc.store(acc.load() * self.scale)

        # 3) R -> S via StMatrix (transpose when swapAB), converting to bf16
        cute.arch.barrier()  # everyone done reading sdQaccum before recast write
        copy_dQ_r2s, _, _ = copy_utils.get_smem_store_C(
            tiled_mma, sdQ if const_expr(not self.swapAB) else sdQt,
            tidx, transpose=self.swapAB,
        )
        copy_dQ_r2s(acc, dst_idx=None)

        # 4) S -> R -> G, coalesced, row-tail predicated
        cute.arch.barrier()
        gmem_thr = gmem_tiled_copy_dQ.get_slice(tidx)
        tdQsdQ = gmem_thr.partition_S(sdQ)
        tdQgdQ = gmem_thr.partition_D(gdQ)
        tdQrdQ = cute.make_fragment_like(tdQsdQ, self.dtype)
        cute.autovec_copy(tdQsdQ, tdQrdQ)
        cdQ = cute.make_identity_tensor(tile_shape)
        tdQcdQ = gmem_thr.partition_S(cdQ)
        row_limit = seqlen_q - m_block * self.tile_rows
        for rest_m in cutlass.range(cute.size(tdQrdQ.shape[1]), unroll_full=True):
            if tdQcdQ[0, rest_m, 0][0] < row_limit:
                cute.copy(
                    gmem_tiled_copy_dQ, tdQrdQ[None, rest_m, None],
                    tdQgdQ[None, rest_m, None],
                )
