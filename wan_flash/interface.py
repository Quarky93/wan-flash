"""Host driver: torch tensors -> cute tensors (from_dlpack, fully static
layouts baked per shape) with a compile cache keyed by (shape, features).

wan_flash_fwd(q, k, v) -> (o, lse)
  q: (b, s_q, h, d) bf16, k/v: (b, s_kv, h, d) bf16 (contiguous)
  o: (b, s_q, h, d) bf16, lse: (b, h, s_q) fp32, NATURAL log.

wan_flash_bwd(q, k, v, o, lse, do) -> (dq, dk, dv)
  lse: NATURAL log, (b, h, s_q) fp32 -- the FA3/FA4 contract, so this bwd is a
  drop-in for FA3-fwd hybrids. dq is fp32-atomically accumulated (FA3/FA4
  default) => nondeterministic, matching FA3/FA4 behavior.

wan_flash_attn(q, k, v) -> o: autograd.Function chaining the two.
"""

import torch

import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from wan_flash import features
from wan_flash.fwd_sm90 import WanFlashFwdSm90

_compile_cache = {}
_bwd_compile_cache = {}


def _to_cute(t: torch.Tensor):
    return from_dlpack(t.detach(), assumed_align=16)


def wan_flash_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    b, sq, h, d = q.shape
    bk, skv, hk, dk = k.shape
    assert (b, h, d) == (bk, hk, dk) and v.shape == k.shape
    assert d == 128, "specialized for head_dim=128"
    q, k, v = (t if t.is_contiguous() else t.contiguous() for t in (q, k, v))

    o = torch.empty_like(q)
    lse = torch.empty(b, h, sq, device=q.device, dtype=torch.float32)

    f = features.get()
    scheduler = f.scheduler
    if scheduler == "auto":
        # measured policy (docs/FEATURES.md): persistent wins everywhere
        # except long-chain large-KV shapes, where the persistent walk keeps
        # each cluster pair coupled for the whole kernel and measures ~1.5%
        # slower than single (pairs retire per tile). Wan shapes: only self
        # h40@75600 lands in the "single" bucket.
        cluster_m = f.cluster_mn[0]
        m_units = -(-sq // (f.tile_m * cluster_m))
        units_per_slot = (b * h * m_units) / max(1, 132 // cluster_m)
        n_blocks = -(-skv // f.tile_n)
        scheduler = "single" if units_per_slot >= 128 and n_blocks >= 16 else "persistent"
    key = (b, sq, skv, h, d, f, scheduler)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    cargs = (_to_cute(q), _to_cute(k), _to_cute(v), _to_cute(o), _to_cute(lse))
    if key not in _compile_cache:
        kernel = WanFlashFwdSm90(
            head_dim=d,
            tile_m=f.tile_m,
            tile_n=f.tile_n,
            num_stages=f.num_stages,
            rescale_skip_threshold=f.rescale_skip_threshold,
            intra_wg_overlap=f.intra_wg_overlap,
            mma_pv_is_rs=f.mma_pv_is_rs,
            scheduler=scheduler,
            cluster_mn=f.cluster_mn,
        )
        _compile_cache[key] = cute.compile(kernel, *cargs, stream)
    _compile_cache[key](*cargs, stream)
    return o, lse


def _auto_nsplit(base_ctas: int, m_blocks: int, sm_count: int = 132) -> int:
    """Split-M factor for the bwd main kernel (docs/SPECIALIZATION.md B5).
    1 when the (n_blocks*h*b) grid fills the GPU well; otherwise the smallest
    split with >= 95% wave efficiency (fills partial waves)."""

    def eff(ctas):
        return ctas / (sm_count * -(-ctas // sm_count))

    if eff(base_ctas) >= 0.85:
        return 1
    best_k, best_eff = 1, 0.0
    for cand in range(1, min(m_blocks, 64) + 1):
        e = eff(base_ctas * cand)
        if base_ctas * cand >= sm_count and e >= 0.95:
            return cand
        if e > best_eff:
            best_k, best_eff = cand, e
    return best_k


def wan_flash_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
):
    """(b,s,h,d) bf16 q/k/v/o/do; lse (b,h,s_q) fp32 natural log ->
    (dq, dk, dv) bf16 (b,s,h,d)."""
    from wan_flash.bwd_sm90 import (
        WanFlashBwdPreprocessSm90,
        WanFlashBwdSm90,
        WanFlashBwdPostprocessSm90,
    )

    assert all(t.dtype == torch.bfloat16 for t in (q, k, v, o, do))
    assert lse.dtype == torch.float32
    b, sq, h, d = q.shape
    bk, skv, hk, dk_ = k.shape
    assert (b, h, d) == (bk, hk, dk_) and v.shape == k.shape
    assert o.shape == q.shape and do.shape == q.shape
    assert lse.shape == (b, h, sq)
    assert d == 128, "specialized for head_dim=128"
    q, k, v, o, do = (
        t if t.is_contiguous() else t.contiguous() for t in (q, k, v, o, do)
    )
    lse = lse if lse.is_contiguous() else lse.contiguous()

    f = features.get_bwd()
    tile_m = f.tile_m
    sq_rounded = (sq + tile_m - 1) // tile_m * tile_m
    m_blocks = sq_rounded // tile_m
    n_blocks = (skv + f.tile_n - 1) // f.tile_n
    skv_rounded = n_blocks * f.tile_n
    nsplit = f.nsplit if f.nsplit > 0 else _auto_nsplit(n_blocks * h * b, m_blocks)
    nsplit = min(nsplit, m_blocks)
    cluster_n = f.cluster_n if nsplit == 1 else 1  # cluster excludes split-M

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dpsum = torch.empty(b, h, sq_rounded, device=q.device, dtype=torch.float32)
    lse_log2 = torch.empty(b, h, sq_rounded, device=q.device, dtype=torch.float32)
    dq_accum = torch.empty(b, h, sq_rounded * d, device=q.device, dtype=torch.float32)
    if nsplit > 1:
        # split-M: dK/dV accumulate in fp32 gmem (must start zeroed)
        dk_accum = torch.zeros(b, h, skv_rounded * d, device=q.device, dtype=torch.float32)
        dv_accum = torch.zeros_like(dk_accum)
        main_dk, main_dv = dk_accum, dv_accum
    else:
        main_dk, main_dv = dk, dv

    key = (b, sq, skv, h, d, f, nsplit, cluster_n)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    pre_args = (
        _to_cute(o), _to_cute(do), _to_cute(lse),
        _to_cute(dpsum), _to_cute(lse_log2), _to_cute(dq_accum),
    )
    main_args = (
        _to_cute(q), _to_cute(k), _to_cute(v), _to_cute(do),
        _to_cute(lse_log2), _to_cute(dpsum), _to_cute(dq_accum),
        _to_cute(main_dk), _to_cute(main_dv),
    )
    post_args = (_to_cute(dq_accum), _to_cute(dq))
    if nsplit > 1:
        post_dk_args = (_to_cute(dk_accum), _to_cute(dk))
        post_dv_args = (_to_cute(dv_accum), _to_cute(dv))
    if key not in _bwd_compile_cache:
        pre = WanFlashBwdPreprocessSm90(head_dim=d, tile_m=tile_m)
        main = WanFlashBwdSm90(
            head_dim=d, tile_m=tile_m, tile_n=f.tile_n, num_stages=f.num_stages,
            nsplit=nsplit, cluster_n=cluster_n, sgemm_rotate=f.sgemm_rotate,
        )
        post = WanFlashBwdPostprocessSm90(
            head_dim=d, tile_rows=tile_m, num_wg=main.num_wg_dQ,
            atom_m=main.AtomLayoutMdQ, swapAB=main.dQ_swapAB,
            scale=main.softmax_scale,
        )
        compiled = [
            cute.compile(pre, *pre_args, stream),
            cute.compile(main, *main_args, stream),
            cute.compile(post, *post_args, stream),
        ]
        if nsplit > 1:
            post_dkv = WanFlashBwdPostprocessSm90(
                head_dim=d, tile_rows=f.tile_n, num_wg=main.num_wg_mma,
                atom_m=main.AtomLayoutNdKV, swapAB=main.dKV_swapAB, scale=1.0,
            )
            compiled.append(cute.compile(post_dkv, *post_dk_args, stream))
        _bwd_compile_cache[key] = compiled
    compiled = _bwd_compile_cache[key]
    compiled[0](*pre_args, stream)
    compiled[1](*main_args, stream)
    compiled[2](*post_args, stream)
    if nsplit > 1:
        compiled[3](*post_dk_args, stream)
        compiled[3](*post_dv_args, stream)
    return dq, dk, dv


class _WanFlashAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        o, lse = wan_flash_fwd(q, k, v)
        ctx.save_for_backward(q, k, v, o, lse)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = wan_flash_bwd(q, k, v, o, lse, do)
        return dq, dk, dv


def wan_flash_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Differentiable attention: our fwd + our bwd. Returns o (b,s,h,d) bf16."""
    return _WanFlashAttnFn.apply(q, k, v)
