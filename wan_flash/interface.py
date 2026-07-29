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
    key = (b, sq, skv, h, d, f)
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
            scheduler=f.scheduler,
        )
        _compile_cache[key] = cute.compile(kernel, *cargs, stream)
    _compile_cache[key](*cargs, stream)
    return o, lse


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

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dpsum = torch.empty(b, h, sq_rounded, device=q.device, dtype=torch.float32)
    lse_log2 = torch.empty(b, h, sq_rounded, device=q.device, dtype=torch.float32)
    dq_accum = torch.empty(b, h, sq_rounded * d, device=q.device, dtype=torch.float32)

    key = (b, sq, skv, h, d, f)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    pre_args = (
        _to_cute(o), _to_cute(do), _to_cute(lse),
        _to_cute(dpsum), _to_cute(lse_log2), _to_cute(dq_accum),
    )
    main_args = (
        _to_cute(q), _to_cute(k), _to_cute(v), _to_cute(do),
        _to_cute(lse_log2), _to_cute(dpsum), _to_cute(dq_accum),
        _to_cute(dk), _to_cute(dv),
    )
    post_args = (_to_cute(dq_accum), _to_cute(dq))
    if key not in _bwd_compile_cache:
        pre = WanFlashBwdPreprocessSm90(head_dim=d, tile_m=tile_m)
        main = WanFlashBwdSm90(
            head_dim=d, tile_m=tile_m, tile_n=f.tile_n, num_stages=f.num_stages
        )
        post = WanFlashBwdPostprocessSm90(
            head_dim=d, tile_m=tile_m, num_wg=main.num_wg_dQ
        )
        _bwd_compile_cache[key] = (
            cute.compile(pre, *pre_args, stream),
            cute.compile(main, *main_args, stream),
            cute.compile(post, *post_args, stream),
        )
    pre_c, main_c, post_c = _bwd_compile_cache[key]
    pre_c(*pre_args, stream)
    main_c(*main_args, stream)
    post_c(*post_args, stream)
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
