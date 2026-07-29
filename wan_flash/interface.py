"""Host driver: torch tensors -> cute tensors (from_dlpack, fully static
layouts baked per shape) with a compile cache keyed by (shape, features).

wan_flash_fwd(q, k, v) -> (o, lse)
  q: (b, s_q, h, d) bf16, k/v: (b, s_kv, h, d) bf16 (contiguous)
  o: (b, s_q, h, d) bf16, lse: (b, h, s_q) fp32, NATURAL log.
"""

import torch

import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from wan_flash import features
from wan_flash.fwd_sm90 import WanFlashFwdSm90

_compile_cache = {}


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
        )
        _compile_cache[key] = cute.compile(kernel, *cargs, stream)
    _compile_cache[key](*cargs, stream)
    return o, lse
