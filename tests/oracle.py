"""Chunked fp32 attention oracle — forward + analytic backward at TRUE Wan
shapes (a naive reference at S=75,600 needs 22.9 GB/head). Ported from the
wan-attn project's battle-tested version; gate convention: candidate error
<= 2x FA3's on identical inputs, per-tensor for o/dq/dk/dv.
"""

import math

import torch

CHUNK = 2048


@torch.no_grad()
def oracle_fwd(q, k, v, scale=None, dtype=torch.float32):
    """q,k,v: (b, h, s, d). Returns (o, lse) in `dtype`; lse natural-log."""
    b, h, sq, d = q.shape
    scale = scale if scale is not None else 1.0 / math.sqrt(d)
    q, k, v = (t.to(dtype) for t in (q, k, v))
    o = torch.empty(b, h, sq, d, device=q.device, dtype=dtype)
    lse = torch.empty(b, h, sq, device=q.device, dtype=dtype)
    for bi in range(b):
        for hi in range(h):
            kq, kk, kv_ = q[bi, hi], k[bi, hi], v[bi, hi]
            for i0 in range(0, sq, CHUNK):
                i1 = min(i0 + CHUNK, sq)
                s = (kq[i0:i1] @ kk.T) * scale
                l = torch.logsumexp(s, dim=-1)
                o[bi, hi, i0:i1] = torch.exp(s - l[:, None]) @ kv_
                lse[bi, hi, i0:i1] = l
    return o, lse


@torch.no_grad()
def oracle_bwd(q, k, v, o, lse, do, scale=None, dtype=torch.float32):
    b, h, sq, d = q.shape
    scale = scale if scale is not None else 1.0 / math.sqrt(d)
    q, k, v, o, lse, do = (t.to(dtype) for t in (q, k, v, o, lse, do))
    dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    for bi in range(b):
        for hi in range(h):
            kq, kk, kv_ = q[bi, hi], k[bi, hi], v[bi, hi]
            ko, kl, kdo = o[bi, hi], lse[bi, hi], do[bi, hi]
            D = (kdo * ko).sum(-1)
            for i0 in range(0, sq, CHUNK):
                i1 = min(i0 + CHUNK, sq)
                p = torch.exp((kq[i0:i1] @ kk.T) * scale - kl[i0:i1, None])
                ds = p * (kdo[i0:i1] @ kv_.T - D[i0:i1, None]) * scale
                dq[bi, hi, i0:i1] = ds @ kk
                dk[bi, hi] += ds.T @ kq[i0:i1]
                dv[bi, hi] += p.T @ kdo[i0:i1]
    return dq, dk, dv


def error_metrics(x, ref):
    x, ref = x.float(), ref.float()
    return {
        "max_abs": (x - ref).abs().max().item(),
        "rel_l2": ((x - ref).norm() / ref.norm().clamp_min(1e-12)).item(),
        "finite": bool(torch.isfinite(x).all().item()),
    }
