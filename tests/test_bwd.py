"""Backward correctness: wan-flash bwd vs the chunked-fp32 oracle, gated at
<= 2x FA3's raw-backward error on identical inputs, per-tensor (dq, dk, dv).
Inputs come from OUR forward (o, lse) -- both kernels get the same o/lse/do,
so the gate compares backward implementations, not forward noise. Also checks
the FA3-fwd hybrid contract (natural-log LSE drop-in) and the autograd chain.
"""

import math
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from tests.oracle import error_metrics, oracle_bwd, oracle_fwd
from wan_flash.shapes import WanShape

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# small shapes for the fast battery; true shapes behind WAN_FLASH_SLOW_TESTS
FAST_SHAPES = [
    WanShape("self", 1, 2, 1024, 1024),
    WanShape("self", 1, 2, 960, 960),      # sq%80==0, skv%128!=0 -> K-tail path
    WanShape("self", 1, 4, 3000, 3000),    # ragged both dims (m-tail 40, K-tail 56)
    WanShape("cross", 1, 2, 2048, 512),
    WanShape("cross", 1, 2, 2048, 257),    # I2V image branch length
]


def _fa3_raw_bwd(q, k, v, o, lse, do):
    import flash_attn_interface as fa3

    dq, dk, dv = (torch.empty_like(t) for t in (q, k, v))
    fa3._flash_attn_backward(
        do.contiguous(), q, k, v, o, lse, None, None, None, None, None, None,
        dq, dk, dv, softmax_scale=q.shape[-1] ** (-0.5), is_causal=False,
        deterministic=False,
    )
    return dq, dk, dv


def _run(shape, seed=0):
    from wan_flash.interface import wan_flash_bwd, wan_flash_fwd

    torch.manual_seed(seed)
    q = torch.randn(shape.b, shape.sq, shape.h, shape.d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape.b, shape.skv, shape.h, shape.d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    do = torch.randn_like(q)

    o, lse = wan_flash_fwd(q, k, v)
    ours = wan_flash_bwd(q, k, v, o, lse, do)
    fa3 = _fa3_raw_bwd(q, k, v, o, lse, do)

    qh, kh, vh, oh, doh = (t.transpose(1, 2) for t in (q, k, v, o, do))
    refs = oracle_bwd(qh, kh, vh, oh, lse.float(), doh)

    for name, x, x3, ref in zip(("dq", "dk", "dv"), ours, fa3, refs):
        e_ours = error_metrics(x.transpose(1, 2), ref)
        e_fa3 = error_metrics(x3.transpose(1, 2), ref)
        assert e_ours["finite"], f"non-finite {name} at {shape.name}"
        assert e_ours["rel_l2"] <= 2.0 * e_fa3["rel_l2"] + 1e-9, (
            f"{shape.name} {name}: ours {e_ours['rel_l2']:.3e} > "
            f"2x FA3 {e_fa3['rel_l2']:.3e}"
        )
    # no NaN leaked through the ragged tails
    for x in ours:
        assert torch.isfinite(x).all()


@requires_gpu
@pytest.mark.parametrize("shape", FAST_SHAPES, ids=lambda s: s.name)
def test_bwd_fast(shape):
    _run(shape)


@requires_gpu
def test_bwd_fa3_fwd_hybrid():
    """Our bwd consuming FA3's forward outputs (the lse-contract drop-in)."""
    import flash_attn_interface as fa3i
    from wan_flash.interface import wan_flash_bwd

    torch.manual_seed(0)
    q = torch.randn(1, 1024, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 1024, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    do = torch.randn_like(q)
    o3, lse3 = fa3i._flash_attn_forward(q, k, v)[:2]
    ours = wan_flash_bwd(q, k, v, o3, lse3, do)
    fa3 = _fa3_raw_bwd(q, k, v, o3, lse3, do)
    qh, kh, vh, oh, doh = (t.transpose(1, 2) for t in (q, k, v, o3, do))
    refs = oracle_bwd(qh, kh, vh, oh, lse3.float(), doh)
    for name, x, x3, ref in zip(("dq", "dk", "dv"), ours, fa3, refs):
        e_ours = error_metrics(x.transpose(1, 2), ref)
        e_fa3 = error_metrics(x3.transpose(1, 2), ref)
        assert e_ours["rel_l2"] <= 2.0 * e_fa3["rel_l2"] + 1e-9, (
            f"hybrid {name}: {e_ours['rel_l2']:.3e} > 2x {e_fa3['rel_l2']:.3e}"
        )


@requires_gpu
def test_autograd_chain():
    """wan_flash_attn end-to-end vs torch SDPA autograd at a small shape."""
    from wan_flash.interface import wan_flash_attn

    torch.manual_seed(0)
    q = torch.randn(1, 960, 2, 128, device="cuda", dtype=torch.bfloat16,
                    requires_grad=True)
    k = torch.randn(1, 960, 2, 128, device="cuda", dtype=torch.bfloat16,
                    requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    do = torch.randn_like(q)

    o = wan_flash_attn(q, k, v)
    o.backward(do)
    grads = (q.grad.clone(), k.grad.clone(), v.grad.clone())

    q2, k2, v2 = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    o_ref = torch.nn.functional.scaled_dot_product_attention(
        q2.transpose(1, 2).float(), k2.transpose(1, 2).float(),
        v2.transpose(1, 2).float(),
    ).transpose(1, 2)
    o_ref.backward(do.float())

    assert (o.float() - o_ref).abs().max().item() < 5e-2
    for g, g_ref in zip(grads, (q2.grad, k2.grad, v2.grad)):
        rel = (g.float() - g_ref).norm() / g_ref.norm().clamp_min(1e-12)
        assert rel.item() < 6e-3, f"autograd grad rel_l2 {rel.item():.3e}"


@requires_gpu
@pytest.mark.skipif(__import__("os").environ.get("WAN_FLASH_SLOW_TESTS") != "1",
                    reason="WAN_FLASH_SLOW_TESTS=1 for true shapes")
@pytest.mark.parametrize("shape", [WanShape("self", 1, 12, 32760, 32760),
                                   WanShape("self", 1, 12, 75600, 75600)],
                         ids=lambda s: s.name)
def test_bwd_true_shapes(shape):
    _run(shape)
