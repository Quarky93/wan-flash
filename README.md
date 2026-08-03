# wan-flash

Greenfield flash attention for **Wan2.1 video-diffusion training** on H100 (sm90),
written in **CuTeDSL** (NVIDIA cutlass-dsl). Not a wrapper: the kernels in
`wan_flash/` are new source, specialized to Wan's exact workload:

- bf16, head_dim 128, MHA (no GQA), non-causal, no mask/dropout/softcap
- self-attention: S_q = S_kv ∈ {32,760 (480p/81f), 75,600 (720p/81f)}, batch 1,
  heads ∈ {12 (1.3B), 40 (14B)}
- cross-attention: S_q × 512 (dense, unmasked umT5 context; I2V adds S_q × 257)
- forward AND backward (training), fp32 accumulation, natural-log LSE
  `(b, h, s)` fp32 output contract (SAC-compatible, drop-in for wan-attn's
  `wan_attention` op)

## Design rule

Every technique choice is a **measured decision**, not an inheritance:
FlashAttention-4's new tricks (CuTeDSL authoring, softmax-rescale skipping,
scheduler design) and FlashAttention-3's Hopper design (warp specialization,
TMA pipelines, pingpong) are implemented behind feature flags in
`wan_flash/features.py` and A/B-tested at Wan shapes by `bench/bench.py`.
Where the FA3-style variant measures faster, the FA3 variant is the default —
`docs/FEATURES.md` records every verdict with numbers.

Baselines benchmarked against: flash-attn-3 3.0.0 (C++ hopper) and
flash-attn-4 4.0.0b23 (CuTeDSL) as installed packages — used for comparison
only; no kernel code is imported from them.

## Layout

```
wan_flash/
  fwd_sm90.py       forward kernel (CuTeDSL, warp-specialized TMA pipeline)
  bwd_sm90.py       backward kernels (preprocess -> main -> postprocess)
  features.py       the A/B feature-flag registry
  interface.py      torch autograd op: wan_flash_attn(q, k, v) -> (o, lse)
  shapes.py         the Wan shape registry + tile arithmetic
tests/              chunked-fp32 oracle + correctness batteries (pytest)
bench/              baseline comparison (fa3 / fa4 / wan) across Wan shapes
tools/ab_feature.py paired-ABBA A/B harness — the protocol behind every
                    verdict in docs/FEATURES.md (run it before changing a
                    default; single unpaired deltas are noise at these margins)
tools/dev*.py       inner-loop dev harnesses (kernel vs oracle, small shapes)
docs/FEATURES.md    per-feature A/B verdicts with measurements, incl. the
                    rejected ideas and why (read before re-proposing one)
docs/ROADMAP.md     remaining headroom: what is locked, by what, and what
                    would unlock it
docs/{FWD,BWD}_STUDY.md, SPECIALIZATION.md, CUTEDSL_COOKBOOK.md
                    architecture notes from the FA3/FA4 source study + the
                    DSL footguns this kernel hit (read before writing CuTeDSL)
```

## Status

**Forward: done and green — faster than FA3 and FA4 at every Wan shape.**
`tests/test_fwd.py` passes (fast battery + alt-config matrix + true shapes
behind `WAN_FLASH_SLOW_TESTS=1`); error vs the chunked-fp32 oracle sits at
the bf16 floor (rel_l2 ≈ 2.2e-3, same as FA3; LSE ≤ 1.5e-6).

H100 SXM, forward, defaults (2-CTA cluster K/V TMA multicast + auto
scheduler + packed bf16x2 converts + rescale-skip 8.0 + intra-WG overlap,
128×128, 2 stages):

| shape | wan-flash | vs FA3 | vs FA4 |
|---|---|---|---|
| self h12 S=32760 | 694.5 TFLOP/s | **1.017x** | **1.058x** |
| self h12 S=75600 | 697.8 TFLOP/s | **1.016x** | **1.050x** |
| self h40 S=32760 | 703.0 TFLOP/s | **1.019x** | **1.063x** |
| self h40 S=75600 | 691.7 TFLOP/s | **1.019x** | **1.043x** |
| cross h12 S=75600×512 | 547.9 TFLOP/s | **1.022x** | **1.200x** |
| cross h40 S=75600×512 | 544.3 TFLOP/s | **1.039x** | **1.203x** |

**Backward: done and green — faster than FA3 everywhere, ahead of or at
parity with FA4 at every Wan shape.** `tests/test_bwd.py` passes (fast
battery + cluster alt-configs, FA3-fwd hybrid, autograd chain, true shapes
behind `WAN_FLASH_SLOW_TESTS=1`); dq/dk/dv vs the chunked-fp32 oracle sit at
the bf16 floor (rel_l2 ≈ 2.3e-3, gated at ≤ 2x FA3's raw backward on
identical inputs). Three-kernel chain (preprocess → dK/dV-stationary
warp-specialized main → dQ convert), FA4's hd128 non-causal config (80×128,
SdP/dQ swapAB, register-resident dK/dV), packed bf16x2 converts, PDL, a
2-CTA cluster with Q/dO TMA multicast (halves the dominant L2 stream of the
dK/dV-stationary loop — neither FA3 nor FA4 does this in their sm90 bwd),
and a split-M schedule that fixes the small-KV cross-attention occupancy
pathology FA3/FA4 both have.

H100 SXM, raw backward (identical o/lse/do fed to all three):

| shape | wan-flash | vs FA3 | vs FA4 |
|---|---|---|---|
| self h12 S=32760 | 622.3 TFLOP/s | **1.038x** | **1.008x** |
| self h12 S=75600 | 641.8 TFLOP/s | **1.055x** | **1.018x** |
| self h40 S=32760 | 622.2 TFLOP/s | **1.029x** | 0.997x |
| self h40 S=75600 | 635.7 TFLOP/s | **1.065x** | **1.014x** |
| cross h12 S=75600×512 | 376.0 TFLOP/s | **1.707x** | **1.660x** |
| cross h40 S=75600×512 | 380.3 TFLOP/s | **1.257x** | **1.245x** |

Full 8-shape tables (fwd + bwd) and per-feature verdicts: `docs/FEATURES.md`.

## Optimization status

These defaults are a measured local optimum on sm90, not a starting point:
the forward runs at 86–90% tensor-core-busy against the 698 W power cap with
per-clock parity to FA3's C++ kernel, and the backward is the fastest of the
three implementations while all three sit in the same 76–81% utilization band.
`docs/ROADMAP.md` records the remaining headroom, what blocks each piece
(compiler pass, missing sm90 primitive, power wall) and what would unlock it.
Ideas that were built and measured but lost are kept on branches
(`dq-dsm-experiment`, `phase3-rotation-experiment`, `phase3-ptx-wgmma`) with
their verdicts in `docs/FEATURES.md` — check there before re-proposing one.

## License / provenance

BSD-3-Clause. Techniques informed by the FlashAttention-2/3/4 papers and by
reading the BSD-licensed FlashAttention CuTeDSL sources; all kernel code here
is written fresh against nvidia-cutlass-dsl 4.6.0.
