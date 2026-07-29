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
bench/              baseline comparison + feature-matrix runner + perf gate
docs/DESIGN.md      kernel architecture (from FA3/FA4 source study)
docs/FEATURES.md    per-feature A/B verdicts with measurements
```

## Status

Under active development — see `docs/FEATURES.md` for what is implemented and
how it currently measures against FA3/FA4 at each shape.

## License / provenance

BSD-3-Clause. Techniques informed by the FlashAttention-2/3/4 papers and by
reading the BSD-licensed FlashAttention CuTeDSL sources; all kernel code here
is written fresh against nvidia-cutlass-dsl 4.6.0.
