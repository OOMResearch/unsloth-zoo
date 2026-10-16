# Runtime cut cross-entropy

Uncapped, unsmoothed BF16 frozen heads use the existing vocabulary-chunked CCE
forward, then recompute native MLX CE derivatives in blocks of at most 128 token
rows. BF16 weights and activations stay BF16. Backward logit workspace is bounded to 128 rows by vocabulary size,
and frozen weight gradients are not requested. Each block depends on the preceding
hidden gradient so its workspace can be released before the next projection.
The final partial block is padded when the batch spans multiple blocks; padded
cotangents are zero and padded gradients are discarded.

This preserves native CE's BF16 rounding in its softmax derivative and GEMMs.
The previous chunk-wise analytic/precomputed derivative was close at the hidden
states but failed strict pretrained adapter-gradient agreement after passing
through the BF16 backbone. Even native CE backward with tiny token chunks can
select different GEMM paths and reproduce the discrepancy; 128-row blocks and
padded tails preserve agreement in the tested cases. Native CE is invoked only
by the backward; the forward still executes the CCE Metal kernels.

Ignored labels have zero cotangent. Invalid labels are validated before integer
narrowing and poison the corresponding gradient row with NaN. Empty inputs and
unused cotangents return zero gradients. Quantized head weights remain frozen.
Float32, trainable-head, softcapped and label-smoothed paths retain their existing
implementations; this fix does not certify those paths on every architecture.
The precompute scheduling option yields to the native BF16 backward in the
eligible case, so it does not precompute a hidden gradient there.

The tradeoff is additional projection work and workspace. On M5 Max, MLX 0.32.1,
a compiled frozen head with 512 tokens, vocabulary 151,936 and hidden width 896
measured approximately 8.34 ms before versus 12.61 ms after, with extra peak
allocation about 428 MiB versus 644 MiB. These are five illustrative head-only
samples, not end-to-end training measurements. The 128-token 4-bit head memory
case measured 12.51 MiB; its regression ceiling is 14 MiB.

`tests/test_mlx_cce_bf16_backward.py` covers eager/compiled execution, dense and
quantized heads, partial/empty blocks, ignored labels, invalid wide labels,
workspace row bounds and forward CCE execution. The sibling benchmark harness's
pretrained Qwen2.5-0.5B 4-bit LoRA tests cover original and trained adapters,
normal and partial-block lengths. The original 127-token BF16 check passes the
unchanged loss/gradient/SGD/cosine gate with zero measured gradient and update
difference. Other hardware and full training trajectories still need validation.
