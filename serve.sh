#!/bin/bash
# Qwen3.8-27B on 2x RTX 3080 20 GB (40 GB total) — full 262,144-token native
# context, DFlash2 speculative decoding, tensor parallelism across both cards.
#
# This is the one config this repo ships: the winner of a full sweep against
# every other option on this exact hardware (see README "How this was
# benchmarked"). DFlash2's 7-token block drafter beats Qwen's own MTP head
# once you have the combined VRAM to run bf16 KV + FlashAttention's split-KV
# verify patch at the model's full context, because DFlash2 gets a higher
# tokens-per-step (3.08-3.25 vs MTP's 2.79-2.83 on this hardware) from
# proposing a whole block in one non-autoregressive pass instead of chaining
# single-token drafts.
#
# Measured, C1 (one request at a time), real chat prompts:
#   133-140 tok/s default sampling, 138-139 tok/s greedy, 262,144 context.
#
# Prerequisites (see README "Setup"):
#   - model requantized: quant_lm_head.py, quant_embed.py (int8 lm_head/embed
#     — DFlash2's drafter shares the target's quantized lm_head directly)
#   - fetch_fast_variant.py: int4-GPTQ lm_head (the "-fast" checkpoint;
#     it also carries an int4 MTP module this repo never uses — harmless,
#     just extra bytes on disk)
#   - fetch_dflash2.py: the W4A16 DFlash2 block drafter itself
#   - every patch in patches/ applied to your vLLM install (verify.sh checks)
#
# TP: GPU count / tensor-parallel degree. Defaults to 2 (this repo's whole
#   point). Both 3080s have no NVLink and no P2P (torch.cuda.can_device_
#   access_peer is False both ways) — pass P2P=1 only if you've confirmed
#   P2P actually works between your cards.
# GPU_UTIL: 0.965 measured stable on 2x 20 GB (soak-tested: a 20,022-token
#   prompt and a 2,251-token generation, no OOM). Lower it if these cards
#   run other work too, or if you see instability under concurrent long
#   generations — headroom at 262144 is exactly 1.00x, no slack for more
#   than one resident full-length request.
# MAX_LEN: override to trade context for a bigger headroom margin.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if [ -z "$MODEL" ] && [ -d "$DIR/models/Qwen3.8-27B-W4A16-AutoRound-fast" ]; then
  MODEL=$DIR/models/Qwen3.8-27B-W4A16-AutoRound-fast
fi
MODEL=${MODEL:-$DIR/models/Qwen3.8-27B-W4A16-AutoRound}
PORT=${PORT:-8000}
MAX_SEQS=${MAX_SEQS:-8}
API_SERVERS=${API_SERVERS:-1}

TP=${TP:-2}
TP_ARGS=""
if [ "$TP" -gt 1 ]; then
  TP_ARGS="--tensor-parallel-size $TP --pipeline-parallel-size 1"
  if [ "${P2P:-0}" != "1" ]; then
    export NCCL_P2P_DISABLE=1
    TP_ARGS="$TP_ARGS --disable-custom-all-reduce"
  fi
fi

GPU_UTIL=${GPU_UTIL:-0.965}
MAX_LEN=${MAX_LEN:-262144}

if [ -z "$DRAFT" ]; then
  for d in Qwen3.8-27B-DFlash2-W4A16 Qwen3.8-27B-DFlash2; do
    [ -f "$DIR/models/$d/model.safetensors" ] && DRAFT=$DIR/models/$d && break
  done
fi
[ -n "$DRAFT" ] || { echo "needs the DFlash2 drafter: venv/bin/python fetch_dflash2.py" >&2; exit 1; }
DRAFT_TOKENS=${DFLASH_TOKENS:-7}
SPEC_CFG="{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$DRAFT_TOKENS}"
CG=${CG:-$((MAX_SEQS * (DRAFT_TOKENS + 1)))}
export VLLM_V2_CUDAGRAPH_MEM_MIB=${VLLM_V2_CUDAGRAPH_MEM_MIB:-1400}
# Lookup-augmented drafting: draft from the request's own context when it's
# reproducing something it was shown (patches/dflash2-lookup-drafting.patch).
export VLLM_DFLASH2_LOOKUP=${LOOKUP:-1}

if [ "${PREFIX_CACHE:-0}" = "1" ]; then
  EXTRA_ARGS="--enable-prefix-caching --mamba-cache-mode align ${EXTRA_ARGS}"
fi

export PATH="$DIR/venv/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_SPEC_DECODE_ATTN=${SPEC_ATTN:-1}

if [ -z "$VLLM_API_KEY" ] && [ -f "$DIR/api_key.txt" ]; then
  export VLLM_API_KEY="$(cat "$DIR/api_key.txt")"
fi

exec venv/bin/vllm serve "$MODEL" \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port $PORT \
  --gpu-memory-utilization $GPU_UTIL \
  --max-model-len $MAX_LEN \
  --max-num-seqs $MAX_SEQS \
  --api-server-count $API_SERVERS \
  --language-model-only \
  --attention-backend FLASH_ATTN --kv-cache-dtype bfloat16 \
  $TP_ARGS \
  --mamba-ssm-cache-dtype float16 \
  --async-scheduling \
  --max-num-batched-tokens 2048 \
  --speculative-config "$SPEC_CFG" \
  --compilation-config "{\"max_cudagraph_capture_size\":$CG,\"custom_ops\":[\"+rms_norm\",\"+silu_and_mul\"]}" \
  --reasoning-parser qwen3 \
  ${EXTRA_ARGS}
