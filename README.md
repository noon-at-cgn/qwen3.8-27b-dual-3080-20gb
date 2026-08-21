# Qwen3.8-27B on 2x RTX 3080 20 GB

Serve [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) on two 20 GB
consumer GPUs (40 GB combined), tensor-parallel across both cards, at the
model's **full native 262,144-token context**, with DFlash2 speculative
decoding for **133-140 tok/s single-stream** generation.

Inspired by / built on top of the quantization and vLLM-patch work in
[syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090)
(Apache-2.0), which targets a single 24 GB card. This repo re-tunes that work
for two smaller cards with no NVLink and picks the one config — tensor
parallelism + DFlash2 + bf16 KV — that came out fastest after benchmarking
every alternative on this exact hardware.

| | |
|---|---|
| single-stream, real chat prompts | **133-140 tok/s** default sampling, **138-139 tok/s** greedy |
| context | **262,144** tokens (the model's full native max) |
| hardware | 2x RTX 3080 20 GB, no NVLink, no GPU-to-GPU P2P |

## Quick start

```bash
git clone <this-repo-url> && cd qwen3.8-27b-dual-3080-20gb
echo "VLLM_API_KEY=$(openssl rand -hex 24)" > .env
docker compose up -d
docker compose logs -f serve
```

Or by hand in a venv — see [Setup](#setup).

## Setup

You need: two Ampere-or-newer NVIDIA GPUs with at least 20 GB each (tested on
2x RTX 3080 20 GB — the non-reference 20 GB boards, not the stock 10/12 GB
3080), a recent driver, Python 3.12, ~40 GB disk.

Check your PCIe/NVLink topology — this repo defaults to assuming no P2P
(the common case for non-3090/professional cards):

```bash
nvidia-smi topo -m
python3 -c "import torch; print(torch.cuda.can_device_access_peer(0, 1))"
```

If that prints `True`, pass `P2P=1` to `serve.sh` to skip the
`NCCL_P2P_DISABLE=1 --disable-custom-all-reduce` fallback.

```bash
git clone <this-repo-url> ~/qwen-dual-3080
cd ~/qwen-dual-3080

python3 -m venv venv
venv/bin/pip install vllm==0.27.1 huggingface_hub hf_transfer ninja pandas

# base model, ~19.5 GB
HF_HUB_ENABLE_HF_TRANSFER=1 venv/bin/hf download \
  dbirks/Qwen3.8-27B-W4A16-AutoRound \
  --local-dir models/Qwen3.8-27B-W4A16-AutoRound

# requantize lm_head + embeddings to int8 (CPU only, a couple minutes) —
# DFlash2's drafter shares these two with the target model directly
venv/bin/python quant_lm_head.py models/Qwen3.8-27B-W4A16-AutoRound
venv/bin/python quant_embed.py   models/Qwen3.8-27B-W4A16-AutoRound
# int4-GPTQ lm_head (serve.sh picks up the "-fast" checkpoint automatically)
venv/bin/python fetch_fast_variant.py
# the W4A16 DFlash2 block drafter
venv/bin/python fetch_dflash2.py

# patch vllm (written against 0.27.1, currently also the latest release)
for p in patches/*.patch; do
  patch -p1 -d venv/lib/python3.12/site-packages/vllm < $p
done

openssl rand -hex 24 > api_key.txt
```

Then `bash verify.sh --no-server` — checks the venv, that every patch applied,
both GPUs are visible with their P2P status, and the model is fully prepared.

```bash
bash serve.sh
```

First start takes several minutes (torch.compile, CUDA graph capture on both
GPUs, JIT-compiled kernels). Later starts reuse the compiled cache and are
much faster.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $(cat api_key.txt)" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8-27b",
       "messages": [{"role": "user", "content": "hej"}],
       "chat_template_kwargs": {"enable_thinking": false}}'
```

Then `bash verify.sh` (also probes the live server) and
`bash bench/run_benchmarks.sh` to reproduce the numbers above on your own
hardware (run it twice — the first pass after any restart reads low from JIT
warmup).

### Config knobs (`serve.sh`)

| var | default | what |
|---|---|---|
| `TP` | `2` | tensor-parallel degree (GPU count) |
| `P2P` | `0` | set `1` if your cards have working P2P |
| `GPU_UTIL` | `0.965` | `--gpu-memory-utilization` |
| `MAX_LEN` | `262144` | context length |
| `MAX_SEQS` | `8` | concurrent request slots |
| `PREFIX_CACHE` | `0` | `1` reuses KV for a shared prompt prefix across requests/turns |
| `VISION` | `0` | `1` enables image input (see [Vision](#vision) below); default is `--language-model-only` |
| `ENABLE_THINKING` | `true` | server-side default for chat clients (OpenWebUI, Hermes, ...) that don't pass `chat_template_kwargs` themselves; `preserve_thinking` is always on so reasoning survives multi-turn tool calling |
| `PORT` | `8000` | |

Tool calling (`--enable-auto-tool-choice --tool-call-parser qwen3_coder`) is
always on — needed for clients that send `tool_choice: "auto"` (OpenWebUI
errors without it: `"auto" tool choice requires --enable-auto-tool-choice and
--tool-call-parser to be set`).

### systemd

```bash
sudo cp -r ~/qwen-dual-3080 /opt/qwen3.8-27b-dual-3080-20gb
sudo cp /opt/qwen3.8-27b-dual-3080-20gb/vllm-qwen3.8-27b.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vllm-qwen3.8-27b
journalctl -u vllm-qwen3.8-27b -f
```

The unit waits for both GPUs to report near-zero memory used before
starting — restarting onto a dirty GPU (previous process still releasing
VRAM) silently shrinks the profiled KV cache pool with no error. Edit
`WorkingDirectory=`/`ExecStart=` if you installed somewhere other than
`/opt/qwen3.8-27b-dual-3080-20gb`.

### Docker

```bash
git clone <this-repo-url> && cd qwen3.8-27b-dual-3080-20gb
echo "VLLM_API_KEY=$(openssl rand -hex 24)" > .env
docker compose up -d
docker compose logs -f serve
```

Requires an NVIDIA driver that speaks CUDA 13 (≥ 580) and Docker with the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The first `up` builds the image, downloads/requantizes the model into
`./models`, then serves. Every `serve.sh` knob works from `.env`; `PORT`,
`MODELS_DIR`, and `GPU_COUNT` (default `2`) are read by compose itself.
`docker compose run --rm serve verify` runs `verify.sh` inside the container.

## Vision

The checkpoint's architecture (`Qwen3_5ForConditionalGeneration`) is
vision-capable, and its `visual.*` tower (~0.9 GB bf16) is kept through
quantization for exactly this reason. `--language-model-only` (the default,
`VISION=0`) doesn't drop the tower or change which model class loads — it
just zeroes the per-prompt image/video limit at the config level
(`vllm/config/multimodal.py get_limit_per_prompt`). Set `VISION=1` to accept
images.

Vision costs KV headroom, so it doesn't fit at the full 262,144-token
context on 40 GB total VRAM — at that context the shortfall is a genuine
~2.3 GB, not closeable by raising `GPU_UTIL` further (it hits a pre-flight
free-memory ceiling around `0.98` first). Lower `MAX_LEN` to make room; one
validated working point on this hardware is `MAX_LEN=220000` with
`GPU_UTIL=0.977` and `PREFIX_CACHE=1`, which leaves a ~1.02x KV concurrency
margin. Re-benchmark and re-soak-test (see [Why this config](#why-this-config))
before trusting a `GPU_UTIL`/`MAX_LEN` pair you haven't run yourself.

## Why this config

Benchmarked against every alternative on this exact hardware before settling
here: plain tensor parallelism with Qwen's own MTP speculative head and fp8
KV cache (the straightforward port of the upstream single-GPU recipe) reached
only ~62-71 tok/s — communication over PCIe with no NVLink dominates at batch
size 1. Switching to bf16 KV + FlashAttention's split-KV verify patch nearly
doubled that at the same full context, and DFlash2's block drafter (proposes
a whole 7-token block per pass instead of chaining single-token drafts) added
another ~6% on top — the config this repo ships. A single GPU alone, no
parallelism, was only marginally faster than the two-GPU config above at 1/8th
the context, and pipeline parallelism isn't supported for this model's
speculative-decoding path in vLLM 0.27.1 at all.

KV cache stays bf16 rather than fp8 for two independent reasons on this
hardware. First, DFlash2's drafter needs non-causal cross-attention, and
until vLLM's FlashInfer-backend fix (PR #43081, merged after this repo's
0.27.1 pin) every attention backend either rejected non-causal attention
outright or excluded fp8 KV dtypes in non-causal mode — a structural
conflict, not a tuning choice. Second, even with that fix, it requires the
FlashInfer backend, which this repo avoids (`VLLM_USE_FLASHINFER_SAMPLER=0`)
due to a CUDA/cub JIT-compile incompatibility on this toolchain. And
RTX 3080 (Ampere) has no native fp8 tensor cores at all — fp8 tensor cores
arrived with Hopper/Ada — so even where fp8 KV does work elsewhere (e.g. the
PR's own RTX 4090 benchmark), it measured *slower* than bf16, not faster.

The `--gpu-memory-utilization 0.965` this repo uses leaves KV headroom at
exactly 1.00x for the full 262144-token context — no slack for more than one
resident full-length request. It's soak-tested (a 20,000+-token prompt and a
2,000+-token generation both completed cleanly), but lower `GPU_UTIL`,
`MAX_LEN`, or `MAX_SEQS` if you need more margin for concurrent long
generations, or if these cards run other work too.

## License

Apache-2.0, same as the upstream repo and the model.
