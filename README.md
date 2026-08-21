# Qwen3.8-27B on 2× RTX 3080 20GB

Run **Qwen3.8-27B** on two 20 GB consumer GPUs (**40 GB total VRAM**) with tensor parallelism, the model's full native **262,144-token context window**, and **DFlash2 speculative decoding**.

On two RTX 3080 20GB cards, this setup reaches **133–140 tok/s** for single-stream generation on real chat prompts.

The project is inspired by and builds on the quantization and vLLM work from [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090) (Apache-2.0), which targets a single 24 GB RTX 3090. This version adapts the approach to two smaller GPUs without NVLink and benchmarks the available configurations to find the fastest combination for this hardware.

|                      |                                         |
| -------------------- | --------------------------------------- |
| **Generation speed** | **133–140 tok/s** with default sampling |
| **Greedy decoding**  | **138–139 tok/s**                       |
| **Context length**   | **262,144 tokens**                      |
| **Hardware**         | **2× RTX 3080 20GB**                    |
| **GPU interconnect** | PCIe, no NVLink, no GPU-to-GPU P2P      |

> **Note:** The RTX 3080 20GB cards used here are non-reference models. Standard RTX 3080 cards have 10GB or 12GB of VRAM.

### Uncensored variant

[noon-at-cgn/Qwen3.8-27B-Uncensored-W4A16-AutoRound](https://huggingface.co/noon-at-cgn/Qwen3.8-27B-Uncensored-W4A16-AutoRound)
is the same quantization recipe applied to
[orcarouter/Qwen3.8-27B-Uncensored](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored)
(an abliterated base) instead — same architecture, same excluded-module set,
same serving config. Drop-in replacement: point `MODEL` at that repo instead
of `dbirks/Qwen3.8-27B-W4A16-AutoRound` and everything else in this README
(config knobs, systemd, Docker) works unchanged, at essentially the same
133-140 tok/s. See that model's card for eval scores and its safety
disclaimer before using it.

## Quick Start

### Docker

```bash
git clone <this-repo-url>
cd qwen3.8-27b-dual-3080-20gb

echo "VLLM_API_KEY=$(openssl rand -hex 24)" > .env

docker compose up -d
docker compose logs -f serve
```

### Manual installation

See [Setup](#setup) below for a step-by-step installation.

Once the server is running:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $(cat api_key.txt)" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "hej"}],
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Run the verification and benchmark scripts to check your installation:

```bash
bash verify.sh
bash bench/run_benchmarks.sh
```

Run the benchmark twice after a restart. The first run can be slower because CUDA kernels and graphs may still be warming up.

---

## Setup

### Requirements

* 2× NVIDIA GPUs with **at least 20 GB VRAM each**
* Ampere architecture or newer
* Recent NVIDIA driver
* Python 3.12
* ~40 GB of free disk space
* vLLM 0.27.1
* CUDA-compatible PyTorch environment

This project was specifically tested on **2× RTX 3080 20GB**.

Before starting, check your PCIe topology and whether GPU P2P is available:

```bash
nvidia-smi topo -m

python3 -c "import torch; print(torch.cuda.can_device_access_peer(0, 1))"
```

If P2P is available, start the server with:

```bash
P2P=1 bash serve.sh
```

Otherwise, the default configuration disables NCCL P2P and custom all-reduce.

### Installation

```bash
git clone <this-repo-url> ~/qwen-dual-3080
cd ~/qwen-dual-3080

python3 -m venv venv

venv/bin/pip install \
  vllm==0.27.1 \
  huggingface_hub \
  hf_transfer \
  ninja \
  pandas
```

Download the base checkpoint:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 \
venv/bin/hf download \
  dbirks/Qwen3.8-27B-W4A16-AutoRound \
  --local-dir models/Qwen3.8-27B-W4A16-AutoRound
```

DFlash2 shares the embedding and language-model head with the target model, so these layers are requantized to int8:

```bash
venv/bin/python quant_lm_head.py \
  models/Qwen3.8-27B-W4A16-AutoRound

venv/bin/python quant_embed.py \
  models/Qwen3.8-27B-W4A16-AutoRound
```

Download the faster int4-GPTQ language-model head and the DFlash2 drafter:

```bash
venv/bin/python fetch_fast_variant.py
venv/bin/python fetch_dflash2.py
```

Finally, apply the vLLM patches:

```bash
for p in patches/*.patch; do
  patch -p1 -d venv/lib/python3.12/site-packages/vllm < "$p"
done
```

Generate an API key:

```bash
openssl rand -hex 24 > api_key.txt
```

Verify the installation:

```bash
bash verify.sh --no-server
```

The verification script checks:

* vLLM and Python environment
* required patches
* model and DFlash2 checkpoints
* GPU visibility
* GPU P2P availability
* model preparation

Start the server:

```bash
bash serve.sh
```

The first startup can take several minutes while `torch.compile`, CUDA graph capture, and JIT-compiled kernels are initialized. Subsequent starts reuse the compiled cache.

---

## Configuration

The main settings are exposed through `serve.sh`:

| Variable          |  Default | Description                                              |
| ----------------- | -------: | -------------------------------------------------------- |
| `TP`              |      `2` | Tensor-parallel degree                                   |
| `P2P`             |      `0` | Set to `1` when GPU P2P is available                     |
| `GPU_UTIL`        |  `0.965` | vLLM GPU memory utilization                              |
| `MAX_LEN`         | `262144` | Maximum context length                                   |
| `MAX_SEQS`        |      `8` | Maximum concurrent request slots                         |
| `PREFIX_CACHE`    |      `0` | Enable prefix caching                                    |
| `VISION`          |      `0` | Enable image input                                       |
| `ENABLE_THINKING` |   `true` | Default thinking mode for clients that do not specify it |
| `PORT`            |   `8000` | API server port                                          |

For example:

```bash
MAX_LEN=131072 MAX_SEQS=4 bash serve.sh
```

### Tool Calling

Tool calling is enabled by default with:

```text
--enable-auto-tool-choice
--tool-call-parser qwen3_coder
```

This is required by clients such as OpenWebUI when they send:

```text
tool_choice: "auto"
```

Without these flags, vLLM rejects such requests.

---

## Systemd

The repository includes a systemd service for running the server automatically:

```bash
sudo cp -r ~/qwen-dual-3080 \
  /opt/qwen3.8-27b-dual-3080-20gb

sudo cp \
  /opt/qwen3.8-27b-dual-3080-20gb/vllm-qwen3.8-27b.service \
  /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable --now vllm-qwen3.8-27b
```

Follow the logs with:

```bash
journalctl -u vllm-qwen3.8-27b -f
```

The service waits for both GPUs to report near-zero memory usage before starting. This prevents a restart from occurring while a previous process is still releasing VRAM.

This matters because vLLM profiles its KV-cache pool during startup. Starting while residual VRAM is still occupied can silently reduce the available KV-cache capacity.

If you install the project somewhere other than `/opt/qwen3.8-27b-dual-3080-20gb`, update `WorkingDirectory=` and `ExecStart=` in the service file.

---

## Docker

The Docker setup requires:

* NVIDIA driver **≥ 580** with CUDA 13 support
* Docker
* NVIDIA Container Toolkit

See the [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Start the complete stack with:

```bash
git clone <this-repo-url>
cd qwen3.8-27b-dual-3080-20gb

echo "VLLM_API_KEY=$(openssl rand -hex 24)" > .env

docker compose up -d
docker compose logs -f serve
```

The first `docker compose up` builds the image and prepares the model. The model is stored under `./models`.

The same `serve.sh` configuration options can be supplied through `.env`.

`PORT`, `MODELS_DIR`, and `GPU_COUNT` are read directly by Docker Compose.

To run verification inside the container:

```bash
docker compose run --rm serve verify
```

---

## Vision

The checkpoint uses `Qwen3_5ForConditionalGeneration` and retains the model's vision tower (~0.9 GB in bf16).

By default, the server runs with:

```text
--language-model-only
```

This does **not** remove the vision tower or change the model class. Instead, it disables image/video inputs at the vLLM configuration level.

Set:

```bash
VISION=1
```

to enable image input.

### Vision + Long Context

Vision requires additional VRAM and therefore does not fit at the full 262,144-token context on 40 GB total VRAM.

At the full context length, the measured shortfall is approximately **2.3 GB**. Increasing `GPU_UTIL` cannot recover this memory because vLLM reaches its pre-flight free-memory limit at roughly `0.98`.

One validated configuration that leaves enough room for vision is:

```bash
MAX_LEN=220000
GPU_UTIL=0.977
PREFIX_CACHE=1
```

This provides approximately a **1.02× KV-cache concurrency margin** on the tested hardware.

If you change `GPU_UTIL` or `MAX_LEN`, benchmark and soak-test the new configuration before relying on it for long-running workloads.

---

## Why This Configuration?

The configuration shipped here was selected after benchmarking several alternatives on the same two RTX 3080 20GB cards.

### Tensor parallelism

The model is split across both GPUs using tensor parallelism:

```text
GPU 0 ─┐
       ├── Tensor Parallelism ── Qwen3.8-27B
GPU 1 ─┘
```

Because these cards do not have NVLink or GPU-to-GPU P2P, communication happens over PCIe. At batch size 1, that communication overhead becomes a major part of the latency.

A straightforward implementation using Qwen's MTP speculative head with fp8 KV cache reached only around **62–71 tok/s**.

### bf16 KV cache

Switching from fp8 KV cache to **bf16 KV cache**, together with the FlashAttention split-KV verification patch, nearly doubled generation speed at the full 262K context.

This may seem counterintuitive because bf16 uses more memory, but on this particular hardware the faster attention path more than compensates for the additional memory consumption.

There are also two implementation reasons for keeping the KV cache in bf16.

First, DFlash2 uses non-causal cross-attention. Before the relevant FlashInfer fix (PR #43081), the available attention backends either rejected non-causal attention or did not support fp8 KV in that mode.

Second, the FlashInfer path requires a toolchain that is problematic with the CUDA/CUB JIT compilation used by this setup, so this project disables the FlashInfer sampler with:

```text
VLLM_USE_FLASHINFER_SAMPLER=0
```

Finally, RTX 3080 is an Ampere GPU and does not have native fp8 tensor cores. FP8 KV therefore does not provide the same hardware acceleration available on newer architectures such as Ada and Hopper. In testing on this hardware, bf16 was actually faster.

### DFlash2

DFlash2 adds another performance improvement on top of the optimized attention path.

Instead of generating speculative tokens one at a time, the DFlash2 drafter proposes a **7-token block** in a single pass. On this setup, that added approximately **6%** more throughput.

The final configuration is therefore:

```text
2× RTX 3080 20GB
        │
        ▼
Tensor Parallelism
        │
        ▼
Qwen3.8-27B W4A16
        │
        ├── bf16 KV cache
        │
        └── DFlash2 speculative decoding
                 │
                 ▼
          133–140 tok/s
```

Pipeline parallelism was not used because the speculative-decoding path for this model is not supported with pipeline parallelism in vLLM 0.27.1.

---

## Memory Considerations

At:

```text
MAX_LEN=262144
GPU_UTIL=0.965
```

the KV-cache pool is essentially sized for **one full-length request**.

The configuration has been soak-tested with:

* 20,000+ token prompts
* 2,000+ token generations

and completed successfully on the tested hardware.

However, this should not be interpreted as a configuration with large memory headroom.

If you need:

* multiple long-context requests,
* longer generations,
* other CUDA workloads running alongside vLLM, or
* additional safety margin,

reduce `MAX_LEN`, `GPU_UTIL`, or `MAX_SEQS` accordingly.

---

## Benchmarking

Run:

```bash
bash bench/run_benchmarks.sh
```

For meaningful results:

1. Restart the server.
2. Run the benchmark once to warm up JIT kernels and CUDA graphs.
3. Run it again and use the second result.
4. Compare results only on the same hardware and software configuration.

The reported **133–140 tok/s** numbers are from single-stream chat-style workloads, not synthetic maximum-throughput benchmarks.

---

## License

Apache-2.0.
