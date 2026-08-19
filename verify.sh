#!/bin/bash
# Check that this repo is installed the way the README numbers assume:
# venv + vLLM version, every patch applied, both GPUs visible with their P2P
# status, the model requantized (lm_head, embed_tokens), the fast variant and
# DFlash2 drafter present, keys/files present, and — if a server is running —
# that it answers and which backend/pool it came up with.
#
#   bash verify.sh            # everything
#   bash verify.sh --no-server
#   bash verify.sh --install  # only the install (venv, vLLM, patches): no GPU,
#                             # model or server checks — what the Docker build runs
# Exit code: 0 all PASS (WARNs allowed), 1 if anything FAILs.
# PY=/path/to/python overrides the interpreter (default: this repo's venv).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
NOSRV=0; INSTALL=0
for a in "$@"; do case "$a" in --no-server) NOSRV=1;; --install) INSTALL=1; NOSRV=1;; esac; done
FAILS=0
ok()   { printf "  PASS  %s\n" "$1"; }
warn() { printf "  WARN  %s\n" "$1"; }
fail() { printf "  FAIL  %s\n" "$1"; FAILS=$((FAILS+1)); }
MODEL=${MODEL:-$HERE/models/Qwen3.8-27B-W4A16-AutoRound}
PY=${PY:-$HERE/venv/bin/python}

echo "== environment"
[ -x "$PY" ] && ok "python: $PY" || { fail "no $PY (see README Setup)"; exit 1; }
VER=$($PY -c "import vllm; print(vllm.__version__)" 2>/dev/null)
[ "$VER" = "0.27.1" ] && ok "vllm $VER" || warn "vllm ${VER:-missing} (patches were written against 0.27.1)"
SP=$($PY -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null)
[ -n "$SP" ] && [ -d "$SP" ] && ok "vllm package at $SP" || { fail "cannot import vllm with $PY"; exit 1; }
if [ $INSTALL = 0 ]; then
$PY - <<'EOF' 2>/dev/null || fail "torch cannot see a CUDA GPU"
import torch; assert torch.cuda.is_available()
n = torch.cuda.device_count()
for i in range(n):
    p=torch.cuda.get_device_properties(i)
    print(f"  PASS  GPU {i}: {p.name}, {p.total_memory/2**30:.1f} GiB, sm{p.major}{p.minor}, torch {torch.__version__}")
if n < 2:
    print("  WARN  only 1 GPU visible — this repo's serve.sh defaults to TP=2")
else:
    p2p = torch.cuda.can_device_access_peer(0, 1)
    print(f"  {'PASS' if p2p else 'WARN'}  P2P GPU0<->GPU1: {p2p} ({'TP will use direct GPU-GPU' if p2p else 'TP all-reduce stages through host memory (expected on RTX 3080 — no NVLink); serve.sh defaults to NCCL_P2P_DISABLE=1'})")
EOF
command -v nvidia-smi >/dev/null && { PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1); ok "power limit ${PL} W"; }
fi
for t in triton flashinfer compressed_tensors; do $PY -c "import $t" 2>/dev/null && ok "python module $t" || fail "python module $t missing"; done

echo "== vLLM patches (patches/*.patch)"
# The reverse dry-run is exact, but two patches touching the same file (the DFlash2 pair)
# can no longer be reversed individually once both are applied; then look for their content.
for p in patches/*.patch; do
  if patch -p1 -R --dry-run -s -d "$SP" < "$p" >/dev/null 2>&1; then ok "$(basename $p) applied"
  elif $PY patches/_check_applied.py "$p" "$SP" 2>/dev/null; then ok "$(basename $p) applied (content check; hunks overlap another patch)"
  elif patch -p1 -N --dry-run -s -d "$SP" < "$p" >/dev/null 2>&1; then fail "$(basename $p) NOT applied (patch -p1 -d $SP < $p)"
  else fail "$(basename $p) neither applied nor applicable — vLLM version mismatch?"; fi
done
grep -q "VLLM_MARLIN_INT8_INCLUDE_RE" "$SP/envs.py" 2>/dev/null && ok "int8 layer-select env vars registered in envs.py" || fail "envs.py lacks VLLM_MARLIN_INT8_INCLUDE_RE"

if [ $INSTALL = 0 ]; then
echo "== model at $MODEL"
if [ ! -f "$MODEL/config.json" ]; then fail "model not found (README Setup: hf download)"; else
$PY - "$MODEL" <<'EOF'
import json, os, sys
d = sys.argv[1].rstrip("/") + "/"
c = json.load(open(d + "config.json"))
qc = c.get("quantization_config", {})
groups = qc.get("config_groups", {})
idx = json.load(open(d + "model.safetensors.index.json"))["weight_map"]
F = 0
def ok(m): print("  PASS ", m)
def fail(m):
    global F
    print("  FAIL ", m); F += 1
# lm_head / embed int8 — DFlash2's drafter shares these directly, so they're required, not optional
if "lm_head.weight_packed" in idx and any(g["targets"] == ["re:.*lm_head$"] and g["weights"]["num_bits"] == 8 for g in groups.values()): ok("lm_head requantized to int8 (quant_lm_head.py)")
else: fail("lm_head not requantized: run quant_lm_head.py")
if any(k.endswith("embed_tokens.weight_packed") for k in idx) and any(g["targets"] == ["re:.*embed_tokens$"] for g in groups.values()): ok("embed_tokens requantized to int8 (quant_embed.py)")
else: fail("embed_tokens not requantized: run quant_embed.py")
missing = [f for f in set(idx.values()) if not os.path.exists(d + f)]
if missing: fail(f"safetensors shards missing: {missing}")
else: ok(f"{len(set(idx.values()))} safetensors shards present")
sys.exit(1 if F else 0)
EOF
[ $? -ne 0 ] && FAILS=$((FAILS+1))
fi

echo "== fast variant (int4-GPTQ lm_head)"
if [ -d "$HERE/models/Qwen3.8-27B-W4A16-AutoRound-fast" ]; then ok "fast variant present (models/Qwen3.8-27B-W4A16-AutoRound-fast)"; else fail "no models/Qwen3.8-27B-W4A16-AutoRound-fast (venv/bin/python fetch_fast_variant.py)"; fi
echo "== DFlash2 drafter"
if [ -f "$HERE/models/Qwen3.8-27B-DFlash2-W4A16/config.json" ]; then
  $PY -c "import json,sys; c=json.load(open('$HERE/models/Qwen3.8-27B-DFlash2-W4A16/config.json')); assert c['architectures']==['DFlash2DraftModel'] and c['quantization_config']['quant_method']=='compressed-tensors'" 2>/dev/null && ok "DFlash2 drafter present, W4A16 (models/Qwen3.8-27B-DFlash2-W4A16)" || fail "models/Qwen3.8-27B-DFlash2-W4A16 is not a quantized DFlash2DraftModel checkpoint"
  [ -f "$SP/model_executor/models/qwen3_dflash2.py" ] || fail "DFlash2 drafter present but patches/dflash2-backport.patch not applied"
else fail "no DFlash2 drafter (venv/bin/python fetch_dflash2.py)"; fi

echo "== keys / units"
[ -s api_key.txt ] || [ -n "${VLLM_API_KEY:-}" ] && ok "API key configured (api_key.txt or VLLM_API_KEY)" || fail "no api_key.txt (openssl rand -hex 24 > api_key.txt)"
if [ -f /.dockerenv ]; then :; elif systemctl is-active vllm-qwen3.8-27b >/dev/null 2>&1; then ok "systemd unit vllm-qwen3.8-27b active"; else warn "vllm-qwen3.8-27b unit not active (fine if you launch serve.sh by hand)"; fi
fi  # INSTALL

if [ $NOSRV = 0 ]; then
  echo "== live server (127.0.0.1:${PORT:-18020})"
  PORT=${PORT:-18020}
  if curl -sf -o /dev/null http://127.0.0.1:$PORT/health; then
    ok "/health 200"
    KEY=${VLLM_API_KEY:-$(cat api_key.txt 2>/dev/null)}
    R=$(curl -s http://127.0.0.1:$PORT/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
        -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Hvad er hovedstaden i Danmark? Svar med ét ord."}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}')
    echo "$R" | grep -qi "københavn\|copenhagen" && ok "chat completion answers ('$(echo "$R" | $PY -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())' 2>/dev/null)')" || fail "chat completion wrong/failed: $(echo "$R" | head -c 200)"
    LOG=$HERE/qwen.log
    if [ -f "$LOG" ]; then
      grep -oE "Using [A-Z_]+ attention backend" "$LOG" | tail -1 | sed 's/^/  INFO  /'
      grep -oE "GPU KV cache size: [0-9,]+ tokens" "$LOG" | tail -1 | sed 's/^/  INFO  /'
      grep -oE "Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" "$LOG" | tail -1 | sed 's/^/  INFO  /'
      grep -q "MarlinLinearKernel" "$LOG" && ok "Marlin kernels in use" || true
    fi
  else warn "no server on :$PORT (start serve.sh, or pass --no-server)"; fi
fi
echo
[ $FAILS = 0 ] && echo "verify: OK ($FAILS failures)" || echo "verify: $FAILS FAILURE(S)"
exit $([ $FAILS = 0 ] && echo 0 || echo 1)
