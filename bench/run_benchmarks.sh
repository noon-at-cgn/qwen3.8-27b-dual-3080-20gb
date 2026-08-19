#!/bin/bash
# Reproduce the README numbers against the server you have running on this
# box: real-prompt cohorts at C1/C2/C4/C8, default sampling and greedy,
# reporting e2e tok/s, decode tok/s, tokens-per-step (DFlash2 acceptance) and
# TTFT. ~5-10 min.
#
#   bash bench/run_benchmarks.sh
#
# Run it twice after a restart and keep the second numbers: the first run
# after start includes JIT warmup and reads 30-50% low.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
cd "$REPO"
export PATH="$REPO/venv/bin:$PATH"
export OPENAI_API_KEY=${VLLM_API_KEY:-$(cat "$REPO/api_key.txt" 2>/dev/null)}
HOST=${HOST:-127.0.0.1}; PORT=${PORT:-8000}
MODEL=${MODEL:-$REPO/models/Qwen3.8-27B-W4A16-AutoRound}
B="venv/bin/vllm bench serve --host $HOST --port $PORT --model $MODEL --served-model-name qwen3.8-27b"
OUT=${OUT:-$HERE/results}; mkdir -p "$OUT"

curl -sf -o /dev/null http://$HOST:$PORT/health || { echo "no server on $HOST:$PORT"; exit 1; }
metrics() { curl -s http://$HOST:$PORT/metrics -H "Authorization: Bearer $OPENAI_API_KEY"; }
spec() { metrics | grep -E "^vllm:spec_decode_num_(drafts|accepted_tokens)_total" | awk '{print $2}' | tr "\n" " "; }
num() { awk "/$1/ {print \$$2}" "$3"; }
tokstep() { python3 -c "
a='$1'.split(); b='$2'.split()
try:
    d=float(b[0])-float(a[0]); acc=float(b[1])-float(a[1]); print(f'{1+acc/d:.2f}' if d>0 else '-')
except Exception: print('-')"; }

echo "# $(date) server=$HOST:$PORT"
$B --dataset-name random --random-input-len 256 --random-output-len 256 --num-prompts 16 --max-concurrency 8 > /dev/null 2>&1   # warmup

for T in default 0; do
  TARG=""; [ "$T" = "0" ] && TARG="--temperature 0"
  for C in 1 2 4 8; do
    S0=$(spec)
    $B --dataset-name custom --dataset-path $HERE/prompts_real.jsonl --custom-output-len 1024 --num-prompts 8 --max-concurrency $C $TARG > $OUT/cohort_T${T}_c$C.log 2>&1
    S1=$(spec)
    L="cohort C$C real prompts T=$T"; F=$OUT/cohort_T${T}_c$C.log
    echo "ROW $L | e2e=$(num "Output token throughput" 5 $F) tok/s | decode(C/meanTPOT)=$(python3 -c "print(f'{$C*1000/$(num "Mean TPOT" 4 $F):.1f}')") | tok/step=$(tokstep "$S0" "$S1") | meanTTFT=$(num "Mean TTFT" 4 $F) ms"
  done
done
echo "# raw logs in $OUT"
