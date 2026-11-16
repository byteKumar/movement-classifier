#!/usr/bin/env bash
# Serving benchmark. Plan fixed before the first run:
#   server : 1 uvicorn worker (uvloop + httptools), ONNX Runtime 1 thread per call,
#            on 127.0.0.1:8000; k6 runs on the same machine
#   warm-up: 10 s of single-window requests, not recorded
#   latency: 60 s, 1 virtual user back to back, for single-window and batch-32 bodies
#   ladder : single-window requests at 250, 500, 1000, 1500, 2000, 3000, 4000 req/s,
#            60 s each; a step passes if p99 < 50 ms, no failed requests and no
#            dropped iterations. The ladder stops after the first failing step.
#   result : the highest passing step's achieved requests/s
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(cd .. && pwd)
OUT=results
mkdir -p "$OUT"

# Shell wrappers (sh/bash/zsh -c ...) are skipped: a watcher loop's own command line
# contains these patterns, and the real work it starts shows up as its own process.
busy=$(pgrep -fl '^k6 |bench/run-|node bench|lsm_bench|db_bench|bench_|playwright|lighthouse|scripts/train_|scripts/evaluate' \
  | grep -v -E '^[0-9]+ (/bin/)?(ba|z)?sh( |$)' || true)
if [ -n "$busy" ]; then echo "another benchmark or training job is running:"; echo "$busy"; exit 1; fi
if lsof -iTCP:8000 -sTCP:LISTEN -n -P >/dev/null 2>&1; then echo "port 8000 is in use"; exit 1; fi

"$ROOT/.venv/bin/python" make_payloads.py
"$ROOT/.venv/bin/uvicorn" movement.serve:create_app --factory --host 127.0.0.1 --port 8000 \
  --workers 1 --log-level warning > "$OUT/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 1 80); do curl -sf http://127.0.0.1:8000/health >/dev/null && break; sleep 0.25; done
curl -sf http://127.0.0.1:8000/health

{ date '+%F %T'; sysctl -n machdep.cpu.brand_string; sysctl -n vm.loadavg; k6 version; } > "$OUT/machine.txt"

k6 run -q -e MODE=latency -e PAYLOAD=single -e DURATION=10s load.js > /dev/null   # warm-up
for p in single batch32; do
  echo "load avg before latency-$p: $(sysctl -n vm.loadavg)"
  k6 run -q -e MODE=latency -e PAYLOAD=$p -e SUMMARY_OUT="$OUT/latency-$p.json" load.js
done

for rate in 250 500 1000 1500 2000 3000 4000; do
  echo "load avg before rate-$rate: $(sysctl -n vm.loadavg)"
  k6 run -q -e MODE=rate -e PAYLOAD=single -e RATE=$rate -e SUMMARY_OUT="$OUT/rate-$rate.json" load.js || true
  pass=$("$ROOT/.venv/bin/python" -c "
import json; r = json.load(open('$OUT/rate-$rate.json'))
print(int(r['latency_ms']['p99'] < 50 and r['failed_rate'] == 0 and r['dropped_iterations'] == 0))")
  echo "rate $rate: pass=$pass"
  [ "$pass" = 1 ] || break
done
curl -sf http://127.0.0.1:8000/metrics > "$OUT/server-metrics.json"
echo "done"
