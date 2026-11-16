// k6 load test for POST /predict.
//   MODE=latency : 1 virtual user sending requests back to back for DURATION
//                  (latency with no queueing)
//   MODE=rate    : open model, RATE requests/s for DURATION regardless of how
//                  fast the server answers (constant-arrival-rate)
// Env: PAYLOAD=single|batch32, RATE, DURATION (default 60s), URL, SUMMARY_OUT
import http from 'k6/http';
import { check } from 'k6';
import { SharedArray } from 'k6/data';

const MODE = __ENV.MODE || 'latency';
const PAYLOAD = __ENV.PAYLOAD || 'single';
const DURATION = __ENV.DURATION || '60s';
const URL = __ENV.URL || 'http://127.0.0.1:8000/predict';
const bodies = new SharedArray(PAYLOAD, () => JSON.parse(open(`./payloads/${PAYLOAD}.json`)));

const scenario = MODE === 'rate'
  ? { executor: 'constant-arrival-rate', rate: Number(__ENV.RATE), timeUnit: '1s', duration: DURATION,
      preAllocatedVUs: 100, maxVUs: 400 }
  : { executor: 'constant-vus', vus: 1, duration: DURATION };

export const options = {
  scenarios: { predict: scenario },
  summaryTrendStats: ['avg', 'min', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
  discardResponseBodies: true,
};

const params = { headers: { 'Content-Type': 'application/json' } };

export default function () {
  const res = http.post(URL, bodies[__ITER % bodies.length], params);
  check(res, { 'status 200': (r) => r.status === 200 });
}

export function handleSummary(data) {
  const m = data.metrics;
  const d = m.http_req_duration.values;
  const out = {
    mode: MODE, payload: PAYLOAD, target_rate: MODE === 'rate' ? Number(__ENV.RATE) : null, duration: DURATION,
    requests: m.http_reqs.values.count, achieved_rps: m.http_reqs.values.rate,
    failed_rate: m.http_req_failed.values.rate,
    dropped_iterations: m.dropped_iterations ? m.dropped_iterations.values.count : 0,
    latency_ms: { avg: d.avg, min: d.min, p50: d.med, p90: d['p(90)'], p95: d['p(95)'], p99: d['p(99)'], max: d.max },
  };
  const result = { stdout: JSON.stringify(out) + '\n' };
  if (__ENV.SUMMARY_OUT) result[__ENV.SUMMARY_OUT] = JSON.stringify(out, null, 2);
  return result;
}
