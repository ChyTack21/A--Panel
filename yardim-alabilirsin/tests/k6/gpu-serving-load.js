import http from "k6/http";
import { check, sleep } from "k6";

const vus = Number(__ENV.K6_VUS || 6);
const duration = __ENV.K6_DURATION || "2m";
const targetUrl = __ENV.TARGET_URL || "http://localhost:9700/infer";
const matrixSize = Number(__ENV.K6_MATRIX_SIZE || 3072);
const steps = Number(__ENV.K6_STEPS || 3);
const batches = Number(__ENV.K6_BATCHES || 2);
const waitSeconds = Number(__ENV.K6_SLEEP_SECONDS || 0.2);

export const options = {
  scenarios: {
    serving_load: {
      executor: "constant-vus",
      vus,
      duration,
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.05"],
    http_req_duration: ["p(95)<4000"],
  },
};

export default function () {
  const payload = JSON.stringify({
    matrix_size: matrixSize,
    steps,
    batches,
    precision: "float16",
  });

  const response = http.post(targetUrl, payload, {
    headers: { "Content-Type": "application/json" },
  });

  check(response, {
    "status is 200": (r) => r.status === 200,
  });

  sleep(waitSeconds);
}
