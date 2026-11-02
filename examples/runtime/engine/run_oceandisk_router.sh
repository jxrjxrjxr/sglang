#!/bin/sh
PREFILL_IP="${PREFILL_IP:?Set PREFILL_IP to the prefill server IP}"
DECODE_IP="${DECODE_IP:?Set DECODE_IP to the decode server IP}"
PREFILL_PORT="${PREFILL_PORT:-31002}"
DECODE_PORT="${DECODE_PORT:-31001}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-31145}"

python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill "http://${PREFILL_IP}:${PREFILL_PORT}" "$BOOTSTRAP_PORT" \
  --decode "http://${DECODE_IP}:${DECODE_PORT}" \
  --host "${ROUTER_HOST:-0.0.0.0}" \
  --port "${ROUTER_PORT:-31345}" \
  --health-check-timeout-secs 1000
