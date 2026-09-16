#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

: "${PAIR_FABRIC_BINARY:?path to the Linux nvpair-compute-fabric binary is required}"
: "${PAIR_CLUSTER_DIR:?path to the already-paired cluster directory is required}"
: "${PAIR_COORDINATOR_URL:?HTTPS coordinator URL is required}"
: "${PAIR_WORKER_ID:?stable worker UUID is required}"
: "${PAIR_NODE_ID:?stable host UUID is required}"
PAIR_ADVERTISE_URL="${PAIR_ADVERTISE_URL:-}"
PAIR_RUNTIME="${PAIR_RUNTIME:-cpu}"
PAIR_BACKEND="${PAIR_BACKEND:-cpu}"
PAIR_RPC_SERVER_PATH="${PAIR_RPC_SERVER_PATH:-}"
PAIR_RPC_PORT="${PAIR_RPC_PORT:-50052}"
PAIR_HTTP_PORT="${PAIR_HTTP_PORT:-14324}"
PAIR_RPC_TARGET="${PAIR_RPC_TARGET:-127.0.0.1:${PAIR_RPC_PORT}}"
PAIR_HEARTBEAT_TIMEOUT="${PAIR_HEARTBEAT_TIMEOUT:-10s}"

case "${PAIR_RUNTIME}" in cpu|cuda|metal) ;; *) echo "PAIR_RUNTIME must be cpu, cuda, or metal" >&2; exit 2 ;; esac
for value in PAIR_FABRIC_BINARY PAIR_CLUSTER_DIR PAIR_COORDINATOR_URL PAIR_WORKER_ID PAIR_NODE_ID PAIR_ADVERTISE_URL PAIR_BACKEND PAIR_RPC_SERVER_PATH; do
  text="${!value:-}"
  case "${text}" in *$'\n'*|*$'\r'*) echo "${value} contains a newline" >&2; exit 2 ;; esac
done

install -d -m 0755 /opt/nvpair/fabric /etc/nvpair
install -m 0755 "${PAIR_FABRIC_BINARY}" /opt/nvpair/fabric/nvpair-compute-fabric
if [ -n "${PAIR_FABRIC_SHA256:-}" ]; then
  printf '%s  %s\n' "${PAIR_FABRIC_SHA256}" /opt/nvpair/fabric/nvpair-compute-fabric | sha256sum -c -
fi

printf '%q\n'   "PAIR_CLUSTER_DIR=${PAIR_CLUSTER_DIR}"   "PAIR_COORDINATOR_URL=${PAIR_COORDINATOR_URL}"   "PAIR_ADVERTISE_URL=${PAIR_ADVERTISE_URL}"   "PAIR_WORKER_ID=${PAIR_WORKER_ID}"   "PAIR_NODE_ID=${PAIR_NODE_ID}"   "PAIR_RUNTIME=${PAIR_RUNTIME}"   "PAIR_BACKEND=${PAIR_BACKEND}"   "PAIR_RPC_SERVER_PATH=${PAIR_RPC_SERVER_PATH}"   "PAIR_RPC_PORT=${PAIR_RPC_PORT}"   "PAIR_HTTP_PORT=${PAIR_HTTP_PORT}"   "PAIR_RPC_TARGET=${PAIR_RPC_TARGET}"   "PAIR_HEARTBEAT_TIMEOUT=${PAIR_HEARTBEAT_TIMEOUT}" > /etc/nvpair/fabric-worker.env
chmod 0600 /etc/nvpair/fabric-worker.env

printf '%s\n' '#!/bin/sh' 'set -eu' '. /etc/nvpair/fabric-worker.env' 'set -- /opt/nvpair/fabric/nvpair-compute-fabric --daemon --heartbeat-timeout "$PAIR_HEARTBEAT_TIMEOUT" --cluster-dir "$PAIR_CLUSTER_DIR" --coordinator-url "$PAIR_COORDINATOR_URL" --advertise-url "$PAIR_ADVERTISE_URL" --worker-id "$PAIR_WORKER_ID" --node-id "$PAIR_NODE_ID" --runtime "$PAIR_RUNTIME" --backend "$PAIR_BACKEND" --http-port "$PAIR_HTTP_PORT" --rpc-target "$PAIR_RPC_TARGET" --rpc-port "$PAIR_RPC_PORT"' 'if [ -n "$PAIR_RPC_SERVER_PATH" ]; then set -- "$@" --rpc-server-path "$PAIR_RPC_SERVER_PATH"; fi' 'exec "$@"' > /opt/nvpair/fabric/run-worker
chmod 0755 /opt/nvpair/fabric/run-worker

printf '%s\n' '[Unit]' 'Description=PAIR distributed compute worker' 'After=network-online.target' 'Wants=network-online.target' '' '[Service]' 'Type=simple' 'ExecStart=/opt/nvpair/fabric/run-worker' 'Restart=always' 'RestartSec=3' 'NoNewPrivileges=true' 'ProtectSystem=full' 'ReadWritePaths=/var/lib/nvpair /tmp' '' '[Install]' 'WantedBy=multi-user.target' > /etc/systemd/system/nvpair-fabric-worker.service
systemctl daemon-reload
systemctl enable --now nvpair-fabric-worker.service
systemctl --no-pager --full status nvpair-fabric-worker.service
