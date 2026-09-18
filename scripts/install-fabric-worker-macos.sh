#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

: "${PAIR_FABRIC_BINARY:?path to the macOS nvpair-compute-fabric binary is required}"
: "${PAIR_CLUSTER_DIR:?path to the already-paired cluster directory is required}"
: "${PAIR_COORDINATOR_URL:?HTTPS coordinator URL is required}"
: "${PAIR_WORKER_ID:?stable worker UUID is required}"
: "${PAIR_NODE_ID:?stable host UUID is required}"
PAIR_ADVERTISE_URL="${PAIR_ADVERTISE_URL:-}"
PAIR_RUNTIME="${PAIR_RUNTIME:-auto}"
PAIR_BACKEND="${PAIR_BACKEND:-}"
PAIR_RPC_SERVER_PATH="${PAIR_RPC_SERVER_PATH:-}"
PAIR_RPC_PORT="${PAIR_RPC_PORT:-50052}"
PAIR_HTTP_PORT="${PAIR_HTTP_PORT:-14324}"
PAIR_RPC_TARGET="${PAIR_RPC_TARGET:-127.0.0.1:${PAIR_RPC_PORT}}"
PAIR_HEARTBEAT_TIMEOUT="${PAIR_HEARTBEAT_TIMEOUT:-10s}"
PAIR_METAL_PAGE_HELPER="${PAIR_METAL_PAGE_HELPER:-}"

if [[ "${PAIR_RUNTIME}" == auto ]]; then
  if command -v system_profiler >/dev/null 2>&1 && system_profiler SPDisplaysDataType >/dev/null 2>&1; then
    PAIR_RUNTIME=metal
  else
    PAIR_RUNTIME=cpu
  fi
fi
if [[ -z "${PAIR_BACKEND}" ]]; then
  PAIR_BACKEND="${PAIR_RUNTIME}"
fi

case "${PAIR_RUNTIME}" in metal|cpu) ;; *) echo "PAIR_RUNTIME must be metal or cpu" >&2; exit 2 ;; esac
for value in PAIR_FABRIC_BINARY PAIR_CLUSTER_DIR PAIR_COORDINATOR_URL PAIR_WORKER_ID PAIR_NODE_ID PAIR_ADVERTISE_URL PAIR_BACKEND PAIR_RPC_SERVER_PATH PAIR_METAL_PAGE_HELPER; do
  text="${!value:-}"
  case "${text}" in *$'\n'*|*$'\r'*) echo "${value} contains a newline" >&2; exit 2 ;; esac
done

config_dir="${HOME}/Library/Application Support/PAIR"
launch_agents="${HOME}/Library/LaunchAgents"
install_dir="${config_dir}/fabric"
plist="${launch_agents}/com.nvidia.pair.fabric-worker.plist"
mkdir -p "${install_dir}" "${launch_agents}"
install -m 0755 "${PAIR_FABRIC_BINARY}" "${install_dir}/nvpair-compute-fabric"

if [[ -n "${PAIR_FABRIC_SHA256:-}" ]]; then
  printf '%s  %s\n' "${PAIR_FABRIC_SHA256}" "${install_dir}/nvpair-compute-fabric" | shasum -a 256 -c -
fi

escape_xml() {
  printf '%s' "$1" | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g; s/"/\&quot;/g; s/'"'"'/\&apos;/g'
}

args=(--daemon --heartbeat-timeout "${PAIR_HEARTBEAT_TIMEOUT}" --cluster-dir "${PAIR_CLUSTER_DIR}" --coordinator-url "${PAIR_COORDINATOR_URL}" --advertise-url "${PAIR_ADVERTISE_URL}" --worker-id "${PAIR_WORKER_ID}" --node-id "${PAIR_NODE_ID}" --runtime "${PAIR_RUNTIME}" --backend "${PAIR_BACKEND}" --http-port "${PAIR_HTTP_PORT}" --rpc-target "${PAIR_RPC_TARGET}" --rpc-port "${PAIR_RPC_PORT}")
if [[ -n "${PAIR_RPC_SERVER_PATH}" ]]; then args+=(--rpc-server-path "${PAIR_RPC_SERVER_PATH}"); fi
if [[ -n "${PAIR_METAL_PAGE_HELPER}" ]]; then args+=(--metal-page-helper "${PAIR_METAL_PAGE_HELPER}"); fi

args_xml=""
for arg in "${args[@]}"; do
  args_xml+="    <string>$(escape_xml "${arg}")</string>\n"
done

cat > "${plist}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.nvidia.pair.fabric-worker</string>
  <key>ProgramArguments</key><array>
    <string>${install_dir}/nvpair-compute-fabric</string>
$(printf '%b' "${args_xml}")  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ProcessType</key><string>Background</string>
  <key>StandardOutPath</key><string>${config_dir}/fabric-worker.log</string>
  <key>StandardErrorPath</key><string>${config_dir}/fabric-worker.err.log</string>
</dict>
</plist>
EOF

launchctl bootout "gui/$(id -u)" "${plist}" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "${plist}"
launchctl kickstart -k "gui/$(id -u)/com.nvidia.pair.fabric-worker"
launchctl print "gui/$(id -u)/com.nvidia.pair.fabric-worker" | sed -n '1,24p'
