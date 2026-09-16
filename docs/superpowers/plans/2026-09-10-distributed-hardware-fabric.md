<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Distributed Hardware Fabric Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add accurate CPU/GPU/backend capability reporting as the first vertical slice toward a secure distributed inference fabric.

**Architecture:** Extend the existing node-info wire response with optional capability metadata. Detect capabilities in platform-specific files, classify them in a platform-neutral helper, and preserve current routing behavior until later distributed-runtime phases.

**Tech Stack:** Go 1.25, existing `nvpair-node-info` service, existing Electron TypeScript bridge, Go unit tests, JSON wire compatibility.

**Spec:** `docs/superpowers/specs/2026-09-10-distributed-hardware-fabric-design.md`

## Global Constraints

- Do not claim or implement a transparent shared OS CPU/GPU.
- Do not change Ollama or LM Studio request routing in Phase 1.
- Preserve optional-field compatibility for older consumers.
- Do not add a network listener or third-party runtime dependency in Phase 1.
- Every changed file retains the two-line SPDX header.
- No logs may contain prompts, messages, response bodies, PINs, or key material.

---

### Task 1: Define the capability contract

**Files:**
- Modify: `services/nvpair-node-info/main.go`
- Test: `services/nvpair-node-info/stats_test.go`

**Interfaces:**
- Produce `HardwareCapabilities` and optional backend metadata in `NodeInfoResponse`.
- Preserve existing `GPUs`, `cpu`, `memory`, telemetry, and identity fields.

- [ ] **Step 1: Write failing JSON tests** for CPU-only, CUDA, Metal, and old-response compatibility.
- [ ] **Step 2: Run** `go test ./...` from `services/nvpair-node-info` and verify the new assertions fail for missing fields.
- [ ] **Step 3: Add minimal structs and response fields** with `omitempty` tags.
- [ ] **Step 4: Run** the focused tests and verify they pass.
- [ ] **Step 5: Run** `go test ./...` and preserve the green baseline.

### Task 2: Implement platform-neutral capability classification

**Files:**
- Create: `services/nvpair-node-info/capabilities.go`
- Test: `services/nvpair-node-info/capabilities_test.go`

**Interfaces:**
- Consume normalized CPU/GPU facts.
- Produce deterministic capability labels and backend records.

- [ ] **Step 1: Write failing tests** for no GPU, NVIDIA GPU, Apple GPU, and unknown GPU.
- [ ] **Step 2: Run** `go test ./... -run 'TestClassifyCapabilities'` and verify the expected failures.
- [ ] **Step 3: Implement** the smallest classifier with no platform calls.
- [ ] **Step 4: Run** the focused tests and verify they pass.
- [ ] **Step 5: Run** all node-info tests.

### Task 3: Connect platform detection to the contract

**Files:**
- Modify: `services/nvpair-node-info/main.go`
- Modify: `services/nvpair-node-info/gpu_linux.go`
- Modify: `services/nvpair-node-info/gpu_darwin.go`
- Modify: `services/nvpair-node-info/gpu_windows.go`
- Modify: `services/nvpair-node-info/gpu_other.go`
- Test: platform-specific existing test files as needed

**Interfaces:**
- Populate backend/vendor facts from existing platform detectors.
- Keep unsupported or unavailable backend facts absent.

- [ ] **Step 1: Add failing platform fixture assertions** using existing detector test data.
- [ ] **Step 2: Run platform-relevant tests and verify failure.
- [ ] **Step 3: Add backend normalization and response assembly.
- [ ] **Step 4: Run Windows/Linux/macOS compile-oriented tests available on the host.
- [ ] **Step 5: Run the complete node-info test suite.

### Task 4: Carry the fields through the desktop bridge

**Files:**
- Modify: `desktop/src/shared/types/hardware.ts`
- Modify: `desktop/src/electron/service-bridge/modular-state.ts`
- Test: existing desktop bridge/unit test location identified by current imports

**Interfaces:**
- Parse optional capability fields without making older nodes invalid.
- Store capabilities with node telemetry for future UI and scheduler use.

- [ ] **Step 1: Add a failing parser test** for optional capabilities.
- [ ] **Step 2: Run the focused desktop test and verify failure.
- [ ] **Step 3: Add typed parsing and state storage.
- [ ] **Step 4: Run `npm run typecheck` from `desktop/`.
- [ ] **Step 5: Run the focused desktop unit tests.

### Task 5: Validate contracts and documentation

**Files:**
- Modify: generated contract sources only if the repository tooling requires it
- Modify: `docs/overview.mdx` or the relevant service documentation

- [ ] **Step 1: Run** `npm run service-contracts:write` only if the JSON-RPC contract changed.
- [ ] **Step 2: Run** `npm run service-contracts:check`.
- [ ] **Step 3: Run** `node scripts/spdx-headers.mjs`.
- [ ] **Step 4: Run** node-info Go tests and desktop typecheck again.
- [ ] **Step 5: Inspect `git diff` and confirm no routing behavior changed.
