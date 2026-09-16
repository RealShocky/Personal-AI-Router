// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package fabricwire

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestWorkerStateTransitions(t *testing.T) {
	valid := [][2]WorkerState{
		{WorkerDiscovered, WorkerAuthenticated},
		{WorkerAuthenticated, WorkerProbing},
		{WorkerProbing, WorkerReady},
		{WorkerReady, WorkerLeased},
		{WorkerLeased, WorkerServing},
		{WorkerServing, WorkerSuspect},
		{WorkerSuspect, WorkerQuarantined},
		{WorkerQuarantined, WorkerProbing},
	}
	for _, pair := range valid {
		if !CanTransition(pair[0], pair[1]) {
			t.Errorf("transition %q -> %q rejected", pair[0], pair[1])
		}
	}
	if CanTransition(WorkerDiscovered, WorkerServing) {
		t.Fatal("discovered worker cannot serve without authentication and probing")
	}
}

func TestHeartbeatJSONRoundTrip(t *testing.T) {
	want := Heartbeat{
		WorkerID:     "worker-a",
		NodeID:       "node-a",
		State:        WorkerReady,
		Epoch:        7,
		Runtime:      "llama.cpp",
		Backends:     []string{"cpu", "cuda"},
		MemoryFree:   6 << 30,
		GPUVramTotal: 8 << 30,
		GPUVramFree:  6 << 30,
		GPUCount:     1,
		QueueDepth:   2,
	}
	body, err := json.Marshal(want)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var got Heartbeat
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("round trip = %+v, want %+v", got, want)
	}
}

func TestCheckpointRejectsStaleEpoch(t *testing.T) {
	checkpoint := Checkpoint{JobID: "job-a", GroupID: "group-a", Epoch: 4, Stage: 2}
	if checkpoint.AcceptsEpoch(3) {
		t.Fatal("checkpoint accepted an older epoch")
	}
	if !checkpoint.AcceptsEpoch(4) {
		t.Fatal("checkpoint rejected its own epoch")
	}
}
