// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"testing"

	"nvpair-shared/fabricwire"
)

func TestJobStoreRejectsStaleCheckpointAndRecoversFromLastStage(t *testing.T) {
	stateDir := t.TempDir()
	store := NewJobStore(stateDir)
	if err := store.Submit(JobRecord{JobID: "job-1", GroupID: "group-1", Epoch: 4}); err != nil {
		t.Fatal(err)
	}
	checkpoint := fabricwire.Checkpoint{JobID: "job-1", GroupID: "group-1", Epoch: 4, Stage: 3}
	if err := store.SaveCheckpoint(checkpoint); err != nil {
		t.Fatal(err)
	}
	if err := store.SaveCheckpoint(fabricwire.Checkpoint{JobID: "job-1", GroupID: "group-1", Epoch: 3, Stage: 4}); err == nil {
		t.Fatal("stale checkpoint accepted")
	}
	recovered, err := store.Recover("job-1", "group-2", 5)
	if err != nil {
		t.Fatal(err)
	}
	if recovered.GroupID != "group-2" || recovered.Epoch != 5 || recovered.Stage != 3 {
		t.Fatalf("recovered checkpoint = %+v", recovered)
	}
	reloaded := NewJobStore(stateDir)
	if _, ok := reloaded.Job("job-1"); !ok {
		t.Fatal("checkpointed job was not persisted")
	}
}

func TestJobStoreStopsRecoveryAfterThreeAttempts(t *testing.T) {
	store := NewJobStore("")
	if err := store.Submit(JobRecord{JobID: "job-recovery", GroupID: "group-1", Epoch: 1}); err != nil {
		t.Fatal(err)
	}
	for attempt := uint32(1); attempt <= 3; attempt++ {
		allowed, err := store.BeginRecovery("job-recovery", 3)
		if err != nil || !allowed {
			t.Fatalf("attempt %d: allowed=%v err=%v", attempt, allowed, err)
		}
	}
	allowed, err := store.BeginRecovery("job-recovery", 3)
	if err != nil {
		t.Fatal(err)
	}
	if allowed {
		t.Fatal("fourth recovery attempt was allowed")
	}
	job, ok := store.Job("job-recovery")
	if !ok || job.State != "failed" || job.RecoveryAttempts != 3 {
		t.Fatalf("job = %+v, want failed after three attempts", job)
	}
}
