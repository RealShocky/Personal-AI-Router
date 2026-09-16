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
