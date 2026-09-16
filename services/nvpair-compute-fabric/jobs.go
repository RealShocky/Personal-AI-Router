// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"nvpair-shared/fabricwire"
)

type JobRecord struct {
	JobID       string                `json:"jobId"`
	GroupID     string                `json:"groupId"`
	ModelDigest string                `json:"modelDigest"`
	Epoch       uint64                `json:"epoch"`
	Checkpoint  fabricwire.Checkpoint `json:"checkpoint"`
	State       string                `json:"state"`
}

type JobStore struct {
	mu   sync.RWMutex
	path string
	jobs map[string]JobRecord
}

func NewJobStore(stateDir string) *JobStore {
	s := &JobStore{jobs: make(map[string]JobRecord)}
	if stateDir != "" {
		s.path = filepath.Join(stateDir, "fabric-jobs.json")
		data, err := os.ReadFile(s.path)
		if err == nil {
			_ = json.Unmarshal(data, &s.jobs)
		}
	}
	return s
}

func (s *JobStore) Submit(job JobRecord) error {
	if job.JobID == "" || job.GroupID == "" || job.Epoch == 0 {
		return fmt.Errorf("job requires jobId, groupId, and positive epoch")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, exists := s.jobs[job.JobID]; exists {
		return fmt.Errorf("job already exists")
	}
	if job.State == "" {
		job.State = "running"
	}
	job.Checkpoint = fabricwire.Checkpoint{JobID: job.JobID, GroupID: job.GroupID, Epoch: job.Epoch}
	s.jobs[job.JobID] = job
	return s.persistLocked()
}

func (s *JobStore) SaveCheckpoint(checkpoint fabricwire.Checkpoint) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	job, ok := s.jobs[checkpoint.JobID]
	if !ok {
		return fmt.Errorf("job not found")
	}
	if !checkpoint.AcceptsEpoch(job.Epoch) || checkpoint.GroupID != job.GroupID || checkpoint.Stage < job.Checkpoint.Stage {
		return fmt.Errorf("checkpoint does not match the active job epoch or moves backward")
	}
	job.Checkpoint = checkpoint
	s.jobs[checkpoint.JobID] = job
	return s.persistLocked()
}

func (s *JobStore) Recover(jobID, groupID string, epoch uint64) (fabricwire.Checkpoint, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	job, ok := s.jobs[jobID]
	if !ok {
		return fabricwire.Checkpoint{}, fmt.Errorf("job not found")
	}
	if groupID == "" || epoch <= job.Epoch {
		return fabricwire.Checkpoint{}, fmt.Errorf("recovery requires a new group and epoch")
	}
	job.GroupID, job.Epoch = groupID, epoch
	job.Checkpoint.GroupID, job.Checkpoint.Epoch = groupID, epoch
	job.State = "running"
	s.jobs[jobID] = job
	if err := s.persistLocked(); err != nil {
		return fabricwire.Checkpoint{}, err
	}
	return job.Checkpoint, nil
}

func (s *JobStore) Job(jobID string) (JobRecord, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	job, ok := s.jobs[jobID]
	return job, ok
}

func (s *JobStore) Jobs() []JobRecord {
	s.mu.RLock()
	defer s.mu.RUnlock()
	jobs := make([]JobRecord, 0, len(s.jobs))
	for _, job := range s.jobs {
		jobs = append(jobs, job)
	}
	return jobs
}
func (s *JobStore) persistLocked() error {
	if s.path == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(s.path), 0o700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(s.jobs, "", "  ")
	if err != nil {
		return err
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, s.path)
}
