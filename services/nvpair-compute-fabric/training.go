// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	"nvpair-shared/fabricwire"
)

type TrainingParallelism string

const (
	TrainingDataParallel TrainingParallelism = "data"
	TrainingFSDP         TrainingParallelism = "fsdp"
)

type TrainingNode struct {
	WorkerID string `json:"workerId"`
	Address  string `json:"address"`
	Backend  string `json:"backend"`
	GPUCount uint32 `json:"gpuCount,omitempty"`
}

type TrainingRequest struct {
	JobID                   string              `json:"jobId"`
	ModelDigest             string              `json:"modelDigest"`
	TrainerPath             string              `json:"trainerPath"`
	ModelPath               string              `json:"modelPath"`
	DatasetPath             string              `json:"datasetPath"`
	CheckpointDirectory     string              `json:"checkpointDirectory"`
	ResumeCheckpoint        string              `json:"resumeCheckpoint,omitempty"`
	RendezvousEndpoint      string              `json:"rendezvousEndpoint"`
	Parallelism             TrainingParallelism `json:"parallelism"`
	ProcessesPerNode        uint32              `json:"processesPerNode"`
	CheckpointIntervalSteps uint64              `json:"checkpointIntervalSteps"`
	Nodes                   []TrainingNode      `json:"nodes"`
}

type TrainingExecution struct {
	JobID    string `json:"jobId"`
	NodeRank uint32 `json:"nodeRank"`
	PID      int    `json:"pid"`
	State    string `json:"state"`
}

type trainingProcess struct {
	execution     TrainingExecution
	cmd           *exec.Cmd
	stopRequested bool
}

type TrainingManager struct {
	mu         sync.Mutex
	ctx        context.Context
	executions map[string]*trainingProcess
	command    func(context.Context, string, ...string) *exec.Cmd
}

type TrainingStartFunc func(context.Context, TrainingNode, TrainingRequest, uint32) (TrainingExecution, error)
type TrainingStopFunc func(context.Context, string, TrainingNode) error
type TrainingStatusFunc func(context.Context, string, TrainingNode) (TrainingExecution, error)

type TrainingCoordinator struct {
	start  TrainingStartFunc
	stop   TrainingStopFunc
	status TrainingStatusFunc
	mu     sync.Mutex
	groups map[string]trainingGroup
	path   string
}

type trainingGroup struct {
	request    TrainingRequest
	state      string
	epoch      uint64
	executions []TrainingExecution
	checkpoint fabricwire.Checkpoint
	attempts   uint32
}

type persistedTrainingGroup struct {
	Request    TrainingRequest       `json:"request"`
	State      string                `json:"state"`
	Epoch      uint64                `json:"epoch"`
	Checkpoint fabricwire.Checkpoint `json:"checkpoint"`
	Attempts   uint32                `json:"recoveryAttempts"`
}

type TrainingGroupStatus struct {
	JobID            string                `json:"jobId"`
	State            string                `json:"state"`
	Epoch            uint64                `json:"epoch"`
	Executions       []TrainingExecution   `json:"executions"`
	Checkpoint       fabricwire.Checkpoint `json:"checkpoint"`
	RecoveryAttempts uint32                `json:"recoveryAttempts"`
}

func NewTrainingCoordinator(start TrainingStartFunc, stop TrainingStopFunc, stateDirs ...string) *TrainingCoordinator {
	coordinator := &TrainingCoordinator{start: start, stop: stop, groups: make(map[string]trainingGroup)}
	if len(stateDirs) > 0 && stateDirs[0] != "" {
		coordinator.path = filepath.Join(stateDirs[0], "training-groups.json")
		coordinator.load()
	}
	return coordinator
}

func (c *TrainingCoordinator) SetStatusFunc(status TrainingStatusFunc) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.status = status
}

func (c *TrainingCoordinator) RefreshStatus(ctx context.Context, jobID string) error {
	c.mu.Lock()
	group, ok := c.groups[jobID]
	statusFunc := c.status
	c.mu.Unlock()
	if !ok {
		return fmt.Errorf("training group not found")
	}
	if statusFunc == nil || group.state != "running" {
		return nil
	}
	type result struct {
		rank      uint32
		execution TrainingExecution
		err       error
	}
	results := make(chan result, len(group.request.Nodes))
	for rank, node := range group.request.Nodes {
		go func(rank uint32, node TrainingNode) {
			execution, err := statusFunc(ctx, jobID, node)
			results <- result{rank: rank, execution: execution, err: err}
		}(uint32(rank), node)
	}
	refreshed := append([]TrainingExecution(nil), group.executions...)
	failed := false
	for range group.request.Nodes {
		current := <-results
		if current.err != nil {
			failed = true
			continue
		}
		if current.rank < uint32(len(refreshed)) {
			refreshed[current.rank] = current.execution
		}
		if current.execution.State == "failed" {
			failed = true
		}
	}
	c.mu.Lock()
	group, ok = c.groups[jobID]
	if !ok {
		c.mu.Unlock()
		return fmt.Errorf("training group not found")
	}
	group.executions = refreshed
	if failed && group.state == "running" {
		group.state = "recoverable"
	}
	c.groups[jobID] = group
	err := c.persistLocked()
	c.mu.Unlock()
	return err
}

func (c *TrainingCoordinator) StartGroup(ctx context.Context, request TrainingRequest) ([]TrainingExecution, error) {
	if err := request.Validate(); err != nil {
		return nil, err
	}
	executions, err := c.launchTrainingWorld(ctx, request)
	if err != nil {
		return nil, err
	}
	c.mu.Lock()
	group := trainingGroup{request: request, state: "running", epoch: uint64(time.Now().UnixNano()), executions: executions}
	c.groups[request.JobID] = group
	if err := c.persistLocked(); err != nil {
		delete(c.groups, request.JobID)
		c.mu.Unlock()
		for _, node := range request.Nodes {
			_ = c.stop(ctx, request.JobID, node)
		}
		return nil, fmt.Errorf("persist distributed training group: %w", err)
	}
	c.mu.Unlock()
	return executions, nil
}

func (c *TrainingCoordinator) launchTrainingWorld(ctx context.Context, request TrainingRequest) ([]TrainingExecution, error) {
	type result struct {
		rank      uint32
		node      TrainingNode
		execution TrainingExecution
		err       error
	}
	results := make(chan result, len(request.Nodes))
	for rank, node := range request.Nodes {
		go func(rank uint32, node TrainingNode) {
			execution, err := c.start(ctx, node, request, rank)
			results <- result{rank: rank, node: node, execution: execution, err: err}
		}(uint32(rank), node)
	}
	launched := make([]result, 0, len(request.Nodes))
	var firstErr error
	for range request.Nodes {
		current := <-results
		if current.err != nil && firstErr == nil {
			firstErr = current.err
		}
		if current.err == nil {
			launched = append(launched, current)
		}
	}
	if firstErr != nil {
		for _, current := range launched {
			_ = c.stop(ctx, request.JobID, current.node)
		}
		return nil, fmt.Errorf("distributed training launch failed: %w", firstErr)
	}
	executions := make([]TrainingExecution, len(launched))
	for _, current := range launched {
		executions[current.rank] = current.execution
	}
	return executions, nil
}

func (c *TrainingCoordinator) StatusGroup(jobID string) (TrainingGroupStatus, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	group, ok := c.groups[jobID]
	if !ok {
		return TrainingGroupStatus{}, false
	}
	return TrainingGroupStatus{JobID: jobID, State: group.state, Epoch: group.epoch, Executions: append([]TrainingExecution(nil), group.executions...), Checkpoint: group.checkpoint, RecoveryAttempts: group.attempts}, true
}

func (c *TrainingCoordinator) SaveCheckpoint(jobID string, checkpoint fabricwire.Checkpoint) error {
	if checkpoint.Filename == "" {
		return fmt.Errorf("training checkpoint filename is required")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	group, ok := c.groups[jobID]
	if !ok {
		return fmt.Errorf("training group not found")
	}
	if checkpoint.JobID != jobID || checkpoint.GroupID != jobID || checkpoint.Epoch != group.epoch || checkpoint.Stage < group.checkpoint.Stage {
		return fmt.Errorf("training checkpoint does not match the active group epoch or moves backward")
	}
	group.checkpoint = checkpoint
	c.groups[jobID] = group
	return c.persistLocked()
}

func (c *TrainingCoordinator) MarkFailed(jobID string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	group, ok := c.groups[jobID]
	if !ok || group.state != "running" {
		return fmt.Errorf("training group is not running")
	}
	group.state = "failed"
	c.groups[jobID] = group
	return c.persistLocked()
}

func (c *TrainingCoordinator) RecoverGroup(ctx context.Context, jobID string, nodes []TrainingNode, rendezvous string) ([]TrainingExecution, error) {
	c.mu.Lock()
	group, ok := c.groups[jobID]
	if !ok || (group.state != "failed" && group.state != "recoverable") {
		c.mu.Unlock()
		return nil, fmt.Errorf("training group is not recoverable")
	}
	if group.checkpoint.Filename == "" {
		c.mu.Unlock()
		return nil, fmt.Errorf("training recovery requires a verified checkpoint")
	}
	if group.attempts >= 3 {
		c.mu.Unlock()
		return nil, fmt.Errorf("training recovery attempt limit reached")
	}
	request := group.request
	request.Nodes = append([]TrainingNode(nil), nodes...)
	request.RendezvousEndpoint = rendezvous
	request.ResumeCheckpoint = group.checkpoint.Filename
	if err := request.Validate(); err != nil {
		c.mu.Unlock()
		return nil, err
	}
	group.state = "recovering"
	group.attempts++
	c.groups[jobID] = group
	if err := c.persistLocked(); err != nil {
		c.mu.Unlock()
		return nil, fmt.Errorf("persist recovery attempt: %w", err)
	}
	c.mu.Unlock()

	executions, err := c.launchTrainingWorld(ctx, request)
	c.mu.Lock()
	group = c.groups[jobID]
	if err != nil {
		group.state = "failed"
		c.groups[jobID] = group
		_ = c.persistLocked()
		c.mu.Unlock()
		return nil, err
	}
	epoch := uint64(time.Now().UnixNano())
	if epoch <= group.epoch {
		epoch = group.epoch + 1
	}
	group.request = request
	group.state = "running"
	group.epoch = epoch
	group.executions = executions
	group.checkpoint.Epoch = epoch
	c.groups[jobID] = group
	if err := c.persistLocked(); err != nil {
		c.mu.Unlock()
		for _, node := range request.Nodes {
			_ = c.stop(ctx, jobID, node)
		}
		return nil, fmt.Errorf("persist recovered training group: %w", err)
	}
	c.mu.Unlock()
	return executions, nil
}

func (c *TrainingCoordinator) StopGroup(ctx context.Context, jobID string) error {
	c.mu.Lock()
	group, ok := c.groups[jobID]
	if !ok || group.state != "running" {
		c.mu.Unlock()
		return fmt.Errorf("training group is not running")
	}
	group.state = "stopping"
	c.groups[jobID] = group
	c.mu.Unlock()
	var firstErr error
	for _, node := range group.request.Nodes {
		if err := c.stop(ctx, jobID, node); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	c.mu.Lock()
	group = c.groups[jobID]
	if firstErr == nil {
		group.state = "stopped"
	} else {
		group.state = "stop-failed"
	}
	c.groups[jobID] = group
	if persistErr := c.persistLocked(); firstErr == nil && persistErr != nil {
		firstErr = persistErr
	}
	c.mu.Unlock()
	if firstErr != nil {
		return fmt.Errorf("stop distributed training group: %w", firstErr)
	}
	return nil
}

func (c *TrainingCoordinator) load() {
	data, err := os.ReadFile(c.path)
	if err != nil {
		return
	}
	var records map[string]persistedTrainingGroup
	if json.Unmarshal(data, &records) != nil {
		return
	}
	for jobID, record := range records {
		state := record.State
		if state == "running" {
			state = "recoverable"
		}
		c.groups[jobID] = trainingGroup{request: record.Request, state: state, epoch: record.Epoch, checkpoint: record.Checkpoint, attempts: record.Attempts}
	}
}

func (c *TrainingCoordinator) persistLocked() error {
	if c.path == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(c.path), 0o700); err != nil {
		return err
	}
	records := make(map[string]persistedTrainingGroup, len(c.groups))
	for jobID, group := range c.groups {
		records[jobID] = persistedTrainingGroup{Request: group.request, State: group.state, Epoch: group.epoch, Checkpoint: group.checkpoint, Attempts: group.attempts}
	}
	data, err := json.MarshalIndent(records, "", "  ")
	if err != nil {
		return err
	}
	tmp := c.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, c.path)
}

func NewTrainingManager(ctx context.Context) *TrainingManager {
	return &TrainingManager{ctx: ctx, executions: make(map[string]*trainingProcess), command: exec.CommandContext}
}

func (m *TrainingManager) Start(request TrainingRequest, nodeRank uint32) (TrainingExecution, error) {
	args, err := BuildTorchRunCommand(request, nodeRank)
	if err != nil {
		return TrainingExecution{}, err
	}
	m.mu.Lock()
	if _, exists := m.executions[request.JobID]; exists {
		m.mu.Unlock()
		return TrainingExecution{}, fmt.Errorf("training job is already running")
	}
	m.mu.Unlock()
	cmd := m.command(m.ctx, args[0], args[1:]...)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return TrainingExecution{}, fmt.Errorf("start torchrun: %w", err)
	}
	execution := TrainingExecution{JobID: request.JobID, NodeRank: nodeRank, PID: cmd.Process.Pid, State: "running"}
	process := &trainingProcess{execution: execution, cmd: cmd}
	m.mu.Lock()
	m.executions[request.JobID] = process
	m.mu.Unlock()
	go m.wait(request.JobID, process)
	return execution, nil
}

func (m *TrainingManager) wait(jobID string, process *trainingProcess) {
	_ = process.cmd.Wait()
	m.mu.Lock()
	defer m.mu.Unlock()
	current, ok := m.executions[jobID]
	if !ok || current != process {
		return
	}
	process.execution.State = "failed"
	if process.stopRequested || m.ctx.Err() != nil {
		process.execution.State = "stopped"
	}
}

func (m *TrainingManager) Stop(jobID string) error {
	m.mu.Lock()
	process, ok := m.executions[jobID]
	if !ok || process.execution.State != "running" {
		m.mu.Unlock()
		return fmt.Errorf("training job is not running")
	}
	process.stopRequested = true
	m.mu.Unlock()
	if err := process.cmd.Process.Kill(); err != nil {
		return fmt.Errorf("stop training job: %w", err)
	}
	return nil
}

func (m *TrainingManager) Status(jobID string) (TrainingExecution, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	process, ok := m.executions[jobID]
	if !ok {
		return TrainingExecution{}, false
	}
	return process.execution, true
}

func (r TrainingRequest) Validate() error {
	if r.JobID == "" || r.ModelDigest == "" || r.TrainerPath == "" || r.ModelPath == "" || r.DatasetPath == "" || r.CheckpointDirectory == "" || r.RendezvousEndpoint == "" {
		return fmt.Errorf("training request requires jobId, modelDigest, trainerPath, modelPath, datasetPath, checkpointDirectory, and rendezvousEndpoint")
	}
	if r.Parallelism != TrainingDataParallel && r.Parallelism != TrainingFSDP {
		return fmt.Errorf("unsupported torchrun parallelism %q", r.Parallelism)
	}
	if r.ProcessesPerNode == 0 || r.CheckpointIntervalSteps == 0 || len(r.Nodes) == 0 {
		return fmt.Errorf("training request requires nodes, positive processesPerNode, and checkpointIntervalSteps")
	}
	backend := r.Nodes[0].Backend
	if backend != "cpu" && backend != "cuda" {
		return fmt.Errorf("torchrun adapter does not support backend %q", backend)
	}
	if len(r.Nodes) > 1 {
		for _, node := range r.Nodes[1:] {
			if node.Backend != backend {
				return fmt.Errorf("torchrun adapter requires homogeneous backends, got %q and %q", backend, node.Backend)
			}
		}
	}
	for _, node := range r.Nodes {
		if node.WorkerID == "" || node.Address == "" {
			return fmt.Errorf("training node requires workerId and address")
		}
		if node.Backend == "cuda" && node.GPUCount == 0 {
			return fmt.Errorf("CUDA training node %q reports no GPU", node.WorkerID)
		}
	}
	return nil
}

func (r TrainingRequest) Backend() fabricwire.Transport {
	if len(r.Nodes) <= 1 {
		return fabricwire.TransportLocal
	}
	return fabricwire.TransportMTLSRPC
}

func BuildTorchRunCommand(request TrainingRequest, nodeRank uint32) ([]string, error) {
	if err := request.Validate(); err != nil {
		return nil, err
	}
	if nodeRank >= uint32(len(request.Nodes)) {
		return nil, fmt.Errorf("node rank %d is outside the training world", nodeRank)
	}
	args := []string{
		"torchrun",
		"--nnodes", strconv.Itoa(len(request.Nodes)),
		"--nproc-per-node", strconv.FormatUint(uint64(request.ProcessesPerNode), 10),
		"--node-rank", strconv.FormatUint(uint64(nodeRank), 10),
		"--rdzv-backend", "c10d",
		"--rdzv-endpoint", request.RendezvousEndpoint,
		request.TrainerPath,
		"--model", request.ModelPath,
		"--dataset", request.DatasetPath,
		"--output-dir", request.CheckpointDirectory,
		"--parallelism", string(request.Parallelism),
		"--checkpoint-interval-steps", strconv.FormatUint(request.CheckpointIntervalSteps, 10),
		"--model-digest", request.ModelDigest,
	}
	if request.ResumeCheckpoint != "" {
		args = append(args, "--resume", request.ResumeCheckpoint)
	}
	return args, nil
}
