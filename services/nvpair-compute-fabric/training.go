// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"sync"

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

type TrainingCoordinator struct {
	start TrainingStartFunc
	stop  TrainingStopFunc
}

func NewTrainingCoordinator(start TrainingStartFunc, stop TrainingStopFunc) *TrainingCoordinator {
	return &TrainingCoordinator{start: start, stop: stop}
}

func (c *TrainingCoordinator) StartGroup(ctx context.Context, request TrainingRequest) ([]TrainingExecution, error) {
	if err := request.Validate(); err != nil {
		return nil, err
	}
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
