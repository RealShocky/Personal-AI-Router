// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	"nvpair-shared/fabricwire"
)

type StartRequest struct {
	JobID          string                  `json:"jobId"`
	ModelDigest    string                  `json:"modelDigest"`
	ServerPath     string                  `json:"serverPath"`
	ModelPath      string                  `json:"modelPath"`
	HTTPPort       int                     `json:"httpPort"`
	SlotSavePath   string                  `json:"slotSavePath,omitempty"`
	CheckpointFile string                  `json:"checkpointFile,omitempty"`
	CheckpointSlot int                     `json:"checkpointSlot,omitempty"`
	Group          fabricwire.GroupRequest `json:"group"`
}

type Execution struct {
	JobID    string `json:"jobId"`
	PID      int    `json:"pid"`
	State    string `json:"state"`
	RPCPeers int    `json:"rpcPeers"`
	HTTPPort int    `json:"httpPort"`
}

type ExecutionManager struct {
	mu         sync.Mutex
	ctx        context.Context
	cluster    string
	executions map[string]*runningExecution
	command    func(context.Context, string, ...string) *exec.Cmd
}

type runningExecution struct {
	execution     Execution
	cmd           *exec.Cmd
	request       StartRequest
	group         fabricwire.GroupRequest
	plan          fabricwire.GroupPlan
	relays        []*rpcRelay
	stopRequested bool
}

func NewExecutionManager(ctx context.Context, clusterDir string) *ExecutionManager {
	return &ExecutionManager{ctx: ctx, cluster: clusterDir, executions: make(map[string]*runningExecution), command: exec.CommandContext}
}

func (m *ExecutionManager) StartWithExecutionPlan(request StartRequest, executionPlan fabricwire.ExecutionPlan) (Execution, error) {
	if err := executionPlan.Validate(); err != nil {
		return Execution{}, fmt.Errorf("invalid execution plan: %w", err)
	}
	if request.ModelDigest != "" && request.ModelDigest != executionPlan.ModelDigest {
		return Execution{}, fmt.Errorf("model digest does not match execution plan")
	}
	if request.ModelDigest == "" {
		request.ModelDigest = executionPlan.ModelDigest
	}
	group := fabricwire.GroupPlan{
		GroupID:   executionPlan.PlanID,
		Runtime:   executionPlan.Runtime,
		Epoch:     executionPlan.Epoch,
		Workers:   make([]string, 0, len(executionPlan.Workers)),
		Endpoints: make(map[string]string),
	}
	for _, assignment := range executionPlan.Workers {
		group.Workers = append(group.Workers, assignment.WorkerID)
		if assignment.Endpoint != "" {
			group.Endpoints[assignment.WorkerID] = assignment.Endpoint
		}
	}
	request.Group.GroupID = group.GroupID
	request.Group.Runtime = group.Runtime
	request.Group.WorkerGoal = uint32(len(group.Workers))
	if request.Group.ModelDigest == "" {
		request.Group.ModelDigest = executionPlan.ModelDigest
	}
	return m.Start(request, group)
}

func (m *ExecutionManager) Start(request StartRequest, plan fabricwire.GroupPlan) (Execution, error) {
	if request.JobID == "" || request.ServerPath == "" || request.ModelPath == "" || request.HTTPPort <= 0 {
		return Execution{}, fmt.Errorf("job start requires jobId, serverPath, modelPath, and positive httpPort")
	}
	if len(plan.Workers) == 0 {
		return Execution{}, fmt.Errorf("job start requires at least one planned worker")
	}

	m.mu.Lock()
	if _, exists := m.executions[request.JobID]; exists {
		m.mu.Unlock()
		return Execution{}, fmt.Errorf("job is already running")
	}
	m.mu.Unlock()

	relays := make([]*rpcRelay, 0, len(plan.Endpoints))
	rpcAddresses := make([]string, 0, len(plan.Endpoints))
	for _, workerID := range plan.Workers {
		endpoint := plan.Endpoints[workerID]
		if endpoint == "" {
			continue
		}
		relay, err := openRPCRelay(m.ctx, "127.0.0.1:0", endpoint, m.cluster)
		if err != nil {
			closeRelays(relays)
			return Execution{}, fmt.Errorf("open relay for %s: %w", workerID, err)
		}
		relays = append(relays, relay)
		rpcAddresses = append(rpcAddresses, relay.Addr())
	}

	args := []string{"--model", request.ModelPath, "--host", "127.0.0.1", "--port", fmt.Sprintf("%d", request.HTTPPort), "--n-gpu-layers", "all"}
	if request.SlotSavePath != "" {
		args = append(args, "--slot-save-path", request.SlotSavePath)
	}
	if len(rpcAddresses) > 0 {
		args = append(args, "--rpc", strings.Join(rpcAddresses, ","))
	}
	cmd := m.command(m.ctx, request.ServerPath, args...)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		closeRelays(relays)
		return Execution{}, fmt.Errorf("start llama server: %w", err)
	}
	execution := Execution{JobID: request.JobID, PID: cmd.Process.Pid, State: "running", RPCPeers: len(rpcAddresses), HTTPPort: request.HTTPPort}
	running := &runningExecution{execution: execution, cmd: cmd, request: request, group: request.Group, plan: plan, relays: relays}
	m.mu.Lock()
	m.executions[request.JobID] = running
	m.mu.Unlock()
	go m.wait(request.JobID, running)
	if request.CheckpointFile != "" && request.SlotSavePath != "" {
		go restoreSlotCheckpoint(m.ctx, request.HTTPPort, request.CheckpointSlot, request.CheckpointFile)
	}
	return execution, nil
}

func (m *ExecutionManager) wait(jobID string, running *runningExecution) {
	_ = running.cmd.Wait()
	closeRelays(running.relays)
	m.mu.Lock()
	defer m.mu.Unlock()
	if current, ok := m.executions[jobID]; !ok || current != running {
		return
	}
	running.execution.State = "failed"
	if running.stopRequested || m.ctx.Err() != nil {
		running.execution.State = "stopped"
	}
}

func (m *ExecutionManager) Stop(jobID string) error {
	m.mu.Lock()
	running, ok := m.executions[jobID]
	if ok {
		running.stopRequested = true
	}
	m.mu.Unlock()
	if !ok {
		return fmt.Errorf("job is not running")
	}
	if running.cmd.Process == nil {
		return nil
	}
	return running.cmd.Process.Kill()
}

func (m *ExecutionManager) Status(jobID string) (Execution, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	running, ok := m.executions[jobID]
	if !ok {
		return Execution{}, false
	}
	return running.execution, true
}

func (m *ExecutionManager) Statuses() []Execution {
	m.mu.Lock()
	defer m.mu.Unlock()
	statuses := make([]Execution, 0, len(m.executions))
	for _, running := range m.executions {
		statuses = append(statuses, running.execution)
	}
	return statuses
}
func closeRelays(relays []*rpcRelay) {
	for _, relay := range relays {
		_ = relay.Close()
	}
}

func (m *ExecutionManager) RecoverFailed(manager *Manager, jobs *JobStore) {
	m.mu.Lock()
	candidates := make([]*runningExecution, 0)
	for _, running := range m.executions {
		unhealthy := false
		for _, workerID := range running.plan.Workers {
			if !manager.Assignable(workerID) {
				unhealthy = true
				break
			}
		}
		if !running.stopRequested && (running.execution.State == "failed" || unhealthy) {
			candidates = append(candidates, running)
		}
	}
	m.mu.Unlock()
	for _, running := range candidates {
		group := running.group
		group.GroupID = fmt.Sprintf("%s-recovery-%d", group.GroupID, running.plan.Epoch)
		plan, err := manager.PlanGroup(group)
		if err != nil {
			continue
		}
		m.mu.Lock()
		current, ok := m.executions[running.execution.JobID]
		if !ok || current != running {
			m.mu.Unlock()
			continue
		}
		delete(m.executions, running.execution.JobID)
		m.mu.Unlock()
		if running.cmd.Process != nil {
			_ = running.cmd.Process.Kill()
		}
		checkpoint, err := jobs.Recover(running.execution.JobID, group.GroupID, plan.Epoch)
		if err != nil {
			continue
		}
		running.request.Group = group
		running.request.CheckpointSlot = checkpoint.SlotID
		running.request.CheckpointFile = checkpoint.Filename
		if _, err := m.Start(running.request, plan); err != nil {
			return
		}
	}
}

func restoreSlotCheckpoint(ctx context.Context, port, slotID int, filename string) {
	client := &http.Client{Timeout: 2 * time.Second}
	deadline := time.NewTimer(30 * time.Second)
	defer deadline.Stop()
	for {
		endpoint := fmt.Sprintf("http://127.0.0.1:%d/slots/%d?action=restore&filename=%s", port, slotID, url.QueryEscape(filename))
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, nil)
		if err == nil {
			resp, requestErr := client.Do(req)
			if requestErr == nil {
				_ = resp.Body.Close()
				if resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices {
					return
				}
			}
		}
		timer := time.NewTimer(250 * time.Millisecond)
		select {
		case <-ctx.Done():
			timer.Stop()
			return
		case <-deadline.C:
			timer.Stop()
			return
		case <-timer.C:
		}
	}
}

func (m *ExecutionManager) SaveCheckpoint(checkpoint fabricwire.Checkpoint) error {
	status, ok := m.Status(checkpoint.JobID)
	if !ok {
		return fmt.Errorf("job is not running")
	}
	if checkpoint.Filename == "" {
		return fmt.Errorf("checkpoint filename is required")
	}
	endpoint := fmt.Sprintf("http://127.0.0.1:%d/slots/%d?action=save&filename=%s", status.HTTPPort, checkpoint.SlotID, url.QueryEscape(checkpoint.Filename))
	ctx, cancel := context.WithTimeout(m.ctx, 10*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, nil)
	if err != nil {
		return err
	}
	resp, err := (&http.Client{Timeout: 10 * time.Second}).Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return fmt.Errorf("llama checkpoint save returned HTTP %d", resp.StatusCode)
	}
	return nil
}
