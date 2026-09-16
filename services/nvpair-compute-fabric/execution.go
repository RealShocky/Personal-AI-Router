// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"nvpair-shared/fabricwire"
)

type StartRequest struct {
	JobID                     string                  `json:"jobId"`
	ModelDigest               string                  `json:"modelDigest"`
	ServerPath                string                  `json:"serverPath"`
	ServerPrefixArgs          []string                `json:"serverPrefixArgs,omitempty"`
	ModelPath                 string                  `json:"modelPath"`
	HTTPPort                  int                     `json:"httpPort"`
	SlotSavePath              string                  `json:"slotSavePath,omitempty"`
	CheckpointFile            string                  `json:"checkpointFile,omitempty"`
	CheckpointSlot            int                     `json:"checkpointSlot,omitempty"`
	CheckpointIntervalSeconds uint64                  `json:"checkpointIntervalSeconds,omitempty"`
	Group                     fabricwire.GroupRequest `json:"group"`
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
	checkpoint func(fabricwire.Checkpoint) error
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

const maxInferenceRecoveryAttempts uint32 = 3
const (
	wslRelayPortFirst uint16 = 52000
	wslRelayPortLast  uint16 = 52999
)

func NewExecutionManager(ctx context.Context, clusterDir string) *ExecutionManager {
	return &ExecutionManager{ctx: ctx, cluster: clusterDir, executions: make(map[string]*runningExecution), command: exec.CommandContext}
}

func (m *ExecutionManager) SetCheckpointSink(sink func(fabricwire.Checkpoint) error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.checkpoint = sink
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
		PeerIDs:   make(map[string]string),
	}
	for _, assignment := range executionPlan.Workers {
		group.Workers = append(group.Workers, assignment.WorkerID)
		if assignment.Endpoint != "" {
			group.Endpoints[assignment.WorkerID] = assignment.Endpoint
		}
		if assignment.PeerID != "" {
			group.PeerIDs[assignment.WorkerID] = assignment.PeerID
		}
	}
	request.Group.GroupID = group.GroupID
	request.Group.Runtime = group.Runtime
	request.Group.WorkerGoal = uint32(len(group.Workers))
	return m.Start(request, group)
}

func (m *ExecutionManager) Start(request StartRequest, plan fabricwire.GroupPlan) (Execution, error) {
	if request.JobID == "" || request.ServerPath == "" || request.ModelPath == "" || request.HTTPPort <= 0 {
		return Execution{}, fmt.Errorf("job start requires jobId, serverPath, modelPath, and positive httpPort")
	}
	if len(plan.Workers) == 0 {
		return Execution{}, fmt.Errorf("job start requires at least one planned worker")
	}
	if request.CheckpointIntervalSeconds > 0 && (request.CheckpointFile == "" || request.SlotSavePath == "") {
		return Execution{}, fmt.Errorf("checkpoint interval requires checkpointFile and slotSavePath")
	}
	if request.CheckpointFile != "" && !validSlotFilename(request.CheckpointFile) {
		return Execution{}, fmt.Errorf("checkpointFile must be a relative filename without path separators")
	}
	if err := ensureSlotSavePath(request.ServerPath, request.ServerPrefixArgs, request.SlotSavePath); err != nil {
		return Execution{}, fmt.Errorf("prepare slot save path: %w", err)
	}

	m.mu.Lock()
	if _, exists := m.executions[request.JobID]; exists {
		m.mu.Unlock()
		return Execution{}, fmt.Errorf("job is already running")
	}
	m.mu.Unlock()

	relays := make([]*rpcRelay, 0, len(plan.Endpoints))
	rpcAddresses := make([]string, 0, len(plan.Endpoints))
	clientHost := ""
	if runtime.GOOS == "windows" && isWSLLauncher(request.ServerPath) && len(plan.Endpoints) > 0 {
		var err error
		clientHost, err = resolveWSLHostGateway(m.ctx, request.ServerPath, request.ServerPrefixArgs)
		if err != nil {
			return Execution{}, fmt.Errorf("resolve WSL host gateway: %w", err)
		}
	}
	listenAddr := "127.0.0.1:0"
	for _, workerID := range plan.Workers {
		endpoint := plan.Endpoints[workerID]
		if endpoint == "" {
			continue
		}
		var relay *rpcRelay
		var err error
		if clientHost != "" {
			relay, err = openWSLRPCRelay(m.ctx, endpoint, m.cluster, clientHost, plan.PeerIDs[workerID])
		} else {
			relay, err = openRPCRelayForClient(m.ctx, listenAddr, endpoint, m.cluster, clientHost, plan.PeerIDs[workerID])
		}
		if err != nil {
			closeRelays(relays)
			return Execution{}, fmt.Errorf("open relay for %s: %w", workerID, err)
		}
		go relay.serve(m.ctx)
		relays = append(relays, relay)
		rpcAddresses = append(rpcAddresses, relay.Addr())
	}

	args := append([]string(nil), request.ServerPrefixArgs...)
	args = append(args, "--model", request.ModelPath, "--host", "127.0.0.1", "--port", fmt.Sprintf("%d", request.HTTPPort), "--n-gpu-layers", "all")
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
	if request.CheckpointIntervalSeconds > 0 && request.CheckpointFile != "" && request.SlotSavePath != "" {
		go m.checkpointLoop(running)
	}
	if request.CheckpointFile != "" && request.SlotSavePath != "" {
		go restoreSlotCheckpoint(m.ctx, request.HTTPPort, request.CheckpointSlot, request.CheckpointFile)
	}
	return execution, nil
}

func wslRelayPort(index int) uint16 {
	return wslRelayPortFirst + uint16(index)
}

func wslRelayListenAddress(index int) (string, error) {
	if index < 0 || wslRelayPort(index) > wslRelayPortLast {
		return "", fmt.Errorf("WSL relay port index %d is outside %d-%d", index, wslRelayPortFirst, wslRelayPortLast)
	}
	return fmt.Sprintf("0.0.0.0:%d", wslRelayPort(index)), nil
}

func openWSLRPCRelay(ctx context.Context, remoteURL, clusterDir, clientHost, peerID string) (*rpcRelay, error) {
	for index := 0; ; index++ {
		listenAddr, err := wslRelayListenAddress(index)
		if err != nil {
			return nil, fmt.Errorf("no WSL relay ports available: %w", err)
		}
		relay, err := openRPCRelayForClient(ctx, listenAddr, remoteURL, clusterDir, clientHost, peerID)
		if err == nil {
			return relay, nil
		}
		if !strings.Contains(strings.ToLower(err.Error()), "address already in use") {
			return nil, err
		}
	}
}

func isWSLLauncher(path string) bool {
	return strings.EqualFold(filepath.Base(path), "wsl.exe")
}

func resolveWSLHostGateway(ctx context.Context, serverPath string, prefixArgs []string) (string, error) {
	args := make([]string, 0, 5)
	for index := 0; index+1 < len(prefixArgs); index++ {
		if prefixArgs[index] != "-d" && prefixArgs[index] != "--distribution" {
			continue
		}
		args = append(args, prefixArgs[index], prefixArgs[index+1])
		break
	}
	args = append(args, "--", "ip", "route", "show", "default")
	output, err := exec.CommandContext(ctx, serverPath, args...).Output()
	if err != nil {
		return "", err
	}
	return parseWSLHostGateway(string(output))
}

func parseWSLHostGateway(output string) (string, error) {
	fields := strings.Fields(output)
	for index := 0; index+1 < len(fields); index++ {
		if fields[index] != "via" {
			continue
		}
		gateway := net.ParseIP(fields[index+1])
		if gateway != nil {
			return gateway.String(), nil
		}
	}
	return "", fmt.Errorf("default route gateway not found")
}

func ensureSlotSavePath(serverPath string, prefixArgs []string, slotPath string) error {
	if slotPath == "" {
		return nil
	}
	if !strings.EqualFold(serverPath, "wsl.exe") && !strings.HasSuffix(strings.ToLower(serverPath), "\\wsl.exe") {
		return os.MkdirAll(slotPath, 0o755)
	}

	args := make([]string, 0, 6)
	for index := 0; index+1 < len(prefixArgs); index++ {
		if prefixArgs[index] != "-d" && prefixArgs[index] != "--distribution" {
			continue
		}
		args = append(args, prefixArgs[index], prefixArgs[index+1])
		break
	}
	args = append(args, "--", "mkdir", "-p", slotPath)
	if err := exec.Command(serverPath, args...).Run(); err != nil {
		return fmt.Errorf("wsl mkdir failed: %w", err)
	}
	return nil
}

func (m *ExecutionManager) checkpointLoop(running *runningExecution) {
	interval := time.Duration(running.request.CheckpointIntervalSeconds) * time.Second
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	var stage uint32
	for {
		select {
		case <-ticker.C:
			stage++
			checkpoint := fabricwire.Checkpoint{
				JobID:    running.execution.JobID,
				GroupID:  running.plan.GroupID,
				Epoch:    running.plan.Epoch,
				Stage:    stage,
				SlotID:   running.request.CheckpointSlot,
				Filename: running.request.CheckpointFile,
			}
			if err := m.SaveCheckpoint(checkpoint); err != nil {
				log.Printf("checkpoint save failed job=%s stage=%d: %v", checkpoint.JobID, checkpoint.Stage, err)
				continue
			}
			m.mu.Lock()
			sink := m.checkpoint
			m.mu.Unlock()
			if sink != nil {
				if err := sink(checkpoint); err != nil {
					log.Printf("checkpoint commit failed job=%s stage=%d: %v", checkpoint.JobID, checkpoint.Stage, err)
				}
			}
		case <-m.ctx.Done():
			return
		}
	}
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
		allowed, err := jobs.BeginRecovery(running.execution.JobID, maxInferenceRecoveryAttempts)
		if err != nil {
			continue
		}
		if !allowed {
			running.execution.State = "failed"
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
		endpoint := fmt.Sprintf("http://127.0.0.1:%d/slots/%d?action=restore", port, slotID)
		body, marshalErr := json.Marshal(map[string]string{"filename": filename})
		if marshalErr != nil {
			return
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, strings.NewReader(string(body)))
		if err == nil {
			req.Header.Set("Content-Type", "application/json")
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
	if !validSlotFilename(checkpoint.Filename) {
		return fmt.Errorf("checkpoint filename must be a relative filename without path separators")
	}
	endpoint := fmt.Sprintf("http://127.0.0.1:%d/slots/%d?action=save", status.HTTPPort, checkpoint.SlotID)
	ctx, cancel := context.WithTimeout(m.ctx, 10*time.Second)
	defer cancel()
	body, err := json.Marshal(map[string]string{"filename": checkpoint.Filename})
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, strings.NewReader(string(body)))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
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

func validSlotFilename(filename string) bool {
	return filename != "" && filename != "." && filename != ".." && filepath.Base(filename) == filename && !strings.ContainsAny(filename, `/\\`)
}
