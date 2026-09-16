// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/tls"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/fabricwire"
	"nvpair-shared/ipc"
)

var Version = "dev"

func main() {
	ipcPath := flag.String("ipc", "", "IPC endpoint: Unix socket or Windows named pipe")
	// Workers normally emit heartbeats every half of their configured timeout.
	// Keep the coordinator lease longer than the 10-second fleet default so a
	// healthy worker is not quarantined merely because one heartbeat is late.
	timeout := flag.Duration("heartbeat-timeout", 15*time.Second, "worker heartbeat lease timeout")
	httpPort := flag.Int("http-port", 0, "authenticated fabric HTTP port (0 disables the network endpoint)")
	clusterDir := flag.String("cluster-dir", "", "cluster trust directory for the authenticated fabric endpoint")
	stateDir := flag.String("state-dir", "", "durable coordinator state directory for checkpoints")
	rpcTarget := flag.String("rpc-target", "", "loopback ggml-rpc-server target for the authenticated raw RPC tunnel")
	relayListen := flag.String("rpc-relay-listen", "", "local TCP address to relay to --rpc-relay-url")
	relayURL := flag.String("rpc-relay-url", "", "HTTPS fabric URL for a remote RPC tunnel")
	relayPeerID := flag.String("rpc-relay-peer-id", "", "trusted peer UUID for a remote RPC tunnel (required when multiple peers are pinned)")
	relaySpecs := flag.String("rpc-relay-specs", "", "semicolon-separated local|HTTPS fabric URL RPC relay pairs")
	rpcServerPath := flag.String("rpc-server-path", "", "explicit ggml-rpc-server executable to supervise")
	rpcPort := flag.Int("rpc-port", 50052, "loopback port for a supervised ggml-rpc-server")
	rpcMemory := flag.String("rpc-memory", "", "optional memory argument passed to ggml-rpc-server")
	llamaServerPath := flag.String("llama-server-path", "", "explicit llama-server executable for distributed inference")
	llamaModel := flag.String("llama-model", "", "GGUF model path for the supervised llama-server")
	llamaRPC := flag.String("llama-rpc", "", "comma-separated local RPC relay addresses passed to llama-server")
	llamaPort := flag.Int("llama-port", 0, "loopback HTTP port for the supervised llama-server (0 disables it)")
	coordinatorURL := flag.String("coordinator-url", "", "HTTPS fabric endpoint to join as a worker")
	advertiseURL := flag.String("advertise-url", "", "HTTPS fabric endpoint advertised to the coordinator for authenticated RPC relays")
	runtimeModelDigest := flag.String("model-digest", "", "SHA-256 model digest advertised by worker mode (computed from --llama-model when omitted)")
	workerID := flag.String("worker-id", "", "stable worker identity for worker mode")
	nodeID := flag.String("node-id", "", "stable host identity for worker mode")
	runtimeName := flag.String("runtime", "cpu", "worker runtime label: cpu, cuda, or metal")
	backend := flag.String("backend", "cpu", "comma-separated worker backends")
	logLevel := flag.String("log-level", "info", "shared service log level: debug, info, warn, or error")
	showVersion := flag.Bool("version", false, "print version and exit")
	daemon := flag.Bool("daemon", false, "run without a JSON-RPC stdin session")
	flag.Parse()
	if !validLogLevel(*logLevel) {
		log.Fatalf("unsupported log level %q", *logLevel)
	}
	if *showVersion {
		fmt.Println(Version)
		return
	}

	var transport io.ReadWriteCloser
	if *ipcPath == "" {
		transport = newStdioTransport()
	} else {
		conn, err := ipc.Dial(*ipcPath)
		if err != nil {
			log.Fatalf("failed to connect to IPC endpoint %q: %v", *ipcPath, err)
		}
		transport = conn
	}
	defer transport.Close()

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	mgr := NewManager(*timeout)
	logicalDevice := NewLogicalDeviceManager(mgr)
	jobs := NewJobStore(*stateDir)
	executions := NewExecutionManager(ctx, *clusterDir)
	executions.SetCheckpointSink(jobs.SaveCheckpoint)
	training := NewTrainingManager(ctx)
	trainingCoordinator := newHTTPTrainingCoordinator(mgr, *clusterDir, *stateDir, 15*time.Second)
	codec := NewCodec(transport)
	if *httpPort != 0 {
		if *clusterDir == "" {
			log.Printf("fabric HTTP endpoint disabled: no cluster directory")
		} else {
			var coordinatorHTTP *TrainingCoordinator
			if *coordinatorURL == "" {
				coordinatorHTTP = trainingCoordinator
			}
			go serveFabricHTTP(ctx, *httpPort, *clusterDir, mgr, *rpcTarget, training, coordinatorHTTP, jobs, executions)
		}
	}
	if *rpcServerPath != "" {
		go superviseRPCServer(ctx, *rpcServerPath, *rpcPort, *rpcMemory)
	}
	if *llamaServerPath != "" {
		if *llamaModel == "" || *llamaPort == 0 {
			log.Fatalf("llama-server mode requires --llama-model and --llama-port")
		}
		go superviseLlamaServer(ctx, *llamaServerPath, *llamaModel, *llamaRPC, *llamaPort)
	}
	if *relayListen != "" || *relayURL != "" {
		if *relayListen == "" || *relayURL == "" || *clusterDir == "" {
			log.Fatalf("RPC relay mode requires --rpc-relay-listen, --rpc-relay-url, and --cluster-dir")
		}
		go serveRPCRelay(ctx, *relayListen, *relayURL, *clusterDir, *relayPeerID)
	}
	for _, spec := range strings.Split(*relaySpecs, ";") {
		parts := strings.SplitN(spec, "|", 2)
		if len(parts) != 2 || parts[0] == "" || parts[1] == "" || *clusterDir == "" {
			if strings.TrimSpace(spec) != "" {
				log.Printf("ignoring invalid RPC relay spec")
			}
			continue
		}
		go serveRPCRelay(ctx, parts[0], parts[1], *clusterDir, "")
	}
	if *coordinatorURL != "" {
		if *workerID == "" || *nodeID == "" || *clusterDir == "" {
			log.Fatalf("worker mode requires --worker-id, --node-id, and --cluster-dir")
		}
		modelDigest := *runtimeModelDigest
		if modelDigest == "" && *llamaModel != "" {
			digest, err := digestFile(*llamaModel)
			if err != nil {
				log.Printf("model digest unavailable: %v", err)
			} else {
				modelDigest = digest
			}
		}
		go runWorkerHeartbeats(ctx, *coordinatorURL, *clusterDir, *workerID, *nodeID, *advertiseURL, *runtimeName, *backend, modelDigest, *llamaServerPath != "", *rpcTarget, *timeout)
	}

	go func() {
		ticker := time.NewTicker(*timeout)
		defer ticker.Stop()
		for {
			select {
			case now := <-ticker.C:
				mgr.Reconcile(now)
				logicalDevice.Reconcile()
				executions.RecoverFailed(mgr, jobs)
			case <-ctx.Done():
				return
			}
		}
	}()

	if *daemon {
		<-ctx.Done()
		return
	}
	if err := codec.Notify("ready", map[string]string{"version": Version}); err != nil {
		log.Fatalf("failed to announce readiness: %v", err)
	}
	for ctx.Err() == nil {
		msg, err := codec.Read()
		if err != nil {
			if err == io.EOF || ctx.Err() != nil {
				return
			}
			log.Printf("JSON-RPC read error: %v", err)
			continue
		}
		handleMessage(codec, mgr, jobs, executions, trainingCoordinator, logicalDevice, cancel, msg)
	}
}

func validLogLevel(level string) bool {
	switch strings.ToLower(level) {
	case "debug", "info", "warn", "error":
		return true
	default:
		return false
	}
}

func runWorkerHeartbeats(ctx context.Context, coordinatorURL, clusterDir, workerID, nodeID, endpoint, runtimeName, backend, modelDigest string, checkpointSupport bool, rpcTarget string, timeout time.Duration) {
	mesh := clustertrust.Open(clusterDir)
	epoch := uint64(time.Now().UnixNano())
	backends := make([]string, 0, 2)
	for _, item := range strings.Split(backend, ",") {
		if item = strings.TrimSpace(item); item != "" {
			backends = append(backends, item)
		}
	}
	if len(backends) == 0 {
		backends = []string{"cpu"}
	}
	client := &http.Client{Timeout: timeout}
	state := fabricwire.WorkerReady
	var probeLatency uint64
	gpuMemory := gpuMemorySnapshot{}
	var transport *http.Transport
	for {
		mesh.Refresh()
		tlsConfig, ok := mesh.ClientTLSConfigAny()
		if !ok {
			log.Printf("worker heartbeat trust unavailable worker=%s", workerID)
		} else {
			if transport == nil {
				transport = &http.Transport{TLSClientConfig: tlsConfig}
				client.Transport = transport
			}
			heartbeat := fabricwire.Heartbeat{
				WorkerID:           workerID,
				NodeID:             nodeID,
				PeerID:             mesh.NodeUUID(),
				Endpoint:           endpoint,
				State:              state,
				Epoch:              epoch,
				Runtime:            runtimeName,
				Backends:           backends,
				ModelDigests:       modelDigests(modelDigest),
				CheckpointSupport:  checkpointSupport,
				RPCSupport:         rpcTargetAvailable(rpcTarget),
				ProbeLatencyMillis: probeLatency,
				MemoryFree:         systemMemoryFree(),
				GPUVramTotal:       gpuMemory.Total,
				GPUVramFree:        gpuMemory.Free,
				GPUCount:           gpuMemory.Count,
			}
			probeStart := time.Now()
			postErr := postFabricJSON(ctx, client, coordinatorURL+"/v1/fabric/heartbeat", heartbeat)
			probeLatency = uint64(time.Since(probeStart).Milliseconds())
			gpuMemory = readGPUMemory()
			if postErr != nil {
				log.Printf("worker heartbeat failed worker=%s err=%v", workerID, postErr)
				epoch++
				if postFabricRejoin(ctx, client, coordinatorURL+"/v1/fabric/rejoin", workerID, epoch) {
					state = fabricwire.WorkerReady
					log.Printf("worker rejoined worker=%s epoch=%d", workerID, epoch)
				} else {
					state = fabricwire.WorkerSuspect
					log.Printf("worker rejoin rejected worker=%s epoch=%d", workerID, epoch)
				}
			} else {
				state = fabricwire.WorkerReady
			}
		}
		timer := time.NewTimer(timeout / 2)
		select {
		case <-ctx.Done():
			timer.Stop()
			return
		case <-timer.C:
		}
	}
}

func rpcTargetAvailable(target string) bool {
	if target == "" {
		return false
	}
	connection, err := net.DialTimeout("tcp", target, 500*time.Millisecond)
	if err != nil {
		return false
	}
	_ = connection.Close()
	return true
}

func modelDigests(digest string) []string {
	if digest == "" {
		return nil
	}
	return []string{digest}
}

func digestFile(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", err
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}
func postFabricJSON(ctx context.Context, client *http.Client, endpoint string, value any) error {
	body, err := json.Marshal(value)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return fmt.Errorf("fabric endpoint returned HTTP %d", resp.StatusCode)
	}
	return nil
}

func newHTTPTrainingCoordinator(mgr *Manager, clusterDir, stateDir string, timeout time.Duration) *TrainingCoordinator {
	mesh := clustertrust.Open(clusterDir)
	start := func(ctx context.Context, node TrainingNode, request TrainingRequest, rank uint32) (TrainingExecution, error) {
		mesh.Refresh()
		tlsConfig, ok := mesh.ClientTLSConfigAny()
		if !ok {
			return TrainingExecution{}, fmt.Errorf("training trust unavailable for worker %s", node.WorkerID)
		}
		client := &http.Client{Timeout: timeout, Transport: &http.Transport{TLSClientConfig: tlsConfig}}
		endpoint, err := trainingEndpoint(node.Address, "/v1/fabric/training/start")
		if err != nil {
			return TrainingExecution{}, err
		}
		body, err := json.Marshal(map[string]any{"request": request, "nodeRank": rank})
		if err != nil {
			return TrainingExecution{}, err
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
		if err != nil {
			return TrainingExecution{}, err
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := client.Do(req)
		if err != nil {
			return TrainingExecution{}, fmt.Errorf("start worker %s: %w", node.WorkerID, err)
		}
		defer resp.Body.Close()
		if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
			detail, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
			message := strings.TrimSpace(string(detail))
			if message == "" {
				return TrainingExecution{}, fmt.Errorf("worker %s returned HTTP %d", node.WorkerID, resp.StatusCode)
			}
			return TrainingExecution{}, fmt.Errorf("worker %s returned HTTP %d: %s", node.WorkerID, resp.StatusCode, message)
		}
		var execution TrainingExecution
		if err := json.NewDecoder(resp.Body).Decode(&execution); err != nil {
			return TrainingExecution{}, fmt.Errorf("decode worker %s response: %w", node.WorkerID, err)
		}
		return execution, nil
	}
	stop := func(ctx context.Context, jobID string, node TrainingNode) error {
		mesh.Refresh()
		tlsConfig, ok := mesh.ClientTLSConfigAny()
		if !ok {
			return fmt.Errorf("training trust unavailable for worker %s", node.WorkerID)
		}
		endpoint, err := trainingEndpoint(node.Address, "/v1/fabric/training/stop")
		if err != nil {
			return err
		}
		endpoint += "?jobId=" + url.QueryEscape(jobID)
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, nil)
		if err != nil {
			return err
		}
		resp, err := (&http.Client{Timeout: timeout, Transport: &http.Transport{TLSClientConfig: tlsConfig}}).Do(req)
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
			return fmt.Errorf("worker %s returned HTTP %d while stopping", node.WorkerID, resp.StatusCode)
		}
		return nil
	}
	status := func(ctx context.Context, jobID string, node TrainingNode) (TrainingExecution, error) {
		mesh.Refresh()
		tlsConfig, ok := mesh.ClientTLSConfigAny()
		if !ok {
			return TrainingExecution{}, fmt.Errorf("training trust unavailable for worker %s", node.WorkerID)
		}
		endpoint, err := trainingEndpoint(node.Address, "/v1/fabric/training/status")
		if err != nil {
			return TrainingExecution{}, err
		}
		query := endpoint + "?jobId=" + url.QueryEscape(jobID)
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, query, nil)
		if err != nil {
			return TrainingExecution{}, err
		}
		resp, err := (&http.Client{Timeout: timeout, Transport: &http.Transport{TLSClientConfig: tlsConfig}}).Do(req)
		if err != nil {
			return TrainingExecution{}, err
		}
		defer resp.Body.Close()
		if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
			return TrainingExecution{}, fmt.Errorf("worker %s returned HTTP %d while reading status", node.WorkerID, resp.StatusCode)
		}
		var execution TrainingExecution
		if err := json.NewDecoder(resp.Body).Decode(&execution); err != nil {
			return TrainingExecution{}, fmt.Errorf("decode worker %s status: %w", node.WorkerID, err)
		}
		return execution, nil
	}
	coordinator := NewTrainingCoordinator(start, stop, stateDir)
	coordinator.SetStatusFunc(status)
	coordinator.SetRecoveryNodeProvider(func(request TrainingRequest) []TrainingNode {
		return readyTrainingReplacementNodes(mgr, request)
	})
	return coordinator
}

func readyTrainingReplacementNodes(mgr *Manager, request TrainingRequest) []TrainingNode {
	if len(request.Nodes) == 0 {
		return nil
	}
	backend := request.Nodes[0].Backend
	wanted := map[string]bool{backend: true}
	used := make(map[string]bool, len(request.Nodes))
	for _, node := range request.Nodes {
		used[node.WorkerID] = true
	}
	replacements := make([]TrainingNode, 0, len(request.Nodes))
	// A worker whose rank process failed may still be healthy and ready. Reuse
	// it first; a machine-level failure is naturally excluded by the heartbeat
	// state and will be replaced by a different ready worker below.
	for _, node := range request.Nodes {
		for _, record := range mgr.Workers() {
			if record.Heartbeat.WorkerID == node.WorkerID && record.State == fabricwire.WorkerReady && record.Heartbeat.Endpoint != "" && hasBackend(record.Heartbeat.Backends, wanted) {
				replacements = append(replacements, TrainingNode{WorkerID: record.Heartbeat.WorkerID, Address: record.Heartbeat.Endpoint, Backend: backend, GPUCount: record.Heartbeat.GPUCount})
				break
			}
		}
	}
	if len(replacements) == len(request.Nodes) {
		return replacements
	}
	for _, record := range mgr.Workers() {
		if record.State != fabricwire.WorkerReady || used[record.Heartbeat.WorkerID] || record.Heartbeat.Endpoint == "" || !hasBackend(record.Heartbeat.Backends, wanted) {
			continue
		}
		replacements = append(replacements, TrainingNode{WorkerID: record.Heartbeat.WorkerID, Address: record.Heartbeat.Endpoint, Backend: backend, GPUCount: record.Heartbeat.GPUCount})
		if len(replacements) == len(request.Nodes) {
			return replacements
		}
	}
	return nil
}

func trainingEndpoint(raw, path string) (string, error) {
	parsed, err := url.Parse(strings.TrimRight(raw, "/"))
	if err != nil || parsed.Scheme != "https" || parsed.Host == "" {
		return "", fmt.Errorf("training endpoint must be an HTTPS URL")
	}
	parsed.Path = path
	parsed.RawQuery = ""
	return parsed.String(), nil
}

func postFabricRejoin(ctx context.Context, client *http.Client, endpoint, workerID string, epoch uint64) bool {
	body, err := json.Marshal(map[string]any{"workerId": workerID, "epoch": epoch})
	if err != nil {
		return false
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return false
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return false
	}
	var result struct {
		Rejoined bool `json:"rejoined"`
	}
	return json.NewDecoder(resp.Body).Decode(&result) == nil && result.Rejoined
}

func serveFabricHTTP(ctx context.Context, port int, clusterDir string, mgr *Manager, rpcTarget string, training *TrainingManager, coordinator *TrainingCoordinator, jobs *JobStore, executions *ExecutionManager) {
	mesh := clustertrust.Open(clusterDir)
	go mesh.Watch(ctx, nil)
	config := mesh.ServerTLSConfig()
	server := &http.Server{
		Addr:      fmt.Sprintf("0.0.0.0:%d", port),
		TLSConfig: config,
		Handler:   fabricHTTPHandlerWithCoordinator(mesh, mgr, rpcTarget, training, coordinator, jobs, executions),
	}
	listener, err := net.Listen("tcp", server.Addr)
	if err != nil {
		log.Printf("fabric HTTP listen failed: %v", err)
		return
	}
	tlsListener := tls.NewListener(listener, config)
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		_ = server.Shutdown(shutdownCtx)
		_ = tlsListener.Close()
	}()
	log.Printf("fabric HTTP endpoint listening on %s", server.Addr)
	if err := server.Serve(tlsListener); err != nil && ctx.Err() == nil {
		log.Printf("fabric HTTP stopped: %v", err)
	}
}

func fabricHTTPHandler(mesh *clustertrust.Mesh, mgr *Manager, rpcTarget string) http.Handler {
	return fabricHTTPHandlerWithExecution(mesh, mgr, rpcTarget, nil, nil, nil)
}

func fabricHTTPHandlerWithTraining(mesh *clustertrust.Mesh, mgr *Manager, rpcTarget string, training *TrainingManager) http.Handler {
	return fabricHTTPHandlerWithExecution(mesh, mgr, rpcTarget, training, nil, nil)
}

func fabricHTTPHandlerWithExecution(mesh *clustertrust.Mesh, mgr *Manager, rpcTarget string, training *TrainingManager, jobs *JobStore, executions *ExecutionManager) http.Handler {
	return fabricHTTPHandlerWithCoordinator(mesh, mgr, rpcTarget, training, nil, jobs, executions)
}

func fabricHTTPHandlerWithCoordinator(mesh *clustertrust.Mesh, mgr *Manager, rpcTarget string, training *TrainingManager, coordinator *TrainingCoordinator, jobs *JobStore, executions *ExecutionManager) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/fabric/status", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		if _, ok := mesh.VerifyClientPin(r); !ok {
			w.WriteHeader(http.StatusForbidden)
			return
		}
		writeJSON(w, map[string]any{"workers": mgr.Workers(), "capacity": mgr.Capacity()})
	})
	if jobs != nil && executions != nil {
		mux.HandleFunc("/v1/fabric/inference/start", func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost {
				w.WriteHeader(http.StatusMethodNotAllowed)
				return
			}
			if _, ok := mesh.VerifyClientPin(r); !ok {
				w.WriteHeader(http.StatusForbidden)
				return
			}
			var request StartRequest
			if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
				http.Error(w, "invalid inference request", http.StatusBadRequest)
				return
			}
			execution, err := startInferenceJob(mgr, jobs, executions, request)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			writeJSON(w, execution)
		})
		mux.HandleFunc("/v1/fabric/inference/status", func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodGet {
				w.WriteHeader(http.StatusMethodNotAllowed)
				return
			}
			if _, ok := mesh.VerifyClientPin(r); !ok {
				w.WriteHeader(http.StatusForbidden)
				return
			}
			execution, ok := executions.Status(r.URL.Query().Get("jobId"))
			if !ok {
				http.Error(w, "inference job not found", http.StatusNotFound)
				return
			}
			writeJSON(w, execution)
		})
		mux.HandleFunc("/v1/fabric/inference/stop", func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost {
				w.WriteHeader(http.StatusMethodNotAllowed)
				return
			}
			if _, ok := mesh.VerifyClientPin(r); !ok {
				w.WriteHeader(http.StatusForbidden)
				return
			}
			if err := executions.Stop(r.URL.Query().Get("jobId")); err != nil {
				http.Error(w, err.Error(), http.StatusNotFound)
				return
			}
			writeJSON(w, map[string]bool{"stopped": true})
		})
	}
	mux.HandleFunc("/v1/fabric/rpc", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodConnect {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		if rpcTarget == "" {
			http.Error(w, "RPC target is not configured", http.StatusNotFound)
			return
		}
		if _, ok := mesh.VerifyClientPin(r); !ok {
			w.WriteHeader(http.StatusForbidden)
			return
		}
		backend, err := net.DialTimeout("tcp", rpcTarget, 5*time.Second)
		if err != nil {
			http.Error(w, "RPC target unavailable", http.StatusBadGateway)
			return
		}
		hijacker, ok := w.(http.Hijacker)
		if !ok {
			_ = backend.Close()
			http.Error(w, "tunnel unsupported", http.StatusInternalServerError)
			return
		}
		clientConn, rw, err := hijacker.Hijack()
		if err != nil {
			_ = backend.Close()
			return
		}
		_, _ = rw.WriteString("HTTP/1.1 200 Connection Established\r\n\r\n")
		_ = rw.Flush()
		bridgeConnectionsFromReader(backend, rw.Reader, clientConn)
	})
	mux.HandleFunc("/v1/fabric/heartbeat", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		if _, ok := mesh.VerifyClientPin(r); !ok {
			w.WriteHeader(http.StatusForbidden)
			return
		}
		var heartbeat fabricwire.Heartbeat
		if err := json.NewDecoder(r.Body).Decode(&heartbeat); err != nil {
			http.Error(w, "invalid heartbeat", http.StatusBadRequest)
			return
		}
		if err := mgr.Heartbeat(heartbeat, time.Now()); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		writeJSON(w, map[string]bool{"accepted": true})
	})
	mux.HandleFunc("/v1/fabric/rejoin", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		if _, ok := mesh.VerifyClientPin(r); !ok {
			w.WriteHeader(http.StatusForbidden)
			return
		}
		var params struct {
			WorkerID string `json:"workerId"`
			Epoch    uint64 `json:"epoch"`
		}
		if err := json.NewDecoder(r.Body).Decode(&params); err != nil {
			http.Error(w, "invalid rejoin", http.StatusBadRequest)
			return
		}
		writeJSON(w, map[string]bool{"rejoined": mgr.Rejoin(params.WorkerID, params.Epoch, time.Now())})
	})
	if training != nil && coordinator == nil {
		mux.HandleFunc("/v1/fabric/training/start", func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost {
				w.WriteHeader(http.StatusMethodNotAllowed)
				return
			}
			if _, ok := mesh.VerifyClientPin(r); !ok {
				w.WriteHeader(http.StatusForbidden)
				return
			}
			var params struct {
				Request  TrainingRequest `json:"request"`
				NodeRank uint32          `json:"nodeRank"`
			}
			if err := json.NewDecoder(r.Body).Decode(&params); err != nil {
				http.Error(w, "invalid training request", http.StatusBadRequest)
				return
			}
			execution, err := training.Start(params.Request, params.NodeRank)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			writeJSON(w, execution)
		})
		mux.HandleFunc("/v1/fabric/training/status", func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodGet {
				w.WriteHeader(http.StatusMethodNotAllowed)
				return
			}
			if _, ok := mesh.VerifyClientPin(r); !ok {
				w.WriteHeader(http.StatusForbidden)
				return
			}
			jobID := r.URL.Query().Get("jobId")
			execution, ok := training.Status(jobID)
			if !ok {
				http.Error(w, "training job not found", http.StatusNotFound)
				return
			}
			writeJSON(w, execution)
		})
		mux.HandleFunc("/v1/fabric/training/stop", func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost {
				w.WriteHeader(http.StatusMethodNotAllowed)
				return
			}
			if _, ok := mesh.VerifyClientPin(r); !ok {
				w.WriteHeader(http.StatusForbidden)
				return
			}
			jobID := r.URL.Query().Get("jobId")
			if err := training.Stop(jobID); err != nil {
				http.Error(w, err.Error(), http.StatusNotFound)
				return
			}
			writeJSON(w, map[string]bool{"stopped": true})
		})
	}
	if coordinator != nil {
		mux.HandleFunc("/v1/fabric/training/start", func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost {
				w.WriteHeader(http.StatusMethodNotAllowed)
				return
			}
			if _, ok := mesh.VerifyClientPin(r); !ok {
				w.WriteHeader(http.StatusForbidden)
				return
			}
			var request TrainingRequest
			if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
				http.Error(w, "invalid training request", http.StatusBadRequest)
				return
			}
			if err := mgr.ValidateTrainingPlacement(request); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			executions, err := coordinator.StartGroup(r.Context(), request)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			writeJSON(w, executions)
		})
		mux.HandleFunc("/v1/fabric/training/status", func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodGet {
				w.WriteHeader(http.StatusMethodNotAllowed)
				return
			}
			if _, ok := mesh.VerifyClientPin(r); !ok {
				w.WriteHeader(http.StatusForbidden)
				return
			}
			jobID := r.URL.Query().Get("jobId")
			if err := coordinator.RefreshStatus(r.Context(), jobID); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			status, ok := coordinator.StatusGroup(jobID)
			if !ok {
				http.Error(w, "training group not found", http.StatusNotFound)
				return
			}
			writeJSON(w, status)
		})
		mux.HandleFunc("/v1/fabric/training/stop", func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost {
				w.WriteHeader(http.StatusMethodNotAllowed)
				return
			}
			if _, ok := mesh.VerifyClientPin(r); !ok {
				w.WriteHeader(http.StatusForbidden)
				return
			}
			if err := coordinator.StopGroup(r.Context(), r.URL.Query().Get("jobId")); err != nil {
				http.Error(w, err.Error(), http.StatusNotFound)
				return
			}
			writeJSON(w, map[string]bool{"stopped": true})
		})
	}
	return mux
}

func superviseRPCServer(ctx context.Context, executable string, port int, memory string) {
	args := []string{"--host", "127.0.0.1", "--port", fmt.Sprintf("%d", port)}
	if memory != "" {
		args = append(args, "--mem", memory)
	}
	args = append(args, "-c")
	superviseChild(ctx, "ggml-rpc-server", executable, args)
}

func superviseLlamaServer(ctx context.Context, executable, model, rpcAddresses string, port int) {
	args := []string{"--model", model, "--host", "127.0.0.1", "--port", fmt.Sprintf("%d", port), "--n-gpu-layers", "all"}
	if rpcAddresses != "" {
		args = append(args, "--rpc", rpcAddresses)
	}
	superviseChild(ctx, "llama-server", executable, args)
}

func superviseChild(ctx context.Context, name, executable string, args []string) {
	backoff := time.Second
	for ctx.Err() == nil {
		cmd := exec.CommandContext(ctx, executable, args...)
		cmd.Env = childEnvironment(executable)
		cmd.Stdout = os.Stderr
		cmd.Stderr = os.Stderr
		err := cmd.Run()
		if ctx.Err() != nil {
			return
		}
		log.Printf("%s stopped: %v; restarting", name, err)
		timer := time.NewTimer(backoff)
		select {
		case <-timer.C:
		case <-ctx.Done():
			if !timer.Stop() {
				<-timer.C
			}
			return
		}
		if backoff < 30*time.Second {
			backoff *= 2
		}
	}
}

func childEnvironment(executable string) []string {
	environment := os.Environ()
	if runtime.GOOS == "windows" {
		return environment
	}
	variable := "LD_LIBRARY_PATH"
	if runtime.GOOS == "darwin" {
		variable = "DYLD_LIBRARY_PATH"
	}
	directory := filepath.Dir(executable)
	current := os.Getenv(variable)
	filtered := make([]string, 0, len(environment)+1)
	prefix := variable + "="
	for _, value := range environment {
		if !strings.HasPrefix(value, prefix) {
			filtered = append(filtered, value)
		}
	}
	if current == "" {
		return append(filtered, variable+"="+directory)
	}
	return append(filtered, variable+"="+current+string(os.PathListSeparator)+directory)
}

func serveRPCRelay(ctx context.Context, listenAddr, remoteURL, clusterDir, peerID string) {
	relay, err := openRPCRelayForClient(ctx, listenAddr, remoteURL, clusterDir, "", peerID)
	if err != nil {
		log.Printf("RPC relay listen failed: %v", err)
		return
	}
	defer relay.Close()
	relay.serve(ctx)
}

func (r *rpcRelay) serve(ctx context.Context) {
	for {
		conn, err := r.listener.Accept()
		if err != nil {
			if ctx.Err() == nil {
				log.Printf("RPC relay accept failed: %v", err)
			}
			return
		}
		go relayRPCConnection(ctx, conn, r.remoteURL, r.clusterDir, r.peerID)
	}
}

type rpcRelay struct {
	listener   net.Listener
	remoteURL  string
	clusterDir string
	clientHost string
	peerID     string
}

func openRPCRelay(ctx context.Context, listenAddr, remoteURL, clusterDir string) (*rpcRelay, error) {
	return openRPCRelayForClient(ctx, listenAddr, remoteURL, clusterDir, "", "")
}

func openRPCRelayForClient(ctx context.Context, listenAddr, remoteURL, clusterDir, clientHost, peerID string) (*rpcRelay, error) {
	listener, err := net.Listen("tcp", listenAddr)
	if err != nil {
		return nil, err
	}
	relay := &rpcRelay{listener: listener, remoteURL: remoteURL, clusterDir: clusterDir, clientHost: clientHost, peerID: peerID}
	go func() {
		<-ctx.Done()
		_ = relay.Close()
	}()
	return relay, nil
}

func (r *rpcRelay) Addr() string {
	address := r.listener.Addr().String()
	if r.clientHost == "" {
		return address
	}
	_, port, err := net.SplitHostPort(address)
	if err != nil {
		return address
	}
	return net.JoinHostPort(r.clientHost, port)
}

func (r *rpcRelay) Close() error {
	return r.listener.Close()
}

func relayRPCConnection(ctx context.Context, local net.Conn, remoteURL, clusterDir, peerID string) {
	defer local.Close()
	mesh := clustertrust.Open(clusterDir)
	mesh.Refresh()
	config, ok := relayTLSConfig(mesh, peerID)
	if !ok {
		return
	}
	u, err := url.Parse(remoteURL)
	if err != nil || u.Host == "" || u.Scheme != "https" {
		log.Printf("RPC relay rejected remote URL peer=%s", peerID)
		return
	}
	log.Printf("RPC relay dialing peer=%s host=%s", peerID, u.Host)
	dialer := &tls.Dialer{NetDialer: &net.Dialer{Timeout: 5 * time.Second}, Config: config}
	remote, err := dialer.DialContext(ctx, "tcp", u.Host)
	if err != nil {
		log.Printf("RPC relay dial failed peer=%s: %v", peerID, err)
		return
	}
	defer remote.Close()
	path := u.EscapedPath()
	if path == "" || path == "/" {
		path = "/v1/fabric/rpc"
	}
	if _, err := fmt.Fprintf(remote, "CONNECT %s HTTP/1.1\r\nHost: %s\r\n\r\n", path, u.Host); err != nil {
		return
	}
	reader := bufio.NewReader(remote)
	response, err := http.ReadResponse(reader, nil)
	if err != nil {
		log.Printf("RPC relay response read failed peer=%s: %v", peerID, err)
		return
	}
	if response.StatusCode != http.StatusOK {
		log.Printf("RPC relay rejected by peer=%s status=%d", peerID, response.StatusCode)
		return
	}
	log.Printf("RPC relay connected peer=%s", peerID)
	bridgeConnectionsFromReader(local, reader, remote)
}

func relayTLSConfig(mesh *clustertrust.Mesh, peerID string) (*tls.Config, bool) {
	if peerID != "" {
		if config, ok := mesh.ClientTLSConfig(peerID); ok {
			return config, true
		}
		// Heartbeats historically carried a display node ID (for example,
		// "spark-b57c") while the trust store is keyed by the node UUID. The
		// Any path still pins the exact presented certificate to a current
		// cluster member, so it is safe and allows mixed-version workers to
		// participate while they upgrade their heartbeat schema.
		log.Printf("RPC relay peer ID is not a UUID peer=%s; resolving pinned certificate", peerID)
	}
	return mesh.ClientTLSConfigAny()
}

func bridgeConnections(a, b net.Conn) {
	bridgeConnectionsFromReader(a, bufio.NewReader(b), b)
}

func bridgeConnectionsFromReader(a net.Conn, reader *bufio.Reader, b net.Conn) {
	done := make(chan struct{}, 2)
	go func() { _, _ = io.Copy(a, reader); _ = a.Close(); _ = b.Close(); done <- struct{}{} }()
	go func() { _, _ = io.Copy(b, a); _ = a.Close(); _ = b.Close(); done <- struct{}{} }()
	<-done
}

func writeJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(value)
}

func handleMessage(codec *Codec, mgr *Manager, jobs *JobStore, executions *ExecutionManager, trainingCoordinator *TrainingCoordinator, logicalDevice *LogicalDeviceManager, cancel context.CancelFunc, msg *Message) {
	if !msg.IsRequest() {
		return
	}
	switch msg.Method {
	case "fabric:heartbeat":
		var heartbeat fabricwire.Heartbeat
		if err := json.Unmarshal(msg.Params, &heartbeat); err != nil {
			_ = codec.RespondError(msg.ID, -32602, "invalid heartbeat")
			return
		}
		if err := mgr.Heartbeat(heartbeat, time.Now()); err != nil {
			_ = codec.RespondError(msg.ID, -32602, err.Error())
			return
		}
		_ = codec.Respond(msg.ID, map[string]bool{"accepted": true})
	case "fabric:rejoin":
		var params struct {
			WorkerID string `json:"workerId"`
			Epoch    uint64 `json:"epoch"`
		}
		if err := json.Unmarshal(msg.Params, &params); err != nil {
			_ = codec.RespondError(msg.ID, -32602, "invalid rejoin")
			return
		}
		_ = codec.Respond(msg.ID, map[string]bool{"rejoined": mgr.Rejoin(params.WorkerID, params.Epoch, time.Now())})
	case "fabric:reconcile":
		mgr.Reconcile(time.Now())
		_ = codec.Respond(msg.ID, map[string]bool{"reconciled": true})
	case "fabric:plan-group":
		var request fabricwire.GroupRequest
		if err := json.Unmarshal(msg.Params, &request); err != nil {
			_ = codec.RespondError(msg.ID, -32602, "invalid group request")
			return
		}
		plan, err := mgr.PlanGroup(request)
		if err != nil {
			_ = codec.RespondError(msg.ID, -32001, err.Error())
			return
		}
		_ = codec.Respond(msg.ID, plan)
	case "fabric:build-execution-plan":
		var params struct {
			Request   fabricwire.GroupRequest  `json:"request"`
			Group     fabricwire.GroupPlan     `json:"group"`
			Strategy  fabricwire.ShardStrategy `json:"shardStrategy"`
			Transport fabricwire.Transport     `json:"transport"`
		}
		if err := json.Unmarshal(msg.Params, &params); err != nil {
			_ = codec.RespondError(msg.ID, -32602, "invalid execution plan request")
			return
		}
		plan, err := mgr.BuildExecutionPlan(params.Request, params.Group, params.Strategy, params.Transport)
		if err != nil {
			_ = codec.RespondError(msg.ID, -32001, err.Error())
			return
		}
		_ = codec.Respond(msg.ID, plan)
	case "fabric:logical-device-plan":
		if logicalDevice == nil {
			_ = codec.RespondError(msg.ID, -32001, "logical device manager unavailable")
			return
		}
		var request fabricwire.LogicalDevicePlanRequest
		if err := json.Unmarshal(msg.Params, &request); err != nil {
			_ = codec.RespondError(msg.ID, -32602, "invalid logical device plan request")
			return
		}
		plan, err := logicalDevice.Plan(request)
		if err != nil {
			_ = codec.RespondError(msg.ID, -32001, err.Error())
			return
		}
		_ = codec.Respond(msg.ID, plan)
	case "fabric:logical-device-describe":
		if logicalDevice == nil {
			_ = codec.RespondError(msg.ID, -32001, "logical device manager unavailable")
			return
		}
		_ = codec.Respond(msg.ID, logicalDevice.Describe())
	case "fabric:logical-device-status":
		if logicalDevice == nil {
			_ = codec.RespondError(msg.ID, -32001, "logical device manager unavailable")
			return
		}
		var params struct {
			PlanID string `json:"planId"`
		}
		if err := json.Unmarshal(msg.Params, &params); err != nil || params.PlanID == "" {
			_ = codec.RespondError(msg.ID, -32602, "invalid logical device status request")
			return
		}
		status, ok := logicalDevice.Status(params.PlanID)
		if !ok {
			_ = codec.RespondError(msg.ID, -32005, "logical device plan not found")
			return
		}
		_ = codec.Respond(msg.ID, status)
	case "fabric:logical-device-transfer":
		if logicalDevice == nil {
			_ = codec.RespondError(msg.ID, -32001, "logical device manager unavailable")
			return
		}
		var request fabricwire.TransferRequest
		if err := json.Unmarshal(msg.Params, &request); err != nil {
			_ = codec.RespondError(msg.ID, -32602, "invalid logical device transfer request")
			return
		}
		transfer, err := logicalDevice.Transfer(request)
		if err != nil {
			_ = codec.RespondError(msg.ID, -32001, err.Error())
			return
		}
		_ = codec.Respond(msg.ID, transfer)
	case "fabric:logical-device-recover":
		if logicalDevice == nil {
			_ = codec.RespondError(msg.ID, -32001, "logical device manager unavailable")
			return
		}
		var params struct {
			PlanID  string   `json:"planId"`
			Workers []string `json:"workers"`
		}
		if err := json.Unmarshal(msg.Params, &params); err != nil || params.PlanID == "" {
			_ = codec.RespondError(msg.ID, -32602, "invalid logical device recovery request")
			return
		}
		plan, err := logicalDevice.Recover(params.PlanID, params.Workers)
		if err != nil {
			_ = codec.RespondError(msg.ID, -32004, err.Error())
			return
		}
		_ = codec.Respond(msg.ID, plan)
	case "fabric:build-training-command":
		var params struct {
			Request  TrainingRequest `json:"request"`
			NodeRank uint32          `json:"nodeRank"`
		}
		if err := json.Unmarshal(msg.Params, &params); err != nil {
			_ = codec.RespondError(msg.ID, -32602, "invalid training command request")
			return
		}
		args, err := BuildTorchRunCommand(params.Request, params.NodeRank)
		if err != nil {
			_ = codec.RespondError(msg.ID, -32001, err.Error())
			return
		}
		_ = codec.Respond(msg.ID, map[string][]string{"args": args})
	case "fabric:training-start":
		var request TrainingRequest
		if err := json.Unmarshal(msg.Params, &request); err != nil {
			_ = codec.RespondError(msg.ID, -32602, "invalid distributed training request")
			return
		}
		if err := mgr.ValidateTrainingPlacement(request); err != nil {
			_ = codec.RespondError(msg.ID, -32004, err.Error())
			return
		}
		executions, err := trainingCoordinator.StartGroup(context.Background(), request)
		if err != nil {
			_ = codec.RespondError(msg.ID, -32004, err.Error())
			return
		}
		_ = codec.Respond(msg.ID, executions)
	case "fabric:training-status":
		var params struct {
			JobID string `json:"jobId"`
		}
		if err := json.Unmarshal(msg.Params, &params); err != nil {
			_ = codec.RespondError(msg.ID, -32602, "invalid distributed training status request")
			return
		}
		_ = trainingCoordinator.RefreshStatus(context.Background(), params.JobID)
		status, ok := trainingCoordinator.StatusGroup(params.JobID)
		if !ok {
			_ = codec.RespondError(msg.ID, -32005, "training group not found")
			return
		}
		_ = codec.Respond(msg.ID, status)
	case "fabric:training-checkpoint":
		var checkpoint fabricwire.Checkpoint
		if err := json.Unmarshal(msg.Params, &checkpoint); err != nil {
			_ = codec.RespondError(msg.ID, -32602, "invalid distributed training checkpoint")
			return
		}
		if err := trainingCoordinator.SaveCheckpoint(checkpoint.JobID, checkpoint); err != nil {
			_ = codec.RespondError(msg.ID, -32003, err.Error())
			return
		}
		_ = codec.Respond(msg.ID, checkpoint)
	case "fabric:training-recover":
		var params struct {
			JobID      string         `json:"jobId"`
			Nodes      []TrainingNode `json:"nodes"`
			Rendezvous string         `json:"rendezvousEndpoint"`
		}
		if err := json.Unmarshal(msg.Params, &params); err != nil {
			_ = codec.RespondError(msg.ID, -32602, "invalid distributed training recovery")
			return
		}
		executions, err := trainingCoordinator.RecoverGroup(context.Background(), params.JobID, params.Nodes, params.Rendezvous)
		if err != nil {
			_ = codec.RespondError(msg.ID, -32004, err.Error())
			return
		}
		_ = codec.Respond(msg.ID, executions)
	case "fabric:training-stop":
		var params struct {
			JobID string `json:"jobId"`
		}
		if err := json.Unmarshal(msg.Params, &params); err != nil {
			_ = codec.RespondError(msg.ID, -32602, "invalid distributed training stop request")
			return
		}
		if err := trainingCoordinator.StopGroup(context.Background(), params.JobID); err != nil {
			_ = codec.RespondError(msg.ID, -32005, err.Error())
			return
		}
		_ = codec.Respond(msg.ID, map[string]bool{"stopped": true})
	case "fabric:job-start":
		var request StartRequest
		if err := json.Unmarshal(msg.Params, &request); err != nil {
			_ = codec.RespondError(msg.ID, -32602, "invalid job start")
			return
		}
		execution, err := startInferenceJob(mgr, jobs, executions, request)
		if err != nil {
			_ = codec.RespondError(msg.ID, -32004, err.Error())
			return
		}
		_ = codec.Respond(msg.ID, execution)
	case "fabric:job-stop":
		var params struct {
			JobID string `json:"jobId"`
		}
		if err := json.Unmarshal(msg.Params, &params); err != nil || executions.Stop(params.JobID) != nil {
			_ = codec.RespondError(msg.ID, -32005, "job is not running")
			return
		}
		_ = codec.Respond(msg.ID, map[string]bool{"stopped": true})
	case "fabric:job-status":
		var params struct {
			JobID string `json:"jobId"`
		}
		if err := json.Unmarshal(msg.Params, &params); err != nil {
			_ = codec.RespondError(msg.ID, -32602, "invalid job status")
			return
		}
		status, ok := executions.Status(params.JobID)
		if !ok {
			_ = codec.RespondError(msg.ID, -32006, "job not found")
			return
		}
		_ = codec.Respond(msg.ID, status)
	case "fabric:get-status":
		var logicalDeviceStatus any
		if logicalDevice != nil {
			logicalDeviceStatus = logicalDevice.Describe()
		}
		_ = codec.Respond(msg.ID, map[string]any{"workers": mgr.Workers(), "capacity": mgr.Capacity(), "jobs": jobs.Jobs(), "executions": executions.Statuses(), "logicalDevice": logicalDeviceStatus})
	case "fabric:job-submit":
		var job JobRecord
		if err := json.Unmarshal(msg.Params, &job); err != nil || jobs.Submit(job) != nil {
			_ = codec.RespondError(msg.ID, -32602, "invalid or duplicate job")
			return
		}
		_ = codec.Respond(msg.ID, job)
	case "fabric:checkpoint":
		var checkpoint fabricwire.Checkpoint
		if err := json.Unmarshal(msg.Params, &checkpoint); err != nil {
			_ = codec.RespondError(msg.ID, -32602, "invalid checkpoint")
			return
		}
		if checkpoint.Filename != "" {
			if err := executions.SaveCheckpoint(checkpoint); err != nil {
				_ = codec.RespondError(msg.ID, -32602, err.Error())
				return
			}
		}
		if err := jobs.SaveCheckpoint(checkpoint); err != nil {
			_ = codec.RespondError(msg.ID, -32602, "invalid checkpoint")
			return
		}
		_ = codec.Respond(msg.ID, checkpoint)
	case "fabric:recover":
		var params struct {
			JobID   string `json:"jobId"`
			GroupID string `json:"groupId"`
			Epoch   uint64 `json:"epoch"`
		}
		if err := json.Unmarshal(msg.Params, &params); err != nil {
			_ = codec.RespondError(msg.ID, -32602, "invalid recovery")
			return
		}
		checkpoint, err := jobs.Recover(params.JobID, params.GroupID, params.Epoch)
		if err != nil {
			_ = codec.RespondError(msg.ID, -32002, err.Error())
			return
		}
		_ = codec.Respond(msg.ID, checkpoint)
	case "shutdown":
		_ = codec.Respond(msg.ID, nil)
		cancel()
	default:
		_ = codec.RespondError(msg.ID, -32601, "method not found")
	}
}
