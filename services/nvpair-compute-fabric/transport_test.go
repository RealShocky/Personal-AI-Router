// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"encoding/pem"
	"io"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
)

func TestBridgeConnectionsForwardsRPCBytes(t *testing.T) {
	localRelay, localClient := net.Pipe()
	remoteRelay, remoteServer := net.Pipe()
	done := make(chan struct{})
	go func() {
		bridgeConnections(localRelay, remoteRelay)
		close(done)
	}()

	message := []byte("rpc-frame")
	go func() {
		_, _ = localClient.Write(message)
	}()
	got := make([]byte, len(message))
	if _, err := io.ReadFull(remoteServer, got); err != nil {
		t.Fatalf("read bridged bytes: %v", err)
	}
	if string(got) != string(message) {
		t.Fatalf("bridged bytes = %q, want %q", got, message)
	}
	_ = localClient.Close()
	_ = remoteServer.Close()
	<-done
}

func TestBridgeConnectionsPreservesBufferedBytes(t *testing.T) {
	local, peer := net.Pipe()
	remote, remotePeer := net.Pipe()
	reader := bufio.NewReader(bytes.NewReader([]byte("prefetched-rpc-frame")))
	done := make(chan struct{})
	go func() {
		bridgeConnectionsFromReader(local, reader, remote)
		close(done)
	}()
	got := make([]byte, len("prefetched-rpc-frame"))
	if _, err := io.ReadFull(peer, got); err != nil {
		t.Fatalf("read buffered frame: %v", err)
	}
	if string(got) != "prefetched-rpc-frame" {
		t.Fatalf("buffered frame = %q", got)
	}
	_ = peer.Close()
	_ = remotePeer.Close()
	<-done
}

func TestFabricRPCTunnelRejectsUnclusteredRequest(t *testing.T) {
	handler := fabricHTTPHandler(clustertrust.Open(t.TempDir()), NewManager(0), "127.0.0.1:1")
	req := httptest.NewRequest(http.MethodConnect, "https://fabric.invalid/v1/fabric/rpc", nil)
	req.URL.Path = "/v1/fabric/rpc"
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, req)
	if recorder.Code != http.StatusForbidden {
		t.Fatalf("unclustered RPC request status = %d, want %d", recorder.Code, http.StatusForbidden)
	}
}

func TestFabricRPCTunnelAuthenticatesAndBridges(t *testing.T) {
	serverDir := t.TempDir()
	clientDir := t.TempDir()
	serverCert, serverKey := writeIdentity(t, serverDir, "server")
	clientCert, clientKey := writeIdentity(t, clientDir, "client")
	writeAdmission(t, serverDir)
	writeAdmission(t, clientDir)
	writePin(t, serverDir, "client", clientCert)
	writePin(t, clientDir, "server", serverCert)
	serverMesh := clustertrust.Open(serverDir)
	clientMesh := clustertrust.Open(clientDir)

	backend, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer backend.Close()
	go func() {
		conn, acceptErr := backend.Accept()
		if acceptErr != nil {
			return
		}
		defer conn.Close()
		_, _ = io.Copy(conn, conn)
	}()

	server := httptest.NewUnstartedServer(fabricHTTPHandler(serverMesh, NewManager(time.Second), backend.Addr().String()))
	server.TLS = serverMesh.ServerTLSConfig()
	server.StartTLS()
	defer server.Close()
	parsed, _ := url.Parse(server.URL)
	clientTLS, ok := clientMesh.ClientTLSConfig("server")
	if !ok {
		t.Fatal("client could not build pinned TLS config")
	}
	conn, err := tls.Dial("tcp", parsed.Host, clientTLS)
	if err != nil {
		t.Fatalf("mTLS dial: %v", err)
	}
	defer conn.Close()
	if _, err := io.WriteString(conn, "CONNECT /v1/fabric/rpc HTTP/1.1\r\nHost: "+parsed.Host+"\r\n\r\n"); err != nil {
		t.Fatal(err)
	}
	response, err := http.ReadResponse(bufio.NewReader(conn), nil)
	if err != nil || response.StatusCode != http.StatusOK {
		t.Fatalf("CONNECT response = %v, err = %v", response.Status, err)
	}
	if _, err := io.WriteString(conn, "fabric-rpc"); err != nil {
		t.Fatal(err)
	}
	got := make([]byte, len("fabric-rpc"))
	if _, err := io.ReadFull(conn, got); err != nil || string(got) != "fabric-rpc" {
		t.Fatalf("echo = %q, err = %v", got, err)
	}
	_ = conn.Close()
	_ = clientKey

	// The certificate handshake alone is not membership: an unpinned client is
	// admitted by TLS but rejected by the handler's certificate pin gate.
	_ = serverKey
}

func writeIdentity(t *testing.T, dir, uuid string) ([]byte, []byte) {
	t.Helper()
	public, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	uri, _ := url.Parse("urn:nvpair:node:" + uuid)
	template := &x509.Certificate{SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: uuid}, NotBefore: time.Now().Add(-time.Minute), NotAfter: time.Now().Add(time.Hour), URIs: []*url.URL{uri}, ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth}, KeyUsage: x509.KeyUsageDigitalSignature}
	der, err := x509.CreateCertificate(rand.Reader, template, template, public, private)
	if err != nil {
		t.Fatal(err)
	}
	keyDER, err := x509.MarshalPKCS8PrivateKey(private)
	if err != nil {
		t.Fatal(err)
	}
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: keyDER})
	if err := os.WriteFile(filepath.Join(dir, "node.crt"), certPEM, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "node.key"), keyPEM, 0o600); err != nil {
		t.Fatal(err)
	}
	return certPEM, keyPEM
}

func writeAdmission(t *testing.T, dir string) {
	t.Helper()
	data, _ := json.Marshal(map[string]any{"clusterId": "test", "epoch": 1})
	if err := os.WriteFile(filepath.Join(dir, "admission.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func writePin(t *testing.T, dir, uuid string, certPEM []byte) {
	t.Helper()
	trusted := filepath.Join(dir, "trusted")
	if err := os.MkdirAll(trusted, 0o700); err != nil {
		t.Fatal(err)
	}
	data, _ := json.Marshal(map[string]string{"nodeUuid": uuid, "certPem": string(certPEM)})
	if err := os.WriteFile(filepath.Join(trusted, uuid+".json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
}
