// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"

	"nvpair-shared/fabricwire"
)

// startInferenceJob is the single coordinator-side inference admission and
// launch path. The broker JSON-RPC surface and the authenticated HTTP surface
// both call this function so they cannot drift in planning, persistence, or
// execution behavior.
func startInferenceJob(mgr *Manager, jobs *JobStore, executions *ExecutionManager, request StartRequest) (Execution, error) {
	// Inference workers receive the model through the execution plan. They do
	// not need to keep a second local model copy just to join the group. Keep
	// the digest on the execution plan for identity and safety, but do not use
	// it as a local-cache admission filter here. Training keeps exact digest
	// matching because every rank must load the same local training model.
	groupRequest := request.Group
	groupRequest.ModelDigest = ""
	group, err := mgr.PlanGroup(groupRequest)
	if err != nil {
		return Execution{}, err
	}
	planningRequest := request.Group
	planningRequest.ModelDigest = request.ModelDigest
	executionPlan, err := mgr.BuildExecutionPlan(planningRequest, group, fabricwire.ShardTensor, fabricwire.TransportMTLSRPC)
	if err != nil {
		return Execution{}, err
	}
	job := JobRecord{JobID: request.JobID, GroupID: group.GroupID, ModelDigest: request.ModelDigest, Epoch: group.Epoch}
	if err := jobs.Submit(job); err != nil {
		return Execution{}, fmt.Errorf("persist inference job: %w", err)
	}
	execution, err := executions.StartWithExecutionPlan(request, executionPlan)
	if err != nil {
		return Execution{}, err
	}
	return execution, nil
}
