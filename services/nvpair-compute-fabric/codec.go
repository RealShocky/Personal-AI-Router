// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import "nvpair-shared/jsonrpc"

type (
	Message = jsonrpc.Message
	Codec   = jsonrpc.Codec
)

var NewCodec = jsonrpc.NewCodec
