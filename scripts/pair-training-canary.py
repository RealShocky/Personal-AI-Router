#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tiny real torch.distributed trainer used to validate the PAIR adapter."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parallelism", choices=("data", "fsdp"), required=True)
    parser.add_argument("--checkpoint-interval-steps", type=int, required=True)
    parser.add_argument("--model-digest", required=True)
    parser.add_argument("--resume")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    if world_size > 1:
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = torch.nn.Linear(4, 2).to(device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if use_cuda else None)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.module.load_state_dict(checkpoint["model"] if hasattr(model, "module") else checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])

    for step in range(start_step + 1, start_step + 3):
        inputs = torch.ones((4, 4), device=device) * (rank + 1)
        target = torch.zeros((4, 2), device=device)
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(inputs), target)
        loss.backward()
        optimizer.step()
        if step % args.checkpoint_interval_steps == 0 or step == start_step + 2:
            raw_model = model.module if hasattr(model, "module") else model
            checkpoint_path = output_dir / f"pair-canary-step-{step}.pt"
            torch.save({"step": step, "model": raw_model.state_dict(), "optimizer": optimizer.state_dict(), "model_digest": args.model_digest}, checkpoint_path)
            if rank == 0:
                manifest = {"step": step, "world_size": world_size, "checkpoint": checkpoint_path.name, "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()}
                (output_dir / "pair-canary-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    print(json.dumps({"rank": rank, "worldSize": world_size, "device": str(device), "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()
