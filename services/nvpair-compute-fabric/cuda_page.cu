// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

int failCuda(const char* operation, cudaError_t error) {
    std::cerr << operation << ": " << cudaGetErrorString(error) << '\n';
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        return fail("usage: pair-cuda-page <input> <output> <device-index>");
    }
    const std::string inputPath = argv[1];
    const std::string outputPath = argv[2];
    char* end = nullptr;
    const unsigned long deviceValue = std::strtoul(argv[3], &end, 10);
    if (end == argv[3] || *end != '\0' || deviceValue > std::numeric_limits<int>::max()) {
        return fail("device-index must be a non-negative integer");
    }
    if (const cudaError_t error = cudaSetDevice(static_cast<int>(deviceValue)); error != cudaSuccess) {
        return failCuda("cudaSetDevice", error);
    }

    std::ifstream input(inputPath, std::ios::binary | std::ios::ate);
    if (!input) {
        return fail("cannot open input page");
    }
    const std::streamsize size = input.tellg();
    if (size < 0) {
        return fail("cannot determine input page size");
    }
    input.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> host(static_cast<std::size_t>(size));
    if (size > 0 && !input.read(reinterpret_cast<char*>(host.data()), size)) {
        return fail("cannot read input page");
    }

    void* device = nullptr;
    if (const cudaError_t error = cudaMalloc(&device, host.size()); error != cudaSuccess) {
        return failCuda("cudaMalloc", error);
    }
    const auto cleanup = [&]() { cudaFree(device); };
    if (!host.empty()) {
        if (const cudaError_t error = cudaMemcpy(device, host.data(), host.size(), cudaMemcpyHostToDevice); error != cudaSuccess) {
            cleanup();
            return failCuda("cudaMemcpy host-to-device", error);
        }
        if (const cudaError_t error = cudaMemcpy(host.data(), device, host.size(), cudaMemcpyDeviceToHost); error != cudaSuccess) {
            cleanup();
            return failCuda("cudaMemcpy device-to-host", error);
        }
    }
    if (const cudaError_t error = cudaDeviceSynchronize(); error != cudaSuccess) {
        cleanup();
        return failCuda("cudaDeviceSynchronize", error);
    }
    cleanup();

    std::ofstream output(outputPath, std::ios::binary | std::ios::trunc);
    if (!output) {
        return fail("cannot open output page");
    }
    if (!host.empty()) {
        output.write(reinterpret_cast<const char*>(host.data()), static_cast<std::streamsize>(host.size()));
    }
    return output ? 0 : fail("cannot write output page");
}
