// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import Metal

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

guard CommandLine.arguments.count == 4 else {
    fail("usage: pair-metal-page <input> <output> <device-index>")
}

let inputURL = URL(fileURLWithPath: CommandLine.arguments[1])
let outputURL = URL(fileURLWithPath: CommandLine.arguments[2])
guard let deviceIndex = Int(CommandLine.arguments[3]), deviceIndex >= 0 else {
    fail("device index must be a non-negative integer")
}
let devices = MTLCopyAllDevices()
guard deviceIndex < devices.count else {
    fail("requested Metal device is unavailable")
}
let device = devices[deviceIndex]
guard let queue = device.makeCommandQueue(), let commandBuffer = queue.makeCommandBuffer() else {
    fail("Metal command queue is unavailable")
}

do {
    let input = try Data(contentsOf: inputURL)
    guard let source = device.makeBuffer(bytes: [UInt8](input), length: input.count, options: .storageModeShared),
          let destination = device.makeBuffer(length: input.count, options: .storageModeShared) else {
        fail("Metal buffer allocation failed")
    }
    guard let blit = commandBuffer.makeBlitCommandEncoder() else {
        fail("Metal blit encoder is unavailable")
    }
    blit.copy(from: source, sourceOffset: 0, to: destination, destinationOffset: 0, size: input.count)
    blit.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    guard commandBuffer.status == .completed else {
        fail("Metal command failed")
    }
    let output = Data(bytes: destination.contents(), count: input.count)
    try output.write(to: outputURL, options: .atomic)
} catch {
    fail("Metal page staging failed: \(error)")
}
