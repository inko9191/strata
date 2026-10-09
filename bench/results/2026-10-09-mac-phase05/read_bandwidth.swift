import Foundation
import Metal

let bufferBytes = 8 * 1024 * 1024 * 1024
let trials = 7
let passes = 8
let lanes = 256
let readsPerThread = 16
let vectorCount = bufferBytes / 16
let threads = vectorCount / readsPerThread
let groups = threads / lanes

guard CommandLine.arguments.count == 2 else { fatalError("Pass the .metal source path") }
guard let device = MTLCreateSystemDefaultDevice(), device.hasUnifiedMemory,
      let queue = device.makeCommandQueue() else { fatalError("Unified Metal device unavailable") }
let source = try String(contentsOfFile: CommandLine.arguments[1], encoding: .utf8)
let library = try device.makeLibrary(source: source, options: nil)
guard let function = library.makeFunction(name: "read_bandwidth") else { fatalError("Kernel unavailable") }
let pipeline = try device.makeComputePipelineState(function: function)
guard pipeline.maxTotalThreadsPerThreadgroup >= lanes,
      let input = device.makeBuffer(length: bufferBytes, options: .storageModeShared),
      let output = device.makeBuffer(length: groups * 4, options: .storageModeShared) else {
    fatalError("Buffer or threadgroup allocation failed")
}
// CPU initialization is outside GPU timing. Touch every page before the warm-up.
let words = input.contents().bindMemory(to: UInt32.self, capacity: bufferBytes / 4)
for index in 0..<(bufferBytes / 4) { words[index] = UInt32(index) }
func run() throws -> Double {
    guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
        fatalError("Command creation failed")
    }
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(input, offset: 0, index: 0)
    encoder.setBuffer(output, offset: 0, index: 1)
    var count = UInt32(vectorCount)
    encoder.setBytes(&count, length: 4, index: 2)
    for _ in 0..<passes {
        encoder.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: lanes, height: 1, depth: 1))
        encoder.memoryBarrier(scope: .buffers)
    }
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    if command.status != .completed { throw command.error ?? NSError(domain: "Metal", code: 1) }
    let checksums = output.contents().bindMemory(to: UInt32.self, capacity: groups)
    for i in 0..<groups {
        let base = UInt64(i * lanes)
        let laneSum = UInt64(lanes) * base + UInt64(lanes * (lanes - 1) / 2)
        let iterationSum = UInt64(readsPerThread * (readsPerThread - 1) / 2)
        let expected = UInt32(truncatingIfNeeded:
            16 * UInt64(readsPerThread) * laneSum + UInt64(lanes) *
            (16 * UInt64(threads) * iterationSum + UInt64(6 * readsPerThread)))
        if checksums[i] != expected { fatalError("Checksum mismatch at group \(i)") }
    }
    let elapsed = command.gpuEndTime - command.gpuStartTime
    if elapsed <= 0 { fatalError("GPU timestamps unavailable") }
    return elapsed
}
let before = ProcessInfo.processInfo.thermalState.rawValue
let warmup = try run()
var measurements: [[String: Any]] = []
for trial in 1...trials {
    let seconds = try autoreleasepool { try run() }
    let bytes = Double(bufferBytes) * Double(passes)
    measurements.append(["trial": trial, "gpu_seconds": seconds,
                         "logical_read_bytes": UInt64(bufferBytes) * UInt64(passes),
                         "logical_write_bytes": groups * 4 * passes,
                         "GB_per_s": bytes / seconds / 1e9,
                         "checksum_pass": true,
                         "thermal_state": ProcessInfo.processInfo.thermalState.rawValue])
}
let values = measurements.map { $0["GB_per_s"] as! Double }.sorted()
let report: [String: Any] = [
    "date_utc": ISO8601DateFormatter().string(from: Date()), "device": device.name,
    "os": ProcessInfo.processInfo.operatingSystemVersionString,
    "physical_memory_bytes": ProcessInfo.processInfo.physicalMemory,
    "recommendedMaxWorkingSetSize": device.recommendedMaxWorkingSetSize,
    "maxBufferLength": device.maxBufferLength, "hasUnifiedMemory": device.hasUnifiedMemory,
    "storage_mode": "shared", "buffer_bytes": bufferBytes, "passes_per_trial": passes,
    "initialization": "Each uint32 input word contains its own index; all group sums verified analytically modulo 2^32.",
    "threadgroup_size": lanes, "thread_execution_width": pipeline.threadExecutionWidth,
    "thermal_state_before": before, "thermal_state_after": ProcessInfo.processInfo.thermalState.rawValue,
    "warmup_gpu_seconds": warmup, "trials": measurements,
    "median_GB_per_s": values[values.count / 2], "min_GB_per_s": values.first!, "max_GB_per_s": values.last!,
    "timer": "MTLCommandBuffer.gpuEndTime - gpuStartTime; command with 8 dispatches and buffer barriers",
    "interpretation": "Effective logical large-buffer GPU read throughput; not a hardware DRAM counter or Strata throughput."
]
let json = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
FileHandle.standardOutput.write(json)
FileHandle.standardOutput.write(Data([10]))
