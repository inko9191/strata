#include <metal_stdlib>
using namespace metal;

// Read each uint4 once per dispatch. Only one 32-bit checksum per group is written.
kernel void read_bandwidth(device const uint4* input [[buffer(0)]],
                           device uint* output [[buffer(1)]],
                           constant uint& vectorCount [[buffer(2)]],
                           uint tid [[thread_position_in_grid]],
                           uint lane [[thread_index_in_threadgroup]],
                           uint group [[threadgroup_position_in_grid]],
                           uint grid [[threads_per_grid]]) {
    uint sum = 0;
    for (uint index = tid; index < vectorCount; index += grid) {
        uint4 value = input[index];
        sum += value.x + value.y + value.z + value.w;
    }
    threadgroup uint partials[256];
    partials[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride /= 2) {
        if (lane < stride) partials[lane] += partials[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) output[group] = partials[0];
}
