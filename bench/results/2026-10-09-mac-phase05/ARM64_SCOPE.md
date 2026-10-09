# arm64 / Darwin build scope — no engine changes

Inspected at Strata `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` on 2026-10-09.
This is a source inspection and scope estimate, not a tested patch.

## CPU expert library

| File / boundary | Current x86 dependency | Necessary for the initial Mac IQ pack |
| --- | --- | --- |
| `CMakeLists.txt:1082–1123` | Always adds `expert.cpp`, `q2_avx2.cpp`; AVX-512/FMA/F16C file flags; AVX-VNNI compile probe | Choose sources and probes by target architecture. Do not send x86 flags to Apple arm64. Keep existing x86 selections unchanged. |
| `CMakeLists.txt:1186–1218` | Adds `iq_avx512.cpp`, `iq_avx2.cpp`, `kq_avx2.cpp`, `kq_avx1.cpp` and ISA flags | Omit these translation units on arm64, while retaining `native_expert.cpp` and pinned ggml-cpu. A successful ggml-cpu build already establishes the NEON build path, not expert runtime parity. |
| `expert.cpp` | AVX-512 VNNI/VBMI decode, AVX/SSE helpers, FP16 conversion, Q8 activation quantization | Do not port the whole S2 fast path just to run IQ. Separate any required portable helpers and explicitly reject unsupported non-native production modes. Legacy x86-only tests need a reported skip; portable references still need to run. |
| `q2_avx2.cpp`, `q2_avx2_rows.inl` | `_mm256_*`, `_mm_*`, FMA/F16C, bitplane/VNNI alternatives | Not the initial native IQ reference path. Remove arm64 link references to these functions; a runtime false ISA predicate alone is not a reliable solution in debug builds. |
| `iq_avx512.cpp`, `iq_avx2.cpp`, `iq_avx2_rows.inl` | AVX-512 or AVX2 gathers, packed dot products, codebooks | Existing ggml CPU traits are the first native IQ implementation on arm64. New NEON fast paths are not needed before measuring. |
| `kq_avx2.cpp`, `kq_avx1.cpp` | AVX / AVX2 K-quant / router lookahead | Guard call sites and source selection. Do not route arm64 through the AVX1 alternative. |
| `expert_layout.cpp` | `<cpuid.h>` / `<intrin.h>`, CPUID leaves, `xgetbv`, CPU brand / hybrid-core detection, AVX-only support refusal | Add explicit arm64 capabilities and native-ggml support. AVX predicates must be false; x86 ISA overrides must not bypass the check. Guard `q2_rows_any` and `act_quant_any` and their AVX symbol references. |
| `native_expert.cpp` | Calls `q8k_quant_avx2`, `iq512_gu_rows`, `iq256_gu_rows`, `kq256_*`, `iq4nl256_down_rows` behind runtime flags | Preserve ggml `from_float` / `vec_dot` fallback as the explicit CPU reference. Compile out AVX references on arm64; validate IQ2_S, IQ2_XXS, IQ1_M and Q2_0 against independent dequantized-weight tests. |
| `pool.cpp` | Unconditional `<immintrin.h>`, `_mm_pause`; non-Windows path calls Linux affinity and topology helpers | Provide a portable waiting policy and a Darwin topology/affinity policy. Darwin affinity tags are not Linux hard pinning to numbered CPUs. Report unsupported pinning rather than claiming it succeeded. |
| `pool_affinity_linux.hpp`, pool topology / affinity tests | `CPU_ALLOC`, `CPU_SET_*`, `pthread_*affinity_np`, Linux `/sys` topology; helper definitions exist only under `__linux__` | Add a Darwin boundary; preserve Linux tests. Identify architecture-specific skips separately from portable pool tests. |

`arm64-intrinsics.json` inventories intrinsic symbol occurrences, including comments.
It is a review aid, not an estimate of changed lines or elapsed engineering time.
The initial target is the native GGUF expert path, with no replacement of all AVX kernels by NEON.
That still requires source selection, call-site guards, feature detection, pool portability and tests together;
deleting compiler flags alone will fail on headers, assembly or unresolved symbols.

## Engine host code beyond strata_kernels_cpu

| File / group | Scope |
| --- | --- |
| `src/core/graph.cpp` | Unconditional `<immintrin.h>` / `_mm_pause` in bounded waits. Abstract CPU wait behavior and preserve timeout semantics. |
| `src/core/verify.cpp` | `_mm_pause` and `_mm_sfence` around mapped GPU doorbells / event waits. Separate host memory ordering from GPU visibility and submission completion; a compile-only substitution is not correctness evidence. |
| `src/core/session.cpp` | Pause include is already guarded, but `_mm_sfence` remains elsewhere. Preserve existing x86 path and implement the selected backend's visibility contract. |
| `src/core/mtp.cpp` | Include guard excludes arm64, while wait sites still use `_mm_pause`. Needs the same portable wait boundary. |
| `src/platform/direct_file.cpp` | The non-Windows path assumes `O_DIRECT`. Darwin needs its own implementation and documented cache semantics. `F_NOCACHE` is not proof of physical SSD I/O. |
| `src/platform/memory.cpp`, `expert_source.cpp`, `expert_cache.cpp`, `conversation_memory.cpp` | Linux memory queries, `/proc`, and UMA capacity detection need Darwin APIs. Do not double-count shared RAM as independent RAM and VRAM. |
| GPU CMake and graph/device headers | GB10 can keep CUDA. Mac requires a separate Metal/runtime boundary and GPU-free host targets. This is additional work beyond CPU ISA portability. |

## Comparison with #1306

[Issue #1306](https://github.com/Niko1221/Strata/issues/1306) was open when inspected.
Its proposal is to guard x86 includes / intrinsic calls and provide an ARM alternative.
No complete fix linked from that issue was available to validate on this Mac.

Common work: ISA-aware CMake, guards for x86 instructions, CPU capability detection, and choosing ggml's
arm64 native expert reference. Mac-only work: Darwin affinity/topology, file I/O, memory accounting,
framework/dylib handling, and replacing CUDA runtime ownership with Metal.
Therefore **the same x86-guard change alone is insufficient**. GB10's Linux/CUDA environment does not
exercise the additional Darwin/Metal boundaries.

Verification after a future patch must include native expert / activation CPU parity, pool tests,
portable host tests, explicit x86-only skips, and CUDA/HIP/SYCL builds of shared-file changes.
Neither a NEON expert runtime nor those cross-backend builds were verified in Phase 0.5.
No arm64 engine fix was written here.
