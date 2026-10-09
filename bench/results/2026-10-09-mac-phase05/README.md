# Phase 0.5: actual IQ2_XS files and pinned llama.cpp on M5 Max

Measured 2026-10-09 on Apple M5 Max, 64 GiB unified memory, macOS 27.0.1 (26A434),
AppleClang 21.0.0.21000099. Strata inspection commit: `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`.
**These inference measurements are unmodified llama.cpp, not Strata.** No Strata engine or llama.cpp source
was changed. The added programs inspect files, capture the public llama API, compare captures, or measure
an independent Metal read kernel. No arm64 engine fix was implemented; Phase 1 was not started.

## Actual files

Source: [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ2_XS),
the revision pinned in Strata's `setup.py`: `ed59f92082b1e93c0e96d60a8b11aab089b52f09`.
Both complete local files passed SHA256 and byte-length checks against the revision's LFS records.

| File | Bytes | Computed SHA256 |
| --- | ---: | --- |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf` | 39,225,954,592 | `92cee27ae5bbadcd732416a0f7a7f0acc092399dbbe8f5a5efa707c2ec0a49d7` |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf` | 28,800,138,432 | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |

Local model location: repository-relative `build-mac-phase05/models/`. Files remain there and are gitignored.
[Revision metadata](huggingface-revision.json), [verification / commands / raw measurements](reference-results.json),
[headers](gguf-headers.json), [tensor CSV](tensors.csv), [tensor JSON](tensors.json),
[manifest comparison](manifest-check.json).

| Header field | Shard 1 | Shard 2 |
| --- | --- | --- |
| Magic / version / endian | GGUF / 3 / little | GGUF / 3 / little |
| Metadata entries / tensors | 73 / 1,223 | 3 / 1 |
| Alignment / data start | 32 / 11,024,672 | 32 / 192 |
| `general.architecture` | **qwen4exp** | Absent; split-shard metadata only |
| Split number / count | 0 / 2 | 1 / 2 |
| Total tensors across splits | 1,224 | 1,224 |

Actual metadata: 48 layers, 512 experts, top-10, shared expert width 640, expert width 640,
Q heads 24 / KV heads 2, trained context 262,144. Shard 2 contains only
`per_layer_token_embd.weight`, IQ4_NL, 28,800,138,240 tensor bytes.
Raw headers are saved locally as `*.gguf.header.bin`; metadata arrays in the JSON have counts,
previews and payload hashes, with the complete arrays retained in the raw headers.

All 48 manifest rows match `tests/data/native_experts/iq2_xs.txt`: gate/up/down type, shape,
absolute offset, expert bytes, and cumulative packed offset. Expert bytes are 35,454,976,000.
Gate/up: IQ2_S in 34 layers, IQ2_XXS in 11, IQ1_M in 3; down: Q2_0 in all 48.
Other tensors also include IQ4_XS, BF16, F32, IQ3_S, IQ4_NL, Q6_K, F16 and Q8_0.
The fixed Strata header tool independently reported zero geometry-bracket errors and zero unknown types.
Its architecture guard passes shard 1; it reports missing architecture on shard 2 when inspected alone.
That is recorded, not filled with an invented value; the split model loads as a pair in llama.cpp.

## Pinned llama.cpp build and workload

Source commit: `3cf03257f219afbe7334045ff7c6a06ac68c627d` from Strata's `third_party/ggml/VERSION.txt`.
Codeload archive SHA256: `c076d7534afa0e5d0ec2a0d425b11e791c16f3de0d727221aea071cef156a280`.
Both CLI binaries report this commit, version 0.4.1-dev / build 350, Darwin arm64 / AppleClang.
[CPU build options](cpu-build-options.json), [Metal build options](metal-build-options.json).
CPU: GGML_METAL=OFF; Metal: GGML_METAL=ON / embedded library; both GGML_BLAS=OFF, GGML_NATIVE=ON.
The original source archive is unchanged.

[Frozen prompts and settings](prompts.json): English sky explanation, Japanese morning-walk benefits,
and a Python max function. The performance prompt is the English one, **12 input tokens / 32 generated
tokens**; the Japanese and code prompts are correctness captures. Context limit 512, batch/ubatch 64,
8 CPU threads for generation and prompt processing, greedy temperature 0, seed 1234, repeat penalty 1,
Flash Attention off, MTP off, fit off, no explicit warm-up.

An explicit identity Jinja template ([file](raw-template.jinja)) emits the raw prompt without role markers
or a thinking prefix. This avoids template differences between probes; it is a diagnostic completion
workload, not a default chat-template quality evaluation. CPU and Metal use identical input token IDs.
CPU uses `-ngl 0`, Metal `-ngl 99`; both use mmap / lazy mode and
`-ot per_layer_token_embd.weight=CPU`, keeping the lookup table on the CPU/SSD side.

`llama-cli` was run as CPU1 / Metal1 / CPU2 / Metal2 / CPU3 / Metal3, each in a fresh process.
The original CLI does not export every generation logit. The additional [capture program](capture_logits.cpp)
links the same pinned library and saves the exact input IDs, 32 greedy picks, and all pre-pick logits.
Those capture timings include logit I/O and are **not** used for throughput.
The model and library were not patched to implement this capture.

The successful Metal capture reports 49/49 layers offloaded, an MTL0 mapped model buffer of
37,398.27 MiB, and CPU mappings for embedding / host tensors and the 27,465.95 MiB PLE table.
It uses shared buffers. Offloaded layers do not mean every operation runs on the GPU: the PLE override
and CPU embedding remain explicit host work. These are startup allocation sizes, not measured GPU peaks.

## Tokens and logits

| Prompt | Generated IDs | CPU vs Metal | First different pick | Largest absolute logit difference | Largest relative L2 |
| --- | ---: | --- | --- | ---: | ---: |
| English | 32 each | **Different** | 22nd token: CPU 1220, Metal 1144 | 0.870231 | 0.0727436 |
| Japanese | 32 each | Identical | None | 0.964064 | 0.0747462 |
| Code | 32 each | Identical | None | 1.647361 | 0.106481 |

All saved logits are finite. Logits are not bitwise equal. English numeric comparison covers the first
22 rows, which have identical input prefixes; after the different pick the prefixes differ, so subsequent
rows are not treated as a direct numeric comparison. Japanese and code cover all 32 rows.
Relative L2 is `||CPU - Metal||2 / ||CPU||2`, measured per full-vocabulary row, with the largest row reported.
No numeric acceptance threshold was relaxed to call these outputs parity.

[CPU token capture](capture-cpu/capture.json), [Metal token capture](capture-metal/capture.json),
[row-by-row comparison and logit hashes](capture-comparison.json).
Raw logits remain locally in each capture directory as `english.logits.bin`, `japanese.logits.bin`,
and `code.logits.bin`: little-endian FP32, shape **[32, 248320]**, 31,784,960 bytes per file.
They are gitignored but saved, not discarded. CLI stdout/stderr logs are also saved locally.

**Full-model token equivalence is not established.** Fixed ggml-cpu remains useful for independent
component references; fixed llama.cpp CPU is now an executable candidate for full-model fixtures.
The Metal build is a second measured implementation, not an interchangeable exact oracle.
Selecting a Strata whole-model oracle still requires the unchanged Strata CUDA captures or an explicit
decision to use the saved CPU fixtures and enforce that chosen contract. The cause of the CPU/Metal
difference was not isolated in this phase.

## llama.cpp throughput and peak memory

These are engine-reported prompt/decode rates, excluding model loading and capture I/O.
All six clean CLI trials exited successfully with no command-buffer or model-loading errors.
No cancelled or error-bearing trial is included in the medians.

| Engine / trial | Prompt tok/s | Output tok/s | Peak RSS, bytes |
| --- | ---: | ---: | ---: |
| llama.cpp CPU 1 | 12.01 | 10.08 | 39,415,234,560 |
| llama.cpp CPU 2 | 12.00 | 10.61 | 39,793,885,184 |
| llama.cpp CPU 3 | 12.10 | 10.51 | 39,794,229,248 |
| llama.cpp Metal 1 | 111.41 | 52.95 | 39,613,104,128 |
| llama.cpp Metal 2 | 80.59 | 52.82 | 39,715,454,976 |
| llama.cpp Metal 3 | 92.20 | 53.20 | 39,716,618,240 |
| **CPU median** | **12.01** | **10.51** | **39,793,885,184** |
| **Metal median** | **92.20** | **52.95** | **39,715,454,976** |

Peak RSS is `/usr/bin/time -l` maximum resident set size (bytes on Darwin); it includes mapped pages
and is not total system RAM plus GPU consumption. The same tool's separate peak-memory-footprint
median is 583,998,368 bytes CPU / 607,772,296 bytes Metal. File-backed mapping accounting differs;
do not treat that smaller footprint as the entire model's memory cost. GPU-resource peak and system-wide
physical peak were not measured. This short 512-token context does not validate 4K/32K memory usage.

[Conditions](conditions.json) include swap and memory snapshots. OS file cache and Metal shader caches
were already warmed by preliminary attempts; they were not purged between runs. Other user workloads
were not controlled. Thermal state was nominal after the runs; per-trial temperatures were unavailable.
Prompt throughput varies even on this small workload. These are baseline observations, not an optimization
claim, a Strata speed estimate, or proof of model quality.

## Independent GPU read bandwidth

Final probe ran after downloads/builds finished, with no model process running.
[Kernel](read_bandwidth.metal), [Swift driver](read_bandwidth.swift), [all seven trials](bandwidth.json).
Input: 8 GiB `storageModeShared`, CPU-initialized with each uint32 word's index. Each dispatch reads all
8 GiB and writes one group checksum; every checksum is verified analytically modulo 2^32.
256 threads/group, 32-wide execution, 8 dispatches per timed command, one untimed warm-up command,
7 measured commands. Logical read bytes per trial: 68,719,476,736; checksum writes: 4,194,304.
CPU initialization, compilation and validation are outside GPU timing.

Timer: `MTLCommandBuffer.gpuEndTime - gpuStartTime`, including reductions and buffer barriers.
Decimal GB/s for logical reads: **median 582.64, minimum 574.03, maximum 585.67**.
This is effective large-buffer GPU read throughput, not a hardware DRAM traffic counter or Strata throughput.
Thermal state before/after: nominal. Sensor temperature: unavailable.
Device readouts: recommended working set 60,817,408,000 bytes, maximum buffer 41,747,087,360 bytes;
`iogpu.wired_limit_mb=58000`. No system setting was changed.

## Invalid preliminary attempts and limits

[Preserved invalid attempts](overlapping-invalid/README.md): restarting the runner left an older process
alive, so model runs overlapped and logs mixed. Metal reported out-of-memory despite exit code zero.
These runs were invalidated and repeated with an exclusive process lock; they are not evidence that one
model cannot fit. The first bandwidth probe also had an active download because the pause command did
not match its process. It was repeated after download completion. Final tables use only the clean runs.
The lock's rejection of a second runner was tested. Future success checks inspect GPU errors as well as exit codes.

Remaining risks: non-identical full-model outputs, no unchanged Strata CUDA full-model oracle,
unverified Strata native-expert runtime / Metal port / Linux backends, and larger-context residency.
[arm64/Darwin file-by-file scope and #1306 comparison](ARM64_SCOPE.md). Engine fixes were not written.

## Reproduction

Use the two verified model files, the pinned source archive, and the saved build options.
The build requires `LLAMA_BUILD_SERVER=ON` for this version's `llama-cli`, and
`LLAMA_BUILD_EXAMPLES=ON` for `llama-debug`; initial tool-only builds without the server had no CLI target.
The two build directories are `build-mac-phase05/llama-cpu` and `llama-metal`.

```sh
# From the Strata repository root, with the pinned source archive already extracted.
llama_source=build-mac-phase0/llama.cpp-3cf03257f219afbe7334045ff7c6a06ac68c627d
for backend in cpu metal; do
  metal_enabled=OFF
  if [ "$backend" = metal ]; then metal_enabled=ON; fi
  llama_build="build-mac-phase05/llama-$backend"
  cmake -S "$llama_source" -B "$llama_build" -DCMAKE_BUILD_TYPE=Release \
    -DGGML_METAL="$metal_enabled" -DGGML_METAL_EMBED_LIBRARY="$metal_enabled" \
    -DGGML_BLAS=OFF -DGGML_NATIVE=ON -DGGML_CPU=ON -DGGML_BACKEND_DL=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_COMMIT=3cf03257f219afbe7334045ff7c6a06ac68c627d
  cmake --build "$llama_build" --target llama-cli llama-debug -j 8
  clang++ -std=c++17 -O2 bench/results/2026-10-09-mac-phase05/capture_logits.cpp \
    -I "$llama_source/include" -I "$llama_source/ggml/include" -I "$llama_source/vendor" \
    -L "$llama_build/bin" -lllama -lggml -lggml-base -lggml-cpu \
    -Wl,-rpath,"@executable_path/llama-$backend/bin" -o "build-mac-phase05/capture-$backend"
done
python bench/results/2026-10-09-mac-phase05/inspect_pack.py \
  build-mac-phase05/models \
  build-mac-phase0/llama.cpp-3cf03257f219afbe7334045ff7c6a06ac68c627d \
  tests/data/native_experts/iq2_xs.txt bench/results/2026-10-09-mac-phase05
swiftc -O bench/results/2026-10-09-mac-phase05/read_bandwidth.swift -o build-mac-phase05/read_bandwidth
build-mac-phase05/read_bandwidth bench/results/2026-10-09-mac-phase05/read_bandwidth.metal
python bench/results/2026-10-09-mac-phase05/run_reference.py
python bench/results/2026-10-09-mac-phase05/compare_captures.py
```

`run_reference.py` contains every CLI argument, verifies complete files, and refuses a concurrent runner.
It overwrites the current local result files when deliberately rerun; preserve a report before a new experiment.
No upstream PR or fork push was made. The supplied fork owner was a placeholder; work remained in the fixed
local upstream checkout. Phase 0.5 ends with this report.
