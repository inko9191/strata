# Phase 0.6: llama.cpp Metal, MTP and long prompts on M5 Max

Measured 2026-10-09 on Apple M5 Max, 64 GiB unified memory, macOS 27.0.1 (26A434),
AppleClang 21.0.0.21000099. Strata base: `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`.
**All inference rates below are unmodified llama.cpp, not Strata.** No Strata engine source or llama.cpp
source was changed. Phase 1 has not started. The authorized publishing scope is this report, Phase 0.5
records, and `docs/METAL_PLAN.md`, on the user's fork [inko9191/strata](https://github.com/inko9191/strata).

## Sources and files

- llama.cpp [tag v0.6.0](https://github.com/ggml-org/llama.cpp/tree/v0.6.0), annotated tag object
  `69ecae922d2b60c1fa8c590279f4e2099af75111`, peeled commit **`d81235049384534c167caea52b85a694f6103d14`**.
  Tag `b11429` points to the same commit. The older head-to-head README lists `8345f33`; that differs from
  the fetched tag, and this report records the actual fetched source rather than repeating that value.
  [Provenance](llama-provenance.json), archive SHA256
  `6b58785f0a82898f4c3442417ff962e2d4b231b1bee5d033902ad90b27901e14`.
  The archive build's banner is `0.6.0-dev`, build 350 (outer checkout build-number fallback); commit is
  explicitly pinned. [Build options](llama-build-options.json): Metal/embedded library ON, CPU/NEON ON,
  BLAS OFF, native ON, dynamic backends OFF. `llama-server` and `llama-cli` built successfully.
- Main model: the exact two IQ2_XS files retained from [Phase 0.5](../2026-10-09-mac-phase05/README.md).
  ISTA-DASLab revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`; both Phase 0.5 full-file SHA256
  checks passed. No re-quantization, replacement, or additional main-model download in this phase.
- Separate self-contained head:
  [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/766911a6b7369840a91dbcd95f9f997acaab6cd6/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf),
  **`MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf`**, revision **`766911a6b7369840a91dbcd95f9f997acaab6cd6`**,
  bytes **4,137,429,120**, computed SHA256 **`58a47d6a216307e9b97172d5bfdc3436a8994b632f5e436b9bb6edf8951da8d1`**, verified against LFS.
  [Download provenance](mtp-provenance.json), [GGUF header/tensors](mtp-header.json): architecture
  `qwen4exp`, 34 tensors, one nextn block (block 48), 512 experts / top-10. This is unsloth's head,
  not Strata's native MTP head.

Models and builds stay under ignored `build-mac-phase05/` and `build-mac-phase06/`. Raw headers/logits
are also ignored. No weights, build artifacts, or large logit arrays are part of the report commit.

## Protocol

The method follows [the existing comparison](../2026-10-06-llamacpp-0.6.0/README.md) and
`bench/bench_vs_llama.py`, with the Mac-specific differences recorded in [conditions](conditions.json).
Existing `prompt-4k.txt` / `prompt-32k.txt` are used unchanged; hashes are in conditions.
Each request adds a nonce plus "Summarize the text above in three sentences." The same nonce is used
for a paired off/mtp3 request, and changes between rounds/cells/reps. Actual prompt counts are **4,042
and 32,040**, including the template, nonce and instruction.

- `/v1/chat/completions`, no thinking, cache_prompt=false, seed=1, temperatures 0 / 0.7.
  At 0.7: top_k=20, top_p=0.95; at 0: those overrides are omitted, following the reference script.
- **256 output tokens in every measured request**, using ignore_eos=true. The original comparison
  specified a maximum of 256 and sometimes stopped early. This is an explicit protocol difference.
- `-ngl 99 -ot per_layer_token_embd.weight=CPU --fit off --lazy-mode on`, mmap default,
  `-b 2048 -ub 2048 -c 36864 -ctk q8_0 -ctv q8_0 -fa on -t 8 -tb 8`, one slot.
  MTP: `-md <head> --spec-type draft-mtp --spec-draft-n-max 3 -ngld 99`.
- Two discarded warm-ups per session (16-token hello, then 32-token short corpus), fresh server per
  configuration/round. Orders: off/mtp3, mtp3/off, off/mtp3. **3 rounds, n=6 for each 4K cell and
  n=3 for each 32K cell**, 36 measured requests total.
- Same exclusive `fcntl.flock` as Phase 0.5, held across a session and across rounds. A second runner's
  rejection was tested. Servers are stopped and waited for before the next begins. GPU/model errors
  are checked independently of exit status; all 36 requests and six sessions passed.
- TTFT here is server `prompt_ms`, matching the non-streaming reference; it is not client streaming
  first-byte latency. Model loading is excluded. The first output token is charged to prompt time;
  the server's decode rate is **255 / predicted_seconds** for these 256-output requests.

[All request measurements and verification traces](trials.jsonl), [validated medians/ranges](summary.json),
full request/response JSON for each trial, and six `*-session.json` files are retained. Console logs remain
locally ignored. `LLAMA_TRACE=1` records each verification's draft length and accepted count.
An extra verbose diagnostic, excluded from medians, [confirmed actual Metal offload](backend-confirmation.json)
for the main model and draft model, and the PLE CPU mapping; embedding also remains CPU work.

## Results: medians

| Prompt | Temp | Config | n | Prompt tok/s | Decode tok/s | TTFT s | Draft accepted/generated | Request peak RSS GiB |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4K | 0 | off | 6 | 1178.53 | 50.52 | 3.430 | - | 39.292 |
| 4K | 0 | mtp3 | 6 | 1075.17 | 60.46 | 3.759 | 0.6724 | 43.974 |
| 4K | 0.7 | off | 6 | 1180.06 | 50.31 | 3.425 | - | 40.144 |
| 4K | 0.7 | mtp3 | 6 | 1080.13 | 54.80 | 3.742 | 0.5814 | 44.458 |
| 32K | 0 | off | 3 | 1101.53 | 49.01 | 29.087 | - | 44.828 |
| 32K | 0 | mtp3 | 3 | 934.20 | 61.25 | 34.297 | 0.7384 | 46.653 |
| 32K | 0.7 | off | 3 | 1102.10 | 48.78 | 29.072 | - | 45.667 |
| 32K | 0.7 | mtp3 | 3 | 911.05 | 55.90 | 35.168 | 0.6264 | 46.633 |

| Prompt | Temp | Config | Prompt tok/s min - max | Decode tok/s min - max |
| --- | ---: | --- | ---: | ---: |
| 4K | 0 | off | 1117.40 - 1183.36 | 49.78 - 50.98 |
| 4K | 0 | mtp3 | 1016.40 - 1084.88 | 56.09 - 62.65 |
| 4K | 0.7 | off | 1178.98 - 1182.22 | 49.94 - 50.74 |
| 4K | 0.7 | mtp3 | 1041.13 - 1086.69 | 48.30 - 59.95 |
| 32K | 0 | off | 1091.00 - 1103.50 | 48.71 - 49.40 |
| 32K | 0 | mtp3 | 917.06 - 937.00 | 57.09 - 63.26 |
| 32K | 0.7 | off | 1101.38 - 1104.04 | 48.62 - 48.81 |
| 32K | 0.7 | mtp3 | 910.54 - 918.20 | 44.65 - 57.49 |

RSS is sampled every 0.5 s from the process tree with Darwin `ps`, including mmap resident pages.
It can miss a shorter peak, and is not physical system-plus-GPU peak. The separate `/usr/bin/time -l`
maximum for each whole server session (loading, warm-up and all six cells):

| Config | Median session peak RSS GiB | Min - max GiB |
| --- | ---: | ---: |
| off | 45.666 | 43.449 - 45.666 |
| mtp3 | 46.876 | 46.146 - 46.957 |

GPU-resource/system-wide physical peak was not measured. OS and shader caches were not purged;
other user workloads were not controlled. Post-run ProcessInfo thermal state was nominal (0), and
`pmset` had no recorded warning; per-trial sensor temperatures/thermal states were not sampled.
No wired-limit setting was changed. Successful 32K llama.cpp runs do not prove Strata's future residency.

MTP's decode medians are higher in these cells, while prompt medians are lower. At temperature 0.7,
MTP's decode spread overlaps the off results, particularly 32K; do not infer a universal improvement
or statistical significance from three 32K trials.

## Output comparison

Paired inputs are identical. **Greedy decoded chat messages match in 4/9 pairs**, with
substantive text divergence in the others. The two short warm-up responses matched between the initial
smoke configurations. All outputs are saved; observed coherent text is not a quality/parity test.
This comparison is of the complete decoded message, not token IDs or hidden states, and **full-model
MTP token parity is not established**. Temperature-0.7 outputs are also recorded, without requiring
identity. No acceptance threshold or oracle was relaxed. The cause of greedy divergence was not isolated.
The Phase 0.5 CPU fixtures / unchanged Strata CUDA capture decision remains necessary before a port.

## Short prompt: old vs new, same day

MTP off, Phase 0.5's exact raw English prompt/template, 12 input tokens, 32 outputs, temperature=0,
seed=1234, context=512, batch/ubatch=64, 8 threads, FA off, same IQ2_XS and PLE CPU/lazy mmap.
CLI order: pinned/new, new/pinned, pinned/new. Fresh process each time, same lock and GPU error check.
[Every command and trial](short-trials.json). Pinned source is `3cf03257f219afbe7334045ff7c6a06ac68c627d`.

| Version | n | Prompt tok/s median | Decode tok/s median | Decode min - max | Peak RSS median GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| pinned | 3 | 114.83 | 53.19 | 49.96 - 53.98 | 36.988 |
| v0.6.0 | 3 | 117.95 | 53.12 | 52.53 - 53.20 | 37.001 |

New/old median differences: prompt **+2.72%**, decode **-0.13%**. The prompt range is
61.63-117.99 for new versus 106.24-116.95 for old; decode ranges also overlap. **No speed improvement
is claimed.** The original Phase 0.5 Metal decode median was 52.95; this same-day rerun is the comparison.

## Native weight-byte accounting

[Calculation](weight_accounting.py), [all tensor memberships and formulas](weight-accounting.json),
using the Phase 0.5 `tensors.csv` and actual main/header metadata. One native read per matrix/selected
expert/embedding row is assumed; this is logical tensor work, not a hardware DRAM traffic measurement.

| Main decode component | Native bytes per token |
| --- | ---: |
| Routed gate/up/down: 48 layers, 10/512 experts | 692,480,000 |
| Shared experts including their gates | 135,221,760 |
| Dense: attention, routers, gated residual, norms, PLE projections/conv | 2,936,153,600 |
| Head including final HC mixer | 350,863,360 |
| One input embedding row | 1,360 |
| **Total, excluding PLE row gathers** | **4,114,720,080** |
| PLE row gathers separately: 16 heads x 90-byte IQ4_NL row | **1,440** |

Do not count all 28.8 GB of PLE per token or the whole embedding table. On this Mac's 16 KiB VM pages,
16 cold, distinct row reads would touch 262,144 bytes if each fits one page, or at most 524,288 bytes
if every 90-byte row crosses a page boundary. Read-ahead can add more traffic; actual SSD bytes,
row alignment crossings and cache hits are not measured. PLE stays on CPU/mmap.

For off: `equivalent_percent = 4,114,720,080 * measured_decode_tok_s / (582.6e9) * 100`.
Denominator is the Phase 0.5 independent GPU-read median rounded to 582.6 GB/s, as requested.

| Prompt | Temp | Off tensor-work equivalent |
| --- | ---: | ---: |
| 4K | 0 | 35.68% |
| 4K | 0.7 | 35.53% |
| 32K | 0 | 34.61% |
| 32K | 0.7 | 34.45% |

This does not show how much physical bandwidth was used: activation/KV/state traffic, repeated tile
reads, CPU repacks and cache reuse are absent. It is an estimate from native weight work and elapsed time.

For MTP, each actual verification has `m = 1 + drafted` positions (usually 4). Let
`R=692,480,000`, `F=3,422,238,720` (shared+dense+head), `E=1,360`:

- No reuse logical operands: `B_verify=m*(F+R+E)`.
- Ideal unique tensor reads with full fixed-weight reuse: `F+R+m*E` to `F+m*R+m*E`.
  The lower end assumes the same top-10 expert set; the upper assumes disjoint sets. Actual router IDs
  and physical reuse were not captured, so these are conditional bounds.
- Sum actual verification batches, including any replay. Divide by the **255 timed output tokens**,
  keeping accepted drafts and correction/bonus tokens separate; the cap can discard a last extra pick.
  Dividing only by accepted drafts would omit the output supplied by the main model.

Each following entry is the median of per-request calculations; medians of counts need not sum.

| Prompt | Temp | Completed verifications | Drafted | Accepted drafts | Accepted/pass | Mean B_verify no reuse GB | Mean ideal unique GB bounds | Target no-reuse equivalent | Target ideal-reuse equivalent bounds | Generated-draft-only equivalent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4K | 0 | 84.5 | 253 | 170 | 2.013 | 16.436 | 4.115 - 6.188 | 56.45% | 14.13% - 21.25% | 8.53% |
| 4K | 0.7 | 93.5 | 280 | 161.5 | 1.741 | 16.439 | 4.115 - 6.189 | 56.51% | 14.16% - 21.28% | 8.54% |
| 32K | 0 | 79 | 237 | 175 | 2.215 | 16.459 | 4.115 - 6.192 | 54.40% | 13.60% - 20.47% | 8.22% |
| 32K | 0.7 | 89 | 265 | 166 | 1.865 | 16.366 | 4.115 - 6.177 | 54.91% | 13.78% - 20.68% | 8.30% |

The self-contained head's single-step native read model is **829,409,952 bytes**, including its own
Q8_0 output projection and top-10 experts. Its generated-draft-only column adds `draft_n * that_size`
to work separately; it is **not the entire draft cost**. The source's `draft_mtp::process()` also performs
catch-up forward passes over target batches, and confidence-rejected attempts can cost work without
increasing draft_n. These extra passes, activation/KV traffic and router-union reuse are not measured.
Thus an exact **whole MTP physical bandwidth utilization is unknown**. Target-only percentages must
not be presented as the complete GPU utilization, and off's per-token weight model must not simply
be multiplied by MTP output tok/s.

## Port targets, not speed predictions

Use the **best measured configuration median for the same workload**, independently per metric:

| Prompt | Temp | Prompt target tok/s (source config) | Decode target tok/s (source config) |
| --- | ---: | ---: | ---: |
| 4K | 0 | 1178.53 (off) | 60.46 (mtp3) |
| 4K | 0.7 | 1180.06 (off) | 54.80 (mtp3) |
| 32K | 0 | 1101.53 (off) | 61.25 (mtp3) |
| 32K | 0.7 | 1102.10 (off) | 55.90 (mtp3) |

The off prompt target and mtp3 decode target are from different configurations; they are not a single
measured combined result. No Strata speed is predicted, and matching a target still requires the chosen
correctness gate. The future MTP implementation must resolve its output contract before claiming parity.

## Reproduction

Download/build before timing. Use the verified Phase 0.5 files at `build-mac-phase05/models/`, place the
verified head at `build-mac-phase06/mtp-Qwen3.8-Flash-Next-Q8_0.gguf`, and extract the pinned archive.

```sh
cmake -S build-mac-phase06/llama.cpp-d81235049384534c167caea52b85a694f6103d14 \
  -B build-mac-phase06/llama-metal -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_BLAS=OFF \
  -DGGML_CPU=ON -DGGML_NATIVE=ON -DGGML_BACKEND_DL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_BUILD_COMMIT=d81235049384534c167caea52b85a694f6103d14
cmake --build build-mac-phase06/llama-metal --target llama-server llama-cli -j 8
python3 bench/results/2026-10-09-mac-phase06/inspect_head.py
python3 bench/results/2026-10-09-mac-phase06/weight_accounting.py
python3 bench/results/2026-10-09-mac-phase06/run_long.py
python3 bench/results/2026-10-09-mac-phase06/run_short.py
python3 bench/results/2026-10-09-mac-phase06/summarize.py
```

The long runner appends rows; preserve/move the existing result set before repeating an experiment.
It rejects concurrent runners rather than stopping unrelated user processes. Raw model files, builds,
logits and console logs remain local. Phase 0.6 stops here; no upstream PR is created.
