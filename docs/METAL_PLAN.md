# Metal / Apple Silicon port plan — Phase 0 / 0.5 / 0.6

調査日: 2026-10-09。対象は **Niko1221/Strata**、MIT。
固定コミット: [`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`](https://github.com/Niko1221/Strata/tree/fb58e0dbc8399662c0e47c76578c6e878b14f6cf) (v0.1.41)。
ggml / llama.cpp は `third_party/ggml/VERSION.txt` の
`3cf03257f219afbe7334045ff7c6a06ac68c627d` を使用する。

これは Strata の Metal 実装・モデル推論・性能測定の完了報告ではない。
Phase 0 の成果はこの文書。Phase 0.5 では配布 GGUF と固定 llama.cpp の調査、
独立した帯域測定・記録用プログラムを追加した。エンジンのソースは変更していない。従来の ro99/strata の作業は残してあるが、
その `mac-port-plan.md`、NVFP4、4bit 未満禁止という前提は本計画に適用しない。
Phase 0 / 0.5 は upstream の固定版で行った。Phase 0.5 で示されたフォーク URL の
owner は `<あなたのユーザー名>` というプレースホルダーで、有効な URL は未確定。
Phase 0.5 はその報告後に停止した。Phase 0.6 でユーザーが有効なフォーク
[`inko9191/strata`](https://github.com/inko9191/strata) と記録の push を承認した。
Phase 0.5 / 0.6 の文書・測定記録だけを `docs/mac-phase05-phase06` ブランチに保存する。
モデル、build、raw logits は push せず、upstream の PR は作らない。
**Phase 0.6 の報告後に停止し、Phase 1 へ進まない**。

## 1. 調査範囲と現時点の結論

最初に確認した資料は `AGENTS.md`、`README.md`、`docs/HOW_IT_WORKS.md`、
`docs/DETAILS.md`、`docs/MODELS.md`、`docs/AMD_HIP.md`、`docs/STRIX_HALO.md`、
`docs/INTEL.md`、`docs/COMMUNITY_BENCHMARKS.md`、`CMakeLists.txt`。
加えて kernel headers / implementations、`generate.cpp`、`verify.cpp`、
CPU experts、expert cache/source、`sycl/`、parity tests、`serve/`、`setup.py`、
固定版 ggml の CPU・Metal・モデル実装を確認した。

- 第 4 バックエンドの追加は可能性があるが、単なる CUDA API の名前置換では済まない。
  ホスト側も CUDA graph、stream、device pointer、x86 ISA、Linux の I/O に結び付いている。
- ggml-cpu はこの arm64 Mac でビルドできた。Strata の CPU expert ライブラリは
  x86 のコンパイラオプションで失敗し、native expert の実行までは確認できていない。
- Strata 全体の CPU-only forward はない。CPU experts は既存 GPU エンジンの一部分である。
- IQ2_XS はパックの名前であり、全テンソルが `GGML_TYPE_IQ2_XS` という意味ではない。
  IQ1_M を含む混在形式と Q2_0 down が最初から必要。
- 独立した比較基準を持つテストはあるが、多くの harness が CUDA を要求する。
  `bench/micro/` の native oracle harness と一部の参照ファイルは固定版の公開ツリーにない。
- Phase 0 時点では GGUF 実ファイルは未確認だった。Phase 0.5 の実ファイルと
  固定 llama.cpp の確認結果は §6 と補足報告を参照。Strata のモデル出力・速度は未検証。

関連する既存の議論も調べた。
[Mac 対応の要望 #518](https://github.com/Niko1221/Strata/issues/518) は closed、
[ARM64 の不足 #1306](https://github.com/Niko1221/Strata/issues/1306) は open だった。
この調査から、完成した native Strata Metal 経路があるとは確認できなかった。
外部の llama.cpp Metal wrapper は、本エンジンの移植や MTP parity の証拠には用いない。
外部への投稿は行っていない。

## 2. この Mac で取得した情報

| 項目 | 読み取り結果 / 取得方法 |
| --- | --- |
| CPU / GPU | Apple M5 Max、arm64 / `sysctl` と Metal device query |
| OS | macOS 27.0.1、build 26A434 / `sw_vers` |
| C++ compiler | AppleClang 21.0.0.21000099 / CMake |
| 物理メモリ | 68,719,476,736 bytes = 64 GiB / `hw.memsize` |
| VM page size | 16,384 bytes / `hw.pagesize` |
| `hasUnifiedMemory` | true / `MTLDevice` |
| `recommendedMaxWorkingSetSize` | **60,817,408,000 bytes** / `MTLDevice` |
| `maxBufferLength` | **41,747,087,360 bytes** / `MTLDevice` |
| `iogpu.wired_limit_mb` | **58000** / `sysctl` の報告値と単位をそのまま記録 |
| SSD 容量・帯域 | SSD 帯域は未測定。1 TB SSD はユーザー指定の環境情報 |
| GPU read 帯域 / サーマル状態 | Phase 0.5 で測定。中央値 582.64 GB/s、probe 前後 nominal。詳細は §6 |

Phase 0 の Metal device query は `swift -e` で上のプロパティを読み取っただけで、
バッファを割り当てたり、推論・帯域ベンチマークを実行したりしていない。
システム設定は変更していない。この値は各起動時にも記録する。
`recommendedMaxWorkingSetSize` は性能上の推奨値であり、確定した空き容量や
割り当ての成功保証ではない。`maxBufferLength` は別の制約である。
[Apple: working set](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize)

## 3. バックエンドの境界

### 3.1 現在の構成

| 境界 | CUDA | HIP | SYCL | Metal への示唆 |
| --- | --- | --- | --- | --- |
| CMake | `STRATA_ENABLE_CUDA` | `STRATA_ENABLE_HIP`、gfx906 分岐 | `STRATA_ENABLE_SYCL` で `add_subdirectory(sycl)` 後に root を return | `STRATA_ENABLE_METAL` を OFF 初期値で追加。複数 GPU backend 同時選択を明示的に拒否 |
| GPU runtime | CUDA runtime / driver / cuBLAS | `cmake/hip_backend.cmake`、`strata_hip_runtime`、compat headers | oneAPI、queue、USM、graph adapter | Objective-C++ の device / buffer / command submission。Metal と Foundation framework を Apple の選択時だけリンク |
| ソース | `src/kernels/cuda/*.cu` | 同じ `.cu` を HIP としてコンパイル、compat header、`STRATA_USE_HIP` / `__HIPCC__` | `sycl/` にミラーと `.dp.cpp`、include override | MSL を別ファイルに実装し、既存 namespace の entry point と対応付ける |
| interface | `include/strata/kernels/*.hpp` の関数と shape / buffer structs | 多くを共有 | ミラー / override | shape と数値契約を再利用。`void* stream` だけでは backend-neutral とは言えない |
| 登録 | CMake の static library ソース列挙、host から直接関数呼び出し | 同左 | 独自 CMake | 共通の動的 kernel registry はない。Metal pipeline table は新 backend 内に閉じる |
| 実行 | `core/graph.*`、`verify.*`、`session.*` が CUDA graph / stream 型を所有 | CUDA 型の互換経路 | queue / graph への置換 | submission と state lifetime の境界が必要。長時間 GPU spin と doorbell を移さない |
| CPU expert | x86 fast path + 固定 ggml-cpu | 同じ host 側 | CPU 側を共有 / ミラー | ISA 選択を arm64 対応にし、ggml-cpu の native 経路を検証 |

CUDA / HIP が OFF の現在の CMake では、artifact reader、planner、speculation の一部、
CPU kernels と独立テストは作られるが、`strata_core`、`strata_engine`、
`strata_prefill`、推論用 `strata` は作られない。Phase 1 ではこれらの
ホスト責務を切り出す。GPU なしの実行ファイルが full forward 対応だと装わず、
GPU が必要な操作には明確な unsupported error を返す。

### 3.2 主なファイル群の扱い

| ファイル / 群 | 方針 |
| --- | --- |
| `src/artifact/`、`include/strata/artifact/`、`include/strata/plan/`、`src/plan/`、`src/spec/` | 基本的に再利用。ビルドと単体テストで確認 |
| `src/core/weights.cpp`、`layout.cpp`、native manifest / geometry / GGUF validation | 計算やフォーマットを再利用し、device storage / allocation を backend 境界へ |
| `include/strata/core/{graph,session,verify}.hpp`、対応 `.cpp`、`src/program/generate.cpp` | CUDA 型、allocation、graph capture、同期、head / embed ownership を境界の裏に分ける |
| `src/core/device.cu`、`pinned.cu` | Metal 用 device / shared buffer 実装を追加。既存ソースは残す |
| `src/core/expert_cache.cpp`、`expert_source.cpp` | cache 容量判定・expert ownership・native pointer table を Metal 用に分ける |
| `src/kernels/cuda/`、`include/strata/kernels/` | 後述の最小集合を MSL で再実装。shape、dtype、数学的意味は維持 |
| `src/kernels/cpu/{expert,expert_layout,native_expert,pool}.cpp`、`iq_avx*`、`q2_avx*`、`kq_avx*` | x86 flag / intrinsics / CPU detection の条件分岐。arm64 は ggml native を検証し、AVX ファイルをリンクしない |
| `src/kernels/cpu/pool_affinity_linux.hpp`、`src/platform/`、`src/core/conversation_memory.cpp` | affinity、`/proc`、`O_DIRECT`、Linux memory query を Darwin 対応。利用不能な任意機能は明示 |
| H2D staging、PCIe miss copies、host/GPU expert split、GPU wait-flag kernels | 全 expert を shared memory から読む Metal 最小経路では使わない。他 backend から削除しない |
| `src/prefill/` | 短い prompt の正しさにも必要。Phase 2 の逐次 prompt か最小 native prefill、Phase 4 の通常 prefill を分けて検証 |

Strix Halo に近いのは `expert_cache.cpp::device_free_bytes` の unified-memory 判定。
ただし現在は Linux の integrated-device 判定と `/proc/meminfo` の `MemAvailable` を使い、
既定 6 GiB の headroom を引く **容量計算**である。expert cache の `cudaMalloc` や
host arena と GPU cache の二重配置まで消す仕組みではない。
Mac で同じ heuristic を写すだけでは、目標の単一メモリ構成にならない。

### 3.3 SYCL を型紙にできる範囲

後付けの `sycl/` は、第 4 の opt-in CMake 分岐、host/runtime の置換範囲、
テスト単位、共有ソースを選別する方法の参考になる。
しかし DPCT の CUDA-to-SYCL 変換、USM、oneMKL、CUDA graph 互換 API は Metal の実装には使えない。
ミラー全体をもう一つ複製すると修正が追従しにくい。
まず host の共通責務と buffer/submission 境界だけを小さく切り出し、
Metal を選択した時だけ新経路へ入る。既存 default の演算順序は変えない。

`docs/INTEL.md` の失敗例も再現テストに使う: null queue、async copy の寿命、
subgroup 幅、IQ decode の aliasing / byte shift、除算と丸め、host/device の可視性。
Metal の `threadExecutionWidth` を取得し、CUDA の lane 演算を無条件に移さない。

## 4. パックとメモリの前提

Phase 0 は **このコミットにある manifest fixture** の調査だった。
Phase 0.5 で Strata の `setup.py` が固定する ISTA-DASLab の IQ2_XS を取得し、
revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09` の実ファイルと照合した。
全 48 layers の gate/up/down の型・形状・絶対 offset・bytes、packed offset が一致した。
IQ3_XXS は引き続き fixture の情報で、実ファイルは未取得。
[header・tensor 一覧・照合結果](../bench/results/2026-10-09-mac-phase05/README.md)を保存した。

| fixture | gate/up の型と layer 数 | down の型 | expert bytes | 64 GiB に対する expert bytes の割合 |
| --- | --- | --- | --- | --- |
| `tests/data/native_experts/iq2_xs.txt` | IQ2_S (22): 34、IQ2_XXS (16): 11、IQ1_M (29): 3 | Q2_0 (42): 48 layers | 35,454,976,000 = 33.0200 GiB | 51.59% |
| `tests/data/native_experts/iq3_xxs.txt` | IQ2_XXS (16)、IQ2_XS (17)、IQ3_XXS (18)、IQ3_S (21)、IQ2_S (22) の混在 | Q2_0 (42) / IQ4_NL (20) の混在 | 42,912,972,800 = 39.9658 GiB | 62.45% |

fixture の各 layer の `blob_bytes × 512` を足して算出。
いずれも 48 layers、512 experts/layer。top-k は既存の **10** のまま。
`docs/MODELS.md` の **39.2 GB / 47.0 GB はモデルの RAM+VRAM Requirements の表示値**であり、
expert だけの数字ではない。同資料は expert 部分を約 35.5 / 43 GB と区別している。
残りの dense / embedding / head、KV、GDN state、scratch、Metal resources、OS、
PLE の一時的な cache pages を足して実測するまでは「64 GiB で収まる」と断定しない。

`include/strata/core/layout.hpp` の既存 geometry は n_embd=2560、n_ff=640、
GDN 36 layers (key heads 16 / value heads 48 / state dim 128)、
QSA 12 layers (Q heads 24 / KV heads 2 / head dim 256)、
indexer query heads 4 / key dim 128、gated-residual streams 4 / bottleneck 320。
型と geometry は loader で照合し、不一致なら停止する。
Q2_0 はこの固定 ggml の type 42、64 elements / 18-byte block であり Q2_K とは別。

### 単一メモリと mmap の検証順

1. 小さな実 GGUF region を page-aligned に mmap し、page 単位の長さで
   `newBufferWithBytesNoCopy` / `storageModeShared` に包む。offset、末尾、読取り専用 mapping、
   GPU read、CPU/GPU completion、release / unmap の順序を試験する。
2. 同じ tensor の decoded bytes / values を CPU で比較する。resident-copy に頼らず
   file-backed mapping を実際に GPU が読めるかを記録する。
3. `maxBufferLength` 以下の複数 buffer views と **buffer identifier + byte offset** の
   descriptor を使う。IQ3_XXS の expert bytes は取得した上限を超えるため、
   単一の巨大 `MTLBuffer` を前提にしない。buffer 分割は物理的な重みの複製を意味しない。
4. GGUF の gate/up/down は manifest に独立した絶対 offset がある。
   CUDA の `unsigned long long grp_ptr` に CPU アドレスを入れても MSL が dereference
   できるとは扱わない。3 role の offset と型・範囲・寿命を検証する。
5. mmap wrapping が失敗する、または block alignment / buffer view の制約で扱えない場合、
   元の量子化 bytes を一度だけ shared allocation に読み込む。
   元 mapping / host arena を解放して常駐 expert copy を一組にする。
   どちらを選んだか、失敗条件、load 時の peak memory を記録する。

[Apple: bytes-no-copy](https://developer.apple.com/documentation/metal/mtldevice/makebuffer(bytesnocopy:length:options:deallocator:))
は page-aligned pointer / length と単一 VM region を要求する。
固定 ggml の `ggml/src/ggml-metal/ggml-metal-device.m` も page alignment と
大きな mapping の複数 view を扱う参考になるが、Mac でこの GGUF を包めたという証拠ではない。

重みの再量子化はしない。永続的な FP16 / FP32 expert copy も作らない。
IQ の layout / codebook は固定版の `ggml-common.h`、`ggml-metal/kernels/` の
`mul_mv.metal`、quant 関連 headers と既存 `iq_kernels.cu` を突き合わせる。
借用部分の MIT notice を保持し、Strata の graph を ggml graph に置き換えない。
PLE の約 28.8 GB は upstream と同様 SSD に置き、必要な n-gram rows を CPU で読み、
小さな shared staging を GPU に渡す。Darwin の I/O 方針は実測前に先読みへ変更しない。

## 5. 最小デコードの kernel 一覧

**これはソースを追った呼出し一覧であり、GPU 実行 trace ではない。**
GGUF と対応機がなく、実 launch 数・checkpoint 固有の dtype 分岐は未確認。
Phase 2 では descriptor と entry point の trace を取り、一覧の漏れを照合する。

最初の比較条件は IQ2_XS、batch=1、短い context、greedy (`--temperature 0`)、
FP16 の resident KV、MTP 無効、lookup 無効、steering / experimental fusion 無効。
native pack の `generate.cpp` は **`--spec 0` を拒否**する。
そのため `--native SHARD1 --spec 2 --suffix-draft 0 --lookup-chain 0 --kv fp16 --kv-resident 0`、`--mtp` なし、
正の `--prefill CHUNK` で、Verifier が実行する **実質 T=1 の window** を対象にする。
既存 `--native` は native router / QSA / GDN / PLE 等を選択する。
capture の容量と実際の proposal 数を混同しない。
Metal だけに no-draft の明示的な指定が必要なら opt-in にし、他 backend の CLI default は変えない。

表のパスは特記がなければ `src/kernels/cuda/`、テストは `src/kernels/`。
「native oracle 欠落」は CMake が `bench/micro/` に要求する harness がないことを表す。

| entry point / 演算 | 実装ファイル | 呼ばれる場所 / 役割 | 既存 parity / 比較基準 |
| --- | --- | --- | --- |
| `iq_embed_rows` | `iq_kernels.cu` | `NativeEmbed::gather_dev`、token embedding の native decode | `native_expert_parity` の embedding / BF16 checks、`iq_parity`、ggml `to_float` |
| `broadcast_streams` | `verify_kernels.cu` | embedding を 4 residual streams に配置 | `verify_parity`、`elementwise_parity` の関連 copy/broadcast 契約 |
| `fused_gr_read_multi` の down/up/inject/norm、必要な `gr_write` / head read | `fused_gr.cu`、`gr.cu`、`native_gr_norm.cu`、`native_gr_postops.cu` | `Verifier::record_window`、48 layers の attention / FFN 入力、最後の HC mix | `gr_parity`、`gr_multi_parity`、`verify_batch_parity`。独立 native GR oracle は欠落 |
| `native_quantize_q8_1`、`native_mmvq` | `native_mmvq.cu`、IQ dispatch は `iq_kernels.cu` | GDN/QSA projections、shared experts、PLE key、output head。header の型で dispatch | `mmvq_multi_parity`、`iq_parity`、`quantize_act_parity`。native MMVQ oracle は欠落 |
| `bf16_gemv_fp32_mmvf`、BF16 conversion | `native_bf16.cu`、`bf16_gemv.cu`、`dequant_bf16.cu` | router、indexer、GR、PLE value、shared gate 等 | `bf16_gemv_parity`、`dequant_bf16_test`、`bf16_bits_test`。native BF16 oracle は欠落 |
| PLE key/value projection、`native_ple_postops`、`ple_history_advance` | `native_ple_postops.cu`、`ple.cu`、`verify_kernels.cu`、projection files | layer 1 の PLE。ngram hash / SSD row read は CPU。履歴も次 token 用に更新 | `ple_parity` A/B/C、`verify_batch_parity`。C の captured block は外部ファイルが必要 |
| `gdn_conv_l2_multi` | `verify_kernels.cu` | GDN 36 layers、depthwise causal conv、Q/K L2 norm。T=1 でも multi entry | `gdn_parity` の conv と state 比較 |
| `gdn_ab_multi` | `verify_kernels.cu` | GDN decay / beta、BF16 A/B projection | `gdn_parity`、fused single 対 multi の比較 |
| `gdn_step_norm_multi` | `verify_kernels.cu` | GDN recurrent update、norm / gate、T=1 self-commit | `gdn_parity`、output と recurrence state 両方 |
| Q/K / indexer norm と RoPE、`native_qsa_rms_norm_rope` または個別 norm / rope | `native_qsa.cu`、`native_rope.cu` | QSA 12 layers。fusion flag による分岐 | `qsa_parity`、`rope_parity`。native QSA / RoPE oracle は欠落 |
| `kv_append_step` 等の KV append、`native_qsa_indexer_append` | `qsa.cu`、`native_qsa_indexer.cu` | FP16 K/V、indexer tail/dead/pooled と block position を更新 | `kv_steps_parity`、`qsa_parity` の native append / pooling。native indexer oracle は欠落 |
| `qsa_block_scores`、`qsa_block_topk` | `qsa_select.cu` | Verifier は短い context でもこの selector を呼ぶ。block capacity / live-tail に注意 | `qsa_parity`、`qsa_topk_parity`、`qsa_topk_active_parity`、selected ids を厳密比較 |
| `qsa_decode_attn_batch` の chunk / merge | `qsa_decode_attn.cu` | Verifier は n=1 でも batch entry。split attention と FP32 merge | `qsa_parity`。`qsa_decode_attn_step` だけ移植して終了とはしない |
| `native_qsa_gate_apply` | `native_qsa.cu` | query gate の sigmoid と attention result の積 | `qsa_parity`。native QSA oracle は欠落 |
| `native_router_top10` | `native_router.cu` | BF16 router projection 後に 512→10。softmax、ties、selected-weight normalization | `router_top10_parity`、`native_multi_parity`。native router oracle は欠落 |
| routed expert plan | `verify_kernels.cu` の `resident_plan`、host planner | all-resident table / top-10 entries。device planning flag により host/GPU 分岐 | `resident_plan_parity`。Metal は raw CUDA address でなく logical descriptor を比較 |
| `quantize_q8_1_rows` | `iq_kernels.cu` | routed expert 入力の activation quantization | `quantize_act_parity`、`iq_parity`、`native_expert_parity` |
| `native_expert_grouped` の GU / SwiGLU / Q8_1 / down | `iq_kernels.cu` | IQ2_S / IQ2_XXS / IQ1_M gate/up、Q2_0 down。`native_gu(_multi)_kernel`、`swiglu_q8_1_entries_kernel`、`native_down(_multi)_kernel` 等 | `native_expert_parity`、`native_grouped_parity`、`iq_parity`、`iq_multi_parity` |
| `shared_expert_multi`、native SwiGLU、scalar sigmoid、row scale | `shared_expert.cu` と projection files | 共有 expert は削除しない。n=1 でも wrapper は multi を使う | `shared_expert_parity`、`native_multi_parity`。native shared oracle は欠落 |
| `moe_combine_parts` → native combine | `native_moe.cu`、`shared_expert.cu`、host `layer.cpp` | top-10 weighted result を選択順に加算し shared result を足す | `native_multi_parity`、`shared_expert_parity`。native MoE oracle は欠落 |
| final mix / Q8_1 / native head projection | GR と MMVQ の上記 files | vocabulary logits を生成 | 部品 parity + 外部の teacher-forced full-model logits |
| `sample_tokens` または `argmax_rows` | `sampler.cu`、`verify_kernels.cu` | greedy pick、同値時の選択規則。CUDA architecture による分岐あり | `sampler_parity`、`verify_parity`、`spec_verify_parity` |

この表は wrapper と内部 launch の対応を記載したもの。
GU/down の launch は型、grouped-v1、old-kernel、CUDA/HIP architecture で変わる。
`STRATA_EXPERT_V2` の S26、AMD LDS / WMMA、PDL、cluster、QFUSE/LFUSE、
compressed KV / FWHT、native flash-attention の診断用経路は最小対象に混ぜない。
未実装オプションは Metal 選択時に理由付きで拒否し、黙って CPU に落とさない。

CUDA にある `wait_flag_*`、mapped copy、`fetch_blobs`、`rebase_ptrs`、
CPU miss rows と GPU hit rows の合流は輸送・分業の仕組みである。
Metal の全 expert resident 経路では、その handshake を GPU 上の spin に置き換えず、
必要な CPU PLE I/O と command completion を host 側で管理する。
T=1 の commit も recurrence / convolution / PLE history の更新を省略してよいという意味ではない。

### MTP / speculative verification で追加されるもの

| 演算 / 状態 | ファイル | parity / 検証 |
| --- | --- | --- |
| MTP embedding、target hidden + embedding の結合、draft layer、QSA / GR / MoE / head | `src/core/mtp.cpp`、上表の projection / residual / attention / expert kernels | 共通部品 parity + 同じ MTP weights / vocabulary mapping からの CUDA draft logits / ids capture |
| `add_streams_broadcast`、`mtp_select`、`ident_hits`、`row_top_prob`、`gather_rows`、`map_ids`、`copy_row_to_first`、必要なら `force_token` | `verify_kernels.cu`、MTP host orchestration | `verify_parity`、`spec_verify_parity` と固定 draft sequence fixture。MTP 全体 capture は未取得 |
| T=2..8 の projections / routes / shared / expert grouping / combine / GR | `native_mmvq.cu`、`native_bf16.cu`、`native_router.cu`、`shared_expert.cu`、`iq_kernels.cu`、`native_moe.cu`、`fused_gr.cu` | `mmvq_multi_parity`、`native_multi_parity`、`native_grouped_parity`、`iq_multi_parity`、`gr_multi_parity`。同じ backend の sequential と byte comparison |
| GDN speculative state と accepted-prefix commit | `verify_kernels.cu` の `gdn_conv_l2_multi` / `gdn_step_norm_multi` / `gdn_conv_commit`、`src/core/verify.cpp::capture_commit` | `gdn_parity`、各 n_keep / reject position、未採択 token が state に残らないこと |
| batched KV / indexer append、query ごとの selection、PLE snapshot / GR postops | `qsa.cu`、`native_qsa_indexer.cu`、`native_ple_postops.cu`、`native_gr_postops.cu`、`verify_kernels.cu` の `copy_indexed` 等 | `kv_steps_parity`、`qsa_parity`、`verify_batch_parity`、history と snapshot も比較 |
| CPU suffix draft / lookup chain / acceptance policy | `src/spec/`、`src/program/generate.cpp` | `suffix_drafter_test`、`controller_test`、`draft_policy_test`、`coupled_draft_test`、`spec_prob_test`。lookup 自体には新 GPU 演算はない |
| speculative sampler / target validation | `sampler.cu`、`verify_kernels.cu`、`src/core/verify.cpp` | `sampler_parity`、`spec_verify_parity`、全 prompt の spec 有無で token 列一致 |

最初は greedy を正しさの出口にする。既存 sampler の FP64 を使う分岐は
MSL に同じまま移せないため、非 greedy を対応済みと表示しない。
Phase 3 で sampling 対応を行うなら乱数・確率補正・精度の契約を別途固定する。
upstream 自体でも CPU/GPU expert 分担や prefill 分岐による丸め差はあり得る。
同一 Metal 経路の spec ON/OFF の一致を実際に確認し、upstream の説明だけで保証しない。

## 6. 正しさの基準と Mac での用意

### 6.1 既存テストの基準

| 基準 | 例 | Mac での扱い |
| --- | --- | --- |
| host の FP32 / double 数式 | `gr_parity`、`gdn_parity`、`qsa_parity`、`rope_parity`、router / sampler | 数式と試験入力は再利用。CUDA allocation / launch 部分だけ Metal test adapter に移す |
| 固定 ggml CPU `to_float` / `vec_dot` / activation quantizer | `native_expert_parity`、`dequant_q5_1_test`、`bf16_bits_test` | arm64 でビルド済み。各 IQ と Q2_0 の runtime reference を Phase 1 で試験する |
| 独立した Python-generated fixture | `tools/iq_fixture.py` の GGUF dequant、`qsa_oracle_vectors.inc`、`ple_oracle_vectors.inc` | generator / fixed revision / hash を記録。Q2_0 の独立 reference 対応も確認 |
| 同一 backend の旧/new、single/multi を memcmp | `mmvq_multi_parity`、`native_multi_parity`、`native_grouped_parity`、`gr_multi_parity`、`verify_batch_parity` | Metal 内でも厳密比較する。これだけでは両方が同じ誤りの可能性を排除できず、独立 oracle を併用 |
| 独立 ggml CUDA oracle | CMake の `STRATA_ORACLE_SOURCE_DIR` / `STRATA_ORACLE_BUILD_DIR`、16 個の `native_*_parity` | CUDA + `ggml-cuda` を要求。**`bench/micro/*_parity.cpp` はこの公開 clone にない**。そのまま Mac で動くと扱わない |
| external captured tensors | `ple_parity` C の `ple_in.bin` / `ple_out.bin` | capture producer、checkpoint hashes、dtype / shape / endian、実行設定を添付して別機から取得。未取得は未検証 |

公開 `ref/` には `load.py` があるが、テストのコメントが参照する
`ref/model.py`、`ref/gdn.py`、`ref/gr.py`、`ref/qsa.py`、`ref/ngram.py` 等はない。
「既存 Python full forward を実行すればよい」という計画にはしない。
`iq_parity` は fixture が全部ない場合 exit 3 を返す。
移植時は missing device / fixture に理由を表示した **CTest skip (77 等)** を使い、
テスト集計の skipped に残す。入力が壊れている場合や数値不一致は failure にする。

### 6.2 推奨する組合せ

1. **各 kernel の独立 oracle:** 固定 ggml-cpu の dequant / dot、既存 host reference と
   committed vectors。Metal の shader と同じコードブック decoder だけを CPU に写したものを
   唯一の基準にはしない。
2. **既存 Strata に最も近い full-model oracle:** 固定コミットの CUDA 機で、同じ GGUF、
   tokenizer、prompt token ids、prefill、greedy 設定を使った teacher-forced logits / selected experts /
   intermediate state / 出力 token capture。現時点では機械も capture も未確保。
3. **Mac で実行できた full-model 候補:** 固定 llama.cpp の
   `src/models/qwen4exp.cpp`。Phase 0.5 の実 GGUF は `general.architecture=qwen4exp` で、
   CPU-only `llama-cli` への読み込みと 32-token greedy 生成に成功した。
   同じ版の Metal build も実行できた。両方で `per_layer_token_embd.weight=CPU`、
   mmap / lazy mode を使い、PLE 全体の RAM 常駐を要求していない。
   これは Strata 自体の CPU-only / Metal 実装ではない。

llama.cpp CPU の native activation dot は Q8_K、Strata GPU experts は Q8_1 を使うなど、
同じ重みでも丸めの経路は違う。CPU logits が CUDA の logits と bitwise 同一だとは要求しない。
変更していない比較系同士の差は下記のとおり保存できた。
選択する oracle を Metal model 実装前に確定する。CUDA capture が得られなければ
CPU oracle を採用した事実を明記し、その oracle と出力 token が一致した範囲だけを
Phase 2 の結果として報告する。llama.cpp Metal を exact oracle として CPU と交換しない。

### Phase 0.5: 実ファイルと実行結果

詳細と再現用 source / raw data は
[Phase 0.5 report](../bench/results/2026-10-09-mac-phase05/README.md) に保存した。
IQ2_XS の 2 shards の revision / SHA256 / 全 tensor descriptors を記録し、
既存 manifest の全 48 layers が一致した。第 2 shard 単体には architecture metadata はない。
第 1 shard の metadata と split 情報を使った llama.cpp の paired load は成功した。

固定 prompt は英語、日本語、コードの 3 本。raw prompt を出す明示的な同一テンプレート、
temperature=0、32 generated tokens、context=512、batch/ubatch=64、CPU threads=8、
Flash Attention off、MTP off。input token IDs は両 backend で一致した。
CLI の生成ログに加え、固定版の public llama API を呼ぶ外部プログラムで
各 step の全 vocabulary logits と greedy token IDs を保存した。llama.cpp の修正はしていない。

| Prompt | CPU / Metal token 列 | 最初の違い | 同一 prefix で比較できた logits rows | 最大 relative L2 / 最大 absolute error |
| --- | --- | --- | ---: | --- |
| English | **不一致**、各 32 tokens | 22 token 目、CPU 1220 / Metal 1144 | 22 | 0.0727436 / 0.870231 |
| Japanese | 32 tokens 一致 | なし | 32 | 0.0747462 / 0.964064 |
| Code | 32 tokens 一致 | なし | 32 | 0.106481 / 1.647361 |

全 logits は finite だが bitwise 一致ではない。relative L2 は vocabulary 全体の row ごとに
`||CPU - Metal||2 / ||CPU||2` を求めた最大値。English の分岐後は input prefix が異なるため
その後の logits を同条件の数値比較に含めていない。
差の原因は未特定であり、これを新しい許容誤差として採用しない。
**full-model token parity は未達**。CPU fixtures を有用な候補として残し、
変更前の Strata CUDA capture と選択した基準の関係を確認する必要がある。

ユーザー指定の独立 baseline 測定として、llama.cpp CLI の同一 English prompt を
CPU / Metal 交互に 3 回ずつ測った。全有効試行の medians:

| llama.cpp のみ | Prompt tok/s | Output tok/s | ピーク RSS の中央値 (bytes) |
| --- | ---: | ---: | ---: |
| CPU | 12.01 | 10.51 | 39,793,885,184 |
| Metal | 92.20 | 52.95 | 39,715,454,976 |

RSS は `/usr/bin/time -l` の値で、mapped pages を含む。GPU を含む system-wide physical peak とは
同じではない。別の peak-memory-footprint 指標も保存した。全 trial、範囲、startup memory、
使用した flags と raw logs は補足報告にある。Metal は 49/49 layers の offload を報告するが、
embedding と明示的な PLE は CPU 側にも残る。Strata の速度や全演算 GPU 化を表す数字ではない。
OS / shader cache は予備試行で暖まっており、温度センサー値は不明、終了後の thermal state は nominal。
長い context のメモリ適合性・quality の評価には使わない。

独立 Metal read kernel の最終測定は 8 GiB shared buffer、各試行 8 回の全 buffer read、
1 warm-up / **7 measured trials**、全 group checksums 一致。
GPU command timestamps による logical read bandwidth は **中央値 582.64 GB/s、
最小 574.03、最大 585.67** (decimal)。モデルの速度や hardware DRAM counter ではない。
downloads / builds 完了後、model process がない状態で測った。wired setting は変更していない。

予備の model 測定は記録プログラムの再起動ミスで二つの runner が重複し、Metal OOM と
ログ混在が発生したため無効化した。終了コード 0 だけでは成功扱いにしない。
single-runner lock と GPU error 検査を入れてやり直し、最終 6 CLI trials / captures はエラーなし。
予備の帯域 probe も download を停止できていなかったため、取得完了後に再測定した。
無効な logs を別フォルダに残し、上の集計には含めていない。

### Phase 0.6: MTP / 長い prompt の比較基準

固定 llama.cpp v0.6.0 の Metal build で、同一 IQ2_XS と unsloth の独立 Q8_0 MTP head を実行した。
タグが実際に指す commit は `d81235049384534c167caea52b85a694f6103d14`。
既存 4K / 32K prompt、no thinking、temperature 0 / 0.7、256 outputs、KV q8_0 / FA on、
context 36864、PLE CPU / lazy mmap。MTP off / `draft-mtp` n_max=3 を交互に 3 rounds。
4K は各条件 6、32K は各条件 3 trials、計 36 trials が GPU/model error なしで完了した。
全条件、範囲、TTFT 定義、MTP head の revision / SHA256 は
[Phase 0.6 report](../bench/results/2026-10-09-mac-phase06/README.md) を参照。

同一入力の greedy decoded messages は **4/9 組だけ一致**し、他は本文が分岐した。
これは token ID / state parity の証明ではない。CPU/Metal の差に加え、Metal 内の MTP on/off も
exact oracle として交換できるとは確認できていない。Strata CUDA captures または明記した
CPU oracle の採用を決め、失敗後に基準を緩めない。Strata の正しさ確認前にその速度を報告しない。
ユーザー指定の独立 llama.cpp 測定値を、Strata の性能や quality の証明には使わない。

### 6.3 数値ゲート

既存テストの尺度・比較方法を維持する。下の数字は既存ソースの基準であり Mac の測定結果ではない。

| 対象 | 基準 |
| --- | --- |
| BF16 / FP16 conversion、Q8 activation bytes、selection ids、ties、plan logical fields | 既存 vector に対して厳密一致。CUDA address 値そのものは Metal の比較対象にしない |
| `iq_parity` | dequant relative L1 < 1e-6、float-reference MMVQ relative L1 < 2e-2、single/multi の相違 0 |
| `native_expert_parity` | CPU/GPU expert 各々の float reference に対する relative L1 < 3e-2、finite。dequant は < 1e-6。これは activation quantization を含む部品の基準 |
| GDN | conv / norm 関連 1e-6、recurrence output / state 1e-5、該当 single/multi・commit cases は memcmp |
| QSA | pooled keys / scores 1e-6 (既存の terms 正規化)、ids 厳密、同一 FP16 cache の attention 1e-5、cache の float 基準との差 1e-3 |
| GR、shared、router、combine、RoPE、PLE | 各既存 test の項目別基準をそのまま採用。融合前後 / column invariance は該当項目の memcmp を維持 |
| full-model 出口 | 事前に固定した複数 prompt、tokenizer、greedy 条件で oracle と token 列 / top-1 が一致。teacher-forced logits と state 差も保存 |

部品の 2% / 3% や Phase 0.5 の CPU/Metal 差を full-model logits の許容値へ流用しない。
full-model logits の数値閾値は、実パックと変更前の oracle 間比較を取得してから
**Metal 実装前に**固定し、未確定のまま合格としない。
閾値や oracle を失敗後に緩めて先へ進めず、違いの原因を報告して phase を止める。

## 7. CPU-only とホストビルドの試験結果

以下は Phase 0 の履歴。Phase 0.5 の固定 llama.cpp 実行・ビルド結果は §6 に追記した。
試験機は §2 の Mac。コードは変更せず、CMake options のみで調べた。
固定 llama.cpp archive は codeload から取得し、指定 hash のディレクトリを
`STRATA_GGML_DIR` に渡した。archive に `.git` がないため ggml の configure banner が
外側の Strata commit を表示する場合があるが、使用したソース revision は上記 VERSION の hash。

| 操作 | 結果 | 限界 |
| --- | --- | --- |
| CUDA/HIP/SYCL OFF、native experts OFF、tests ON の configure | 成功 | 全体の CPU-only inference target は生成されない |
| 同構成の全 build | **失敗** | `strata_kernels_cpu` が arm64 に `-mavx512*`、`-mf16c`、`-mfma` を渡す |
| CPU library 不要の host tests | **16 / 16 pass** | 以下の 4 expert/pool tests を明示的に除外。全 CPU test 成功ではない |
| native experts ON、固定 ggml、Metal/BLAS OFF、`ggml-cpu` target | **build 成功** | ggml の arm64 / NEON build。Strata native expert 実行は未確認 |
| 同構成の `strata_kernels_cpu` target | **失敗** | 同じ x86 flags の問題 |
| 同構成の `dequant_q5_1_test` | **1 / 1 pass** | 4,096 blocks、ggml reference に対し max_abs 0。IQ expert 全体の証明ではない |
| Python server unit tests の一部 | **import errors 3、Windows-only skips 3** | Python 3.10.4 に `jinja2` がない。依存はインストールしていない |
| CUDA / HIP / SYCL build、Metal kernel、full model、MTP | **未検証** | 必要な GPU / 実装 / GGUF / captures がない |

16 tests: `conversation_cache_test`、`conversation_split_failure_test`、`conversation_memory_test`、
`conversation_file_test`、`gguf_reader_test`、`gguf_split_test`、`tuning_header_arch_family_test`、
`arch_defaults_test`、`suffix_drafter_test`、`controller_test`、`draft_policy_test`、
`message_boundary_test`、`conv_cache_test`、`coupled_draft_test`、`spec_prob_test`、`bf16_bits_test`。
除外は `expert_multi_test`、`expert_parity`、`pool_test`、`pool_stress`。
除外を success や runtime skip と数えていない。
`conversation_memory_test` が通っても、Mac の memory pressure / automatic parking が利用可能とは言えない。

再現用 (repository root、build directories は gitignore 対象):

```sh
cmake -S . -B build-mac-phase0 \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_SYCL=OFF \
  -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_BUILD_CONVERSATION_TESTS=ON
cmake --build build-mac-phase0 --parallel 4
# 上の全 build は x86 flags で失敗。独立 target を build してから以下を実行した。
cmake --build build-mac-phase0 --target \
  conversation_cache_test conversation_split_failure_test conversation_memory_test \
  conversation_file_test gguf_reader_test gguf_split_test tuning_header_arch_family_test \
  arch_defaults_test suffix_drafter_test controller_test draft_policy_test \
  message_boundary_test conv_cache_test coupled_draft_test spec_prob_test bf16_bits_test
ctest --test-dir build-mac-phase0 --output-on-failure --timeout 120 \
  -E '^(expert_multi_test|expert_parity|pool_test|pool_stress)$'

cmake -S . -B build-mac-native-phase0 \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_SYCL=OFF \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_GGML_DIR="$PWD/build-mac-phase0/llama.cpp-3cf03257f219afbe7334045ff7c6a06ac68c627d" \
  -DGGML_METAL=OFF -DGGML_BLAS=OFF
cmake --build build-mac-native-phase0 --target ggml-cpu --parallel 6
cmake --build build-mac-native-phase0 --target strata_kernels_cpu --parallel 2
# 最後の CPU expert library build は失敗。
cmake --build build-mac-native-phase0 --target dequant_q5_1_test
ctest --test-dir build-mac-native-phase0 -R '^dequant_q5_1_test$' --output-on-failure
python -m unittest serve.test_runconfig serve.test_monitor serve.test_vram serve.test_winjob -v
```

local evidence: `build-mac-phase0/host-tests.log`、`python-tests.log`、
`build-mac-native-phase0/ggml-cpu-build.log`、`native-expert-build.log`、
両 build directory の `Testing/Temporary/LastTest.log`。
これらは local logs で、配布 GGUF や性能結果を文書に添付したものではない。

## 8. serve / setup の macOS 不足

| 箇所 | 問題 | 対応の段階 |
| --- | --- | --- |
| `serve/server.py`、`runconfig.py` の HTTP / subprocess / engine protocol | 基本構造は再利用できるが、Mac engine での end-to-end 未試験 | Phase 1 host checks、Phase 2 text serving smoke test |
| `serve/telemetry.py` | NVML の DLL/SO と AMD sysfs が中心。Metal GPU memory / residency reader がない。`/proc` fallback も Mac では使えない | 明示的な Metal reader / portable process metrics。利用不能な値は null / unavailable と表示 |
| `server.py::child_env` / lib dirs | HIP 以外を CUDA とする環境処理、non-Windows の `LD_LIBRARY_PATH`。Darwin framework / dylib と一致しない | Metal backend を明示し、runtime 探索を platform 別に。GPU headroom を NVIDIA の VRAM guard で判定しない |
| min-free-VRAM / reserve / expert resize controls | discrete VRAM を前提。UMA の物理メモリと GPU 使用量を二重に予算化できない | Metal で単一 budget にする。未対応操作は理由付きで拒否 |
| `serve/winjob.py` と Windows tests | Windows Job Object は Darwin で使わない | 現行 platform guard を維持。skip は skip と記録 |
| vision helper / tools build | CUDA 選択・GPU dependency の分岐がある | 初期 text-only。未対応 vision を opt-in しても黙って有効扱いにしない |
| `setup.py::ram_gb` / CPU detection | non-Windows を Linux と扱い `/proc/meminfo`、cpuinfo / AVX 前提 | installer 対応時に Darwin / arm64 分岐。今回 setup は実行しない |
| setup prebuilt / backend choices | Linux x64 または Windows binaries、cuda/hip/sycl、NVCC / ROCm / oneAPI / apt / Docker assumptions | installer は後回し。Mach-O arm64 + Metal を後で明示的に追加 |
| setup memory sizing / library search | RAM+VRAM、Linux SO search に依存 | UMA の予算、Darwin dylib / framework に対応。パックの dtype は変更しない |

現時点の Python import error は依存不足であり、それ自体を macOS 非互換とは分類しない。
Phase 1 では既存 Python dependencies を用意して server / setup の非 GPU テストを実行し、
失敗と platform skip を区別する。production dependency の追加は現時点で提案しない。

## 9. フェーズごとの出口と停止条件

各 phase の終わりで結果・未検証・リスクを報告し、次 phase を自動で始めない。

### Phase 1 — host / NEON

- フォーク URL を受け取り、この固定版に対する差分として小さな変更を行う。
- ISA flags / CPUID / intrinsics / pause / affinity を分け、arm64 の ggml native expert を build / test。
  scalar / ggml CPU reference は明示的なテスト経路であり、Metal kernel の隠れた fallback にはしない。
- host loader / sampler reference / planner / server を GPU 無しで build する。
  full engine CPU forward はこの phase の新機能として作らない。
- GGUF header と manifest、split shards、PLE、MTP の実ファイルを照合する。
  IQ2_XS の全形式、とくに IQ1_M / Q2_0 に ggml-cpu の reference が用意できるか確認。
- 非 GPU tests と CPU parity をすべて実行。AVX 専用のテストは理由を表示して skip。
  portable な CPU test を AVX のせいで一括 skip することはしない。
- CUDA/HIP/SYCL を共有ファイル変更の前後で build し、同じ fixture / default output を byte 比較。
  手元で実行できない build は未検証として報告し、完了済みの review checks と称しない。

停止条件: native type support 不足、公開されていない oracle、platform API の未処理、
既存 default の差分を説明できない場合。テスト除外だけで Phase 1 成功にしない。

### Phase 2 — opt-in Metal と最小 decode

- `STRATA_ENABLE_METAL` OFF default。unsupported selection / options は明確な error。
- 先に小さな mmap wrapping 試験と単一メモリの ownership / view / completion を確定。
- BF16 / FP16 / Q8 の丸め、IQ decode / dots、GR、GDN、QSA、router、shared / combine、
  PLE、head / greedy sampler の順に、各 kernel を独立基準で合格させる。
- shader の FP32 reduction order、FMA contraction、division、exp の契約を記録。
  高速 math を最初から追加しない。GPU 間の bitwise 一致と数値一致を混同しない。
- 短い prompt の ingest を用意する。逐次 T=1 prompt 処理なら oracle も同じ処理にする。
  既存 batched prefill と同じだと主張しない。通常 prefill の差は別に記録する。
- §6 の oracle / logits 閾値 / prompt ids を事前固定し、複数 prompt の greedy token 列一致。
  intermediate states と routed ids も保存。速度は報告しない。

仮説: GGUF native bytes を shared buffer で直接読み、同じ数学的契約を満たせる。
kill 条件: mmap の読取り・寿命が成立しない場合は一度読み込みへ切り替えて理由を記録。
どちらも memory budget を満たさない、部品 gate が通らない、token 列が違う場合は停止。
experts / top-k / dtype / 精度を変えて出口を満たしたことにしない。

### Phase 3 — MTP / lookup chain

- 同じ MTP、vocabulary mapping、lookup / suffix rules と acceptance policy を使う。
- T=1..8、全 accepted-prefix 長、途中 rejection、EOS / cancel / reset を試験。
  GDN / KV / PLE state を採択 prefix だけ更新し、draft state と target state を分ける。
- spec OFF / MTP / lookup chain の token 列一致、single/multi invariance、独立 draft capture を確認。
- 選択した sampling 契約を満たせない場合は、その mode を unsupported とし黙って変えない。

kill 条件: commit 後の state 不一致、未採択 token の漏れ、spec ON/OFF token 不一致。

### Phase 4 — 計測と最適化

- 正しさの gates が通ってから、まず Metal の通常設定で測る。
  IQ2_XS を先に、IQ3_XXS を次に、4K / 32K prompt、batch 1。
  full prefill もその時点で correctness を確認する。
- `docs/COMMUNITY_BENCHMARKS.md` に合わせ、各条件 3 回以上、比較は交互実行。
  prompt tok/s と output tok/s を分離し、中央値、全試行、範囲を保存。
  実 token 数、reasoning / EOS、context 上限、reuse と warm-up / load の扱いを記録。
- OS / compiler / Metal device / commits / GGUF hashes、sampling temperature、背景負荷、
  thermal state (取得不能なら不明)、RAM peak / page faults / swap、
  `recommendedMaxWorkingSetSize` / `maxBufferLength` / wired limit を記録。
- メモリ帯域は別の測定 workload / buffer size / CPU or GPU / cache 条件と対で報告。
  theoretical bandwidth を実測値にしない。
- GPU command timing、CPU serial work、PLE I/O、expert / KV traffic を区別する。
  physical SSD bytes と logical read bytes を混同しない。測れない counter は未測定。
- 最適化前に仮説と中止条件を書く。一度に 1 変更、同じ correctness gates を再実行。
  variation 内の差を改善と呼ばず、先読み・独自 cache・新 fusion は計測で必要性が出てから。

Strata の tok/s の予測や未検証 backend の速度は掲載しない。
§6 の Phase 0.5 / 0.6 baseline はユーザーが指定した独立 llama.cpp の実測である。

## 10. 工数が大きい順のリスク

| 順位 | リスク | 根拠 / 解消方法 |
| --- | --- | --- |
| 1 | 全演算と量子化混在形式の Metal 数値 parity | 実 IQ2_XS には IQ1_M / Q2_0 と mixed dense types がある。固定 llama.cpp でも English が 22 token 目に分岐し、full-model exact oracle の交換はできない。Phase 0.6 の MTP on/off greedy messages も 4/9 組だけ一致。部品の independent oracle と変更前 Strata captures を使い、数値契約を固定する |
| 2 | host runtime / graph / pointer table と speculation の状態管理 | CUDA 型と device addresses が engine 全体にある。buffer descriptor / submit / completion を小さな境界にし、T=1 後に accepted-prefix commit を実装 |
| 3 | prompt 処理と長い context の正しさ | 短い prompt にも ingest が必要。batched prefill / native MMQ は CUDA/HIP 依存で、逐次 decode と丸めが違う。段階を分けて比較 |
| 4 | 64 GiB UMA の実適合性、mmap / residency | llama.cpp は IQ2_XS + unsloth MTP、context 36864 / 32K prompt でも実行できた。RSS は GPU/system peak や Strata の適合性の証明ではない。IQ3 実ファイルは未取得で expert fixture だけでも single-buffer limit 超過。複数 view / lifetime を検証し、wired 設定を自動変更しない。重複した無効試行の OOM を単体モデル不適合の証拠にしない |
| 5 | oracle harness / Strata captures の不足 | CPU llama qwen4exp は読み込み・生成でき、全 logits を保存した。一方 `bench/micro` / `ref` の欠落と Strata CUDA capture 未取得は残る。CPU/Metal の実際の差を理由に基準を緩めない |
| 6 | arm64 / Darwin host portability と他 backend の回帰 | x86 source/flags 選択、CPUID/xgetbv、AVX symbol references、pause/sfence、Linux affinity / memory / O_DIRECT。#1306 の x86 guards だけでは Darwin の不足は解消しない。[file scope](../bench/results/2026-10-09-mac-phase05/ARM64_SCOPE.md) を列挙済み。修正と CUDA/HIP/SYCL build は未実施 |
| 7 | serve metrics / controls / installer | Linux/Windows assumptions、discrete VRAM budget、prebuilt x64。最初は source build / text-only、installer は後回し |
| 8 | 性能計測の再現性 | shared memory pressure / page cache / thermal / prompt reuse が変動する。条件を保存し、交互の試行で差とばらつきを比較 |

日数や Strata tok/s は見積もっていない。Phase 0.5 では配布 IQ2_XS の実ファイル確認、
固定 llama.cpp の CPU/Metal baseline、独立 read kernel、arm64 scope の調査を追加した。
Strata の Metal kernels、native expert runtime、full-model correctness、fork 上のエンジン変更、
既存 GPU backend の build / default output checks は今後の作業である。Phase 0.5 の停止後、
承認された Phase 0.6 の結果を以下へ追記した。


## 11. Phase 0.6 の実測と移植の目標

機械は §2 の Apple M5 Max / 64 GiB / macOS 27.0.1。エンジンの source は変更していない。
新しい版は v0.6.0 (peeled commit `d81235049384534c167caea52b85a694f6103d14`)。
unsloth head は `MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf`、revision
`766911a6b7369840a91dbcd95f9f997acaab6cd6`、SHA256
`58a47d6a216307e9b97172d5bfdc3436a8994b632f5e436b9bb6edf8951da8d1`。
実ファイル全体の hash / length 一致を確認した。main の GGUF は Phase 0.5 の 2 shards を再使用。
PLE の 29 GB table は CPU / mmap に残る。GPU offload は別の verbose probe で確認し、
その probe は集計から除外した。exclusive lock と GPU error 検査を使用した。

以下は **Strata ではなく llama.cpp Metal**、全有効試行の中央値。
TTFT は参照報告と同じ server prompt time。RSS は 0.5 s の process-tree samples で、
GPU を含む physical system peak ではない。全 server-session の time -l 最大 RSS も保存した。

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

4K の input count は 4,042、32K は 32,040。指定された本文に nonce / instruction / chat template
が加わる。参照との差は IQ2_XS、Metal / PLE override、**ignore_eos=true で各 256 outputs を保証**、
paired nonce、Darwin RSS、LLAMA_TRACE=1。no thinking、cache_prompt=false、seed=1、temp 0.7 では
top_k=20 / top_p=0.95。warm-up は各 server 2 requests。cache / shader caches は purge していない。
サーマル状態は終了後 nominal、各 trial の温度・thermal state は未計測。wired 設定は変更しない。
MTP の temp 0.7 はばらつきが大きく off と範囲が重なるため、普遍的な改善とは主張しない。

### 11.1 同日の短い prompt、固定版 vs v0.6.0

Phase 0.5 の exact CLI settings (raw English、12 input / 32 outputs、context 512、batch 64、
FA off、temp 0、seed 1234、8 threads、MTP off) を交互に各 3 trials で再実行した。

| Version | n | Prompt tok/s median | Decode tok/s median | Decode min - max | Peak RSS median GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| pinned | 3 | 114.83 | 53.19 | 49.96 - 53.98 | 36.988 |
| v0.6.0 | 3 | 117.95 | 53.12 | 52.53 - 53.20 | 37.001 |

固定版は `3cf03257f219afbe7334045ff7c6a06ac68c627d`。decode median の差は -0.13%。
prompt は +2.72% だが新しい版の範囲は 61.63-117.99、固定版 106.24-116.95。
両方ともばらつきの範囲内で、改善とは呼ばない。Phase 0.5 の 52.95 tok/s とのみ比較せず、
この同日再測定を比較値にした。

### 11.2 重みの読み出し量と帯域との比

`tensors.csv` の native bytes から 1 decode token あたりを計算した:

| Component | Bytes/token |
| --- | ---: |
| routed experts、48 layers x top-10/512 | 692,480,000 |
| shared expert と gate | 135,221,760 |
| dense、attention / router / HC / norms / PLE projections | 2,936,153,600 |
| head と final HC mixer | 350,863,360 |
| input embedding 1 row | 1,360 |
| 合計 (PLE rows を除く) | **4,114,720,080** |
| PLE row gathers、16 x 90 bytes | **1,440** |

MTP off の `W * decode_tok/s / 582.6e9` は条件ごとに **34.45-35.68%**。
これは native weight work と時間からの換算で、DRAM counter で測った利用率ではない。
activation / KV / state、CPU repack、tile 再読、cache reuse は含まれない。

MTP verification は実 trace の `m=1+drafted` (通常 4) を使う。
同じ重みを token ごとに読む logical operands は約 **16.4 GB/pass**、
fixed tensors と expert set を完全再利用する ideal unique reads は約 **4.115-6.192 GB/pass**。
実際の expert union / DRAM reuse を取っていないため後者は仮定付きの範囲。
採択された draft と correction/bonus を分け、first token の prompt time を除いた
**255 timed outputs** を分母にする。actual accepted/pass の中央値は条件ごとに約 1.74-2.22。
target verification の換算は no-reuse 約 **54.40-56.51%**、ideal unique-reuse 約 **13.60-21.28%**。
これは target のみ。unsloth head は single-step native read model が **829,409,952 bytes** で、
generated drafts だけでも別に約 **8.22-8.54%** 相当。catch-up forward や confidence-rejected attempts
の追加 work は含めていない。**MTP 全体の physical bandwidth 利用率は不明**で、
MTP output tok/s に単純に 4.115 GB を掛けて全利用率とはしない。
全 per-trial bytes / verification counts / 採択数 / formulas は補足報告に保存した。

### 11.3 移植の目標値

目標を **同じ workload での llama.cpp の最良 configuration median** とする。
best single trial は使わない。prompt と decode の最良 config は別なので、
この両方を同時に実現した測定ではない。

| Prompt | Temp | Prompt target tok/s (source config) | Decode target tok/s (source config) |
| --- | ---: | ---: | ---: |
| 4K | 0 | 1178.53 (off) | 60.46 (mtp3) |
| 4K | 0.7 | 1180.06 (off) | 54.80 (mtp3) |
| 32K | 0 | 1101.53 (off) | 61.25 (mtp3) |
| 32K | 0.7 | 1102.10 (off) | 55.90 (mtp3) |

**Strata の速度予測は書かない。** 目標達成の報告には先に選択した oracle の正しさゲートを通す。
MTP on/off の出力契約と数値差は未解決。Phase 1 と既存 GPU backends の build は未実施。
Phase 0.5 / 0.6 の記録だけを fork へ push し、モデル・build・大きな logits は含めず、
upstream の PR は作らない。**Phase 0.6 はここで停止する。**
