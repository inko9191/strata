# Phase 0.7: llama.cpp Metal の時間内訳

2026-10-09。**Strata の性能測定ではない**。Strata / llama.cpp の source と、Phase 0.6 の
original executable / libraries は変更していない。外部診断を別ディレクトリへ build / link した。
Phase 1 は未着手。raw trace / XML / TSV、モデル、build、大きな logits は Git に含めない。

## 条件と集計

- Mac Studio、Apple M5 Max、64 GiB (68,719,476,736 bytes)、macOS 27.0.1 (26A434)、arm64。
- Xcode 26.4.1 (17E202)、Instruments 16.0 (17E202)、Apple Clang 21.0.0.21000099。
- llama.cpp v0.6.0、commit `d81235049384534c167caea52b85a694f6103d14`。
  [source-integrity.json](source-integrity.json) は固定 archive と 3,687 files の byte hash 一致を確認。
- Phase 0.5 の IQ2_XS GGUF 2 shards。ファイル revision / SHA256 / inventory は
  [Phase 0.5](../2026-10-09-mac-phase05/README.md)、MTP Q8_0 head は
  [Phase 0.6](../2026-10-09-mac-phase06/README.md) と同一。PLE table は CPU / mmap。
  パック名 IQ2_XS に対して routed tensor の実型は IQ2_S / Q2_0 / IQ2_XXS / IQ1_M。
- Phase 0.6 と同じ `prompt-4k.txt` 本文、固定 prefix `[bench mac-phase07 4k]` と同じ末尾 instruction。
  chat template / no thinking を含め **4,034 input tokens**。temp 0、seed 1、ignore_eos=true、
  cache_prompt=false。短い context の batch 1 decode。server の全 argv は各 trial JSON に保存。
- MTP off と `draft-mtp`, n_max=3。CPU PLE override、ngl=99、FA on、Q8_0 KV、8 threads、
  context 36,864、batch/ubatch 2,048、lazy-mode on。2 warm-up requests、page/shader caches は purge しない。
- Passive 診断は各 3 trials / 64 outputs、stage 診断は各 3 trials / 32 outputs。
  mode 内で off/mtp3、mtp3/off、off/mtp3 と交互に行った。before/profiled/after は同じ prompt。
  off の最初の出力は prefill に含まれ、64 outputs の **63 decode graphs**を集計。
  MTP は first m=4 prompt tail を除いた **25 full-width verifications/trial**。
  stage は off 31 graphs / MTP 9 full-width verifications/trial、m=1/2 tail は比較から除く。
- 各 trial 内の median を先に計算し、3 trial medians の median / min / max を記載。
  token を独立試行として pool しない。各区分の median の和は全体 median と一致しない。
- `reference.lock` を全 runner / diagnostic rebuild で共有。重複 runner の拒否を
  [lock-check.json](lock-check.json) で確認。完了した採用試行で GPU error は検出されていない。
- before/profiled/after の生成 **text** は一致。chat API は token-ID 列を返しておらず、
  この検査を token-ID / logits parity と呼ばない。MTP on/off parity は別の未解決事項。
- recommendedMaxWorkingSetSize=60,817,408,000、maxBufferLength=41,747,087,360、
  `iogpu.wired_limit_mb=58000`。システム設定は変更していない。温度 (℃) は未計測。

## 通常の graph 構成を保った GPU 時間

外部 `metal_probe.mm` の passive mode は command encoder の構成を変えず、実際に選択された
pipeline と公開 API `GPUStartTime` / `GPUEndTime` を記録する。関数の呼び出しは version 固定の
DYLD interpose を用い、ggml-metal / ggml-cpu の **同じ object files** を `-Wl,-interposable` だけ
追加して別 output へ relink した。engine の source patch / 再コンパイルは行わない。

| 区間 | 単位 | median ms | trial median min–max ms |
| --- | --- | ---: | ---: |
| MTP off main GPU graph span | 1 decode token | 19.705750 | 19.540750–19.800083 |
| MTP off command buffer interval union | 1 decode token | 17.955083 | 17.471333–17.981042 |
| MTP off command buffer 間の隙間 | 1 decode token | 1.687542 | 1.595792–2.244125 |
| MTP off CPU encode/submit wall time | 1 graph | 3.822459 | 3.729500–4.419667 |
| MTP main GPU graph span | 1 m=4 verification | 39.615250 | 38.334750–39.681542 |
| MTP main command buffer interval union | 1 m=4 verification | 39.279292 | 38.334125–39.427000 |
| MTP CPU encode/submit wall time | 1 m=4 graph | 4.399208 | 3.400625–4.538917 |

ratio は **2.01034**。4 tokens を検証する graph と single token graph の比較で、
accepted token 当たりの時間ではない。draft head / catch-up / sampling の時間もこの ratio から除く。
CPU encode は GPU と overlap するので GPU 時間へ足さない。CB interval には kernel 間の隙間も含む。
passive off の server decode は before 20.474 / profiled 21.836 / after 20.473 ms/output。
記録処理にも負荷があり、無介入の tok/s benchmark と同一ではない。

CPU graph の passive median は off **0.031333 ms**、m=4 **0.059167 ms**。
PLE と input embedding の gather を含む CPU graph 全体で、PLE だけの時間ではない。
native row operands は off PLE **1,440 bytes** + embedding **1,360 bytes**、
m=4 で **5,760 + 5,440 bytes**。page faults / physical SSD reads / cache hits はこの値から分からない。

## Instruments と shader の時間

標準 Metal System Trace の unmodified original-build pilot では 15 decode graphs を検査した。
main GPU span median **19.847292 ms**、active Compute interval union **17.432584 ms**、
その span 内の no-active-Compute **1.998416 ms**、graph 間 **0.976792 ms**。
[小さい区間記録](instruments-decode-summary.json) に全 15 graphs を保存した。
GPU Compute interval は driver が報告する execution interval で、ALU occupancy や DRAM utilization ではない。
CPU 処理 / command issuance / driver wait へ隙間を因果的に分割することは、この集計だけではできない。

初期 template は counter profile が M5 Max に対応せず、Shader Timeline が空だった。
その後 Instruments UI で **Performance Limiters、Shader Timeline ON、Deferred、Default performance state**
を選び、M5 用 counter profile=13 の template を保存した。エージェントから installer は実行していない。
正しい設定では shader rows が取得できた。設定は [instruments-settings.json](instruments-settings.json)。
**Sample Duration は sampled shader work であり、dispatch の網羅的な wall time ではない**。
足りない時間を比例配分して graph span を埋めたり、そこから実効 DRAM bandwidth を算出したりしない。
HC / head / PLE / router は同じ matvec pipeline を共有するため、shader 名だけで全 semantic roles は分離できない。
追加の通常実行は各 3 trials / 32 outputs。off/mtp3、mtp3/off、off/mtp3 と交互に実行し、
off は 31 decode graphs/trial、MTP は 9 m=4 verifications/trial。最後の2 trials は12秒、それ以前は25秒の recording。
request 全体が capture 内にあることを CB counts / timing / draft events で検査した。
`shader-off-r3` は保存 timeout で除外し、fresh label `shader-off-r3-retry` に取り直した。
[除外記録](measurement-failures.json)、[全6 trialsの集計](instruments-summary.json)、
[shader samples CSV](shader-samples.csv)、各 `shader-*-instruments.json` を参照。
最初の追加 off trial 中は別 trace の XML export が一部重なった。CPU encoding / gaps に対する交絡として残し、
無負荷での精密な throughput や serial 部分の比較には使わない。それ以降の大きい export は inference 完了後に行った。

| 通常 encoder の Instruments 区間 | off single median ms (trial min–max) | MTP m=4 median ms (trial min–max) |
| --- | ---: | ---: |
| main GPU span | 20.384 (20.132–20.720) | 39.794 (39.727–40.190) |
| active Compute union | 18.015 (17.676–18.221) | 38.670 (38.602–38.956) |
| span 内で対象processの Compute が観測されない時間 | 2.564 (2.413–2.675) | 1.298 (1.094–1.310) |
| CPU CB encoding wall interval union | 4.763 (4.645–4.822) | 5.400 (5.325–5.421) |
| shader Sample Duration の和 | 1.161 (1.105–1.176) | 2.616 (2.613–3.015) |

span ratio は **1.95217**。off の graph 間 no-active-Compute は **1.178 ms**。
これらの gaps は対象processの Compute の非観測区間で、GPU 全体の idle を意味しない。
encoder 内の kernel 間の隙間はこの粒度では分離できない。CPU encoding は GPU と overlap する。
全6 traces は thermal state **Nominal**、induced performance state なし、温度 (℃) は未計測。
server off の before/profiled/after は **20.364 / 21.706 / 20.786 ms/output**、
MTP は **16.322 / 17.136 / 16.361 ms/output**。計測の負荷と揺れを含む。

shader samples では IQ4_XS matvec (head と attention/GDN の投影を完全に分離不能) の median が off **0.625 ms**、
m=4 **1.616 ms**。routed expert matvec は **0.141 / 0.465 ms**、BF16 matvec は **0.053 / 0.138 ms**。
これらは sampled time。IQ3_S 等の sample が0でも実行時間が0という意味ではなく、passive pipeline list に実行が記録されている。
sample の和が graph span の約6%に留まることから、全区分 wall time をここから復元しない。
normal full-width verifications は各 trial **20 accepted drafts / 27 proposed drafts**。
採択数を verification ごとの JSON に対応付けたが、expert union / physical traffic との対応は未取得。

## 各区分の timestamp 診断 (通常実行の時間ではない)

この M5 Max の公開 counter sampling は atStageBoundary をサポートし、atDispatchBoundary はサポートしない。
そのため stage mode では **既存の fused op group ごとに encoder を区切る**。
演算 / routing / batch / 型を変えず、stage attachment の timestamps を 2 回の CPU/GPU timestamp sample で
CPU nanoseconds に較正する。これは dependency / concurrency / cache / encoder overhead を変える診断である。
timestamp の較正は [Apple の手順](https://developer.apple.com/documentation/metal/converting-gpu-timestamps-into-cpu-time)を参照。
shader sampling の解釈は [Apple の profiling 解説](https://developer.apple.com/documentation/xcode/optimizing-gpu-performance)を参照。

その影響は大きい。off の server 時間は before **20.466** / profiled **74.625** / after **20.420 ms/output**。
MTP は **16.988 / 33.039 / 17.114 ms/output**。従って下表を通常実行の部品別時間や改善余地の % と扱わない。
全 op stage timestamps の median 合計は single **24.912966 ms**、m=4 **47.264707 ms**。
一部の空 CB は GPU start timestamp が 0 で、その CB span は欠測として除外した。
stage の incomplete graph span を ratio の計算に使わない。

native bytes は GGUF の論理的な重み operand。GB/s 列は **native weights / diagnostic stage time** の換算で、
DRAM counter の実測ではない。activation / KV / GDN state、再読、LUT、cache、tile traffic を含まない。
GDN / QSA の weight-only bytes は norm / parameter の部分だけで、その処理の traffic を表さないので換算は「不明」。
その他 0 bytes は分類された native weights がないという意味で、実際の転送が 0 という意味ではない。

| 区分 | single stage ms | m=4 stage ms | native weights bytes/single | weight-equivalent GB/s (診断のみ) |
| --- | ---: | ---: | ---: | ---: |
| GDN | 0.805 | 1.523 | 5,930,496 | 不明 |
| HC | 4.945 | 11.505 | 1,270,087,680 | 256.8 |
| PLE GPU | 0.209 | 0.227 | 65,740,800 | 314.7 |
| QSA | 1.491 | 2.353 | 36,864 | 不明 |
| head | 0.636 | 1.633 | 350,863,360 | 551.3 |
| non-expert matvec BF16 | 0.894 | 1.662 | 182,845,440 | 204.4 |
| non-expert matvec IQ3_S | 0.568 | 1.387 | 146,995,200 | 258.9 |
| non-expert matvec IQ4_XS | 3.163 | 6.384 | 1,191,403,520 | 376.7 |
| non-expert matvec Q6_K | 0.228 | 0.239 | 73,113,600 | 320.3 |
| other | 5.632 | 7.171 | 0 | 不明 |
| routed experts | 4.570 | 10.524 | 692,480,000 | 151.5 |
| shared expert | 1.823 | 2.432 | 135,221,760 | 74.2 |

物理的な読み出しバイト数・GB/s は **全区分で未計測**。通常経路の各区分 wall time も未確定。
行列は tensor 名を inventory と照合して分類する。匿名中間ノードは graph の並べ替えで意味を特定できないため
other に残す。従って HC / GDN / QSA 等は識別できた op の部分集計で、完全な区分時間ではない。
fused op は細分せず先頭 op の区分へ割り当てる。
[CSV](diagnostic-category-times.csv)、各 `final-stage-*-op-audit.csv` と
[集計コード](summarize.py) で分類を確認できる。time / native byte totals の意味は異なり、配分に精度の限界がある。

## 非エキスパート dense 2.94 GB の inventory

shared expert / head / input embedding / PLE table を除いた **2,936,153,600 bytes**。
shared と head は別表、PLE の key/value projections / conv / norms はこの dense 内。
45 kind/type rows は [dense-by-kind-type.csv](dense-by-kind-type.csv)。すべて native bytes、再量子化しない。

| 型 | bytes |
| --- | ---: |
| BF16 | 1,514,536,960 |
| IQ4_XS | 1,191,403,520 |
| IQ3_S | 146,995,200 |
| Q6_K | 73,113,600 |
| F32 | 10,022,400 |
| F16 | 81,920 |

この dense 集合に Q8_0 はない。観測された Q8_0 matvec / MMA は shared 等の別集合。
F16 は PLE conv 81,920 bytes。HC attn/FFN の down/up 4種類だけで **1,258,291,200 bytes (42.855%)**。

| Tensor kind | 型 | bytes |
| --- | --- | ---: |
| attn_gate.weight | IQ3_S | 74,342,400 |
| attn_gate.weight | IQ4_XS | 183,828,480 |
| attn_gate.weight | Q6_K | 38,707,200 |
| attn_k.weight | IQ3_S | 563,200 |
| attn_k.weight | IQ4_XS | 5,570,560 |
| attn_k.weight | Q6_K | 3,225,600 |
| attn_k_norm.weight | F32 | 12,288 |
| attn_output.weight | IQ3_S | 6,758,400 |
| attn_output.weight | IQ4_XS | 83,558,400 |
| attn_output.weight | Q6_K | 12,902,400 |
| attn_q.weight | IQ3_S | 13,516,800 |
| attn_q.weight | IQ4_XS | 183,828,480 |
| attn_q_norm.weight | F32 | 12,288 |
| attn_qkv.weight | IQ3_S | 11,264,000 |
| attn_qkv.weight | IQ4_XS | 487,424,000 |
| attn_v.weight | IQ4_XS | 4,874,240 |
| attn_v.weight | Q6_K | 5,376,000 |
| ffn_gate_inp.weight | BF16 | 125,829,120 |
| hc_attn_down.weight | BF16 | 314,572,800 |
| hc_attn_inject.weight | BF16 | 3,932,160 |
| hc_attn_norm.weight | F32 | 1,966,080 |
| hc_attn_up.weight | BF16 | 314,572,800 |
| hc_ffn_down.weight | BF16 | 314,572,800 |
| hc_ffn_inject.weight | BF16 | 3,932,160 |
| hc_ffn_norm.weight | F32 | 1,966,080 |
| hc_ffn_up.weight | BF16 | 314,572,800 |
| indexer.k_norm.weight | F32 | 6,144 |
| indexer.k_proj.weight | BF16 | 7,864,320 |
| indexer.q_norm.weight | F32 | 6,144 |
| indexer.q_proj.weight | BF16 | 31,457,280 |
| ple_conv1d.weight | F16 | 81,920 |
| ple_key.weight | BF16 | 52,428,800 |
| ple_norm_conv.weight | F32 | 40,960 |
| ple_norm_key.weight | F32 | 40,960 |
| ple_norm_query.weight | F32 | 40,960 |
| ple_value.weight | BF16 | 13,107,200 |
| ssm_a | F32 | 6,912 |
| ssm_alpha.weight | BF16 | 8,847,360 |
| ssm_beta.weight | BF16 | 8,847,360 |
| ssm_conv1d.weight | F32 | 5,898,240 |
| ssm_dt.bias | F32 | 6,912 |
| ssm_norm.weight | F32 | 18,432 |
| ssm_out.weight | IQ3_S | 40,550,400 |
| ssm_out.weight | IQ4_XS | 242,319,360 |
| ssm_out.weight | Q6_K | 12,902,400 |

## MTP の kernel と再利用

実際の選択 pipeline は passive traces から取得した。m=4 target は 1 回の main graph、4 回の host forward ではない。
`kernel_mul_mm*` は **m=4 verification では観測されない**。
[pinned dispatch predicates](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/ggml/src/ggml-metal/ggml-metal-common.cpp#L35)
では通常 MM は src1 columns >8、expert MM は >=32 が条件。

| 実際の kernel family | m=4 での扱い |
| --- | --- |
| `mul_mv_ext_bf16...r1_4`, 一部 IQ4_NL / Q2_0 ext | 1 decoded weight vector と最大4 input rows を同じ内側 loop で使う、明示的な再利用 |
| `mul_mv_mma_q6_K`, `mul_mv_mma_q8_0` | few-row MMA。`mul_mm` という名前ではなくても複数 columns をまとめる |
| generic `mul_mv_iq4_xs`, `mul_mv_iq3_s` | input row を grid で別々に処理。4-token register-level reuse はこの実装にない |
| expert `mul_mv_id_iq2_s/q2_0/iq2_xxs/iq1_m` | token x selected-expert ごとの grid、内部 matvec は1 row。top-10 は変えない |

[ext / expert shader](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/ggml/src/ggml-metal/kernels/mul_mv.metal)
の r1ptg loop (L602–700) と iid1 / inner ne11=1 (L3295 以降) を確認した。
「mul_mv という名前だから重み共有がない」とは言えない。generic/expert の cache による物理再利用は不明。
output.weight では src1 ne[1]=1 の observation もあるため、lazy-mode の head selection を無視して全 head を4回読むと断定しない。
probe は全 src1 dimensions / expert router IDs を保存していない。

MTP verification の native no-reuse operand model は Phase 0.6 の約16.46 GB/pass、ideal union reuse は約4.115–6.192 GB/pass。
今回の pipeline 観測により、その両端のいずれも physical traffic の実測として採用できないと分かった。
normal m=4 39.62 ms と 582.6 GB/s を使って「実際の帯域利用率」を埋めない。
検証1回の GPU時間、draft の採択数、出力数は別の量であり、MTP 全体を main graph の ratio だけでは説明できない。

## 結論と不足する証拠

**kernel 改善だけで十分か、Strata の構成が必要かは断定できない。**
1.26 GB の HC BF16 行列と、1.19 GB のその他 IQ4_XS 行列は調査対象として大きい。
ただし stage 診断の ranking を通常実行の ranking と置き換えると誤る。
llama.cpp はすでに m=4 target graph、few-row weight reuse、HC pre/post、routing/reduction、norm 等の融合を持つ。
したがって「llama.cpp は token ごとに forward を繰り返すだけで、Strata のみが batch/fusion を持つ」は誤り。

必要な追加証拠は、encoder を変えない区分別の elapsed GPU time、physical read traffic と stall counters の区分別対応、
全 operand shapes / head selection、検証ごとの expert union / physical traffic と accepted tokens の対応。
これらが欠ける間は、kernel だけの仮説 / Strata 構成必須の仮説を採択しない。
最適化は導入せず、Strata の速度予測は書かない。通常経路の完全な区分別 timing と実効 bandwidth は未検証として残る。

## 再現

同じ archive source / Phase 0.6 build / GGUF を用意して repository root から実行する。
必要なのは既存の Xcode / Metal / Python stdlib。新しい production dependency はない。

```sh
python3 bench/results/2026-10-09-mac-phase07/build_probe.py
python3 bench/results/2026-10-09-mac-phase07/run_trace.py off final-passive-off-r1 --probe passive --outputs 64
python3 bench/results/2026-10-09-mac-phase07/run_trace.py mtp3 final-passive-mtp3-r1 --probe passive --outputs 64
python3 bench/results/2026-10-09-mac-phase07/run_trace.py off final-stage-off-r1 --probe stage --outputs 32
python3 bench/results/2026-10-09-mac-phase07/run_trace.py mtp3 final-stage-mtp3-r1 --probe stage --outputs 32
# R2 は mtp3/off、R3 は off/mtp3。全12 valid records / TSV が揃ってから集計。
python3 bench/results/2026-10-09-mac-phase07/summarize.py
```

Instruments は上述の設定を UI で選び Save as Template。

```sh
python3 bench/results/2026-10-09-mac-phase07/run_trace.py off shader-off-r1 --outputs 32 --template build-mac-phase07/Phase07-M5-Shader.tracetemplate
python3 bench/results/2026-10-09-mac-phase07/analyze_instruments.py shader-off-r1
```

source / model / argv が変われば新しい label を用いる。生 trace の export は inference の実行中を避け、全 capture 完了後に行う。
小さい JSON / CSV と tools のみ公開し、大きい trace / XML / TSV は local に残す。
