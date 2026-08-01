# quick_test_online.py:pic_a3 在「dataset 模式」vs「legacy 硬编码模式」下的缓存差异

本文对比下面两条命令里 **`pic_a3` 运行时缓存行为的区别**。两条命令唯一的差别是有没有 `--dataset`:

```bash
# 命令 A —— dataset 模式(带 --dataset)
python quick_test_online.py --modes full_recompute pic_a3 \
    --dataset hotpotqa --n-samples 8 --a3-recomp-ratio 0.15 --tp 8

# 命令 B —— legacy 硬编码模式(不带 --dataset)
python quick_test_online.py --modes full_recompute pic_a3 \
    --a3-recomp-ratio 0.15 --tp 8
```

`--dataset` 是否出现,决定走两条**完全不同的 prompt 构造 / warmup / 打分**路径(`quick_test_online.py:1260-1269`、`main()` 里 `if args.dataset:` 分支)。`pic_a3` 的服务器 flag(`--pic-enable --enable-a3 --recomp-ratio 0.15`)两边完全一样——**变的是「喂给它什么 prompt、怎么预热缓存、怎么判分」**。

> 命令 B(legacy)下 `--n-samples` 被忽略(它只跑一个固定 prompt)。

---

## 0. 一句话结论

- **命令 B(legacy)**:用**固定的合成文档**(cats/dogs/birds),**3 个手工 warmup**把 SYS+C1+C2+C3 全部预缓存,**跑一次**测试 prompt(全命中),用 **FDT(首个分歧 token)** 判是否和全量重算一致。—— 面向「机制是否正确」。
- **命令 A(dataset)**:用**每条真实 hotpotqa 样本自己的文档**,**逐样本 `prime_sample`**(每个 chunk 一个 warmup 请求)预缓存该样本的 chunk,**跑 8 次**(每样本一次),用 **F1** 对真实答案打分。—— 面向「精度到底多少」。

---

## 1. 并排对比

| 维度 | 命令 B:legacy 硬编码 | 命令 A:dataset(hotpotqa) |
|---|---|---|
| prompt 构造 | 固定字符串(`quick_test_online.py:75-118`) | 逐样本 `build_prompt(chunks, query)`(`:872`) |
| SYS 段 | `"You are a helpful AI assistant. "×9`(精确 64 tok) | `"Read the following documents and answer…"`(`:859`) |
| 文档内容 | `C1=cats×800`,`C2=dogs×800`,`C3=birds×800`(各 ~4032 tok,**3 个独立段**) | 每条样本自己的真实 hotpotqa 文档——**~10-12 篇维基段落合并成 1 个 chunk**(`hotpotqa.py:54` `chunks=["\n\n".join(all_docs)]`) |
| warmup 方式 | 3 个固定多文档 prompt:`[pic_w1, pic_w2, pic_w3]`(`:1568`) | 逐样本 `prime_sample(chunks)`,**每 chunk 一个请求**(`:888`) |
| warmup 预缓存了谁 | SYS、C1、C2、C3(一次性) | SYS(共享,只缓存一次)+ 当前样本的每个 chunk |
| 测试请求数 | **1** 次 | **N=8** 次(每样本 1 次) |
| 单次测试命中 | SYS+C1+C2+C3 全命中,只有 Q fresh(4/5 段) | 该样本的 chunk 命中,query fresh(1-chunk 样本 = 2/3 段) |
| 跨请求缓存累积 | 无(warmup 完只测一次) | 有(服务器缓存跨样本累积;SYS 全程复用,chunk 逐样本各不相同) |
| 判分 | FDT(与 full_recompute 逐 token 比)+ 文本预览 | F1(答案 vs ground_truth),8 样本求均值 |
| 目的 | 机制正确性(受控合成场景) | 真实 QA 精度 |

---

## 2. 命令 B(legacy)缓存时间线

固定 prompt(`:116-118`,`:1416-1423` 做 64 对齐):

```
PIC_PROMPT = SYS <SEP> C1 <SEP> C2 <SEP> C3 <SEP> Q     # 测试
PIC_W1     = SYS <SEP> C1 <SEP> Q                        # warmup1
PIC_W2     = SYS <SEP> C2 <SEP> Q                        # warmup2
PIC_W3     = SYS <SEP> C3 <SEP> Q                        # warmup3
```

`pic_a3` 的 `warmup_prompts = [pic_w1, pic_w2, pic_w3]`(`:1568`,`run_mode` 逐个发,`:538-540`),再发 1 次 `PIC_PROMPT` 测 TTFT + 正确性:

```
warmup1  SYS+C1+Q   → SYS 未缓存 → 无 hit 段 → normal path → 缓存 SYS、C1
warmup2  SYS+C2+Q   → SYS 命中   → 有 hit 段 → 新路径     → 缓存 C2
warmup3  SYS+C3+Q   → SYS 命中   → 有 hit 段 → 新路径     → 缓存 C3
test     SYS+C1+C2+C3+Q → SYS/C1/C2/C3 全命中,Q fresh → 测 FDT
```

段布局:5 段(SYS、C1、C2、C3、Q),非末段 4 个可缓存,测试时全部命中;末段 Q 永不缓存、每次现算。判分只有**这一个数据点**。

---

## 3. 命令 A(dataset)缓存时间线

每个样本(`run_mode_dataset`,`:847` 循环):`sys_padded`(`:859`)+ `spec.chunks`(该样本真实文档,`:862`)+ `query_padded`(`:870`),用 `build_prompt` 拼成 `SYS<SEP>chunk1<SEP>…<SEP>chunkK<SEP>query`。

> **注意 hotpotqa 的 chunk 结构**:适配器把一条问题的 ~10-12 篇维基文档(gold 支撑 + distractor)**全部合并成 1 个 chunk**(`hotpotqa.py:54`,`chunks=["\n\n".join(all_docs)]`)。所以 hotpotqa 恒为 **1 个文档段**(不是没文档,也不是 1 篇文档,而是「12 篇合并成 1 段」),PIC 段布局 = `SYS + [合并文档块] + query` = 3 段 → 命中 `2/3`。每条问题的文档组合各不相同,**跨样本只有 SYS 复用**,文档段的命中全靠下面的逐样本 `prime_sample` 自预热。

warmup 用 `prime_sample`(`:888`;实现见 `python/sglang/test/pic_bench_lite/warmup.py`):**对每个 chunk 发一个** `SYS <SEP> chunk_i <SEP> "warmup"` 请求。之后测试请求命中这些 chunk。

服务器缓存**跨样本持续存在**,所以时间线(以每样本 1 chunk 为例)是:

```
样本1  prime: SYS+chunk1+warmup → SYS 未缓存 → 无 hit → normal path → 缓存 SYS、chunk1
       test : SYS+chunk1+query  → SYS/chunk1 命中,query fresh → F1
样本2  prime: SYS+chunk2+warmup → SYS 命中 → 新路径 → 缓存 chunk2
       test : SYS+chunk2+query  → 命中 chunk2 → F1
…
样本8  prime: SYS+chunk8+warmup → SYS 命中 → 新路径 → 缓存 chunk8
       test : SYS+chunk8+query  → 命中 chunk8 → F1
```

关键点:
- **SYS 只在样本1第一次 prime 时缓存一次**,之后所有 prime/test 都命中同一个 SYS 段。
- **每个样本的 chunk 各不相同**,逐样本新缓存,累积在服务器里(不 evict 的话)。
- 除了样本1的第一个 prime(SYS 还没缓存 → normal path),**其余所有 chunk 都是在「SYS 已命中 → 新路径」的 forward 里被缓存的**。

---

## 4. 缓存路径与「本次 bug/fix」的交互(重要)

pic_a3 有两条缓存写入路径,取决于该请求**有没有命中段**:
- **无 hit 段** → `pic_a3_new_path=False` → normal path → miss 段 public KV **正常回写**。
- **有 hit 段** → `pic_a3_new_path=True` → 新路径 → miss 段 public KV 回写。**本次修复前,新路径这一步是坏的(写全零)**,详见 `scripts/PIC_A3_L2PLUS_WRITEBACK_BUGFIX.md`。

把它套到上面两条时间线,就能解释**修复前**两种模式各自的崩法:

**命令 B(legacy)修复前**:
- C1 由 warmup1 缓存(那次没 hit → normal path)→ **C1 复用正确**。
- C2、C3 由 warmup2/3 缓存(SYS 已命中 → 新路径)→ **C2、C3 public 全零 → 复用崩**。
- 测试 prompt 里 C2/C3 命中到零 K → 输出崩。

**命令 A(dataset)修复前**:
- 样本1 的 chunk 由第一个 prime 缓存(SYS 尚未缓存 → normal path)→ **样本1 正确**。
- 样本2-8 的 chunk 都在「SYS 已命中 → 新路径」下缓存 → **public 全零 → 复用崩**。
- 这正好解释了修复前日志里的精确现象:**样本1 F1=1.000,样本2-8 F1≈0**(见 `/tmp/base_r*.log`)——不是随机崩,是「第一个进缓存的段走了 normal path 侥幸正确,其余全走新路径全坏」。

**修复后**:两条路径的 miss→public 回写都正确 → 两种模式所有段复用都对(命令 A 的 pic_a3 F1 0.13→0.622)。

> 一句话:**两种模式的缓存「结构」不同,但都因为「SYS 很早就进了缓存,导致之后几乎所有段都在新路径下被缓存」而在修复前集体崩塌;区别只是「哪一个段侥幸走了 normal path」。**

---

## 5. 何时用哪个

- **命令 B(legacy)**:快速验证「PIC 机制在受控合成 prompt 上是否与全量重算逐 token 一致」。文档巨大(每段 ~4032 tok)、命中率拉满,适合看机制对错、看 TTFT 上限、做 K 级诊断(`scripts/pic_a3_layer01_error.py` 就是基于这套 SYS/C1/C2/C3 布局)。
- **命令 A(dataset)**:测真实任务精度(F1)。文档是真实 hotpotqa、逐样本预热,更贴近生产;`--n-samples` 越大 F1 越稳。

---

## 6. 关键代码位置

| 作用 | 位置 |
|---|---|
| 硬编码 SYS/C1/C2/C3/Q、PIC_PROMPT/W1/W3 | `quick_test_online.py:75-118` |
| legacy 单 prompt 跑法(warmup→TTFT→FDT) | `quick_test_online.py:475`(`run_mode`) |
| legacy `pic_a3` 的 warmup=[w1,w2,w3] | `quick_test_online.py:1555-1570` |
| dataset 逐样本跑法(prime→test→F1) | `quick_test_online.py:701`(`run_mode_dataset`) |
| dataset 逐样本 prompt 构造 | `quick_test_online.py:859-878`(`build_prompt`) |
| dataset warmup(每 chunk 一请求) | `quick_test_online.py:888` + `python/sglang/test/pic_bench_lite/warmup.py`(`prime_sample`) |
| 分支切换(有无 --dataset) | `quick_test_online.py` `main()` 的 `if args.dataset:` |
| 两条缓存写入路径的 bug/fix | `scripts/PIC_A3_L2PLUS_WRITEBACK_BUGFIX.md` |

---

## 更新(2026-07-20):pool 已移植,默认对齐 pic_bench

本文前面「我们的测试没有 pool」已过时 —— 现已把 pic_bench 的 Zipf distractor 池移植进 `--dataset` 路径,默认开启:

- 新增 `python/sglang/test/pic_bench_lite/pool.py`(`fill_distractor_pool`,`pic_bench/runner.py:238-384` 的忠实移植;参数一致 min8192/max16384/pool200/zipf1.0)。
- `quick_test_online.py`:新增 `--dataset-pool`(**默认开**)/`--no-dataset-pool`、`--pool-min-tokens`/`--pool-max-tokens`/`--seed`;`main()` 一次性建池(mode 无关,公平);`run_mode_dataset` 用 `pool_chunks + own` shuffle 拼 measure prompt,并在 measure loop 前对所有样本的自有 chunk 做 pre-warm(保证 distractor 命中)。
- 实测(N=20,pool 默认开):每请求 `ptok≈8.2–8.9K`、平均 5.3 个 distractor、pic_a3 PIC 命中 99.3%(7.3/8.3 段)、pic_a3 F1=0.602、TTFT 相对 full_recompute **4.19×**。已进入 pic_bench 的多文档共享缓存 regime。
- 复现 pic_bench 的 200-pool 规模建议 `--n-samples ≥ 50`(池从已载入样本构建);`--no-dataset-pool` 可退回本文前面描述的单文档行为。

---

*基于 2026-07-20 QS1J(GLM-5.2-FP8)上的代码与实测。*
