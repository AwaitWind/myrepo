# pic_a3_oracle 测试脚本参数速查(开 / 不开的区别)

> 脚本:`quick_test_online.py`。测试机 QS1J,GLM-5.2-FP8,`--tp 8`。
> 环境变量分两类:**shell 里 export**(经 `os.environ.copy()` 传到 server + TP worker)
> 和 **脚本 mode 循环自动设置**(pic_a3_oracle 模式)。CLI 参数直接跟在命令后。
> 结论标注:✅=数据集上有用 / ⚠️=诊断用 / ❌=会崩或无益。

---

## 0. 一句话背景(理解下面参数的前提)

- **pic_a3**(单层):layer-1 挑 15% imp + clip,深层不再重选 → 数据集上稳(~0.8,4.16×)。
- **pic_a3_oracle**(多层):在 pic_a3 之上,再在 `--oracle-layers` 的深层做 keepalive 重选 +
  隔离 hidden 捕获/注入。聚焦内容上赢 pic_a3;数据集重 pool 上会崩(见下)。

---

## 1. 模式 & 数据集(CLI 参数)

| 参数 | 默认 | 开启 / 设值 | 不设 |
|---|---|---|---|
| `--modes` | full_recompute prefix_cache pic | 指定要跑的模式(可多选:`full_recompute pic_a3 pic_a3_oracle`) | — |
| `--dataset hotpotqa` | 无(合成) | 走真实数据集(N 样本 + F1 打分),warmup 用 `prime_sample` | 走合成 3-doc 固定 prompt + FDT 判定 |
| `--n-samples` | 10 | 数据集每模式跑几个样本 | — |
| `--dataset-pool` / `--no-dataset-pool` | 开 | 每样本填 distractor 池到 8-16K(多段长上下文) | 单段(合并 chunk,~1.4K) |
| `--pool-max-tokens N` | 数据集默认(hotpotqa 16384) | 限制 pool 填充上限(调小→小 pool,让 oracle 落进可用区) | 用数据集默认(大) |
| `--seed` | 0 | pool 构建 + 每样本 shuffle 的种子(跨模式确定性) | — |

---

## 2. pic_a3_oracle 核心开关(脚本 mode 循环**自动**设置)

> 跑 `--modes pic_a3_oracle` 时,脚本自动 export 下面这些(除非你用诊断开关覆盖)。

| 环境变量 | oracle 模式默认 | 开启效果 | 不开启效果 |
|---|---|---|---|
| `SGLANG_PIC_A3_KEEP_ALIVE` | =1(自动) | 深层 keepalive 窗口化重选(多层 oracle 生效) | 退化成单层 pic_a3(只 layer-1) |
| `SGLANG_PIC_A3_ORACLE` | =1(自动) | warmup 捕获隔离 hidden、measure 命中即注入 | 只有 keepalive,无隔离 inject |
| `SGLANG_PIC_A3_CLIP_CAPTURE` | =1(自动) | Route-A clip:每窗口收窄到 miss∪imp → 加速(~3.5×) | full-length keepalive(不 clip,零加速) |

> 三者都需要 `--oracle-layers` 有多个值(≥2)才真正进多层 Phase B;`--oracle-layers 1`(单值)
> 会让 `pic_a3_multiselect=False` → 上面三个即使 =1 也不生效 → 等价单层 pic_a3。

---

## 3. imp 选择策略(决定"重算哪 15%")

| 环境变量 | 默认 | 开启效果 | 不开启效果 | 备注 |
|---|---|---|---|---|
| `SGLANG_PIC_A3_HIT_ONLY_IMP` | 关 | **每段各 15%** 预算(page-align 64),保证每段都分到名额 | **全局 top-15%**(从全部 hit 一起选) | 数据集:oracle 0.000→0.493(防崩),但 pic_a3 0.8→0.29(伤单层) |
| `SGLANG_PIC_A3_HYBRID_IMP` | 关 | layer-1 全局 + 深层每段 | 各层一致(按 HIT_ONLY_IMP) | ❌ 实测更糟(跨层 imp 不一致→崩 0.025) |
| `--recomp-ratio R` | 0.15 | 重算比例(0.05~0.30);调大→重算更多、更准但更慢 | 0.15 | 全局/每段都按这个比例 |

> 默认(两个都关)= **全局 top-15% 从全部 hit 选**(= 你要的那个,也是 pic_a3 用的)。

---

## 4. imp 平滑(ragkv avg_pool)

| 环境变量 | 默认 | 开启效果 | 不开启效果 |
|---|---|---|---|
| `SGLANG_PIC_A3_IMP_SMOOTH_K` | 5 | topk 前对 question→context 的注意力打分做 `avg_pool1d(k)` 平滑(倾向选连续答案片段,对齐 ragkv);覆盖 pic_a3 + oracle 的全局 imp | K<=1 关闭(选零散尖峰,原行为) |

> ⚠️ 原「自动降级 `SGLANG_PIC_A3_KEEPALIVE_MAX_TOKENS`」**已按需求删除** —— oracle 现在**永远走深层多层 keepalive,没有兜底**,数据集大 pool 上会直接崩(0.000)。

---

## 5. 诊断开关(隔离变量用,`quick_test_online.py` 读)

| 环境变量 | 默认 | 开启效果 | 不开启效果 |
|---|---|---|---|
| `PIC_A3_NO_INJECT` | 关 | oracle 模式只开 keepalive、不开 inject(隔离 keepalive vs inject) | 完整 oracle(含 inject) |
| `PIC_A3_NO_CLIP` | 关 | oracle 模式退回 full-length keepalive(不 clip,零加速;验证 clip 是否引入问题) | clip 开(有加速) |
| `PIC_A3_FORCE_ALL_IMP` | 关 | 所有位置都 imp = 全部重算(无 cached-K 复用;≈ layer-2+ full recompute)。**框架 sanity**:若还崩=框架 bug,不崩=复用/选择问题 | 正常 15% |
| `PIC_A3_KEEPALIVE_FIXED_IMP` | 关 | 深层复用 layer-1 的 imp(不逐层重选) | 逐层重选 |
| `PIC_A3_CLIP_REAL_L1` | 关(=0) | clip 路径 layer-1 用真实 pick_imp(而非隔离 reselect_full) | 隔离 reselect(实测两者都崩,无差别) |
| `PIC_A3_MISS_ONLY_IMP` | 关 | imp 只含 miss(hit 全走 prepop) | 正常(topk from hit + miss) |

---

## 6. 合成测试专用(不传 `--dataset` 时,构造固定 prompt)

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `PIC_SYNTH_NUM_DOCS` | 3 | 合成文档段数 N(答案段在 index N//2);最多 8 |
| `PIC_SYNTH_DOC_REPEAT` | 12 | 每段基础文字重复次数(调长度,每段 ~1800 tok) |
| `PIC_SYNTH_SHUFFLE` | 关 | measure 段顺序打乱(warmup 仍逐段隔离捕获) |
| `PIC_SYNTH_VARIED` | 关 | 每段长度不一(模拟数据集变长 chunk) |
| `PIC_SYNTH_REAL_SAMPLE=<idx>` | 无 | 用真实 hotpotqa 第 idx 样本的段落当合成固定文档 + 它的多跳问题当 Q(走忠实 run_mode 路径) |
| `PIC_SYNTH_REAL_DISTRACTORS=N` | 0 | 给上面再塞 N 个"其它样本整块 context"当 distractor(模拟 pool) |
| `PIC_SYNTH_REAL_MERGED` | 关 | 目标样本用 1 个 merged 整块(=数据集分段);不开则拆成段落 |
| `--oracle-warmup-docs K` | 0(=全部) | oracle warmup 只覆盖前 K 段(诊断 isolated inject 从第几段崩) |

---

## 7. 段对齐 / K-dump / 其它

| 参数 | 默认 | 作用 |
|---|---|---|
| `PIC_PAD_TO_64` | 1 | 每段 pad 到 64 倍数(DSA page_size=64 必需;关了会崩) |
| `PIC_FORCE_PAD_ALL` | 0 | 让所有模式(含 full_recompute)都 pad(隔离 padding 对 F1 的影响) |
| `SGLANG_PIC_KDUMP_DIR=<dir>` | 无 | 在深层 dump K buffer(ref/clip/keepalive 三路 + `_c{n}` 计数),用 `kdump_prof.py` 对比余弦 |
| `--answer-extractor` | flexible | flexible(抓 "Answer:" 后内容,对 GLM CoT 鲁棒) / pic_bench(与参考实现对齐) |
| `--tp` | 8 | 张量并行数 |

---

## 8. 常用组合(照抄即可)

```bash
# 环境前缀(QS1J 必带)
export PATH=/opt/dynamo/venv/bin:$PATH
export PYTHONPATH=/root/sglang/python
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/opt/dynamo/venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
cd /root/sglang
```

```bash
# A) 数据集正常对比(oracle 默认 = 深层全局 imp + 平滑;无兜底,大 pool 会崩)
python quick_test_online.py --dataset hotpotqa --n-samples 10 --dataset-pool \
  --modes full_recompute pic_a3 pic_a3_oracle --tp 8

# B) 关闭平滑做 A/B(K<=1)——看平滑对 pic_a3 到底有没有用
SGLANG_PIC_A3_IMP_SMOOTH_K=1 \
  python quick_test_online.py --dataset hotpotqa --n-samples 10 --dataset-pool \
  --modes full_recompute pic_a3 pic_a3_oracle --tp 8

# C) 每段 15%(防崩,但伤 pic_a3)
SGLANG_PIC_A3_HIT_ONLY_IMP=1 \
  python quick_test_online.py --dataset hotpotqa --n-samples 10 --dataset-pool \
  --modes full_recompute pic_a3 pic_a3_oracle --tp 8

# D) oracle 只在第 1 层重算(= 单层 pic_a3)
python quick_test_online.py --modes pic_a3_oracle --oracle-layers 1 --tp 8

# E) 框架 sanity(全部重算,应 == full_recompute)
SGLANG_PIC_A3_FORCE_ALL_IMP=1 \
  python quick_test_online.py --dataset hotpotqa --n-samples 10 --dataset-pool \
  --modes full_recompute pic_a3_oracle --tp 8

# F) 真实内容合成(oracle 在聚焦内容上赢 pic_a3 的证据)
PIC_SYNTH_REAL_SAMPLE=1 \
  python quick_test_online.py --modes full_recompute pic_a3 pic_a3_oracle \
  --oracle-layers 1 20 40 60 --tp 8
```

---

## 9. 你要的那个配置

**"pic_a3_oracle 深层也从全部 hit 全局选 15% 重算"** = **默认**(不 export `HIT_ONLY_IMP`、不 export
`HYBRID_IMP`;深层 keepalive 现在恒开、无兜底)。即上面的组合 **A**。
注意:该配置在数据集大 pool 上会崩(答案被 distractor 挤出全局 15%);聚焦内容(组合 F)上正常且赢 pic_a3。
`avg_pool` 平滑默认开(`SGLANG_PIC_A3_IMP_SMOOTH_K=5`),用组合 B 可关掉做对比。
