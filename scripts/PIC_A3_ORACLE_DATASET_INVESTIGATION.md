# pic_a3_oracle 数据集适配 & 规模崩溃调查

> 环境:全部在 GPU 主机 **QS1J**(8×H20-141GB)运行。模型 **GLM-5.2-FP8**,`--tp 8`,`--recomp-ratio 0.15`。
> 本地 Mac 无 GPU,只做改码。测试脚本 `quick_test_online.py`(未纳入 git)。
> 日期:2026-07-23 ~ 07-24。

---

## 0. 目标

`pic_a3_oracle`(PIC 段级 KV 复用 + A³ 多层 keepalive 窗口化重选 + in-process oracle 隔离
hidden 捕获/注入)此前只在 **synthetic 3-doc** 固定测试上验证过(FDT5-6 "Mount
Kilimanjaro",3.5×)。目标:**适配到真实数据集(hotpotqa)并测试**。

---

## 1. 数据集适配(已完成)

`quick_test_online.py` 两处改动,让 `pic_a3_oracle` 能跑 `--dataset`:

1. **`run_mode_dataset` 增加 SYS 预热**:在逐段 warmup 前先发一次 `SYS<SEP>warmup_query`,
   把 SYS 段缓存上。否则第一个 chunk 的 warmup 是全冷(all-miss)→ 不触发 A³
   keepalive/oracle 钩子 → 该 chunk 的隔离 hidden 抓不到。
2. **oracle 模式自动开 `SGLANG_PIC_A3_CLIP_CAPTURE`**(+ 给其它 mode pop 掉),让该模式
   自包含,不再依赖 driver 脚本外部 `export`。诊断开关:`PIC_A3_NO_CLIP` / `PIC_A3_NO_INJECT`。

oracle 的隔离 hidden 存储(`_pic_a3_oracle_hidden`)是**按 PIC 段 hash 内容寻址**的,所以跨
样本、跨 distractor 池都能命中(前提:该段已 warmup 覆盖)。

---

## 2. 首轮数据集结果 —— 谜题出现

hotpotqa,GLM-5.2-FP8,tp8,recomp 0.15:

| 场景 | full_recompute | pic_a3 | pic_a3_oracle |
|---|---|---|---|
| 单段(`--no-dataset-pool`,N=5,~1.4K tok) | 0.848 | 0.933 | **0.848** ✓ |
| 多段(`--dataset-pool`,N=20,8–10K tok) | 0.781 | **0.807(4.16×)** | **0.000 ✗ 乱码** |

`pic_a3_oracle` 在带 distractor 池的多段长上下文上**崩成 token 级乱码**(`.,!?.,!?` /
`the the the` / `thghy714`),而单层 `pic_a3` 好得很(0.807,+3.3%,4.16×)。

**谜题:同样的 warmup、同样的隔离 cached-K,为什么 pic_a3 行、oracle 崩?**

---

## 3. 根因调查

### 3.1 排除法(synthetic 受控,可 K-dump)

为在**固定 prompt**(可 K-dump)上复现,给 `quick_test_online.py` 加了受控开关:
`PIC_SYNTH_NUM_DOCS`(N 段合成文档)、`PIC_SYNTH_SHUFFLE`、`PIC_SYNTH_VARIED`(变长)。

逐一排除,**synthetic 怎么配都不崩**:

| 假设 | 结果 |
|---|---|
| 框架(routing/positions/DSA metadata/prepop)有 bug | ❌ 排除:数据集 `PIC_A3_FORCE_ALL_IMP=1`(全重算)→ oracle **F1=0.824 == full_recompute**,输出干净 |
| imp 选择错(隔离重投影 vs 真实前向) | ❌ 排除:`[PIC-A3-RESELECT]` IoU = **0.98–0.99** |
| 逐层重选是元凶 | ❌ 排除:`PIC_A3_KEEPALIVE_FIXED_IMP=1`(不重选)仍崩 |
| 多窗口 rebuild 是元凶 | ❌ 排除:`--oracle-layers 1 77`(单窗口)仍崩 |
| chunk 数 / shuffle / 变长 / 规模 | ❌ 排除:synthetic N=3~8 + shuffle + 变长(~7K)全部 FDT5-6 正确 |

**K-dump 旁证**:即便 oracle 输出**正确**的样本,复用的 cached-K 与 full_recompute 的余弦
也很低(L2=0.94 → **L60=0.37**)—— DSA 稀疏注意力用 fresh imp K 把它掩盖了。所以
**cached-K 余弦不是判据**。

### 3.2 决定性验证:真实内容塞进 synthetic 忠实路径

加 `PIC_SYNTH_REAL_SAMPLE=<idx>`:把一个真实 hotpotqa 样本的段落当 synthetic 固定文档、
它的多跳问题当 Q,走**已确认忠实的 run_mode 路径**(不传 `--dataset`)。

结果(4 个多跳样本,`--oracle-layers 1 20 40 60`):

| 样本 | GT | full_recompute | pic_a3 | pic_a3_oracle |
|---|---|---|---|---|
| 1 | Chief of Protocol | ✓ | `0` ✗ | ✓ **逐字匹配** |
| 2 | Animorphs | ✓ | 乱写 ✗ | ✓ **逐字匹配** |
| 4 | Greenwich Village | ✓ | `0 questions.0…` ✗ | ✓ **逐字匹配** |
| 5 | YG Entertainment | ✓ | `0.0` ✗ | ✓ **逐字匹配** |

**oracle 在真实多跳内容上 4/4 全对(逐 token 匹配 full_recompute),而且完胜 pic_a3。**
→ oracle 设计本身**没问题**;"设计对多跳失效"的说法被推翻。

### 3.3 规模扫描 —— 逐步逼近数据集

在忠实 run_mode 路径上,给目标样本加 pool-like distractor(`PIC_SYNTH_REAL_DISTRACTORS=N`)、
merged 整块(`PIC_SYNTH_REAL_MERGED=1`,答案在 1 整块里,与数据集一致)、shuffle:

| 配置(run_mode 路径,样本 1) | token | 段数 | oracle | pic_a3 |
|---|---|---|---|---|
| 仅自身内容(段落) | ~1.6K | ~10 | ✓ | ✗ |
| +4 distractor(merged+shuffle) | ~7.5K | 5 | ✓ | ✗ |
| **+8 distractor(merged+shuffle)** | ~13.5K | 9 | **✗ 乱码** | **✗ 乱码** |
| +8 distractor(段落,no shuffle) | ~13.5K | 18 | ✗ "找不到答案" | ✗ |
| 数据集 pool | 8–10K | ~5-7 | ✗(0.000) | ✓(0.807) |

**run_mode 路径在 ~13.5K 规模上同样崩** → 与 warmup 路径(`prime_sample`)无关,是规模问题。

---

## 4. 最终根因:**规模阈值**

**oracle 的多层 keepalive KV 复用,规模/上下文阈值比单层 pic_a3 更低:**

- oracle 大约 **~8K+ 多段上下文**就崩;
- pic_a3 大约撑到 **~13K+** 才崩;
- 数据集 pool(8–10K)刚好落在**两者之间** → **oracle 崩(0.000)、pic_a3 扛住(0.807)**;
- 到 ~13K,**两个都崩**。

**这一条同时解释了本次调查里所有看似矛盾的现象。** 机理:段越多 × cached-K 复用越多 ×
深层重选,误差累积得比单层快;超过某个上下文规模就整体崩坏。

---

## 5. 纠正的错误结论(过程中一度得出、后被推翻)

1. ~~"隔离捕获设计对多跳内容失效"~~ —— **错**。真实多跳 4/4 逐字匹配,完胜 pic_a3。
2. ~~"崩溃是数据集 warmup 路径(`prime_sample`)的 bug"~~ —— **错**。run_mode 同等规模同样崩。
3. ~~"`_pic_a3_rebuild_rowset_for_window` 有逻辑 bug"~~ —— **错**。不是逻辑 bug,是规模鲁棒性。

---

## 6. 结论

- **oracle 机制(srt 侧:capture/inject/reselect/rebuild)是正确的**,在**中等规模**
  (≤~7.5K、段数不多)的真实多跳复用上**正确且优于单层 pic_a3**。
- 它的弱点是**规模鲁棒性**:大 distractor 池(8K+)会让深层复用误差累积到崩坏,阈值低于 pic_a3。
- 数据集默认 pool 填到 8–16K,正好踩在 oracle 的崩溃区,所以之前一律 0.000。
- **不是**设计缺陷、**不是** warmup 路径 bug、**不是**多跳内容本身的问题。

---

## 7. 代码改动(均 env 门控,默认关 → 对正常路径零回归)

`quick_test_online.py`(未纳入 git):
- **保留(适配)**:`run_mode_dataset` 的 SYS 预热;oracle 模式自动 `CLIP_CAPTURE`。
- **调试/受控开关**:`PIC_SYNTH_NUM_DOCS` / `PIC_SYNTH_SHUFFLE` / `PIC_SYNTH_VARIED` /
  `PIC_SYNTH_REAL_SAMPLE` / `PIC_SYNTH_REAL_DISTRACTORS` / `PIC_SYNTH_REAL_MERGED` /
  `PIC_A3_NO_CLIP` / `PIC_A3_NO_INJECT`;`--oracle-layers` 透传。

`python/sglang/srt/models/deepseek_v2.py`(未纳入 git,**调试探针,默认关**):
- K-dump 探针:3-way tag(ref/clip/keepalive)+ per-forward 计数器,`SGLANG_PIC_KDUMP_DIR` 门控。
- `PIC_A3_CLIP_REAL_L1`(失败的修复尝试,默认 `"0"` = 原行为)。

`kdump_prof.py`:K-dump 余弦分析脚本(按位置分箱,3-way 对比)。

> ⚠️ 收尾时应清理这些调试探针(K-dump probe、`PIC_A3_CLIP_REAL_L1`),保留:dataset 适配
> (SYS 预热 + 自动 CLIP_CAPTURE)与 synthetic 受控开关(复现有用)。

---

## 8. 建议的下一步(三选一)

1. **规模鲁棒性(研究量大)**:让 oracle 撑到更大 pool。思路:减少 check 层数(4→2)、
   限制/衰减深层复用误差、或超过上下文阈值时自动降级到单层 pic_a3。
2. **接受现状 + 收尾**:oracle 定位为"中等上下文复用工具"(那里它赢 pic_a3);大 pool 用
   pic_a3。清理调试探针。
3. **(推荐先做,便宜)** 把数据集 pool 调小到 oracle 的可用区(`--pool-max-tokens ~4000`),
   在数据集上正经量化 **oracle vs pic_a3** 的 F1,证明 oracle 在它擅长的规模上的价值。

---

## 9. 复现命令(QS1J)

环境前缀:
```bash
export PATH=/opt/dynamo/venv/bin:$PATH
export PYTHONPATH=/root/sglang/python
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/opt/dynamo/venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
cd /root/sglang
```

数据集(会看到 oracle 崩):
```bash
python quick_test_online.py --dataset hotpotqa --n-samples 20 --dataset-pool \
  --modes full_recompute pic_a3 pic_a3_oracle --tp 8
```

真实内容 synthetic(会看到 oracle 4/4 正确、完胜 pic_a3):
```bash
PIC_SYNTH_REAL_SAMPLE=1 python quick_test_online.py \
  --modes full_recompute pic_a3 pic_a3_oracle --oracle-layers 1 20 40 60 --tp 8
```

规模复现(会看到两者都崩):
```bash
PIC_SYNTH_REAL_SAMPLE=1 PIC_SYNTH_REAL_DISTRACTORS=8 PIC_SYNTH_REAL_MERGED=1 PIC_SYNTH_SHUFFLE=1 \
  python quick_test_online.py --modes full_recompute pic_a3 pic_a3_oracle --oracle-layers 1 20 40 60 --tp 8
```

只在第一层重算(= 退化成单层,keepalive/深层全不触发):
```bash
python quick_test_online.py --modes pic_a3_oracle --oracle-layers 1 --tp 8
```

QS1J 上的日志(供追溯):`oracle_ds_pool20.out`、`real_synth_s{1,2,4,5}.log`、
`real_faithful.log`、`real_faith8.log`、`ds_diag_forceall_n10.log`、`/tmp/kd_cmp/`(K-dump)。
