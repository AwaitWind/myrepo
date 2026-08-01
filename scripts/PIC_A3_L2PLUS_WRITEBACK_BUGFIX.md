# PIC-A³ 新路径:miss 段 public KV 回写丢失 Bug —— 定位与修复

**状态**:已修复并端到端验证(2026-07-20,GLM-5.2-FP8,H20×8)
**改动文件**:`python/sglang/srt/model_executor/model_runner.py`(仅此一个文件,两处)
**影响模式**:`pic_a3`(A³)与 `pic_cacheblend`(CacheBlend)—— 即所有走「选择性重算新路径」的 PIC 模式
**症状**:`--recomp-ratio < 1.0` 时精度崩塌;`ratio = 1.0`(等价全量重算)时正常

---

## TL;DR

> pic_a3 新路径下,**在一次「新路径 forward」里首次被缓存的 miss 段,其 public KV 从未被写入(全零)**。后续请求复用该段时读到全零 K → attention 变垃圾 → 精度崩。
>
> 根因:`forward_extend` 是在 `forward_batch` 的 **`_eager_fb_view` 副本** 上跑模型的。layer-1 边界把「miss→public 回写」所需的 slot 张量写到了那个副本上,forward 返回后副本连同这些改动一起被丢弃;而 forward **之后** 的 writeback 跑在**原始** `forward_batch` 上,拿到的是没被改过的 legacy 全 -1 张量 → 回写等于空操作。
>
> 修复:layer-1 边界改为把回写 slot 张量额外存到**共享的 `req`**(通过 `_reqs_ref` 跨副本共享)上;writeback 在新路径下从 `req` 读取。

端到端效果(hotpotqa,N=8,ratio=0.15):`pic_a3` F1 从崩塌的 **~0.13 → 0.622**,与 full_recompute(0.633)基本持平。

---

## 1. 背景:pic_a3 新路径的两段式执行

pic_a3 / pic_cacheblend 的核心思路是「只重算重要 token」:

| 阶段 | Q(query)行数 | K(key)空间 | K 写到哪 | public 回写 |
|---|---|---|---|---|
| **layer 0-1** | `full_len`(全部) | `full_len` | `l01_scratch` | ✗ 不写 |
| **layer 2+** | `n_q = miss ∪ imp`(收窄) | `full_len`(仍全长) | `l2plus_miss` / `l2plus_imp` | ✓ miss→public |

- **layer 0-1 全量跑**:所有 token 都算,K 写进 `l01_scratch`。layer 1(check layer)顺便 stash 出 Q/K latent(`forward_mla.py:563-581`),供挑 imp 用。**这两层不写 public**(`pic_alloc.py:376`,`pic_public_out_loc` 全 -1)。
- **layer-1 → 2 边界**(`deepseek_v2.py:2543-2621`):挑 imp(`_pic_a3_pick_imp`)→ 重写 req_to_token(`_pic_a3_rewrite_req_to_token_pool_for_l2plus`)→ prepop(把 hit-非-imp 位置用「缓存 KV + delta-RoPE」填进 layer 2+ 缓冲)→ 裁剪 Q 到 `(miss+imp)` 行。
- **layer 2+ 只把 `(miss+imp)` 当 query**,但 K 空间保持全长(非对称 attention),所以每个存活 query 仍能 attend 到完整上下文。miss/imp 的 fresh K 写进 `l2plus_*` slot。
- **forward 之后**:`_pic_writeback_mla_kv` 把 miss 段的 fresh K 从 `l2plus_miss` 拷到 public(`l2plus_miss_pub`),供**将来的请求**复用。

> 段级复用要能工作,前提是:一个 miss 段这次被算出来后,它的 public KV 必须被正确落盘;下次同样内容的段命中时,才能从 public 读回(+ delta-RoPE 位置校正)。**Bug 就出在「这次落盘」这一步。**

---

## 2. 症状

- `pic_a3 --recomp-ratio 1.0`:正确(F1 ≈ full_recompute)。因为 ratio=1.0 时**所有** hit token 都被选为 imp、全部现算,没有任何位置依赖 public KV,bug 不触发。
- `pic_a3 --recomp-ratio < 1.0`:精度崩塌(hotpotqa F1 ≈ 0.13,而 full_recompute ≈ 0.63)。ratio 越小,依赖 public KV 的 hit-非-imp 位置越多,崩得越彻底。

即:**「省算」一旦真的开启(ratio<1.0),结果就错** —— 这让选择性重算完全不可用。

---

## 3. Bug 在哪

**位置**:`model_runner.py::_pic_writeback_mla_kv`(forward 后调用,`model_runner.py:4526` 处)。

它靠 `forward_batch.pic_public_out_loc`(public 目标 slot)+ `forward_batch.out_cache_loc`(源 slot)做 `buf[pub] = buf[priv]` 的逐层拷贝。

用一个可复现的三段式实验精确定位(测试 prompt:`SYS+C1+C2+C3+Q`;warmup1=`SYS+C1+Q`、warmup3=`SYS+C3+Q`):

| forward | 是否新路径 | writeback 看到的 `pic_public_out_loc` | 结果 |
|---|---|---|---|
| warmup1(SYS+C1+Q,无 hit) | 否(normal path) | 长度 4160,`n_valid=4096` | **C1 public 正确写入** ✓ |
| warmup3(SYS-hit+C3+Q) | 是(新路径) | 长度 4160,**`n_valid=0`** | **C3 public 从未写入 → 全零** ✗ |
| test(复用 C3) | 是(新路径) | —— | C3 prepop 读 public 源 `aliveFrac=0.000` |

关键点:**新路径下 writeback 看到的 `pic_public_out_loc` 是「legacy 全 -1」张量,所以 `valid=(pub>=0)` 全 False,一个 slot 都不拷。** 于是任何**首次在新路径 forward 里被缓存的 miss 段**(例子里的 C3,以及 test 里的 C2),public KV 永远是零。

> 为什么 C1 没事、C3 有事?纯属巧合:C1 是在 warmup1(无 hit,走 **normal** path,正常回写)里缓存的;C3 是在 warmup3(有 SYS hit,走 **新路径**)里缓存的。跟位置 delta 无关(见下节「排查过程」里被推翻的误判)。

---

## 4. 为什么出现(根因)

新路径其实是**有**建回写张量的:`_pic_a3_rewrite_req_to_token_pool_for_l2plus`(`model_runner.py:4041`)在 layer-1 边界会构造正确的 `pic_a3_l2plus_pub_out_loc` / `pic_a3_l2plus_out_cache_loc`(实测 `n_valid=4032`,完全正确),并在 `deepseek_v2.py:2596-2601` 把它们 swap 到 `forward_batch.out_cache_loc` / `pic_public_out_loc` 上。

**但这些改动全部作用在一个副本上。**

```python
# model_runner.py  forward_extend / forward_decode
forward_batch = self._eager_fb_view(forward_batch, pp_proxy_tensors)   # ← 重新绑定成副本!

# _eager_fb_view (model_runner.py:3175)
def _eager_fb_view(self, forward_batch, pp_proxy_tensors=None):
    if envs.SGLANG_EAGER_INPUT_NO_COPY.get():   # pic_a3 下为 True
        return replace(forward_batch)           # dataclasses.replace = 浅拷贝(新对象)
    ...
```

于是执行链变成:

```
dispatch 持有 fb_orig
   └─ forward_extend(fb_orig)
         └─ fb_copy = _eager_fb_view(fb_orig)     # 浅拷贝,新对象
         └─ model.forward(fb_copy)                # layer-1 边界改的是 fb_copy:
                                                  #   fb_copy.pic_a3_l2plus_pub_out_loc = ...
                                                  #   fb_copy.pic_public_out_loc = <swap>
         └─ return (只返回模型输出,fb_copy 被丢弃)
   └─ _pic_writeback_mla_kv(fb_orig)              # ← 用的是 fb_orig!
                                                  #   fb_orig.pic_public_out_loc 还是 legacy 全 -1
                                                  #   fb_orig.pic_a3_l2plus_pub_out_loc 还是 None
```

**为什么 out_cache_loc 的 swap「看起来生效了」而 public 的没生效?**
因为 `out_cache_loc` 是在 forward **进行中**被 kernel 消费的(逐层写 K 到 l2plus slot,所以 imp/miss 的 K 确实算对了);而 `pic_public_out_loc` 只在 forward **结束后** 被 writeback 消费——那时改动已随副本蒸发。这也是为什么 bug 只表现在「复用」而不是「当次生成」。

一句话:**「在 forward 里改 forward_batch,期望 forward 后还能读到」这个隐含假设,被 `_eager_fb_view` 的浅拷贝打破了。**

---

## 5. 排查过程(含两次被推翻的结论,供后人少走弯路)

1. **前一个 agent 的结论「两条路径对同一 SYS token 算出 cos 0.94 不同」是错的。** 逐层逐段测 K 的 cosine 后发现:SYS、C1 全程 cos≈1.0(干净);真正崩的只有 **C3**——那个「在别处缓存、这里换位置复用」的段。SYS 在后面层的漂移是 C3 污染经残差流传播过去的**次生现象**,被误当成了主因。
2. **「delta-RoPE 算错」也是假象。** 把 K 拆成 `k_nope`(位置无关,前 512 维)和 `k_pe`(RoPE,后 64 维)分别测 cosine:`cos_nope ≈ cos_pe ≈ cos_full ≈ 0.27`。如果是 RoPE 旋转错,应该只有 `k_pe` 崩、`k_nope` 干净。三者一起崩 + 范数塌到 ~1/4,说明是**整段 token 被清零**,不是旋转错。
3. **定位到「零填充」**:C3 在 layer 2 有 73% 的 token 范数恰好 = 0(27% 存活的其实是被选为 imp、现算的那批)。
4. **加 `SGLANG_PIC_PREPOP_DBG` 环境变量门控的探针**([PREPOP-DBG]/[WB-DBG]/[RW-DBG]),用同一个三段式实验跑三次,逐步坐实:rewrite 建的 pub_out_loc 是对的(`n_valid=4032`)→ 但 writeback 看到的是 `n_valid=0` → 说明中间丢了 → 最终定位到 `_eager_fb_view` 浅拷贝。

**诊断工具**(仍在仓库/机器上):
- `scripts/pic_a3_layer01_error.py`(逐层逐段 K L2 误差;支持 `--skip-capture` 秒级复用已有 dump)
- `scripts/run_pic_a3_layer01_error.sh`(一键跑上面那个)
- 临时分析脚本 `pic_two_path_analyze.py`(cosine + nope/pe 拆分 + alive 比例;当前在机器 `/tmp`,段布局硬编码到上面那个测试 prompt)

---

## 6. 修复方案

只改 `model_runner.py`,两处,思路是**绕开会蒸发的 forward_batch,改用跨副本共享的 `req`**(`_reqs_ref` 在浅拷贝里是同一个 list、同一批 `Req` 对象;新路径本来就依赖它——rewrite/prepop 都通过它读 req)。

**改动 1** —— rewrite 里把回写张量额外存到 `req` 上(`_pic_a3_rewrite_req_to_token_pool_for_l2plus`,`model_runner.py:4182`):

```python
forward_batch.pic_a3_l2plus_out_cache_loc = _out_cache_loc
forward_batch.pic_a3_l2plus_pub_out_loc = _pub_out_loc
# 新增:forward_extend 在 forward_batch 的 _eager_fb_view 副本上跑模型,
# 所以上面这两行 forward_batch 上的改动到 post-forward writeback 时已丢失;
# req(经 _reqs_ref)与副本共享,故也存一份到 req,让 writeback 从 req 读回。
req.pic_a3_l2plus_pub_out_loc = _pub_out_loc
req.pic_a3_l2plus_out_cache_loc = _out_cache_loc
```

**改动 2** —— writeback 里,新路径下从 `req` 读回(`_pic_writeback_mla_kv`,`model_runner.py:3494`);非新路径分支**原样不动**:

```python
_a3_new = getattr(forward_batch, "pic_a3_new_path", False)
_a3_req = None
if _a3_new:
    _reqs = getattr(forward_batch, "_reqs_ref", None)
    _a3_req = _reqs[0] if _reqs else None
    pub_loc = getattr(_a3_req, "pic_a3_l2plus_pub_out_loc", None)
else:
    pub_loc = getattr(forward_batch, "pic_public_out_loc", None)   # legacy,零改动
if pub_loc is not None:
    out_loc = (
        _a3_req.pic_a3_l2plus_out_cache_loc
        if _a3_new
        else forward_batch.out_cache_loc
    )
    ...  # 后续 buf[pub]=buf[priv] 逐层拷贝逻辑不变
```

**为什么这样是对的**:
- `_reqs_ref` 在 `_eager_fb_view` 浅拷贝下是同一批 `Req` 对象(且新路径本来就靠它跑,否则 rewrite 的 `assert reqs is not None` 早就崩了)。因此 rewrite(在副本上)写的 `req.xxx` 和 writeback(在原始上)读的 `req.xxx` 是**同一个 `Req`**。
- writeback 逐层拷贝时,layer 0-1 的 `l2plus_miss` slot 是零(那两层写的是 `l01_scratch`),会把 public 的 layer 0-1 也写成零——**无害**,因为复用时 layer 0-1 是全量重算的、根本不读 public 的 layer 0-1;prepop 只读 layer 2..N(`max(start_layer, 2)`)。
- 覆盖 `pic_a3` 与 `pic_cacheblend`:两者都置 `pic_a3_new_path=True`。
- 顺带修好了同源问题:test 里的 C2(在新路径 forward 里新缓存的 miss 段)现在也能正确回写 public。

诊断探针(`SGLANG_PIC_PREPOP_DBG` 门控的 [PREPOP-DBG]/[WB-DBG]/[RW-DBG])已全部移除,仅保留上述修复。

---

## 7. 验证(用移除探针后的「出厂」代码)

**K 级**(C3 = 换位置复用、原本全崩的那段):

| 指标 | 修复前 | 修复后 |
|---|---|---|
| C3 layer-2 cos(full, a3) | 0.269 | **0.998** |
| C3 layer-2 alive 比例 | 0.269 | **1.000** |
| C3 public 源 aliveFrac | 0.000 | **1.000** |
| layer 3-5 传播 | 全段被污染 | 全段 cos≈1.0 |

**端到端 F1**(hotpotqa,N=8,ratio=0.15):

| 模式 | F1_mean |
|---|---|
| full_recompute(参照) | 0.633 |
| **pic_a3 @ ratio=0.15** | **0.622**(修复前 ~0.13) |

---

## 8. 复现命令(QS1J)

环境前置:
```bash
cd /root/sglang
export PATH=/opt/dynamo/venv/bin:$PATH
export PYTHONPATH=/root/sglang/python:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/opt/dynamo/venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
```

端到端 F1(表 7 后两行):
```bash
python quick_test_online.py --modes full_recompute pic_a3 --dataset hotpotqa \
    --n-samples 8 --a3-recomp-ratio 0.15 --tp 8 \
    --model /workspace/models/GLM-5.2-FP8 --port 30001
```

K 级逐段误差(表 7 前几行):
```bash
DUMP_ROOT=/tmp/pic_a3_l01_err LAYERS="2" ./scripts/run_pic_a3_layer01_error.sh
# 若要 cos/alive 精确值,再跑 /tmp/pic_two_path_analyze.py --dump-root /tmp/pic_a3_l01_err --layers 2 3 5
```

复现「修复前」:把 `_pic_writeback_mla_kv`(`model_runner.py:3494`)里 `_a3_new = getattr(...)` 临时改成 `_a3_new = False` 即可回到 legacy(崩)行为。

---

## 9. 影响范围与后续

- **正确性**:`pic_a3` / `pic_cacheblend` 在 `ratio < 1.0` 下从「不可用」变为「可用」。normal `pic` 与 `full_recompute` 路径零改动(走 writeback 的 `else` 分支)。
- **TTFT**:本次小样本(hotpotqa,单 chunk)上 pic_a3 TTFT 0.93×(新路径开销 > 省算);TTFT 收益需在**长上下文**上才显现——属独立的性能调优项,非本 bug。
- **潜在遗留风险**:任何「在 forward 中改 `forward_batch`、期望 forward 后再读」的逻辑都会踩同样的 `_eager_fb_view` 浅拷贝坑。新路径里 out_cache_loc 的 swap 之所以没事,只是因为它在 forward 内被消费。排查此类问题时应优先怀疑「改的是副本」。

---

*本文档基于 2026-07-20 在 QS1J(GLM-5.2-FP8,H20×8)上的调查。诊断脚本见 `scripts/pic_a3_layer01_error.py`。*
