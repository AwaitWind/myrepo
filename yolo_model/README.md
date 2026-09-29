# 两个 YOLO：元件检测 + 文本检测

论文（arXiv 2608.27923v1）的 Stage 1 里有**两个独立的 YOLO**，容易看漏：
一个检测元件，一个检测文本区域。本目录实现这两个。

```
原图
 ├─→ YOLO #1 元件检测 ──→ components.bbox
 └─→ YOLO #2 文本检测 ──→ 文本框 ──→ OCR ──→ Name / value / pinname
```

不在本目录范围内：U-Net 导线分割（第三个要训的模型）、引脚几何定位、
连通推断、文本-元件配对、LLM 纠错。

## 目录

| 路径 | 作用 |
|---|---|
| `common/cases.py` | 统一样本发现与清洗，屏蔽两种数据集的命名差异；**y 翻转在这里统一完成** |
| `common/tiling.py` | 切片起点、标签裁剪、data.yaml 生成，两个 YOLO 共用 |
| `common/weights.py` | 权重下载，断点续传 + 完整性校验；也可单独跑来预下载 |
| `comp_det/export.py` | YOLO #1 数据集导出 |
| `comp_det/train.py` | YOLO #1 训练 |
| `text_det/pseudo_label.py` | YOLO #2 的伪标签生成（**需要联网机器**） |
| `text_det/export.py` | YOLO #2 数据集导出 |
| `text_det/train.py` | YOLO #2 训练 |

## YOLO #1 元件检测

标签现成，直接能训。

```bash
# 官方 200 例 + 自采 594 例，单类，1024 切片
python3 yolo_model/comp_det/export.py --out out/comp_det

# 只用官方数据（做数据扩充的对照臂）
python3 yolo_model/comp_det/export.py --out out/comp_det_official --harvest ""

# 2x 上采样对照（切片数约翻 4 倍）
python3 yolo_model/comp_det/export.py --out out/comp_det_s2 --scale 2

# 训练。权重缺失时自动下载；--model *.yaml 表示随机初始化
python3 yolo_model/comp_det/train.py --data out/comp_det/data.yaml --epochs 2 --device cpu
python3 yolo_model/comp_det/train.py --data out/comp_det/data.yaml --epochs 150 \
    --model yolo11m.pt --device 0 --batch 16
```

默认单类（类别无关）。原因是类别口径目前对不上：data/ 19 类、官方
`200_train_cases` 41 类、`10_GTcase` 20 类，而且直接冲突（前者用
`mosfet`/`bjt`，后两者用 `mosfet_npn`/`bjt_npn`）。单类把这摊事绕开，
分类留给下一步。词表统一后可以用 `--classes multi`。

**看召回率，不看 mAP。** ComponentF1 的 bbox 判正阈值是 20px，很宽松；
漏检才是硬损失，而且 F1 下漏检和误检等价，置信度阈值要按 F1 最优点重调。

## YOLO #2 文本检测

**这个没有现成标签，必须先造。** 官方 200 份 target.json 的顶层键只有
`{components, nets, pins}`（已核对 200/200），没有任何文本区域标注。
论文的数据集标了 10 万个 text region，但赛题放出来的那份不含这一层。

做法是位置信任 OCR、内容归 GT 判定：OCR 出候选框 → 用 GT 的
`Name`/`value`/`pinname` 当白名单筛选 → 命中的留作训练标签。

```bash
# 第一步必须在能联网的机器上跑（要装 OCR 后端）
pip install paddleocr        # 或 easyocr
python3 yolo_model/text_det/pseudo_label.py --out out/text_pseudo
python3 yolo_model/text_det/pseudo_label.py --out out/text_pseudo --backend easyocr --limit 5

# 产出是纯 JSON，拷回离线机继续
python3 yolo_model/text_det/export.py --pseudo out/text_pseudo --out out/text_det
python3 yolo_model/text_det/train.py --data out/text_det/data.yaml --epochs 120 --device 0
```

这一步的价值参照论文 Table II：整图直接 OCR 的 CER 是 21.35%，先用 YOLO
框出文本区域再识别降到 6.33%。所以评价 #2 要看下游 OCR 的 CER 改善，
不只看框的 mAP。

## 权重

写死在 `common/weights.py` 的 `URLS` 里，换到能联网的机器直接跑即可。

```bash
python3 yolo_model/common/weights.py yolo11m.pt yolo11n.pt   # 预下载到 weights/
python3 yolo_model/comp_det/train.py --model weights/yolo11m.pt --no-download
```

本机实测 `github.com/ultralytics/assets` 只有 ~63 KB/s，5.6 MB 的
`yolo11n.pt` 在默认超时下会中途断掉。所以这里自己实现了断点续传，
中断后重跑同一命令即可接着下。

**判定陷阱（实际踩过）：** `curl -I` 返回 200 不代表能下载。HEAD 请求对
YOLO 权重返回 200，但 body 传输会超时、文件从未落地。所以校验的是
"完整字节数 + 能被 zipfile 打开"，不看 HTTP 状态码。

## 三条必须记住的数据约定

**一、GT 是左下原点，必须翻 y。** `y_img = H - y_gt`，且翻转后 y 的上下界
互换。实测依据：842 个 bbox 翻转后平均墨迹密度 0.1862，不翻转 0.0506，
而全图随机基线是 0.0467 —— 不翻转等于随机位置。翻转统一在
`common/cases.py: image_boxes()` 里做，**调用方不要再翻一次**。

反过来，`text_det` 的伪标签框是 OCR 直接在图上跑出来的，已经是图像坐标，
所以那条线**不做翻转**。这是两个 YOLO 最容易搞混的地方。

**二、判前景必须用"非白"判据，不能用灰度阈值。** jlc 来源的图是彩色的，
实测主绘图色 RGB(207,127,127) 灰度 151、RGB(127,195,127) 灰度 167，
比灰度 128 更亮。任何"二值化取暗像素"的预处理在 jlc 上会完全失效，
而 jlc 占官方训练集 99/200。判据用"与纯白的最大通道差 > 24"。
同理训练增强里 `hsv_h` 必须为 0。

**三、文本绝对禁止归一化。** 评分是严格字符串匹配，GT 的 `value` 是原图
逐字文本，`10kΩ`/`10K`/`10k`/`10.0k` 并存，单是 r/c 的值就有 652 种写法。
`pseudo_label.py` 里的 `norm_key()` 只用于**放宽白名单匹配**，产出的
`gt_text` 永远是 GT 原文，不要拿归一化结果去填输出字段。

## 数据清洗（自动做掉的）

跑 export 时会打印，也写进 `export_report.json`：

- 跳过 macOS AppleDouble 残留（`._*`），两个数据集里都有
- 剔除 components 为空的样本（实测 16 个，全在 data/）
- 内容重复的图按哈希分组，同组永不跨 split（实测 4 组 / 9 个样本，
  官方和自采里都有）。论文用 design-level 8:1:1 划分，这一步是它的前提，
  否则验证集会看到训练集的图

实测：794 个目录 → 778 个可用样本，按 `--val-ratio 0.1` 划分为
train 700 / val 78，组交叉 0。

## data/ 的可用范围（重要）

`data/` 的 594 个样本**不能整体当官方数据用**，它和官方口径有系统性差异：

| 用途 | 能否用 data/ |
|---|---|
| bbox 检测（本目录 YOLO #1） | 可以 |
| Name / value / pinname 文本（YOLO #2 白名单） | 可以 |
| type 分类 | 需先建 19↔41 类别映射 |
| 匿名元件的 pin | **不行**，编号与命名口径不同 |
| nets / edges | **不行**，31.7% 网络 edges 为空 |

匿名元件的具体差异：官方 `∅580 → pin key "pin_578" → pinname "P_198"`
（全局索引 + 占位符），data/ 是 `∅346 → "pin_346_1" → "GND"`
（复合编号 + 真实网络名）。匿名元件占 data/ 的 48.3%，`P_NN` 占位符在
data/ 全库只有 1 个而官方有 5848 个。具名元件两边一致。

本目录只用到 bbox 和文本，所以不受影响。但做 pins / nets 相关的工作时
必须记住这条。

## 已验证 / 未验证

已在本机实跑验证：

- `common/cases.py` 的发现、清洗、分组划分（778 样本，组交叉 0，分层配额精确）
- `comp_det/export.py` 全流程，并用"非白"判据抽查导出标签：
  791 个框里 785 个框内有前景像素（99.24%），随机位置基线墨迹密度 0.088
- `text_det/export.py` 的切片与划分管道（用人造伪标签测的，只验管道）
- `pseudo_label.py` 的白名单匹配与 `norm_key()` 归一化

**未验证**：`pseudo_label.py` 的 OCR 后端。本机没装 paddleocr/easyocr/doctr，
且 PaddleOCR 权重站不可达，所以这部分代码只写好了、没跑过，
第一次在联网机器上跑时要留意。训练脚本也只在 CPU 冒烟规模下验证过参数可接受，
未做完整长跑。
