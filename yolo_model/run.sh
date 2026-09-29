#!/usr/bin/env bash
# 两个 YOLO 的端到端运行脚本：元件检测 + 文本检测。
#
# 用法:
#   bash yolo_model/run.sh                      # 跑元件检测（导出+训练）
#   bash yolo_model/run.sh --stage export       # 只导出
#   bash yolo_model/run.sh --device 0 --epochs 150 --model yolo11m.pt
#   bash yolo_model/run.sh --what text          # 跑文本检测（需要先有伪标签）
#   bash yolo_model/run.sh --what text --stage pseudo   # 生成伪标签（需联网+OCR）
#
# 在数据所在的根目录执行（能看见 data/ 或 200_train_cases/ 的地方）。
# 官方数据目录会自动探测，也可用 PCB_OFFICIAL_DIR / PCB_HARVEST_DIR 指定。

set -euo pipefail

WHAT=comp          # comp | text
STAGE=all          # all | export | train | pseudo
DEVICE=cpu
EPOCHS=""
MODEL=""
BATCH=""
OUT=""
SCALE=1
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --what)   WHAT="$2"; shift 2 ;;
    --stage)  STAGE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --model)  MODEL="$2"; shift 2 ;;
    --batch)  BATCH="$2"; shift 2 ;;
    --out)    OUT="$2"; shift 2 ;;
    --scale)  SCALE="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=${PYTHON:-python3}

# 官方数据的目录布局各机器不同：本地嵌在 赛题六公开数据集/ 下，
# 远程服务器上直接是 200_train_cases/。Python 侧会自动探测，这里只做存在性兜底。
FOUND=""
for d in 200_train_cases 赛题六公开数据集/200_train_cases data; do
  [[ -d "$d" ]] && FOUND="yes"
done
if [[ -z "$FOUND" && -z "${PCB_OFFICIAL_DIR:-}${PCB_HARVEST_DIR:-}" ]]; then
  echo "错误: 在 $ROOT 下找不到任何数据目录。"
  echo "      找过: 200_train_cases/  赛题六公开数据集/200_train_cases/  data/"
  echo "      可设环境变量 PCB_OFFICIAL_DIR / PCB_HARVEST_DIR 指定。"
  exit 1
fi

# /dev/shm 太小时 DataLoader 多进程会报 "No space left on device"。
# 本机实测 shm 只有 64M，而 batch 8 的 512px 张量一批就 25M。
SHM_MB=$(df -m /dev/shm 2>/dev/null | awk 'NR==2{print $2}' || echo 0)
WORKERS=4
if [[ "${SHM_MB:-0}" -lt 512 ]]; then
  WORKERS=0
  echo "[run] /dev/shm 仅 ${SHM_MB}MB，自动将 workers 设为 0"
fi

step() { echo; echo "=========== $* ==========="; }

if [[ "$WHAT" == "comp" ]]; then
  OUT=${OUT:-out/comp_det}
  if [[ "$STAGE" == "all" || "$STAGE" == "export" ]]; then
    step "1/2 元件检测 导出数据集"
    $PY yolo_model/comp_det/export.py --out "$OUT" --scale "$SCALE" "${EXTRA[@]+"${EXTRA[@]}"}"
  fi
  if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
    step "2/2 元件检测 训练"
    # 没有 GPU 时默认随机初始化的小模型，避免无谓地等权重下载
    if [[ -z "$MODEL" ]]; then
      [[ "$DEVICE" == "cpu" ]] && MODEL=yolo11n.yaml || MODEL=yolo11m.pt
    fi
    [[ -z "$EPOCHS" ]] && { [[ "$DEVICE" == "cpu" ]] && EPOCHS=2 || EPOCHS=150; }
    [[ -z "$BATCH" ]] && { [[ "$DEVICE" == "cpu" ]] && BATCH=4 || BATCH=16; }
    $PY yolo_model/comp_det/train.py --data "$OUT/data.yaml" --model "$MODEL" \
      --epochs "$EPOCHS" --batch "$BATCH" --device "$DEVICE" --workers "$WORKERS"
    echo
    echo "[run] 看召回率(R)而不是 mAP —— ComponentF1 的 bbox 阈值 20px 很宽松，"
    echo "      漏检才是硬损失，置信度阈值要按 F1 最优点重调，不要用默认 0.25。"
  fi

elif [[ "$WHAT" == "text" ]]; then
  PSEUDO=out/text_pseudo
  OUT=${OUT:-out/text_det}
  if [[ "$STAGE" == "pseudo" ]]; then
    step "文本检测 生成伪标签（需要联网 + OCR 后端）"
    echo "[run] 官方 target.json 里没有任何文本区域标注（已核对 200/200），"
    echo "      所以 YOLO #2 的标签必须自己造：OCR 出候选框，用 GT 的"
    echo "      Name/value/pinname 当白名单筛选。"
    $PY yolo_model/text_det/pseudo_label.py --out "$PSEUDO" "${EXTRA[@]+"${EXTRA[@]}"}"
    exit 0
  fi
  if [[ ! -d "$PSEUDO" ]]; then
    echo "错误: 找不到 $PSEUDO"
    echo "      先在能联网的机器上跑: bash yolo_model/run.sh --what text --stage pseudo"
    echo "      产出是纯 JSON，拷回本机即可继续。"
    exit 1
  fi
  if [[ "$STAGE" == "all" || "$STAGE" == "export" ]]; then
    step "1/2 文本检测 导出数据集"
    $PY yolo_model/text_det/export.py --pseudo "$PSEUDO" --out "$OUT" "${EXTRA[@]+"${EXTRA[@]}"}"
  fi
  if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
    step "2/2 文本检测 训练"
    if [[ -z "$MODEL" ]]; then
      [[ "$DEVICE" == "cpu" ]] && MODEL=yolo11n.yaml || MODEL=yolo11m.pt
    fi
    [[ -z "$EPOCHS" ]] && { [[ "$DEVICE" == "cpu" ]] && EPOCHS=2 || EPOCHS=120; }
    [[ -z "$BATCH" ]] && { [[ "$DEVICE" == "cpu" ]] && BATCH=4 || BATCH=16; }
    $PY yolo_model/text_det/train.py --data "$OUT/data.yaml" --model "$MODEL" \
      --epochs "$EPOCHS" --batch "$BATCH" --device "$DEVICE" --workers "$WORKERS"
  fi

else
  echo "错误: --what 只能是 comp 或 text"; exit 1
fi

echo
echo "[run] 完成。"
