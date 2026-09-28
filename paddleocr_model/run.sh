#!/usr/bin/env bash
# PaddleOCR 这条线的运行脚本。
#
# 用法:
#   bash paddleocr_model/run.sh --stage weights     # 准备权重
#   bash paddleocr_model/run.sh --stage eval        # 评 OCR 质量（先跑这个）
#   bash paddleocr_model/run.sh --stage pseudo      # 造 YOLO #2 的训练标签
#   bash paddleocr_model/run.sh --stage recognize   # YOLO #2 训好后的正式识别
#   bash paddleocr_model/run.sh                     # weights -> eval -> pseudo
#
#   bash paddleocr_model/run.sh --backend easyocr   # PaddleOCR 装不上时的备选
#   bash paddleocr_model/run.sh --backend stub --stage pseudo   # 只验管道接线
#
# 在数据所在的根目录执行（能看见 data/ 或 200_train_cases/ 的地方）。

set -euo pipefail

STAGE=default
BACKEND=paddleocr
LANG_=ch
MODEL_DIR=""
LIMIT=""
GPU=""
PSEUDO_OUT=out/text_pseudo
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)     STAGE="$2"; shift 2 ;;
    --backend)   BACKEND="$2"; shift 2 ;;
    --lang)      LANG_="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --limit)     LIMIT="$2"; shift 2 ;;
    --gpu)       GPU="--gpu"; shift ;;
    --out)       PSEUDO_OUT="$2"; shift 2 ;;
    -h|--help)   sed -n '2,18p' "$0"; exit 0 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=${PYTHON:-python3}
D="$(dirname "$0")"; D="${D#./}"

FOUND=""
for d in 200_train_cases 赛题六公开数据集/200_train_cases data; do
  [[ -d "$d" ]] && FOUND="yes"
done
if [[ -z "$FOUND" && -z "${PCB_OFFICIAL_DIR:-}${PCB_HARVEST_DIR:-}" ]]; then
  echo "错误: 在 $ROOT 下找不到任何数据目录。"
  echo "      找过: 200_train_cases/  赛题六公开数据集/200_train_cases/  data/"
  exit 1
fi

OPT=(--backend "$BACKEND" --lang "$LANG_")
[[ -n "$MODEL_DIR" ]] && OPT+=(--ocr-model-dir "$MODEL_DIR")
[[ -n "$GPU" ]] && OPT+=("$GPU")

step() { echo; echo "=========== $* ==========="; }

if [[ "$STAGE" == "weights" || "$STAGE" == "default" ]]; then
  step "准备 PaddleOCR 权重"
  if [[ "$BACKEND" == "paddleocr" ]]; then
    echo "[run] bj.bcebos.com 是百度国内 CDN，国内机器通常直连可达，"
    echo "      不需要 source /etc/network_turbo（那是给 github.com 用的）。"
    $PY "$D/weights.py" || {
      echo
      echo "[run] 自动下载没成。可选:"
      echo "      $PY $D/weights.py --manual"
      echo "      或改用 easyocr:  bash $D/run.sh --backend easyocr --stage eval"
      exit 1; }
  else
    echo "[run] 后端是 $BACKEND，跳过 PaddleOCR 权重准备。"
  fi
  [[ "$STAGE" == "weights" ]] && exit 0
fi

if [[ "$STAGE" == "eval" || "$STAGE" == "default" ]]; then
  step "评测 OCR 质量（整图模式，先摸底）"
  echo "[run] 看 GT 文本召回率，不要看候选框留存率 —— 图上的注释、标题栏、"
  echo "      页码不在 GT 白名单里，被丢掉是正常的。"
  $PY "$D/eval_ocr.py" "${OPT[@]}" ${LIMIT:+--limit "$LIMIT"} \
      "${EXTRA[@]+"${EXTRA[@]}"}"
  [[ "$STAGE" == "eval" ]] && exit 0
fi

if [[ "$STAGE" == "pseudo" || "$STAGE" == "default" ]]; then
  step "生成 YOLO #2 的伪标签"
  echo "[run] 官方 target.json 里没有任何文本区域标注（已核对 200/200），"
  echo "      所以 YOLO #2 的标签只能这样造：位置信任 OCR，内容归 GT 判定。"
  $PY "$D/pseudo_label.py" --out "$PSEUDO_OUT" "${OPT[@]}" \
      ${LIMIT:+--limit "$LIMIT"} "${EXTRA[@]+"${EXTRA[@]}"}"
  echo
  echo "[run] 接下来训 YOLO #2:"
  echo "      bash yolo_model/run.sh --what text --device 0"
fi

if [[ "$STAGE" == "recognize" ]]; then
  step "正式识别（YOLO #2 框 + OCR 只认字）"
  $PY "$D/recognize.py" "${OPT[@]}" ${LIMIT:+--limit "$LIMIT"} \
      "${EXTRA[@]+"${EXTRA[@]}"}"
fi

echo
echo "[run] 完成。"
