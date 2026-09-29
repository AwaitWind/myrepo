#!/usr/bin/env bash
# U-Net 导线分割的端到端运行脚本。
#
# 用法:
#   bash u-net_mode/run.sh                      # 全流程（审计→渲染→切片→自检→训练）
#   bash u-net_mode/run.sh --stage audit        # 只看数据能用多少，不产出文件
#   bash u-net_mode/run.sh --stage sanity       # 只跑接线自检（几秒钟）
#   bash u-net_mode/run.sh --device 0 --base 64 --epochs 80
#   bash u-net_mode/run.sh --line-width 1       # 线宽对照的另一个臂
#
# 在数据所在的根目录执行（能看见 data/ 或 200_train_cases/ 的地方）。
# 官方数据目录会自动探测，也可用 PCB_OFFICIAL_DIR / PCB_HARVEST_DIR 指定。
#
# 流程说明:
#   render_mask  从 GT 的 nets[*].edges 渲染导线 mask，并按 edges 完整度过滤。
#                实测 778 个样本里只有 286 个可用（官方 200 全可用，
#                data/ 的 578 个只有 86 个达标，因为 31.7% 的网络 edges 为空）。
#   export_tiles 预切片落盘。**必须做** —— 在线切片在 286 个样本时会因反复
#                解码大原图（中位数 1MP、最大 17MP）退化成纯 I/O 瓶颈。
#   test_sanity  单批次过拟合，验证接线没被改坏。
#   train        训练。

set -euo pipefail

STAGE=all
DEVICE=cpu
ARCH=unet
BASE=""
EPOCHS=""
BATCH=""
WINDOW=512
LINE_WIDTH=3
MASK_DIR=out/wire_mask
TILES_DIR=out/wire_tiles
OUT=out/unet_runs/wire
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)      STAGE="$2"; shift 2 ;;
    --device)     DEVICE="$2"; shift 2 ;;
    --arch)       ARCH="$2"; shift 2 ;;
    --base)       BASE="$2"; shift 2 ;;
    --epochs)     EPOCHS="$2"; shift 2 ;;
    --batch)      BATCH="$2"; shift 2 ;;
    --window)     WINDOW="$2"; shift 2 ;;
    --line-width) LINE_WIDTH="$2"; shift 2 ;;
    --mask-dir)   MASK_DIR="$2"; shift 2 ;;
    --tiles-dir)  TILES_DIR="$2"; shift 2 ;;
    --out)        OUT="$2"; shift 2 ;;
    -h|--help)    sed -n '2,25p' "$0"; exit 0 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=${PYTHON:-python3}
D="$(dirname "$0")"
D="${D#./}"

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

# CPU 上默认缩小规模，否则跑不完。实测 base=64/512px 单步 7.2 秒，
# 一个 epoch 2554 秒，80 轮要 56.8 小时。
if [[ "$DEVICE" == "cpu" ]]; then
  BASE=${BASE:-16}; EPOCHS=${EPOCHS:-2}; BATCH=${BATCH:-8}
else
  BASE=${BASE:-64}; EPOCHS=${EPOCHS:-80}; BATCH=${BATCH:-8}
fi

step() { echo; echo "=========== $* ==========="; }

if [[ "$STAGE" == "audit" ]]; then
  step "审计 edges 完整度（不产出文件）"
  $PY "$D/render_mask.py" --audit-only "${EXTRA[@]+"${EXTRA[@]}"}"
  exit 0
fi

if [[ "$STAGE" == "sanity" ]]; then
  step "接线自检"
  $PY "$D/test_sanity.py" "${EXTRA[@]+"${EXTRA[@]}"}"
  exit 0
fi

if [[ "$STAGE" == "all" || "$STAGE" == "mask" ]]; then
  step "1/4 渲染导线 mask（线宽 $LINE_WIDTH）"
  $PY "$D/render_mask.py" --out "$MASK_DIR" --line-width "$LINE_WIDTH"
fi

if [[ "$STAGE" == "all" || "$STAGE" == "tiles" ]]; then
  step "2/4 预切片落盘（window $WINDOW）"
  $PY "$D/export_tiles.py" --mask-dir "$MASK_DIR" --out "$TILES_DIR" --window "$WINDOW"
fi

if [[ "$STAGE" == "all" ]]; then
  step "3/4 接线自检（用真实切片，几秒）"
  $PY "$D/test_sanity.py" --mask-dir "$MASK_DIR" || {
    echo "[run] 自检未通过，先修接线再训练。"; exit 1; }
fi

if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
  step "4/4 训练（arch=$ARCH base=$BASE epochs=$EPOCHS device=$DEVICE）"
  if [[ "$DEVICE" == "cpu" ]]; then
    echo "[run] 提示: CPU 上这只是冒烟。实测 base=16 一个 epoch 326 秒、"
    echo "      base=64（论文口径）一个 epoch 2554 秒，正式训练需要 GPU。"
  fi
  $PY "$D/train.py" --tiles-dir "$TILES_DIR" --arch "$ARCH" --base "$BASE" \
    --epochs "$EPOCHS" --batch "$BATCH" --device "$DEVICE" --workers "$WORKERS" \
    --out "$OUT"
  echo
  echo "[run] 看召回率优先于 IoU —— 断一条线会把一个 net 拆成两个，"
  echo "      匈牙利匹配下一次错误同时损伤两个 net。"
fi

echo
echo "[run] 完成。注意: U-Net 只输出 mask，从 mask 到 nets.hyperGraph 还需要"
echo "      骨架化 → 打断 → 并查集 → 引脚吸附，这条后处理链尚未实现。"
