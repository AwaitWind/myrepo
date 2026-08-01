"""
Layer 0 —— 批量数据生成（M1），进程隔离版。

【为什么用进程隔离】
PyAEDT 的 gRPC / C 扩展长时间运行后会累积 refcount 泄漏，
最终触发 `Fatal Python error: bool_dealloc`（C 层崩溃，Python try/except 拦不住）。
同进程内 release+reopen 只能缓解、不能根治（实测 ~260 次后仍崩）。

解决：主进程（orchestrator）不直接碰 HFSS，而是反复拉起「worker 子进程」，
每个 worker 只仿真一小批样本（BATCH_PER_WORKER 个）就正常退出，
由操作系统彻底回收所有 C 扩展资源。worker 即使崩溃，主进程也能：
  - 检测到子进程异常退出
  - 靠 index.csv 断点续跑
  - 对反复导致崩溃的「毒样本」记 attempts，超过 MAX_ATTEMPTS 后永久跳过

用法：
    python gen_dataset.py                    # 采样 DEFAULT_N_SAMPLES 个（推荐）
    python gen_dataset.py --n 500            # 指定样本数
    python gen_dataset.py --batch-size 20    # 每个 worker 处理多少样本
    python gen_dataset.py --aggregate-only   # 只聚合已有 s2p，不跑 HFSS
    python gen_dataset.py --worker --batch 12,13,14   # （内部）worker 模式
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np

# --- 路径处理：让本目录与上级项目目录都可被 import ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_DIR = os.path.dirname(_THIS_DIR)     # AI-RF-filter5/
for _p in (_THIS_DIR, _PROJ_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import id_config as C
from sampling import sample_params
from dataio import (
    s2p_to_y,
    append_index_row,
    load_done_indices,
    load_attempts,
    save_dataset,
    load_dataset,
)

# 每个 worker 处理多少样本后正常退出（进程隔离粒度）
BATCH_PER_WORKER = 40
# 单个样本最多尝试次数（超过则永久跳过）
MAX_ATTEMPTS = 3


# =====================================================================
# 参数点集合：固定生成并落盘，断点续跑复用同一套
# =====================================================================
def get_or_make_params(n):
    C.ensure_dirs()
    if os.path.exists(C.PARAMS_NPY):
        existing = np.load(C.PARAMS_NPY)
        if existing.shape[0] >= n:
            print(f"[params] 复用已有采样点 {C.PARAMS_NPY}（取前 {n} 个）")
            return existing[:n]
        print(f"[params] 已有 {existing.shape[0]} 个不足 {n} 个，重新采样")

    xs = sample_params(n, seed=C.SAMPLING_SEED)
    np.save(C.PARAMS_NPY, xs)
    print(f"[params] 已采样 {n} 个合法参数点 → {C.PARAMS_NPY}")
    return xs


# =====================================================================
# HFSS 会话管理（仅 worker 子进程使用）
# =====================================================================
def _open_hfss():
    from ansys.aedt.core import settings as pyaedt_settings
    from HFSS_funcs import init_hfss

    pyaedt_settings.use_grpc_api = True
    hfss, setup, setup_name, sweep_name, sweeps_name = init_hfss(C.CENTER_FREQ_GHZ)
    return hfss, setup_name, sweeps_name


def _release_hfss(hfss):
    try:
        hfss.release_desktop()
    except Exception:
        pass


def simulate_one(hfss, setup_name, sweep_name, idx, x):
    """用参数 x 建模 + 仿真，返回 (s2p_path, sim_sec)。失败抛异常。"""
    from HFSS_funcs import filter_layout_generate, hfss_simulate

    kwargs = C.param_to_kwargs(x)
    filter_layout_generate(hfss, **kwargs)

    s2p_path = os.path.join(C.S2P_DIR, f"sample_{idx:05d}.s2p")
    sim_sec = hfss_simulate(
        hfss, s2p_path,
        setup_name=setup_name, sweep_name=sweep_name,
        cores=C.SIM_CORES, tasks=C.SIM_TASKS,
    )
    if not os.path.exists(s2p_path):
        raise RuntimeError(f"仿真完成但未生成 s2p: {s2p_path}")
    return s2p_path, sim_sec


# =====================================================================
# Worker 子进程：仿真指定的一批 idx，然后退出
# =====================================================================
def run_worker(batch):
    params = np.load(C.PARAMS_NPY)
    done = load_done_indices()
    todo = [i for i in batch if i not in done]
    if not todo:
        print("[worker] 本批已全部完成，退出。")
        return

    print(f"[worker] PID={os.getpid()} 处理 {len(todo)} 个样本: {todo}")
    hfss, setup_name, sweeps_name = _open_hfss()
    sweep_name = sweeps_name[0]

    for idx in todo:
        x = params[idx]
        try:
            s2p_path, sim_sec = simulate_one(hfss, setup_name, sweep_name, idx, x)
            _ = s2p_to_y(s2p_path)  # 解析校验
            append_index_row(idx, "ok", s2p_path, sim_sec, x)
            print(f"  [{idx:05d}] OK  ({sim_sec:.1f}s)  x={np.round(x,3).tolist()}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            append_index_row(idx, "fail", "", 0.0, x)
            print(f"  [{idx:05d}] FAIL: {e}  → 重建 HFSS 后继续本批")
            _release_hfss(hfss)
            hfss, setup_name, sweeps_name = _open_hfss()
            sweep_name = sweeps_name[0]

    _release_hfss(hfss)
    print(f"[worker] PID={os.getpid()} 本批完成，正常退出。")


# =====================================================================
# 主进程 orchestrator：反复拉起 worker 子进程，直到全部完成
# =====================================================================
def orchestrate(n, batch_size=BATCH_PER_WORKER):
    C.ensure_dirs()
    params = get_or_make_params(n)

    t_start = time.time()
    round_no = 0

    while True:
        done = load_done_indices()
        attempts = load_attempts()
        pending = [
            i for i in range(n)
            if i not in done and attempts.get(i, 0) < MAX_ATTEMPTS
        ]
        skipped = [
            i for i in range(n)
            if i not in done and attempts.get(i, 0) >= MAX_ATTEMPTS
        ]

        print(f"\n[orchestrate] 进度: 完成 {len(done)}/{n} | 待做 {len(pending)} | "
              f"跳过(毒样本) {len(skipped)}")

        if not pending:
            print("[orchestrate] 无待做样本，结束循环。")
            if skipped:
                print(f"[orchestrate] 以下样本尝试 {MAX_ATTEMPTS} 次仍失败被跳过: {skipped}")
            break

        batch = pending[:batch_size]
        round_no += 1
        print(f"[orchestrate] 第 {round_no} 轮，本批 {len(batch)} 个: {batch}")

        # 预写 'try' 占位（记 attempts）—— 即使 worker 立刻崩溃，attempts 也已推进，
        # 保证循环最终会收敛（毒样本达到 MAX_ATTEMPTS 后被排除）。
        for idx in batch:
            append_index_row(idx, "try", "", 0.0, params[idx])

        # 拉起 worker 子进程
        cmd = [sys.executable, os.path.abspath(__file__),
               "--worker", "--batch", ",".join(str(i) for i in batch)]
        try:
            ret = subprocess.run(cmd, cwd=_THIS_DIR)
            code = ret.returncode
        except Exception as e:
            print(f"[orchestrate] 启动 worker 失败: {e}")
            code = -999

        if code == 0:
            print(f"[orchestrate] worker 正常退出 (code=0)")
        else:
            print(f"[orchestrate] ⚠ worker 异常退出 (code={code})，"
                  f"可能触发 PyAEDT C 层崩溃；将重启新进程继续（断点续跑）。")

    elapsed = time.time() - t_start
    done_final = load_done_indices()
    print(f"\n[orchestrate] 全部结束: 成功 {len(done_final)}/{n} | 耗时 {elapsed:.1f}s")
    aggregate()


# =====================================================================
# 聚合
# =====================================================================
def aggregate():
    done = sorted(load_done_indices())
    if not done:
        print("[aggregate] 没有成功样本，跳过聚合。")
        return

    X_list, Y_list, ok_idx = [], [], []
    params = np.load(C.PARAMS_NPY) if os.path.exists(C.PARAMS_NPY) else None

    for idx in done:
        s2p_path = os.path.join(C.S2P_DIR, f"sample_{idx:05d}.s2p")
        if not os.path.exists(s2p_path):
            continue
        try:
            y = s2p_to_y(s2p_path)
        except Exception as e:
            print(f"[aggregate] 跳过 {idx}: 解析失败 {e}")
            continue
        if params is None or idx >= params.shape[0]:
            continue
        X_list.append(params[idx])
        Y_list.append(y)
        ok_idx.append(idx)

    if not X_list:
        print("[aggregate] 无可聚合样本。")
        return

    X = np.array(X_list, float)
    Y = np.array(Y_list, float)
    save_dataset(X, Y)
    print(f"[aggregate] 已聚合 {len(ok_idx)} 个样本 → {C.DATASET_NPZ}")
    print(f"            X shape={X.shape}, Y shape={Y.shape}")

    d = load_dataset()
    print(f"[aggregate] 复核读取: X={d['X'].shape}, Y={d['Y'].shape}, "
          f"freqs={d['freqs'].shape}, include_s11={d['include_s11']}")


# =====================================================================
# CLI
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="逆向设计数据生成 (Layer 0 / M1, 进程隔离版) — filter5")
    parser.add_argument("--n", type=int, default=C.DEFAULT_N_SAMPLES,
                        help=f"样本总数（默认 {C.DEFAULT_N_SAMPLES}）")
    parser.add_argument("--batch-size", type=int, default=BATCH_PER_WORKER,
                        help=f"每个 worker 子进程处理的样本数（默认 {BATCH_PER_WORKER}）")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="只把已有 s2p 聚合成 dataset.npz，不跑 HFSS")
    parser.add_argument("--worker", action="store_true",
                        help="（内部）worker 子进程模式")
    parser.add_argument("--batch", type=str, default="",
                        help="（内部）worker 处理的 idx 列表，逗号分隔")
    args = parser.parse_args()

    if args.worker:
        batch = [int(s) for s in args.batch.split(",") if s.strip() != ""]
        run_worker(batch)
    elif args.aggregate_only:
        aggregate()
    else:
        orchestrate(args.n, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
