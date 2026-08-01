"""inspect_data.py —— id_data 数据体检（合并训练前先跑一次）。

目的：判断两次采样（--n 2000 后又 --n 8000）是否发生了
      「params.npy 被覆盖 → dataset.npz 里前一批 X/Y 错配」的问题。

在远程服务器的 freq_query_design 目录下运行：
    python inspect_data.py
"""
import csv
import os

import numpy as np

import id_config as C


def _exists(p):
    return "存在" if os.path.exists(p) else "缺失"


def _load_params():
    return np.load(C.PARAMS_NPY) if os.path.exists(C.PARAMS_NPY) else None


def _index_stats():
    """统计 index.csv 各状态计数、唯一 ok idx 集合、最大 ok idx。"""
    ok, fail, try_ = set(), 0, 0
    if not os.path.exists(C.INDEX_CSV):
        return ok, fail, try_
    with open(C.INDEX_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            st = row.get("status")
            if st == "ok":
                try:
                    ok.add(int(row["idx"]))
                except (ValueError, KeyError):
                    pass
            elif st == "fail":
                fail += 1
            elif st == "try":
                try_ += 1
    return ok, fail, try_


def main():
    print("=" * 60)
    print("id_data 体检")
    print("=" * 60)
    print(f"DATA_DIR    = {C.DATA_DIR}")
    print(f"params.npy  : {_exists(C.PARAMS_NPY)}")
    print(f"index.csv   : {_exists(C.INDEX_CSV)}")
    print(f"dataset.npz : {_exists(C.DATASET_NPZ)}")
    print(f"N_PARAMS={C.N_PARAMS}  Y_DIM={C.Y_DIM}")

    params = _load_params()
    if params is not None:
        print(f"\n[params.npy] shape={params.shape}  → 当前采样点集共 {params.shape[0]} 个")

    ok, n_fail, n_try = _index_stats()
    if ok:
        oks = sorted(ok)
        print(f"\n[index.csv] status=ok 唯一样本 {len(ok)} 个 | fail 行 {n_fail} | try 行 {n_try}")
        print(f"            ok idx 范围 [{oks[0]} .. {oks[-1]}]")

    # s2p 计数
    n_s2p = 0
    if os.path.isdir(C.S2P_DIR):
        n_s2p = len([f for f in os.listdir(C.S2P_DIR) if f.endswith(".s2p")])
    print(f"\n[s2p] 目录内 .s2p 文件 {n_s2p} 个")

    # dataset.npz 内容
    if os.path.exists(C.DATASET_NPZ):
        d = np.load(C.DATASET_NPZ, allow_pickle=True)
        X, Y = np.asarray(d["X"], float), np.asarray(d["Y"], float)
        n_bad = int((~np.isfinite(Y).all(axis=1)).sum())
        print(f"\n[dataset.npz] X{X.shape}  Y{Y.shape}  含 NaN/Inf 的样本 {n_bad} 个")

        # ---- 关键一致性检查：dataset 的 X 是否都能在当前 params.npy 中找到 ----
        # 若 dataset.npz 是用「当前」params.npy 聚合的，则每一行 X 都应命中 params。
        # 出现命中不了的行 → 这些 X 来自「另一套」params（旧的一批），
        #   说明 params.npy 曾被覆盖，dataset 里混入了「用旧 s2p + 新 params 错配」的样本。
        if params is not None:
            pkey = set(map(tuple, np.round(params, 6)))
            xkey = np.round(X, 6)
            miss = sum(1 for r in xkey if tuple(r) not in pkey)
            print("\n" + "-" * 60)
            if miss == 0:
                print("[一致性] ✅ dataset.npz 的所有 X 均能在当前 params.npy 中找到。")
                print("         —— 单套采样，未见错配迹象。")
            else:
                print(f"[一致性] ⚠️  dataset.npz 中有 {miss}/{X.shape[0]} 行 X 不在当前 params.npy 里！")
                print("         几乎可以确定：params.npy 曾被第二次 --n 覆盖，")
                print("         这些行是「旧 s2p 曲线 + 新 params」错配的坏样本，训练前必须剔除。")
                print("         见文末「补救」说明。")
            print("-" * 60)

    print("\n判读指引：")
    print("  · 若你其实保存了两个独立的 dataset.npz（如 dataset_2000.npz / dataset_8000.npz），")
    print("    直接用 merge_datasets.py 合并即可（各自内部 X/Y 一致，安全）。")
    print("  · 若只有一个 id_data、且上面出现 ⚠️ 错配：run1 的 2000 条已无法找回正确 params，")
    print("    只有 run2 的 ~1500 条是干净的。补救见下：")
    print("    ——把 run1、run2 分别放到独立目录/独立 params.npy 重新 --aggregate-only 生成两份 npz，")
    print("      或干脆只用干净的一批 + 再补采样。需要的话把本脚本输出发我，我给你精确的清洗脚本。")


if __name__ == "__main__":
    main()
