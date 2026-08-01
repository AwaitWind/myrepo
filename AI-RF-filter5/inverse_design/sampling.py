"""
参数空间采样：在 19 维设计空间内铺点，并用几何硬约束过滤。

优先用拉丁超立方（Latin Hypercube, scipy.stats.qmc）；
没 scipy 就回退到 numpy 均匀随机。

对外主接口：
    sample_params(n, seed=...) -> np.ndarray  形状 (n, 19)，全部满足 check_constraints。
"""

import numpy as np

from id_config import (
    LOWER,
    UPPER,
    N_PARAMS,
    SAMPLING_SEED,
    check_constraints,
    PARAM_NAMES,
)


def _unit_lhs(n, d, seed):
    """在 [0,1]^d 上生成 n 个 LHS 样本。缺少 scipy 时回退纯随机。"""
    try:
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=d, seed=seed)
        return sampler.random(n)
    except Exception:
        rng = np.random.default_rng(seed)
        return rng.random((n, d))


def _scale_to_bounds(unit):
    return LOWER + unit * (UPPER - LOWER)


def sample_params(n, seed=SAMPLING_SEED, max_oversample=200):
    """
    采样 n 个满足几何约束的参数点。
    「过采样 + 约束过滤」：一次多生成一些候选，过滤越界的凑够 n 个。
    max_oversample 限制最大放大倍数，防止约束过紧时死循环。
    """
    accepted = []
    batch_seed = seed
    factor = 4

    while len(accepted) < n and factor <= max_oversample:
        n_try = (n - len(accepted)) * factor
        unit = _unit_lhs(n_try, N_PARAMS, batch_seed)
        cand = _scale_to_bounds(unit)

        for x in cand:
            ok, _ = check_constraints(x)
            if ok:
                accepted.append(x)
                if len(accepted) >= n:
                    break

        batch_seed += 1
        factor *= 2

    if len(accepted) < n:
        raise RuntimeError(
            f"采样失败: 仅得到 {len(accepted)}/{n} 个满足约束的点。"
            f"检查 PARAM_BOUNDS 与 check_constraints 是否互斥。"
        )
    return np.array(accepted[:n], dtype=float)


def constraint_pass_rate(n=2000, seed=SAMPLING_SEED):
    """诊断：估计随机采样下约束的通过率。"""
    unit = _unit_lhs(n, N_PARAMS, seed)
    cand = _scale_to_bounds(unit)
    ok_count = sum(1 for x in cand if check_constraints(x)[0])
    return ok_count / n


if __name__ == "__main__":
    rate = constraint_pass_rate()
    print(f"约束通过率（随机估计）: {rate*100:.1f}%")

    xs = sample_params(8)
    print(f"\n采样 8 个合法参数点（列顺序 {PARAM_NAMES}）:")
    for i, x in enumerate(xs):
        vals = "  ".join(f"{v:6.3f}" for v in x)
        print(f"  [{i}] {vals}")
