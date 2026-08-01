"""
Layer 1 —— 正向代理模型（Forward Surrogate, M2）。

学习映射:  x(19 维参数)  →  y(S 参数曲线, Y_DIM 维)
训练好后即「毫秒级 HFSS 仿真器」，是后续 CMA-ES / 梯度逆向求解的引擎。

【本版针对 filter5 电路定制】
本电路 = CPW + 3 段矩形谐振器耦合 + via 打孔阵列 —— 三阶带通结构。
3 对矩形谐振器（re_1_x, re_2_x, re_3_x）从主导带 tl_c 中挖出，形成通带内 3 个匹配谷；
via 阵列（沿 PCB 边缘阵列打孔）会产生**传输零点**（S21 深陷波）。
再叠加 0.01~9.91 GHz 宽频扫描（100 点），S 曲线呈现
「一个通带 + 通带内多个匹配谷 + 通带外传输零点 + 高频寄生通带」的复杂形态。

因此相比通用配置：

  1. Y 逐维 mean/std 归一化（S21 段 / S11 段各自尺度，避免深匹配谷/陷波被压平）
  2. ResMLP：hidden 512、depth 6、GELU + skip —— 19 维输入 + 多谐振输出需要足够容量
  3. 频段自适应加权 MSE：用训练集 |ΔS21/Δf|、|ΔS11/Δf| 分布放大陡变频点，
     sharp_boost 默认 6.0（本电路传输零点/匹配谷比普通通带边更陡）
  4. 一阶差分正则（diff_reg 默认 0.15）：鼓励陷波/边沿斜率对齐
  5. clip_db（默认 -60）：截断 HFSS 偶发的 -70dB 级数值噪声深谷，robust 化 loss
  6. Ensemble（--ensemble N）：via/wc 等参数对零点位置强非线性 → 多 seed 降方差，
     推理取均值降 MAE，std 供 inverse_optimize 做「代理不确定度惩罚」
  7. 训练超参：AdamW，lr 3e-4，wd 1e-4，warmup + cosine，更长 epoch

用法:
    python forward_net.py --train                # 训 1 个模型
    python forward_net.py --train --ensemble 5   # 训 5 个 seed 组集成（推荐）
    python forward_net.py --eval                 # 在数据集上评估（自动识别 ensemble）
"""

import argparse
import glob
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import id_config as C
from dataio import load_dataset, split_y

FORWARD_CKPT = os.path.join(C.DATA_DIR, "forward_net.pt")
ENSEMBLE_GLOB = os.path.join(C.DATA_DIR, "forward_net_seed*.pt")


# =====================================================================
# 网络定义：ResMLP（容量针对 filter5 19维输入 + 100频点输出）
# =====================================================================
def _build_module(in_dim, out_dim, hidden=512, depth=6, dropout=0.05):
    """
    输入 → Linear(hidden) → GELU → [ResBlock] * (depth-1) → Linear(out_dim)
    每个 ResBlock: x + Dropout(Linear(GELU(Linear(x))))
    """
    import torch.nn as nn

    class ResBlock(nn.Module):
        def __init__(self, h, dp):
            super().__init__()
            self.fc1 = nn.Linear(h, h)
            self.fc2 = nn.Linear(h, h)
            self.act = nn.GELU()
            self.dp = nn.Dropout(dp)

        def forward(self, x):
            return x + self.dp(self.fc2(self.act(self.fc1(x))))

    class ResMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU())
            self.blocks = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(depth - 1)])
            self.head = nn.Linear(hidden, out_dim)

        def forward(self, x):
            x = self.stem(x)
            for b in self.blocks:
                x = b(x)
            return self.head(x)

    return ResMLP()


def _build_legacy_mlp(in_dim, out_dim, hidden=256, depth=3):
    """兼容旧版训练的纯 Sequential MLP（键名 0/2/4/…）。"""
    import torch.nn as nn
    layers = [nn.Linear(in_dim, hidden), nn.GELU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.GELU()]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


# =====================================================================
# 频段自适应权重（放大陡变频点：传输零点 / 匹配谷 / 通带边）
# =====================================================================
def build_freq_weight(Y, sharp_boost=6.0, base=1.0):
    """
    根据训练集 Y 的一阶差分幅度，给每个频点分配权重（越陡的地方权重越高）。
    对 S21 段和 S11 段分别构造，避免相互压制。

    本电路的传输零点/S11 匹配谷两侧斜率极陡，sharp_boost 默认较大（6.0），
    让网络把「学准陷波位置/深度」当成高优先级任务。

    返回 shape=(Y_DIM,) 的 numpy 数组，均值≈1（这样 loss 数量级不变）。
    """
    W = np.ones(C.Y_DIM, dtype=np.float32) * base

    def _one_channel(sl):
        y = Y[:, sl]                              # (N, N_FREQ)
        dy = np.abs(np.diff(y, axis=1))           # (N, N_FREQ-1)
        # 每个频点的"平均陡变" —— 取相邻两个差分中最大者
        d_left = np.pad(dy, ((0, 0), (1, 0)), mode="edge")   # (N, N_FREQ)
        d_right = np.pad(dy, ((0, 0), (0, 1)), mode="edge")
        dmag = np.maximum(d_left, d_right).mean(axis=0)      # (N_FREQ,)
        # 归一到 [0, 1] 后线性映射到 [base, base + sharp_boost]
        if dmag.max() > 1e-6:
            dmag /= dmag.max()
        return base + sharp_boost * dmag

    W[:C.N_FREQ] = _one_channel(slice(0, C.N_FREQ))
    if C.INCLUDE_S11:
        W[C.N_FREQ:2 * C.N_FREQ] = _one_channel(slice(C.N_FREQ, 2 * C.N_FREQ))

    # 归一：均值=1，保证跟旧 MSE 数量级可比
    W *= (Y.shape[1] / W.sum())
    return W


# =====================================================================
# 单模型训练
# =====================================================================
def train_one(
    X, Y, x_lo, x_hi, y_mu, y_std, freq_w,
    ckpt_path,
    hidden=512, depth=6, dropout=0.05,
    epochs=800, batch=256, lr=3e-4, weight_decay=1e-4,
    warmup=20, patience=100, val_ratio=0.15,
    diff_reg=0.15, huber_delta=0.0,
    seed=42, device=None, verbose=True, freqs=None, include_s11=True,
):
    """训练单个模型并保存 ckpt。返回 (val_mae_db, ckpt_path)。"""
    import torch
    import torch.nn as nn

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    N = X.shape[0]
    X01 = ((X - x_lo) / (x_hi - x_lo + 1e-12)).astype(np.float32)
    Yn = ((Y - y_mu) / y_std).astype(np.float32)

    idx = np.random.permutation(N)
    n_val = max(1, int(N * val_ratio))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    Xtr = torch.tensor(X01[tr_idx], device=device)
    Ytr = torch.tensor(Yn[tr_idx], device=device)
    Xval = torch.tensor(X01[val_idx], device=device)
    Yval = torch.tensor(Yn[val_idx], device=device)

    fw = torch.tensor(freq_w, dtype=torch.float32, device=device)  # (Y_DIM,)
    y_mu_t = torch.tensor(y_mu, dtype=torch.float32, device=device)
    y_std_t = torch.tensor(y_std, dtype=torch.float32, device=device)

    model = _build_module(C.N_PARAMS, C.Y_DIM, hidden, depth, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / max(1, warmup)
        # cosine decay
        t = (ep - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, t)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    def weighted_loss(pred, target):
        # 频段加权：陡变频点（传输零点/匹配谷/边沿）权重高
        err = pred - target
        if huber_delta > 0:
            # Huber：对个别极端残差（深谷数值噪声）鲁棒
            absr = err.abs()
            quad = torch.clamp(absr, max=huber_delta)
            per = 0.5 * quad ** 2 + huber_delta * (absr - quad)
            loss = (per * fw).mean()
        else:
            loss = (err ** 2 * fw).mean()
        # 一阶差分正则（标准化空间）—— 鼓励陷波/边沿斜率对齐
        if diff_reg > 0:
            dp = pred[:, 1:] - pred[:, :-1]
            dt = target[:, 1:] - target[:, :-1]
            loss = loss + diff_reg * ((dp - dt) ** 2).mean()
        return loss

    best_val = float("inf")
    best_state = None
    bad = 0
    n_tr = Xtr.shape[0]

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        ep_loss = 0.0
        for s in range(0, n_tr, batch):
            bi = perm[s:s + batch]
            xb, yb = Xtr[bi], Ytr[bi]
            pred = model(xb)
            loss = weighted_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += loss.item() * xb.shape[0]
        sched.step()

        model.eval()
        with torch.no_grad():
            pv = model(Xval)
            # 用「反归一化到 dB」的 MAE 作为 early stopping 指标（可解释）
            pv_db = pv * y_std_t + y_mu_t
            yv_db = Yval * y_std_t + y_mu_t
            val_mae_db = float((pv_db - yv_db).abs().mean().item())
            val_mae_s21 = float((pv_db[:, :C.N_FREQ] - yv_db[:, :C.N_FREQ]).abs().mean().item())

        if val_mae_db < best_val - 1e-4:
            best_val = val_mae_db
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if verbose and ((ep + 1) % 25 == 0 or ep == 0):
            cur_lr = opt.param_groups[0]["lr"]
            print(f"  [seed{seed}] ep {ep+1:4d}/{epochs}  train_loss={ep_loss/n_tr:.4f}  "
                  f"val_MAE={val_mae_db:.3f} dB  val_MAE_S21={val_mae_s21:.3f} dB  lr={cur_lr:.2e}")

        if bad >= patience:
            if verbose:
                print(f"  [seed{seed}] early stop @ ep {ep+1} (val 连续 {patience} 轮未改善)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pv = model(Xval)
        pv_db = (pv * y_std_t + y_mu_t).cpu().numpy()
        yv_db = (Yval * y_std_t + y_mu_t).cpu().numpy()
        final_mae = float(np.mean(np.abs(pv_db - yv_db)))
        final_mae_s21 = float(np.mean(np.abs(pv_db[:, :C.N_FREQ] - yv_db[:, :C.N_FREQ])))

    ckpt = {
        "model": model.state_dict(),
        "arch": "resmlp",                       # 标识新架构，供加载时选择正确的 build 函数
        "hidden": hidden, "depth": depth, "dropout": dropout,
        "x_lo": x_lo, "x_hi": x_hi,
        "y_mu": y_mu, "y_std": y_std,          # 均为 (Y_DIM,) 向量
        "freq_w": freq_w,
        "freqs": freqs, "include_s11": include_s11,
        "y_dim": C.Y_DIM, "n_freq": C.N_FREQ,
        "param_names": C.PARAM_NAMES,
        "val_mae_db": final_mae,
        "val_mae_s21_db": final_mae_s21,
        "seed": seed,
    }
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    import torch as _t
    _t.save(ckpt, ckpt_path)
    if verbose:
        print(f"  [seed{seed}] 完成: val MAE={final_mae:.3f} dB (S21 {final_mae_s21:.3f} dB)  → {ckpt_path}")
    return final_mae, ckpt_path


# =====================================================================
# 顶层训练入口（支持单模型或 ensemble）
# =====================================================================
def train(
    npz_path=C.DATASET_NPZ,
    ckpt_path=FORWARD_CKPT,
    ensemble=1,
    hidden=512, depth=6, dropout=0.05,
    epochs=800, batch=256, lr=3e-4, weight_decay=1e-4,
    warmup=20, patience=100, val_ratio=0.15,
    diff_reg=0.15, sharp_boost=6.0, base_w=1.0,
    clip_db=-60.0, huber_delta=0.0,
    seed=42, device=None,
):
    """
    ensemble=1 → 保存到 ckpt_path（单文件）
    ensemble>1 → 保存到 <dir>/forward_net_seed{k}.pt，同时更新 ckpt_path 作为「元 index」
    """
    d = load_dataset(npz_path)
    X = d["X"].astype(np.float32)
    Y = d["Y"].astype(np.float32)
    N = X.shape[0]
    print(f"[train] 数据集 N={N}, X={X.shape}, Y={Y.shape}, ensemble={ensemble}")
    if N < 20:
        print("[train][警告] 样本过少，建议先 gen_dataset 生成更多数据。")

    # ---- robust：截断 HFSS 偶发的极深数值噪声（如 -70dB 陷波底噪）----
    if clip_db is not None:
        below = (Y < clip_db).sum()
        if below > 0:
            print(f"[train] clip: 将 {int(below)} 个 < {clip_db} dB 的点截断到 {clip_db} dB "
                  f"(占比 {below / Y.size * 100:.3f}%)")
        Y = np.maximum(Y, clip_db).astype(np.float32)

    # ---- 归一化统计（逐维，S21/S11 各自尺度）----
    x_lo = C.LOWER.astype(np.float32)
    x_hi = C.UPPER.astype(np.float32)
    y_mu = Y.mean(axis=0).astype(np.float32)                # (Y_DIM,)
    y_std = (Y.std(axis=0) + 1e-6).astype(np.float32)       # (Y_DIM,)

    # ---- 频段权重（陡变处高权重：传输零点/匹配谷/边沿）----
    freq_w = build_freq_weight(Y, sharp_boost=sharp_boost, base=base_w)
    print(f"[train] freq_w S21 min/max/mean = "
          f"{freq_w[:C.N_FREQ].min():.2f} / {freq_w[:C.N_FREQ].max():.2f} / "
          f"{freq_w[:C.N_FREQ].mean():.2f}")

    ckpt_dir = os.path.dirname(ckpt_path) or "."
    os.makedirs(ckpt_dir, exist_ok=True)

    if ensemble <= 1:
        val_mae, _ = train_one(
            X, Y, x_lo, x_hi, y_mu, y_std, freq_w,
            ckpt_path=ckpt_path,
            hidden=hidden, depth=depth, dropout=dropout,
            epochs=epochs, batch=batch, lr=lr, weight_decay=weight_decay,
            warmup=warmup, patience=patience, val_ratio=val_ratio,
            diff_reg=diff_reg, huber_delta=huber_delta, seed=seed, device=device,
            freqs=d["freqs"], include_s11=d["include_s11"],
        )
        print(f"[train] 完成。val MAE = {val_mae:.3f} dB → {ckpt_path}")
        return val_mae

    # ---- ensemble ----
    maes, paths = [], []
    for k in range(ensemble):
        seed_k = seed + k * 1000 + 1
        pk = os.path.join(ckpt_dir, f"forward_net_seed{k}.pt")
        print(f"\n[train] ===== ensemble {k+1}/{ensemble} (seed={seed_k}) =====")
        mae, p = train_one(
            X, Y, x_lo, x_hi, y_mu, y_std, freq_w,
            ckpt_path=pk,
            hidden=hidden, depth=depth, dropout=dropout,
            epochs=epochs, batch=batch, lr=lr, weight_decay=weight_decay,
            warmup=warmup, patience=patience, val_ratio=val_ratio,
            diff_reg=diff_reg, huber_delta=huber_delta, seed=seed_k, device=device,
            freqs=d["freqs"], include_s11=d["include_s11"],
        )
        maes.append(mae)
        paths.append(p)

    # 元 index：把 ensemble 成员列表写进主 ckpt_path
    import torch as _t
    _t.save({
        "ensemble_paths": paths,
        "ensemble_val_mae": maes,
        "y_dim": C.Y_DIM, "n_freq": C.N_FREQ,
        "param_names": C.PARAM_NAMES,
        "freqs": d["freqs"], "include_s11": d["include_s11"],
        "x_lo": x_lo, "x_hi": x_hi,
        "y_mu": y_mu, "y_std": y_std,
        "is_ensemble_index": True,
    }, ckpt_path)
    mean_mae = float(np.mean(maes))
    print(f"\n[train] ensemble 完成。成员 val MAE = "
          f"{['%.3f' % m for m in maes]}, 平均 = {mean_mae:.3f} dB")
    print(f"[train] index 已写入 → {ckpt_path}")
    return mean_mae


# =====================================================================
# 推理引擎（供 inverse_optimize / validate 使用）
# =====================================================================
class ForwardModel:
    """
    统一封装：单模型 or ensemble。

      fm = ForwardModel("id_data/forward_net.pt")     # 单模型 or ensemble index
      fm = ForwardModel(["id_data/forward_net_seed0.pt", ...])  # 显式集成
      fm.predict(x_real, return_std=False)  → (B, Y_DIM) 或 (mean, std)
      fm.forward_unit(x01)                  → torch tensor（若集成，返回均值）
      fm.forward_unit_all(x01)              → 每个成员的预测（用于计算 std）
    """

    def __init__(self, ckpt=FORWARD_CKPT, device=None):
        import torch
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        paths = self._resolve_paths(ckpt)
        self.models = []
        meta = None
        for p in paths:
            ck = torch.load(p, map_location=self.device, weights_only=False)
            if ck.get("is_ensemble_index"):
                # 展开 index
                for pp in ck["ensemble_paths"]:
                    if not os.path.exists(pp):
                        # 兼容：如果绝对路径失效（换机器），尝试同目录同名
                        pp = os.path.join(os.path.dirname(p), os.path.basename(pp))
                    ck_m = torch.load(pp, map_location=self.device, weights_only=False)
                    self._append_model(ck_m)
                    meta = meta or ck_m
                continue
            self._append_model(ck)
            meta = meta or ck

        if not self.models:
            raise RuntimeError(f"未从 {ckpt} 加载到任何模型")

        # ---- 统计量（用第一个 ckpt 或 index 的） ----
        self.x_lo = torch.tensor(np.asarray(meta["x_lo"], dtype=np.float32), device=self.device)
        self.x_hi = torch.tensor(np.asarray(meta["x_hi"], dtype=np.float32), device=self.device)

        y_mu = np.asarray(meta["y_mu"], dtype=np.float32)
        y_std = np.asarray(meta["y_std"], dtype=np.float32)
        # 兼容旧的标量归一化 ckpt
        if y_mu.ndim == 0 or y_mu.size == 1:
            y_mu = np.full(C.Y_DIM, float(y_mu), dtype=np.float32)
        if y_std.ndim == 0 or y_std.size == 1:
            y_std = np.full(C.Y_DIM, float(y_std), dtype=np.float32)
        self.y_mu = torch.tensor(y_mu, device=self.device)
        self.y_std = torch.tensor(y_std, device=self.device)
        self.y_mu_np = y_mu
        self.y_std_np = y_std

        self.freqs = np.asarray(meta.get("freqs", C.FREQ_GHZ))
        self.include_s11 = bool(meta.get("include_s11", C.INCLUDE_S11))
        self.y_dim = int(meta.get("y_dim", C.Y_DIM))
        self.n_freq = int(meta.get("n_freq", C.N_FREQ))
        self.param_names = list(meta.get("param_names", C.PARAM_NAMES))
        self.is_ensemble = len(self.models) > 1

    # ---------- 内部工具 ----------
    def _resolve_paths(self, ckpt):
        if isinstance(ckpt, (list, tuple)):
            return list(ckpt)
        if isinstance(ckpt, str):
            if os.path.isdir(ckpt):
                return sorted(glob.glob(os.path.join(ckpt, "forward_net_seed*.pt")))
            if any(ch in ckpt for ch in "*?["):
                return sorted(glob.glob(ckpt))
            return [ckpt]
        raise TypeError(f"不支持的 ckpt 类型: {type(ckpt)}")

    def _append_model(self, ck):
        sd = ck["model"]
        arch = ck.get("arch")
        # 自动识别：新 ckpt 里带 arch=resmlp；旧 ckpt 的 state_dict 键是 "0.weight/2.weight/..."
        is_resmlp = arch == "resmlp" or any(
            k.startswith(("stem.", "blocks.", "head.")) for k in sd.keys()
        )
        if is_resmlp:
            m = _build_module(C.N_PARAMS, C.Y_DIM,
                              ck.get("hidden", 512),
                              ck.get("depth", 6),
                              ck.get("dropout", 0.0)).to(self.device)
        else:
            print(f"[ForwardModel][兼容] 检测到旧单模型 ckpt (Sequential MLP), "
                  f"hidden={ck.get('hidden', 256)} depth={ck.get('depth', 3)}. "
                  "建议重训以享受新版精度提升。")
            m = _build_legacy_mlp(C.N_PARAMS, C.Y_DIM,
                                  ck.get("hidden", 256),
                                  ck.get("depth", 3)).to(self.device)
        m.load_state_dict(sd)
        m.eval()
        self.models.append(m)

    # ---------- 归一化 ----------
    def to_unit(self, x_real):
        t = self.torch.as_tensor(x_real, dtype=self.torch.float32, device=self.device)
        return (t - self.x_lo) / (self.x_hi - self.x_lo + 1e-12)

    def from_unit(self, x01):
        t = self.torch.as_tensor(x01, dtype=self.torch.float32, device=self.device)
        return self.x_lo + t * (self.x_hi - self.x_lo)

    # ---------- 前向 ----------
    def forward_unit(self, x01_tensor):
        """输入 [0,1] 归一化参数，返回**标准化 y**（若集成，取均值）。可微。"""
        if x01_tensor.dim() == 1:
            x01_tensor = x01_tensor.unsqueeze(0)
        preds = [m(x01_tensor) for m in self.models]
        if len(preds) == 1:
            return preds[0]
        return self.torch.stack(preds, dim=0).mean(dim=0)

    def forward_unit_all(self, x01_tensor):
        """返回 (n_models, B, Y_DIM) —— 用于计算成员间的 std。"""
        if x01_tensor.dim() == 1:
            x01_tensor = x01_tensor.unsqueeze(0)
        return self.torch.stack([m(x01_tensor) for m in self.models], dim=0)

    def predict(self, x_real, return_std=False):
        """numpy 接口 —— 输入真实参数，返回 dB 曲线 (B, Y_DIM)。"""
        x_real = np.atleast_2d(np.asarray(x_real, dtype=np.float32))
        with self.torch.no_grad():
            x01 = self.to_unit(x_real)
            all_pred = self.forward_unit_all(x01)          # (M, B, Y)
            all_db = all_pred * self.y_std + self.y_mu
            mean_db = all_db.mean(dim=0).cpu().numpy()
            if return_std:
                std_db = all_db.std(dim=0).cpu().numpy()
                return mean_db, std_db
            return mean_db

    def predict_curves(self, x_real):
        """预测并拆成 (freqs, s21_db, s11_db)。x_real 为单点。"""
        y = self.predict(x_real)[0]
        s21_db, s11_db = split_y(y)
        return self.freqs, s21_db, s11_db


# =====================================================================
# CLI
# =====================================================================
def _eval(npz_path, ckpt_path):
    fm = ForwardModel(ckpt_path)
    d = load_dataset(npz_path)
    X, Y = d["X"], d["Y"]
    pred = fm.predict(X)
    mae = float(np.mean(np.abs(pred - Y)))
    mae_s21 = float(np.mean(np.abs(pred[:, :C.N_FREQ] - Y[:, :C.N_FREQ])))
    tag = f"ensemble({len(fm.models)})" if fm.is_ensemble else "single"
    print(f"[eval] [{tag}] 全曲线 MAE={mae:.3f} dB | 仅S21 MAE={mae_s21:.3f} dB | N={X.shape[0]}")


def main():
    ap = argparse.ArgumentParser(description="正向代理模型 (Layer1 / M2) — filter5 电路定制")
    ap.add_argument("--train", action="store_true", help="训练模型")
    ap.add_argument("--eval", action="store_true", help="在数据集上评估")
    ap.add_argument("--data", default=C.DATASET_NPZ)
    ap.add_argument("--ckpt", default=FORWARD_CKPT)
    ap.add_argument("--ensemble", type=int, default=1, help="集成成员个数（>=2 启用）")
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--diff_reg", type=float, default=0.15,
                    help="一阶差分正则强度（0 关闭）")
    ap.add_argument("--sharp_boost", type=float, default=6.0,
                    help="频段自适应权重强度：陡变频点最大权重 = base + sharp_boost")
    ap.add_argument("--clip_db", type=float, default=-60.0,
                    help="训练前把 S 参数低于该值截断（robust；设很小如 -200 可关闭）")
    ap.add_argument("--huber_delta", type=float, default=0.0,
                    help="Huber loss 阈值（标准化空间，0=用纯加权 MSE）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.train:
        train(
            args.data, args.ckpt,
            ensemble=args.ensemble,
            hidden=args.hidden, depth=args.depth, dropout=args.dropout,
            epochs=args.epochs, batch=args.batch, lr=args.lr, weight_decay=args.wd,
            warmup=args.warmup, patience=args.patience,
            diff_reg=args.diff_reg, sharp_boost=args.sharp_boost,
            clip_db=args.clip_db, huber_delta=args.huber_delta,
            seed=args.seed,
        )
    if args.eval:
        _eval(args.data, args.ckpt)
    if not args.train and not args.eval:
        print("请指定 --train 或 --eval")


if __name__ == "__main__":
    main()
