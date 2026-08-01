"""
Layer 1 —— 正向代理模型（Forward Surrogate, M2）· **Frequency-Query 版**。

【与旧 inverse_design/forward_net.py 的根本区别】
  旧版 ResMLP：  x(11维参数)              → 一次性回归整条 200 维曲线
  本版 FQ-Net：  (x, f) —— 参数 + 频率查询 → 该频点的 (S21, S11)

  即把"频率"从"输出维度的下标"提升为"网络的连续输入"，逐频点查询后拼成整条曲线。
  这与整曲线 MLP 在原理上完全不同，灵感来自
  《Frequency-Query Enhanced EM Surrogate Modeling for Bandpass Filter Inverse Design》。

【三大定制（针对 filter5 多谐振 + 传输零点曲线）】
  1. 傅里叶特征编码频率  γ(f) = [sin(2^k·πf), cos(2^k·πf)]，频带对齐奈奎斯特。
     振荡基天生适配「通带内多匹配谷 + 通带外传输零点」这种振荡结构。
  2. FiLM 条件调制：参数 x 经超网络（hypernet）生成每层的 (scale, shift)，
     调制频率坐标网络。参数以"逐层特征仿射"方式注入，而非简单拼接。
  3. 物理软上界：输出 dB = -softplus(-raw)，恒 ≤ 0 dB（无源器件 |S|≤1）。
     从根本上杜绝代理外推出 >0 dB 的"假通带"（这是逆向被对抗性欺骗的元凶）。

对外仍暴露与旧版**完全一致**的 ForwardModel 接口
（forward_unit / forward_unit_all / predict / y_mu_np / ...），
因此 inverse_optimize.py / validate.py 无需改动即可复用，可与旧方案公平对比。

用法:
    python forward_net.py --train                # 训 1 个模型
    python forward_net.py --train --ensemble 5   # 5 个 seed 集成（推荐）
    python forward_net.py --eval                 # 数据集上评估
"""

import argparse
import glob
import math
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

# ---- 数据复用：本目录若无 dataset.npz，自动回退到旧 inverse_design 的数据 ----
#      （新代理只换内核，沿用同一份 HFSS 数据，无需重跑仿真，且便于公平对比）
DATASET_NPZ = C.DATASET_NPZ
if not os.path.exists(DATASET_NPZ):
    _LEGACY_NPZ = os.path.join(os.path.dirname(_THIS_DIR),
                               "inverse_design", "id_data", "dataset.npz")
    if os.path.exists(_LEGACY_NPZ):
        DATASET_NPZ = _LEGACY_NPZ


# =====================================================================
# 频率归一化工具
# =====================================================================
def _freqs_to_norm(freqs):
    """把频率网格线性归一化到 [0,1]，供傅里叶特征编码。"""
    f = np.asarray(freqs, dtype=np.float32)
    lo, hi = float(f[0]), float(f[-1])
    return ((f - lo) / (hi - lo + 1e-12)).astype(np.float32)


# =====================================================================
# 网络定义：Frequency-Query（傅里叶特征 + FiLM 调制 + 物理软上界）
# =====================================================================
def _build_freq_query(param_dim, y_dim, f_norm,
                      n_bands=10, hidden=256, depth=5, dropout=0.05):
    """
    构建 Frequency-Query 网络：
      输入   x01 (B, param_dim)   —— [0,1] 归一化的设计参数
      内部   对固定频率网格 f_norm 的每个频点做傅里叶编码 + FiLM 调制
      输出   (B, y_dim) 的 dB 曲线（[S21(N_FREQ), S11(N_FREQ)] 或仅 S21），恒 ≤ 0 dB
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    n_freq = len(f_norm)
    out_per_freq = 2 if y_dim == 2 * n_freq else 1

    class FreqQueryNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_freq = n_freq
            self.out_per_freq = out_per_freq
            self.depth = depth
            self.hidden = hidden

            # 频率网格 + 傅里叶频带（对齐奈奎斯特：最高频带 ~ n_freq/2 个周期）
            self.register_buffer(
                "f_norm", torch.as_tensor(f_norm, dtype=torch.float32))
            max_log2 = math.log2(max(2.0, n_freq / 2.0))
            bands = (2.0 ** torch.linspace(0.0, max_log2, n_bands)) * math.pi
            self.register_buffer("bands", bands)

            ff_dim = 2 * n_bands
            self.in_proj = nn.Linear(ff_dim, hidden)
            self.act = nn.GELU()
            self.dp = nn.Dropout(dropout)
            self.fcs = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(depth)])

            # 超网络：设计参数 → 每层 FiLM 的 (scale, shift)
            self.hyper = nn.Sequential(
                nn.Linear(param_dim, hidden), nn.GELU(),
                nn.Linear(hidden, depth * 2 * hidden),
            )
            # 零初始化最后一层 → 初始 film=0 → scale=1/shift=0（起步为纯频率网络，稳定）
            nn.init.zeros_(self.hyper[-1].weight)
            nn.init.zeros_(self.hyper[-1].bias)

            self.head = nn.Linear(hidden, out_per_freq)

        def _fourier(self):
            # (n_freq, 2*n_bands)
            proj = self.f_norm.unsqueeze(-1) * self.bands.unsqueeze(0)  # (F, n_bands)
            return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

        def forward(self, x01):
            if x01.dim() == 1:
                x01 = x01.unsqueeze(0)
            B = x01.shape[0]

            feat = self._fourier()                    # (F, 2n)
            h = self.act(self.in_proj(feat))          # (F, H)
            h = h.unsqueeze(0).expand(B, -1, -1)      # (B, F, H)

            film = self.hyper(x01).view(B, self.depth, 2, self.hidden)
            for i, fc in enumerate(self.fcs):
                scale = 1.0 + film[:, i, 0].unsqueeze(1)   # (B,1,H) → 广播到 F
                shift = film[:, i, 1].unsqueeze(1)
                h = h + self.dp(self.act(fc(h) * scale + shift))   # FiLM + 残差

            raw = self.head(h)                        # (B, F, out_per_freq)
            db = -F.softplus(-raw)                    # 物理软上界: dB <= 0
            if self.out_per_freq == 1:
                return db.squeeze(-1)                 # (B, F) = (B, Y_DIM)
            s21 = db[..., 0]                          # (B, F)
            s11 = db[..., 1]                          # (B, F)
            return torch.cat([s21, s11], dim=1)       # (B, 2F) = (B, Y_DIM)

    return FreqQueryNet()


# =====================================================================
# 频段自适应权重（放大陡变频点：传输零点 / 匹配谷 / 通带边）
# =====================================================================
def build_freq_weight(Y, sharp_boost=6.0, base=1.0):
    """
    根据训练集 Y 的一阶差分幅度给每个频点分配权重（越陡权重越高）。
    S21 段 / S11 段分别构造，避免相互压制。返回 (Y_DIM,)，均值≈1。
    """
    W = np.ones(C.Y_DIM, dtype=np.float32) * base

    def _one_channel(sl):
        y = Y[:, sl]
        dy = np.abs(np.diff(y, axis=1))
        d_left = np.pad(dy, ((0, 0), (1, 0)), mode="edge")
        d_right = np.pad(dy, ((0, 0), (0, 1)), mode="edge")
        dmag = np.maximum(d_left, d_right).mean(axis=0)
        if dmag.max() > 1e-6:
            dmag /= dmag.max()
        return base + sharp_boost * dmag

    W[:C.N_FREQ] = _one_channel(slice(0, C.N_FREQ))
    if C.INCLUDE_S11:
        W[C.N_FREQ:2 * C.N_FREQ] = _one_channel(slice(C.N_FREQ, 2 * C.N_FREQ))

    W *= (Y.shape[1] / W.sum())
    return W


# =====================================================================
# 单模型训练
# =====================================================================
def train_one(
    X, Y, x_lo, x_hi, y_mu, y_std, freq_w, f_norm,
    ckpt_path,
    n_bands=10, hidden=256, depth=5, dropout=0.05,
    epochs=800, batch=256, lr=3e-4, weight_decay=1e-4,
    warmup=20, patience=100, val_ratio=0.15,
    diff_reg=0.15, huber_delta=0.0,
    seed=42, device=None, verbose=True, freqs=None, include_s11=True,
):
    """训练单个 Frequency-Query 模型并保存 ckpt。返回 (val_mae_db, ckpt_path)。"""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    N = X.shape[0]
    X01 = ((X - x_lo) / (x_hi - x_lo + 1e-12)).astype(np.float32)
    # 直接在 dB 空间训练（网络输出即 dB）：避免除以极小 y_std（近似常数频点 std≈0）
    # 导致标准化目标爆炸、loss 达 1e13、优化被个别频点绑架。
    Ydb = Y.astype(np.float32)

    idx = np.random.permutation(N)
    n_val = max(1, int(N * val_ratio))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    Xtr = torch.tensor(X01[tr_idx], device=device)
    Ytr = torch.tensor(Ydb[tr_idx], device=device)
    Xval = torch.tensor(X01[val_idx], device=device)
    Yval = torch.tensor(Ydb[val_idx], device=device)

    fw = torch.tensor(freq_w, dtype=torch.float32, device=device)   # (Y_DIM,)

    model = _build_freq_query(C.N_PARAMS, C.Y_DIM, f_norm,
                              n_bands=n_bands, hidden=hidden,
                              depth=depth, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / max(1, warmup)
        t = (ep - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, t)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    def weighted_loss(pred_db, target_db):
        """直接在 dB 空间的频段加权损失（fw 均值≈1），数值稳定且可解释。"""
        err = pred_db - target_db
        if huber_delta > 0:
            absr = err.abs()
            quad = torch.clamp(absr, max=huber_delta)
            per = 0.5 * quad ** 2 + huber_delta * (absr - quad)
            loss = (per * fw).mean()
        else:
            loss = (err ** 2 * fw).mean()
        if diff_reg > 0:
            dp = pred_db[:, 1:] - pred_db[:, :-1]
            dt = target_db[:, 1:] - target_db[:, :-1]
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
            pred = model(xb)                       # (B, Y_DIM) dB
            loss = weighted_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += loss.item() * xb.shape[0]
        sched.step()

        model.eval()
        with torch.no_grad():
            pv_db = model(Xval)                     # dB
            yv_db = Yval                            # 已是 dB
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
        pv_db = model(Xval).cpu().numpy()
        yv_db = Yval.cpu().numpy()                  # 已是 dB
        final_mae = float(np.mean(np.abs(pv_db - yv_db)))
        final_mae_s21 = float(np.mean(np.abs(pv_db[:, :C.N_FREQ] - yv_db[:, :C.N_FREQ])))

    ckpt = {
        "model": model.state_dict(),
        "arch": "freq_query",                  # 架构标识
        "n_bands": n_bands, "hidden": hidden, "depth": depth, "dropout": dropout,
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
    torch.save(ckpt, ckpt_path)
    if verbose:
        print(f"  [seed{seed}] 完成: val MAE={final_mae:.3f} dB (S21 {final_mae_s21:.3f} dB)  → {ckpt_path}")
    return final_mae, ckpt_path


# =====================================================================
# 顶层训练入口（支持单模型或 ensemble）
# =====================================================================
def train(
    npz_path=DATASET_NPZ,
    ckpt_path=FORWARD_CKPT,
    ensemble=1,
    n_bands=10, hidden=256, depth=5, dropout=0.05,
    epochs=800, batch=256, lr=3e-4, weight_decay=1e-4,
    warmup=20, patience=100, val_ratio=0.15,
    diff_reg=0.15, sharp_boost=6.0, base_w=1.0,
    clip_db=-60.0, huber_delta=0.0,
    seed=42, device=None,
):
    d = load_dataset(npz_path)
    X = d["X"].astype(np.float32)
    Y = d["Y"].astype(np.float32)
    N = X.shape[0]
    print(f"[train] 数据集 {npz_path}")
    print(f"[train] N={N}, X={X.shape}, Y={Y.shape}, ensemble={ensemble} (arch=freq_query)")
    if N < 20:
        print("[train][警告] 样本过少，建议先 gen_dataset 生成更多数据。")

    # robust：截断 HFSS 偶发的极深数值噪声深谷
    if clip_db is not None:
        below = (Y < clip_db).sum()
        if below > 0:
            print(f"[train] clip: 将 {int(below)} 个 < {clip_db} dB 的点截断到 {clip_db} dB "
                  f"(占比 {below / Y.size * 100:.3f}%)")
        Y = np.maximum(Y, clip_db).astype(np.float32)

    x_lo = C.LOWER.astype(np.float32)
    x_hi = C.UPPER.astype(np.float32)
    y_mu = Y.mean(axis=0).astype(np.float32)
    # floor 1.0 dB：近似常数频点 std≈0 会让标准化中间量爆炸，抬高下限更稳
    y_std = np.maximum(Y.std(axis=0), 1.0).astype(np.float32)

    freq_w = build_freq_weight(Y, sharp_boost=sharp_boost, base=base_w)
    f_norm = _freqs_to_norm(d["freqs"])
    print(f"[train] freq_w S21 min/max/mean = "
          f"{freq_w[:C.N_FREQ].min():.2f} / {freq_w[:C.N_FREQ].max():.2f} / "
          f"{freq_w[:C.N_FREQ].mean():.2f} | 傅里叶频带数 n_bands={n_bands}")

    ckpt_dir = os.path.dirname(ckpt_path) or "."
    os.makedirs(ckpt_dir, exist_ok=True)

    if ensemble <= 1:
        val_mae, _ = train_one(
            X, Y, x_lo, x_hi, y_mu, y_std, freq_w, f_norm,
            ckpt_path=ckpt_path,
            n_bands=n_bands, hidden=hidden, depth=depth, dropout=dropout,
            epochs=epochs, batch=batch, lr=lr, weight_decay=weight_decay,
            warmup=warmup, patience=patience, val_ratio=val_ratio,
            diff_reg=diff_reg, huber_delta=huber_delta, seed=seed, device=device,
            freqs=d["freqs"], include_s11=d["include_s11"],
        )
        print(f"[train] 完成。val MAE = {val_mae:.3f} dB → {ckpt_path}")
        return val_mae

    maes, paths = [], []
    for k in range(ensemble):
        seed_k = seed + k * 1000 + 1
        pk = os.path.join(ckpt_dir, f"forward_net_seed{k}.pt")
        print(f"\n[train] ===== ensemble {k+1}/{ensemble} (seed={seed_k}) =====")
        mae, p = train_one(
            X, Y, x_lo, x_hi, y_mu, y_std, freq_w, f_norm,
            ckpt_path=pk,
            n_bands=n_bands, hidden=hidden, depth=depth, dropout=dropout,
            epochs=epochs, batch=batch, lr=lr, weight_decay=weight_decay,
            warmup=warmup, patience=patience, val_ratio=val_ratio,
            diff_reg=diff_reg, huber_delta=huber_delta, seed=seed_k, device=device,
            freqs=d["freqs"], include_s11=d["include_s11"],
        )
        maes.append(mae)
        paths.append(p)

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
# 推理引擎（接口与旧 ResMLP 版完全一致，供 inverse_optimize / validate 复用）
# =====================================================================
class ForwardModel:
    """
    统一封装：单模型 or ensemble。对外接口与旧版一致：
      fm.predict(x_real, return_std=False)  → (B, Y_DIM) dB 或 (mean, std)
      fm.forward_unit(x01)                  → 标准化 y（可微；集成取均值）
      fm.forward_unit_all(x01)              → (n_models, B, Y_DIM) 标准化
      fm.y_mu_np / fm.y_std_np / fm.device / fm.is_ensemble / fm.freqs ...
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
                for pp in ck["ensemble_paths"]:
                    if not os.path.exists(pp):
                        pp = os.path.join(os.path.dirname(p), os.path.basename(pp))
                    ck_m = torch.load(pp, map_location=self.device, weights_only=False)
                    self._append_model(ck_m)
                    meta = meta or ck_m
                continue
            self._append_model(ck)
            meta = meta or ck

        if not self.models:
            raise RuntimeError(f"未从 {ckpt} 加载到任何模型")

        self.x_lo = torch.tensor(np.asarray(meta["x_lo"], dtype=np.float32), device=self.device)
        self.x_hi = torch.tensor(np.asarray(meta["x_hi"], dtype=np.float32), device=self.device)

        y_mu = np.asarray(meta["y_mu"], dtype=np.float32)
        y_std = np.asarray(meta["y_std"], dtype=np.float32)
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
        freqs = np.asarray(ck.get("freqs", C.FREQ_GHZ))
        f_norm = _freqs_to_norm(freqs)
        m = _build_freq_query(
            C.N_PARAMS, C.Y_DIM, f_norm,
            n_bands=ck.get("n_bands", 10),
            hidden=ck.get("hidden", 256),
            depth=ck.get("depth", 5),
            dropout=ck.get("dropout", 0.0),
        ).to(self.device)
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

    # ---------- 前向（返回标准化 y，可微） ----------
    def _one_norm(self, m, x01_tensor):
        """单模型: [0,1] 参数 → 标准化 y（dB 输出减均值除标准差）。"""
        y_db = m(x01_tensor)                       # (B, Y_DIM) dB
        return (y_db - self.y_mu) / self.y_std

    def forward_unit(self, x01_tensor):
        if x01_tensor.dim() == 1:
            x01_tensor = x01_tensor.unsqueeze(0)
        preds = [self._one_norm(m, x01_tensor) for m in self.models]
        if len(preds) == 1:
            return preds[0]
        return self.torch.stack(preds, dim=0).mean(dim=0)

    def forward_unit_all(self, x01_tensor):
        if x01_tensor.dim() == 1:
            x01_tensor = x01_tensor.unsqueeze(0)
        return self.torch.stack([self._one_norm(m, x01_tensor) for m in self.models], dim=0)

    def predict(self, x_real, return_std=False):
        """numpy 接口 —— 输入真实参数，返回 dB 曲线 (B, Y_DIM)。"""
        x_real = np.atleast_2d(np.asarray(x_real, dtype=np.float32))
        with self.torch.no_grad():
            x01 = self.to_unit(x_real)
            all_norm = self.forward_unit_all(x01)          # (M, B, Y) 标准化
            all_db = all_norm * self.y_std + self.y_mu
            mean_db = all_db.mean(dim=0).cpu().numpy()
            if return_std:
                std_db = all_db.std(dim=0).cpu().numpy()
                return mean_db, std_db
            return mean_db

    def predict_curves(self, x_real):
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
    over = float((pred[:, :C.N_FREQ] > 0.0).mean()) * 100.0
    tag = f"ensemble({len(fm.models)})" if fm.is_ensemble else "single"
    print(f"[eval] [{tag}] 全曲线 MAE={mae:.3f} dB | 仅S21 MAE={mae_s21:.3f} dB | "
          f"预测 S21>0 占比={over:.2f}% | N={X.shape[0]}")


def main():
    ap = argparse.ArgumentParser(description="正向代理 (Layer1/M2) — Frequency-Query 版")
    ap.add_argument("--train", action="store_true", help="训练模型")
    ap.add_argument("--eval", action="store_true", help="在数据集上评估")
    ap.add_argument("--data", default=DATASET_NPZ)
    ap.add_argument("--ckpt", default=FORWARD_CKPT)
    ap.add_argument("--ensemble", type=int, default=1, help="集成成员个数（>=2 启用）")
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--n_bands", type=int, default=10,
                    help="傅里叶频率编码的频带数（越大能表达越高频振荡）")
    ap.add_argument("--hidden", type=int, default=256, help="FiLM 主干隐藏维")
    ap.add_argument("--depth", type=int, default=5, help="FiLM 主干层数")
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
            n_bands=args.n_bands, hidden=args.hidden, depth=args.depth, dropout=args.dropout,
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
