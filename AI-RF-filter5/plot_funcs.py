import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

matplotlib.use("Agg")  # 无显示器环境也能保存 PNG
import matplotlib.pyplot as plt

import skrf as rf


# 绘制s2p文件的s参数图 解析 .sNp 文件并绘制 S11 / S21 dB 曲线, 保存到 out_path。
# s2p_path: 输入 Touchstone 文件路径 (.s1p / .s2p / .sNp 均可)
# out_path: 输出图片路径 (.png)
# 返回bool: True 表示成功, False 表示失败
def plot_s2p(s2p_path: str, out_path: str) -> bool:
    try:
        nw = rf.Network(s2p_path)
    except Exception as e:
        print(f"[plot_sp] 解析失败: {s2p_path} -> {e}")
        return False

    f_ghz = nw.f / 1e9
    s11 = nw.s[:, 0, 0]
    s21 = nw.s[:, 1, 0] if nw.nports >= 2 else None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(f_ghz, 20 * np.log10(np.abs(s11) + 1e-12), label="S11 (dB)", color="#2196F3", linewidth=1.5)
    if s21 is not None:
        ax.plot(f_ghz, 20 * np.log10(np.abs(s21) + 1e-12), label="S21 (dB)", color="#F44336", linewidth=1.5)

    ax.axhline(-10, color="gray", linestyle="--", alpha=0.5, label="-10 dB")
    ax.set_xlabel("Frequency (GHz)", fontsize=12)
    ax.set_ylabel("|S| (dB)", fontsize=12)
    ax.set_title(os.path.basename(s2p_path), fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")

    # 纵轴: 固定 -50 ~ 0 dB, 主刻度间隔 10
    ax.set_ylim(-50, 0)
    ax.set_yticks(np.arange(-50, 0 + 1e-9, 10))

    # 横轴: 显示全频段 (去掉 matplotlib 默认边距, 曲线贴边)
    ax.set_xlim(f_ghz[0], f_ghz[-1])
    ax.margins(x=0)

    try:
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
    except Exception as e:
        # print(f"[plot_sp] 保存失败: {out_path} -> {e}")
        plt.close(fig)
        return False
    finally:
        plt.close(fig)

    # print(f"[plot_sp] OK: {s2p_path} -> {out_path}")
    return True


# 将两张图片进行拼接, 主要是拼接版图和s曲线图
def plot_combine(layout_path: str, sp_path: str, out_path: str, gap: int = 30):
    layout = Image.open(layout_path).convert("RGB")
    sp = Image.open(sp_path).convert("RGB")

    # 按高度对齐
    target_h = max(layout.height, sp.height)
    if layout.height != target_h:
        layout = layout.resize((int(layout.width * target_h / layout.height), target_h))
    if sp.height != target_h:
        sp = sp.resize((int(sp.width * target_h / sp.height), target_h))

    out = Image.new(
        "RGB",
        (layout.width + gap + sp.width, target_h),
        "white",
    )
    out.paste(layout, (0, 0))
    out.paste(sp, (layout.width + gap, 0))
    out.save(out_path)
    return out_path


if __name__ == "__main__":
    print("plot_funcs库")
