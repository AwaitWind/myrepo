"""PaddleOCR 权重准备。

三条路,按推荐顺序:

  1. 自动下载（推荐）
     `paddleocr` 包首次运行会自己从 paddleocr.bj.bcebos.com 拉权重。
     那是百度云的**国内** CDN，国内机器（AutoDL 等）通常直连没问题，
     不需要学术加速——学术加速是给 github.com 用的。
     本项目的开发沙箱访问不到该域名，但那是沙箱的网络限制，不是普遍情况。

  2. 手动下载 inference 模型（离线机器）
     用 `python3 paddleocr_model/weights.py --manual` 下载下面 URLS 里的 tar 并解包。
     注意：PaddleOCR 的模型版本迭代很快（PP-OCRv3/v4/v5...），下面的地址可能失效。
     失效时去官方仓库的模型列表查最新地址，用 --url-det / --url-rec 覆盖。

  3. 第三方镜像（bcebos 不通时）
     ModelScope（魔搭，国内可达）或 HF 镜像站都有 PaddleOCR 模型。
     下好后用 --model-dir 指过去，本模块不内置这类地址——它们变动频繁，
     写死在代码里反而会误导排查。

校验用"能解包 + 目录里有 inference 模型文件"，不看 HTTP 状态码。
踩过的坑：`curl -I` 返回 200 不代表能下载，body 传输可能超时而文件从未落地。
"""

from __future__ import annotations

import argparse
import os
import shutil
import tarfile
import time
import urllib.error
import urllib.request

UA = "Mozilla/5.0 (compatible; pcb-schematic-tooling/1.0)"

# 可能失效，失效时用 --url-det / --url-rec 覆盖。det 其实我们用不上
# （文本框由 YOLO #2 给），列在这里是为了能完整离线打包。
URLS = {
    "det": "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar",
    "rec": "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar",
    "cls": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
}

DEFAULT_DIR = os.environ.get("PCB_OCR_MODEL_DIR", "weights/paddleocr")
# 和 yolo_model/common/weights.py 同样的镜像机制
MIRROR_ENV = "PCB_OCR_MIRROR"


def resolve(url: str) -> str:
    m = os.environ.get(MIRROR_ENV, "").strip()
    if not m:
        return url
    tail = url.split("://", 1)[-1].split("/", 1)[-1]
    return m.rstrip("/") + "/" + tail


def _looks_like_model(d: str) -> bool:
    """inference 模型目录里应当有 .pdmodel/.pdiparams，或新版的 .json/.pdiparams。"""
    if not os.path.isdir(d):
        return False
    names = os.listdir(d)
    has_param = any(n.endswith(".pdiparams") for n in names)
    has_graph = any(n.endswith((".pdmodel", ".json")) for n in names)
    return has_param and has_graph


def download(url: str, dest: str, retries: int = 6, timeout: float = 60.0) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    part = dest + ".part"
    url = resolve(url)
    print(f"[ocr-weights] {url}")
    for attempt in range(1, retries + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        headers = {"User-Agent": UA}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=timeout
            ) as resp:
                mode = "ab" if (have and resp.status == 206) else "wb"
                with open(part, mode) as fh:
                    shutil.copyfileobj(resp, fh, 1 << 18)
            break
        except (urllib.error.URLError, OSError, TimeoutError) as err:
            got = os.path.getsize(part) if os.path.exists(part) else 0
            print(f"[ocr-weights] 第 {attempt}/{retries} 次中断于 {got} 字节: {err}")
            if attempt == retries:
                raise RuntimeError(
                    f"下载未完成（已落地 {got} 字节）。可选办法:\n"
                    "  - 重跑本命令继续续传\n"
                    "  - 直接 `pip install paddleocr` 让它自动下（国内机器通常可行）\n"
                    f"  - 设 {MIRROR_ENV}=<镜像前缀> 走镜像\n"
                    "  - 从 ModelScope / HF 镜像下好后用 --model-dir 指过去\n"
                    "  - 或改用 easyocr 后端（识别率差些，但造伪标签够用）"
                ) from err
            time.sleep(min(2 ** attempt, 15))
    shutil.move(part, dest)
    return dest


def extract(tar_path: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            # 防目录穿越：tar 里的相对路径不能跳出目标目录
            p = os.path.normpath(os.path.join(out_dir, m.name))
            if not p.startswith(os.path.abspath(out_dir) + os.sep) and p != os.path.abspath(out_dir):
                raise RuntimeError(f"tar 内含可疑路径，已拒绝: {m.name}")
        tf.extractall(out_dir)
    # tar 里通常带一层同名目录，找到真正含模型文件的那层
    if _looks_like_model(out_dir):
        return out_dir
    for name in sorted(os.listdir(out_dir)):
        sub = os.path.join(out_dir, name)
        if _looks_like_model(sub):
            return sub
    return out_dir


def probe_auto() -> bool:
    """试探 paddleocr 能否自动拿到权重。这是最省事的一条路，先试它。"""
    try:
        from paddleocr import PaddleOCR
    except ImportError as err:
        print(f"[ocr-weights] paddleocr 未安装: {err}")
        print("[ocr-weights] 装法: pip install paddlepaddle paddleocr")
        print("[ocr-weights] 建议装 CPU 版（不带 -gpu），避免和 torch 的 CUDA 打架；"
              "伪标签生成是一次性批处理，慢点无妨。")
        return False
    try:
        PaddleOCR(lang="ch")
        print("[ocr-weights] paddleocr 自动下载成功，权重就绪，无需手动准备。")
        return True
    except Exception as err:   # 下载/初始化的异常类型不可预知
        print(f"[ocr-weights] 自动下载失败: {type(err).__name__}: {err}")
        return False


def manual(a) -> None:
    urls = dict(URLS)
    if a.url_det:
        urls["det"] = a.url_det
    if a.url_rec:
        urls["rec"] = a.url_rec
    os.makedirs(a.model_dir, exist_ok=True)
    for kind in a.kinds:
        sub = os.path.join(a.model_dir, kind)
        if _looks_like_model(sub):
            print(f"[ocr-weights] {kind} 已就绪: {sub}")
            continue
        tar_path = os.path.join(a.model_dir, f"{kind}.tar")
        if not os.path.exists(tar_path):
            download(urls[kind], tar_path)
        real = extract(tar_path, sub)
        if _looks_like_model(real):
            print(f"[ocr-weights] {kind} 解包完成: {real}")
        else:
            print(f"[ocr-weights] 警告: {sub} 里没找到 inference 模型文件，"
                  "可能是地址过期下到了错误内容，请核对 URL")
    print(f"\n[ocr-weights] 用法: --ocr-model-dir {a.model_dir}")


def main():
    ap = argparse.ArgumentParser(description="准备 PaddleOCR 权重")
    ap.add_argument("--manual", action="store_true",
                    help="手动下载 inference 模型（离线机器用）。默认只试自动下载")
    ap.add_argument("--model-dir", default=DEFAULT_DIR)
    ap.add_argument("--kinds", nargs="+", default=["rec", "det"],
                    choices=["det", "rec", "cls"],
                    help="det 我们其实用不上（文本框由 YOLO #2 给），"
                         "列出来是为了能完整离线打包")
    ap.add_argument("--url-det", default=None)
    ap.add_argument("--url-rec", default=None)
    a = ap.parse_args()

    if a.manual:
        manual(a)
        return
    if not probe_auto():
        print("\n[ocr-weights] 自动方式没成。试试:")
        print("  python3 paddleocr_model/weights.py --manual")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
