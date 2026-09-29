"""权重获取：本地优先，缺了就下载，支持断点续传。

为什么要自己写下载而不用 ultralytics 的自动下载:
  1. 本机实测 github.com/ultralytics/assets 的传输速度只有 ~63 KB/s，
     5.6 MB 的 yolo11n.pt 在默认超时下会中途断掉、留下半个文件；
     ultralytics 自动下载不续传，重跑就从零开始。
  2. 赛会评测平台很可能没有外网（`赛题六_模型设计.md` §3.0.1），
     权重必须能提前下好、随代码走本地路径。

判定陷阱（实际踩过）:
  `curl -I` 返回 200 **不代表能下载**。HEAD 请求对 YOLO 权重返回 200，
  但 body 传输会超时、文件从未落地。所以这里的校验是
  "完整字节数 + 能被 zipfile 打开"，不看 HTTP 状态码。
"""

from __future__ import annotations

import os
import shutil
import time
import urllib.error
import urllib.request
import zipfile

UA = "Mozilla/5.0 (compatible; pcb-schematic-tooling/1.0)"

# 写死在代码里，换到能联网的机器直接跑即可
URLS = {
    "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
    "yolo11s.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
    "yolo11m.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
    "yolo11l.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt",
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
    "yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt",
}

DEFAULT_CACHE = os.environ.get("PCB_WEIGHTS_DIR", "weights")

# 国内机器（如 AutoDL）直连 github.com 常见 503 / 超时。设这个环境变量可以走镜像：
#   export PCB_GH_MIRROR=https://<你的镜像前缀>/
# 代码会把 https://github.com/ 替换成该前缀。具体用哪个镜像由你决定，
# 这里不内置任何第三方镜像地址（它们经常失效，写死反而误导）。
# AutoDL 用户更简单的办法是先 `source /etc/network_turbo` 开学术加速。
GH_PREFIX = "https://github.com/"


def resolve_url(url: str) -> str:
    mirror = os.environ.get("PCB_GH_MIRROR", "").strip()
    if mirror and url.startswith(GH_PREFIX):
        return mirror.rstrip("/") + "/" + url[len(GH_PREFIX):]
    return url


def _remote_size(url: str, timeout: float) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            n = resp.headers.get("Content-Length")
            return int(n) if n else None
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _valid(path: str, expect: int | None) -> bool:
    """完整性校验：字节数对得上，且是合法的 torch 存档（zip 容器）。"""
    if not os.path.exists(path):
        return False
    size = os.path.getsize(path)
    if size == 0:
        return False
    if expect and size != expect:
        return False
    if path.endswith(".pt"):
        return zipfile.is_zipfile(path)
    return True


def download(url: str, dest: str, retries: int = 8, chunk_timeout: float = 60.0,
             verbose: bool = True) -> str:
    """带断点续传的下载。每轮从已有字节数接着拉，直到字节数对上。"""
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    part = dest + ".part"
    real = resolve_url(url)
    if verbose and real != url:
        print(f"[weights] 走镜像 PCB_GH_MIRROR")
    url = real
    expect = _remote_size(url, chunk_timeout)
    if verbose:
        print(f"[weights] {url}")
        print(f"[weights] 目标大小: {expect if expect else '未知'}")

    for attempt in range(1, retries + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if expect and have >= expect:
            break
        headers = {"User-Agent": UA}
        if have:
            headers["Range"] = f"bytes={have}-"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=chunk_timeout) as resp:
                # 服务端忽略 Range 时会回 200 并从头给，此时要重开文件
                mode = "ab" if (have and resp.status == 206) else "wb"
                if mode == "wb":
                    have = 0
                with open(part, mode) as fh:
                    while True:
                        buf = resp.read(1 << 18)
                        if not buf:
                            break
                        fh.write(buf)
                        have += len(buf)
        except (urllib.error.URLError, OSError, TimeoutError) as err:
            got = os.path.getsize(part) if os.path.exists(part) else 0
            if verbose:
                pct = f" ({got * 100 // expect}%)" if expect else ""
                print(f"[weights] 第 {attempt}/{retries} 次中断于 {got} 字节{pct}: {err}")
            if attempt == retries:
                raise RuntimeError(
                    f"下载未完成，已落地 {got} 字节。\n"
                    "  - 重跑本命令可继续续传（已下载的字节不会重来）\n"
                    "  - AutoDL 等国内机器直连 github.com 常见 503/超时，"
                    "可先 `source /etc/network_turbo` 开学术加速\n"
                    "  - 或设 PCB_GH_MIRROR=<镜像前缀> 走镜像\n"
                    f"  - 或在别的机器下好后放到 {dest}\n"
                    "  - 也可以先用 --model <名字>.yaml 随机初始化跑起来，"
                    "那正是 §3.0.2 迁移有效性对照的一条臂"
                ) from err
            time.sleep(min(2 ** attempt, 15))
            continue
        if expect is None:
            break

    if not _valid(part, expect):
        got = os.path.getsize(part) if os.path.exists(part) else 0
        raise RuntimeError(
            f"下载校验失败: {part} 有 {got} 字节，期望 {expect}，"
            "或不是合法的 torch 存档。重跑可续传。"
        )
    shutil.move(part, dest)
    if verbose:
        print(f"[weights] 完成: {dest} ({os.path.getsize(dest)} 字节)")
    return dest


def ensure(model: str, cache_dir: str = DEFAULT_CACHE, allow_download: bool = True,
           verbose: bool = True) -> str:
    """解析 --model 参数为一个可直接喂给 ultralytics 的路径。

    三种取值:
      1. 现成路径（.pt 或 .yaml）      -> 原样返回，不联网
      2. URLS 里的已知名字（yolo11m.pt）-> 缓存目录里找，缺了就下载
      3. 架构 yaml 名（yolo11m.yaml）   -> 原样返回，表示随机初始化，不需要权重
    """
    if os.path.exists(model):
        return model
    if model.endswith(".yaml"):
        return model  # 随机初始化臂，ultralytics 内置架构定义
    if model not in URLS:
        raise SystemExit(
            f"未知权重 '{model}'。已知: {', '.join(sorted(URLS))}；"
            "或直接传一个本地 .pt / .yaml 路径"
        )
    dest = os.path.join(cache_dir, model)
    if _valid(dest, None):
        if verbose:
            print(f"[weights] 命中本地: {dest}")
        return dest
    if not allow_download:
        raise SystemExit(f"本地缺少 {dest} 且已禁用下载（--no-download）")
    return download(URLS[model], dest, verbose=verbose)


def main():
    import argparse

    ap = argparse.ArgumentParser(description="预下载权重，供离线机器使用")
    ap.add_argument("models", nargs="*", default=["yolo11m.pt"],
                    help=f"要下载的权重名，可选: {', '.join(sorted(URLS))}")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    a = ap.parse_args()
    for m in a.models:
        ensure(m, a.cache_dir)


if __name__ == "__main__":
    main()
