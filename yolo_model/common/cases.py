"""统一的样本发现与清洗，屏蔽两种数据集的目录/命名差异。

支持的两种布局:
  官方   赛题六公开数据集/200_train_cases/<case>/<stem>_target.json + <stem>.png
  自采   data/<id>/<id>.json + <id>.png

清洗规则（每条都有实测依据，见 docstring 末尾）:
  1. 跳过 macOS AppleDouble 残留（`._*`），两个数据集里都有；
  2. 剔除 components 为空的样本（data/ 里实测 16 个）；
  3. 按图像内容哈希分组，内容重复的图必须落在同一 split
     （data/ 里实测 042-09-03/08/13 同图、042-09-09/14 同图）。

坐标约定 —— 最容易错的一条:
  GT 的 bbox / point 是**左下原点、y 向上**，转图像坐标必须 `y_img = H - y_gt`，
  且翻转后 y 的上下界互换。实测依据: 842 个 bbox 翻转后平均墨迹密度 0.1862，
  不翻转 0.0506，而全图随机基线是 0.0467 —— 不翻转等于随机位置。
  本模块统一在 `image_boxes()` 里做翻转，调用方不要自己再翻一次。
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

OFFICIAL_STEM = re.compile(r"^(\d+)-(.+)$")


@dataclass
class Case:
    case_id: str
    image_path: str
    json_path: str
    source: str
    dataset: str
    group: str = ""  # 图像内容哈希，去重分组用
    n_components: int = 0
    meta: dict = field(default_factory=dict)


def _visible(names):
    """滤掉 AppleDouble 残留和隐藏文件。"""
    return [n for n in names if not n.startswith("._") and not n.startswith(".")]


def _pick_pair(case_dir: str):
    """返回 (png, json)；找不到唯一配对时返回 None。"""
    try:
        names = _visible(os.listdir(case_dir))
    except OSError:
        return None
    pngs = [n for n in names if n.lower().endswith(".png")]
    jsons = [n for n in names if n.lower().endswith(".json")]
    if len(pngs) != 1 or not jsons:
        return None
    # 官方用 *_target.json；自采用 <目录名>.json
    targets = [n for n in jsons if n.endswith("_target.json")]
    if targets:
        pick = targets[0]
    else:
        same = [n for n in jsons if n[:-5] == os.path.basename(case_dir)]
        if len(jsons) == 1:
            pick = jsons[0]
        elif same:
            pick = same[0]
        else:
            return None
    return os.path.join(case_dir, pngs[0]), os.path.join(case_dir, pick)


def _source_label(stem: str, dataset: str) -> str:
    """官方文件名带来源后缀（0135-jlc），自采没有，统一成一个可分层的标签。"""
    m = OFFICIAL_STEM.match(stem)
    if dataset == "official" and m:
        src = m.group(2).lower().strip()
        return "kicad" if src.startswith("kicad") else src
    return "harvest"


def _file_hash(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_target(json_path: str) -> dict:
    with open(json_path, encoding="utf-8") as fh:
        return json.load(fh)


def discover(
    roots,
    hash_cache: str | None = None,
    min_components: int = 1,
    verbose: bool = True,
):
    """扫描若干根目录，返回清洗后的 Case 列表和一份统计。

    roots: [(路径, dataset 标签)]，dataset 用 'official' / 'harvest'
    min_components: 元件数下限，低于此值的样本剔除（默认剔除空样本）
    """
    stats = Counter()
    dropped = defaultdict(list)
    cache: dict = {}
    if hash_cache and os.path.exists(hash_cache):
        with open(hash_cache, encoding="utf-8") as fh:
            cache = json.load(fh)

    cases: list[Case] = []
    for root, dataset in roots:
        if not os.path.isdir(root):
            raise SystemExit(f"数据根目录不存在: {root}")
        subs = sorted(_visible(os.listdir(root)))
        for sub in subs:
            case_dir = os.path.join(root, sub)
            if not os.path.isdir(case_dir):
                continue
            stats["scanned"] += 1
            pair = _pick_pair(case_dir)
            if pair is None:
                stats["dropped_no_pair"] += 1
                dropped["no_pair"].append(sub)
                continue
            png, js = pair
            try:
                comps = load_target(js).get("components", {})
            except (OSError, ValueError) as err:
                stats["dropped_bad_json"] += 1
                dropped["bad_json"].append(f"{sub}: {err}")
                continue
            if len(comps) < min_components:
                stats["dropped_too_few_components"] += 1
                dropped["too_few_components"].append(sub)
                continue
            stem = os.path.basename(png)[:-4]
            cases.append(
                Case(
                    case_id=stem.replace(" ", "_"),
                    image_path=png,
                    json_path=js,
                    source=_source_label(stem, dataset),
                    dataset=dataset,
                    n_components=len(comps),
                )
            )

    # 内容哈希分组：同图必须同 split，否则验证集泄漏
    for c in cases:
        key = f"{c.image_path}:{os.path.getsize(c.image_path)}"
        if key not in cache:
            cache[key] = _file_hash(c.image_path)
        c.group = cache[key]
    if hash_cache:
        os.makedirs(os.path.dirname(os.path.abspath(hash_cache)) or ".", exist_ok=True)
        with open(hash_cache, "w", encoding="utf-8") as fh:
            json.dump(cache, fh)

    by_group = defaultdict(list)
    for c in cases:
        by_group[c.group].append(c)
    stats["duplicate_image_groups"] = sum(1 for v in by_group.values() if len(v) > 1)
    stats["cases_in_duplicate_groups"] = sum(len(v) for v in by_group.values() if len(v) > 1)
    stats["kept"] = len(cases)

    if verbose:
        print(f"[cases] 扫描 {stats['scanned']} 个目录，保留 {stats['kept']}")
        for reason in ("no_pair", "bad_json", "too_few_components"):
            if dropped[reason]:
                shown = ", ".join(dropped[reason][:8])
                more = "" if len(dropped[reason]) <= 8 else f" ...(共{len(dropped[reason])})"
                print(f"[cases] 剔除 {reason}: {shown}{more}")
        if stats["duplicate_image_groups"]:
            print(
                f"[cases] 内容重复图 {stats['duplicate_image_groups']} 组 / "
                f"{stats['cases_in_duplicate_groups']} 个样本，已按组绑定 split"
            )
        by_src = Counter(c.source for c in cases)
        print("[cases] 来源分布: " + "  ".join(f"{k}={v}" for k, v in sorted(by_src.items())))
    return cases, dict(stats), {k: v for k, v in dropped.items() if v}


def split(cases, n_val: int = 0, val_ratio: float = 0.0, seed: int = 0):
    """按来源分层、按图像内容分组划分 train/val。

    分组是硬约束：同一内容哈希的样本永远同侧，否则验证集会看到训练集的图。
    n_val 与 val_ratio 二选一，n_val 优先。论文用的是 design-level 8:1:1，
    对应 val_ratio=0.1（我们不单独留 test，官方 10_GTcase 另作 sanity check）。
    """
    groups = defaultdict(list)
    for c in cases:
        groups[c.group].append(c)
    # 一个组的来源取组内多数，避免同组跨来源时分层口径摇摆
    items = []
    for g, lst in groups.items():
        src = Counter(c.source for c in lst).most_common(1)[0][0]
        items.append((g, src, lst))

    total = len(cases)
    if n_val > 0:
        target = n_val
    else:
        target = int(round(total * val_ratio))
    target = max(0, min(target, total))

    by_src = defaultdict(list)
    for g, src, lst in items:
        by_src[src].append((g, lst))

    # 最大余数法分配各来源的验证集配额，避免小来源被向上取整成 1 而整体超额
    sizes = {src: sum(len(lst) for _, lst in b) for src, b in by_src.items()}
    exact = {src: (target * n / total if total else 0.0) for src, n in sizes.items()}
    quota = {src: int(v) for src, v in exact.items()}
    left = target - sum(quota.values())
    for src, _ in sorted(exact.items(), key=lambda kv: (-(kv[1] - int(kv[1])), kv[0])):
        if left <= 0:
            break
        quota[src] += 1
        left -= 1

    rng = random.Random(seed)
    val_groups: set[str] = set()
    for src in sorted(by_src):
        if quota[src] <= 0:
            continue
        bucket = sorted(by_src[src], key=lambda t: t[0])
        rng.shuffle(bucket)
        got = 0
        for g, lst in bucket:
            if got >= quota[src]:
                break
            val_groups.add(g)
            got += len(lst)

    train = [c for c in cases if c.group not in val_groups]
    val = [c for c in cases if c.group in val_groups]
    return train, val


def image_boxes(target: dict, height: int, scale: float = 1.0):
    """GT components -> 图像坐标系 bbox。**此处完成 y 翻转，调用方不要重复翻。**

    返回 [(x1, y1, x2, y2, key, type)]，均为左上原点、y 向下。
    """
    out = []
    for key, comp in (target.get("components") or {}).items():
        box = comp.get("bbox")
        if not box or len(box) != 4:
            continue
        try:
            bx1, by1, bx2, by2 = (float(v) for v in box)
        except (TypeError, ValueError):
            continue
        x1, x2 = sorted((bx1, bx2))
        # 左下原点 -> 左上原点；翻转后上下界互换，所以这里重新排序
        y1, y2 = sorted((height - by2, height - by1))
        if x2 <= x1 or y2 <= y1:
            continue
        out.append(
            (x1 * scale, y1 * scale, x2 * scale, y2 * scale, key, comp.get("type") or "")
        )
    return out


def gt_texts(target: dict):
    """汇总一个样本里所有 GT 文本，作为伪标签的白名单。

    返回 {文本: [(来源字段, 元件key)]}。
    不做任何归一化 —— 评分是严格字符串匹配，`10k`/`10K`/`10 k` 是不同答案，
    这里保留原文，归一化只在匹配时临时做（见 text_det/pseudo_label.py）。
    """
    table: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key, comp in (target.get("components") or {}).items():
        for fld in ("Name", "value"):
            val = comp.get(fld)
            if isinstance(val, str) and val.strip():
                table[val].append((fld, key))
    for key, pins in (target.get("pins") or {}).items():
        if not isinstance(pins, dict):
            continue
        for pin in pins.values():
            if not isinstance(pin, dict):
                continue
            name = pin.get("pinname")
            if isinstance(name, str) and name.strip():
                table[name].append(("pinname", key))
    return dict(table)


# 官方数据在不同机器上的目录布局不一样：本地是嵌在 赛题六公开数据集/ 下面，
# 远程服务器上直接就是 200_train_cases/。这里按顺序探测，省得每台机器都要传 --official。
# 也可以用环境变量 PCB_OFFICIAL_DIR / PCB_HARVEST_DIR 显式指定。
OFFICIAL_CANDIDATES = [
    "200_train_cases",
    "赛题六公开数据集/200_train_cases",
    "data/200_train_cases",
]
HARVEST_CANDIDATES = ["data", "harvest"]


def find_root(candidates, env_var: str | None = None) -> str | None:
    if env_var:
        val = os.environ.get(env_var)
        if val:
            if not os.path.isdir(val):
                raise SystemExit(f"{env_var}={val} 不是一个目录")
            return val
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def default_roots():
    """自动探测可用的数据根目录。两个都找不到才报错。"""
    roots = []
    official = find_root(OFFICIAL_CANDIDATES, "PCB_OFFICIAL_DIR")
    if official:
        roots.append((official, "official"))
    harvest = find_root(HARVEST_CANDIDATES, "PCB_HARVEST_DIR")
    if harvest and harvest != official:
        roots.append((harvest, "harvest"))
    if not roots:
        raise SystemExit(
            "找不到数据目录。尝试过: "
            + ", ".join(OFFICIAL_CANDIDATES + HARVEST_CANDIDATES)
            + f"（当前目录 {os.getcwd()}）\n"
            "可以用 --official / --harvest 显式指定，"
            "或设环境变量 PCB_OFFICIAL_DIR / PCB_HARVEST_DIR"
        )
    return roots


def roots_from_args(official: str | None, harvest: str | None):
    """显式传入优先；传空字符串表示不用该数据集；都不传则自动探测。

    注意 None 和 "" 的区别：None 是"没指定，去探测"，"" 是"明确不要"。
    """
    if official is None and harvest is None:
        return default_roots()
    roots = []
    if official:
        if not os.path.isdir(official):
            raise SystemExit(f"--official 指定的目录不存在: {official}")
        roots.append((official, "official"))
    elif official is None:
        found = find_root(OFFICIAL_CANDIDATES, "PCB_OFFICIAL_DIR")
        if found:
            roots.append((found, "official"))
    if harvest:
        if not os.path.isdir(harvest):
            raise SystemExit(f"--harvest 指定的目录不存在: {harvest}")
        roots.append((harvest, "harvest"))
    elif harvest is None:
        found = find_root(HARVEST_CANDIDATES, "PCB_HARVEST_DIR")
        if found:
            roots.append((found, "harvest"))
    if not roots:
        raise SystemExit("没有可用的数据根目录（--official 和 --harvest 都为空）")
    return roots
