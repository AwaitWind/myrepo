import os
import sys
import time
import platform
import getpass
import subprocess
import numpy as np
import math


from config import AEDT_VERSION, PROJECT_PATH, DATA_FOLDER, DEBUG
from ansys.aedt.core import Hfss
from ansys.aedt.core import settings as pyaedt_settings

## 全局定义
# 空气盒子大小
Air_Box_Size = 8
# 求解频率范围设置
Frequency_Start = 0.01
Frequency_Stop = 5.0
Frequency_Step = 0.1


# =====================================================================
# 幂等工具：允许 filter3_layout_generate 在同一个 HFSS 会话里被反复调用
#   （反向设计的 gen_dataset 会用同一个 HFSS 连续跑 40 个样本，
#    没有这些兜底就会因为 Rad_airbox / PerfE_* / P1 P2 同名冲突而崩）
# =====================================================================
def _safe_delete_object(hfss, name):
    try:
        if name in hfss.modeler.object_names:
            hfss.modeler.delete(name)
    except Exception:
        pass


def _safe_delete_boundary(hfss, name):
    try:
        for b in list(hfss.boundaries):
            if getattr(b, "name", str(b)) == name:
                try:
                    b.delete()
                except Exception:
                    pass
    except Exception:
        pass


def _safe_delete_port(hfss, port_name):
    try:
        pnames = getattr(hfss, "port_names", []) or []
        if port_name in list(pnames):
            hfss.delete_port(port_name)
    except Exception:
        pass


def _refresh(hfss):
    """
    强制刷新 pyaedt 对象缓存，使 modeler.object_names 与 AEDT 真实几何一致。
    subtract/unite 会消耗 tool 对象，但 pyaedt 内部缓存可能没同步移除；
    不刷新会导致 create_pec_sheet 误判「已存在」而跳过。
    """
    try:
        hfss.modeler.refresh_all_ids()
    except Exception:
        try:
            hfss.modeler.cleanup_objects()
        except Exception:
            pass


# =====================================================================
# 创建指定平面上指定尺寸的 PEC gnd sheet（幂等版）
#   - 每张 sheet 使用「独立」边界名 PerfE_{sheet_name}（不再共用 "PerfE_GND"，
#     否则第二次 assign 时会因边界名冲突失败）
#   - assign 前先安全删除同名边界，允许在半残状态下重跑
# orientation "XY" 暂时不支持指定平面, 这里设死是 XY 平面
def create_pec_sheet(hfss, sheet_name, orientation, x, y, length, width, height):
    if sheet_name in hfss.modeler.object_names:
        return False
    gnd_sheet = hfss.modeler.create_rectangle(
        orientation=orientation,
        origin=[x, y, height],
        sizes=[length, width],
        name=sheet_name,
    )
    bnd_name = f"PerfE_{sheet_name}"
    _safe_delete_boundary(hfss, bnd_name)
    hfss.assign_perfecte_to_sheets(gnd_sheet, name=bnd_name)
    return True


# 相减操作封装
# 由于hfss相减后自动取主体的名字, 所以这里不更改名字
def model_sub(hfss, main_model, tool_model, keep_old, new_name=""):
    new_model = hfss.modeler.subtract(
        blank_list=main_model,  # 被减体（保留）
        tool_list=tool_model,  # 减去的体
        keep_originals=keep_old,  # 减完只保留结果
    )


# 相加操作封装
def model_unite(hfss, main_model, tool_model, keep_old, new_name):
    new_model = hfss.modeler.unite(main_model + tool_model, keep_originals=keep_old)
    hfss.modeler.objects[new_model].name = new_name


# 创建指定平面上指定尺寸的不规则形状 gnd sheet
# orientation "XY" 暂时不支持指定平面, 这里设死是 XY 平面
# pts 传入指定平面的二维坐标 不能传入三维
# height 传入指定平面法线的高
# 返回是否创建成功
# todo 验证传参pts闭合
def create_polygon_pec_sheet(hfss, sheet_name, orientation, pts, height):
    if sheet_name in hfss.modeler.object_names:
        return False
    if orientation == "XY":
        points_3d = [[x, y, height] for x, y in pts]
    elif orientation == "YZ":
        points_3d = [[height, y, z] for y, z in pts]  # 用 height 当 X 轴
    elif orientation == "XZ":
        points_3d = [[x, height, z] for x, z in pts]  # 用 height 当 Y 轴
    else:
        raise ValueError(f"Unsupported orientation: {orientation}")
    points_closed = points_3d + [points_3d[0]]  # 显式闭合
    sheet = hfss.modeler.create_polyline(points=points_closed, name=sheet_name, cover_surface=True)
    # 幂等：每张 sheet 独立边界名 PerfE_{sheet_name}，assign 前先清同名残留
    bnd_name = f"PerfE_{sheet_name}"
    _safe_delete_boundary(hfss, bnd_name)
    hfss.assign_perfecte_to_sheets(sheet, name=bnd_name)
    return True


# 绕任意点旋转一个物体（原地替换）
# cx, cy, cz : 旋转中心
# axis       : "X" / "Y" / "Z"（旋转轴方向）
# angle      : 角度，units 决定是 deg 还是 rad
def model_rotate_around_point(hfss, obj_name, cx, cy, cz, angle=90, axis="Z", units="deg"):
    """
    把 obj_name 绕 (cx, cy, cz) 沿 axis 转 angle。
    """
    # 1. 平移物体使旋转中心落到全局原点
    hfss.modeler.move(assignment=[obj_name], vector=[-cx, -cy, -cz])

    # 2. 绕主轴旋转，会产生新物体；rotate() 不会原地修改
    hfss.modeler.rotate(
        assignment=[obj_name],
        axis=axis,
        angle=angle,
        units=units,
    )

    # 3. 把旋转后的物体平移回 (cx, cy, cz)
    hfss.modeler.move(assignment=[obj_name], vector=[cx, cy, cz])
    return obj_name


def get_rect_spiral_vertices(a, b1, b2, start=(0.0, 0.0), direction="right_up"):
    """
    生成矩形螺旋条带的所有边界顶点坐标。

    参数
    ----
    a : float
        几何参数，使得起始中心点到最外侧顶边/底边的距离为 a + b2。

    b1 : float
        螺旋条带宽度。

    b2 : float
        相邻条带之间的间距。

    start : tuple[float, float]
        螺旋起始中心点坐标，默认是 (0, 0)。

    direction : str
        螺旋展开方向，可选：
        "right_up" / "右上"
        "left_up" / "左上"
        "right_down" / "右下"
        "left_down" / "左下"

    返回
    ----
    vertices : list[tuple[float, float]]
        螺旋条带边界顶点坐标，按边界顺序排列。
        不重复最后一个闭合点。
    """

    if b1 <= 0:
        raise ValueError("b1 必须大于 0")
    if b2 < 0:
        raise ValueError("b2 不能小于 0")
    if a <= 0:
        raise ValueError("a 必须大于 0")

    direction_map = {
        "right_up": (1.0, 1.0),
        "ru": (1.0, 1.0),
        "右上": (1.0, 1.0),
        "left_up": (-1.0, 1.0),
        "lu": (-1.0, 1.0),
        "左上": (-1.0, 1.0),
        "right_down": (1.0, -1.0),
        "rd": (1.0, -1.0),
        "右下": (1.0, -1.0),
        "left_down": (-1.0, -1.0),
        "ld": (-1.0, -1.0),
        "左下": (-1.0, -1.0),
    }

    direction_key = str(direction).lower()

    if direction_key not in direction_map:
        raise ValueError("direction 只能是 'right_up', 'left_up', 'right_down', 'left_down'，" "或中文 '右上', '左上', '右下', '左下'")

    sx, sy = direction_map[direction_key]

    eps = 1e-12

    x_start, y_start = start

    half_w = b1 / 2.0
    pitch = b1 + b2

    # 先在局部坐标系中生成一个“右上旋”结构
    # 最后再通过 sx, sy 镜像到左上、右下、左下
    outer_center_len = a + b2 - half_w

    if outer_center_len <= 0:
        raise ValueError("a + b2 必须大于 b1 / 2，否则尺寸不成立")

    # 局部右上旋方向：
    # 先向上，再向右，再向下，再向左
    directions = [
        (0.0, 1.0),
        (1.0, 0.0),
        (0.0, -1.0),
        (-1.0, 0.0),
    ]

    # =========================
    # 1. 生成中心线折点
    # =========================
    center_points = [(0.0, 0.0)]

    x, y = 0.0, 0.0
    i = 0

    while True:
        # 每两段缩短一个 pitch
        seg_len = outer_center_len - (i // 2) * pitch

        # 这里只判断 <= 0，保留最后短枝节
        if seg_len <= eps:
            break

        dx, dy = directions[i % 4]

        x += dx * seg_len
        y += dy * seg_len

        center_points.append((x, y))
        i += 1

    if len(center_points) < 2:
        raise ValueError("参数无法生成有效螺旋")

    # =========================
    # 2. 工具函数
    # =========================
    def cross(u, v):
        return u[0] * v[1] - u[1] * v[0]

    def line_intersection(p, d, q, e):
        """
        求直线 p + t d 和 q + s e 的交点
        """
        denom = cross(d, e)

        if abs(denom) < eps:
            raise ValueError("相邻线段平行，无法求交点")

        qp = (q[0] - p[0], q[1] - p[1])
        t = cross(qp, e) / denom

        return p[0] + t * d[0], p[1] + t * d[1]

    def unit_direction(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        length = math.hypot(dx, dy)

        if length <= eps:
            raise ValueError("中心线中存在零长度线段")

        return dx / length, dy / length

    def left_normal(d):
        return -d[1], d[0]

    # =========================
    # 3. 计算每一段的方向和法向
    # =========================
    dirs = []
    normals = []

    for j in range(len(center_points) - 1):
        d = unit_direction(center_points[j], center_points[j + 1])
        n = left_normal(d)

        dirs.append(d)
        normals.append(n)

    # =========================
    # 4. 生成左右两侧边界点
    # =========================
    left_side = []
    right_side = []

    # 起点两侧
    p_start = center_points[0]
    n_start = normals[0]

    left_side.append(
        (
            p_start[0] + half_w * n_start[0],
            p_start[1] + half_w * n_start[1],
        )
    )

    right_side.append(
        (
            p_start[0] - half_w * n_start[0],
            p_start[1] - half_w * n_start[1],
        )
    )

    # 中间拐角
    for j in range(1, len(center_points) - 1):
        p = center_points[j]

        d_prev = dirs[j - 1]
        d_next = dirs[j]

        n_prev = normals[j - 1]
        n_next = normals[j]

        # 左边界拐角
        p1_left = (
            p[0] + half_w * n_prev[0],
            p[1] + half_w * n_prev[1],
        )

        p2_left = (
            p[0] + half_w * n_next[0],
            p[1] + half_w * n_next[1],
        )

        left_corner = line_intersection(
            p1_left,
            d_prev,
            p2_left,
            d_next,
        )

        # 右边界拐角
        p1_right = (
            p[0] - half_w * n_prev[0],
            p[1] - half_w * n_prev[1],
        )

        p2_right = (
            p[0] - half_w * n_next[0],
            p[1] - half_w * n_next[1],
        )

        right_corner = line_intersection(
            p1_right,
            d_prev,
            p2_right,
            d_next,
        )

        left_side.append(left_corner)
        right_side.append(right_corner)

    # 终点两侧
    p_end = center_points[-1]
    n_end = normals[-1]

    left_side.append(
        (
            p_end[0] + half_w * n_end[0],
            p_end[1] + half_w * n_end[1],
        )
    )

    right_side.append(
        (
            p_end[0] - half_w * n_end[0],
            p_end[1] - half_w * n_end[1],
        )
    )

    # =========================
    # 5. 保留最后短枝节，但去掉中心多出来的尖点
    # =========================
    last_p1 = center_points[-2]
    last_p2 = center_points[-1]

    last_len = math.hypot(
        last_p2[0] - last_p1[0],
        last_p2[1] - last_p1[1],
    )

    if last_len <= b1 + eps and len(right_side) >= 3:
        d_last = unit_direction(last_p1, last_p2)
        n_last = left_normal(d_last)

        clip_point = (
            last_p2[0] + (half_w - last_len) * n_last[0],
            last_p2[1] + (half_w - last_len) * n_last[1],
        )

        local_vertices = left_side + [clip_point] + right_side[:-3][::-1]

    else:
        local_vertices = left_side + right_side[::-1]

    # =========================
    # 6. 去掉连续重复点
    # =========================
    clean_local_vertices = []

    for p in local_vertices:
        if not clean_local_vertices:
            clean_local_vertices.append(p)
        else:
            last = clean_local_vertices[-1]
            dist = math.hypot(p[0] - last[0], p[1] - last[1])

            if dist > 1e-9:
                clean_local_vertices.append(p)

    if len(clean_local_vertices) > 1:
        first = clean_local_vertices[0]
        last = clean_local_vertices[-1]

        if math.hypot(first[0] - last[0], first[1] - last[1]) <= 1e-9:
            clean_local_vertices.pop()

    # =========================
    # 7. 根据 direction 做镜像和平移
    # =========================
    vertices = []

    for x, y in clean_local_vertices:
        vertices.append(
            (
                x_start + sx * x,
                y_start + sy * y,
            )
        )

    # 最后统一交换所有点的 X、Y 坐标
    # 因为HFSS的坐标体系和普通的坐标体系是反过来的
    # vertices = [(-y, x) for x, y in vertices]
    return vertices


# 创建一个仿真模板 包含求解频率设置
# 所有模型的创建都不放在这个函数内
# port_back 由于结构为带地结构，所以需要将PCB往两边各延申port_back长度
# port_width port端口的宽 , pcb_x_length, pcb_y_length, pcb_z_length, port_back=5
def init_hfss(freq_ghz):

    print("启动HFSS")
    hfss = Hfss(
        version=AEDT_VERSION,
        non_graphical=False,
        new_desktop=False,
        project=PROJECT_PATH,
        design="HFSSDesign1",
        solution_type="Terminal",
        remove_lock=True,
        close_on_exit=False,  # 出错时不要立刻杀 AEDT，保留窗口方便排查
    )

    # 打印 AEDT 进程信息，方便确认会话是否一致
    if DEBUG:
        try:
            pid = hfss.odesktop.GetProcessID()
            print(
                f"Hfss() 构造完成 ({time.time()-t0:.1f}s) " f"AEDT PID={pid}，用户={getpass.getuser()}，平台={platform.system()}",
                flush=True,
            )
        except Exception as e:
            print(
                f"Hfss() 构造完成 ({time.time()-t0:.1f}s) " f"[警告] 取 PID 失败: {e}",
                flush=True,
            )
    setup_name = "Setup1"
    sweep_name = "Sweep2"

    if setup_name in hfss.setup_names:
        hfss.delete_setup(setup_name)

    # 创建setup
    if DEBUG:
        t0 = time.time()
        print("创建 Setup")
    setup = hfss.create_setup(name=setup_name)
    setup.props["Frequency"] = f"{freq_ghz}GHz"
    setup.props["MaximumPasses"] = 12
    setup.props["MinimumPasses"] = 2
    setup.props["MaxDeltaS"] = 0.02
    setup.props["BasisOrder"] = 1
    setup.props["PercentRefinement"] = 30
    setup.props["SolveType"] = "Single"
    setup.props["UseHPC"] = True
    setup.props["NumberOfCores"] = 8  # 128
    setup.update()

    # 创建扫频
    if DEBUG:
        print("创建 frequency sweep")
    freqs_ghz = np.round(np.arange(Frequency_Start, Frequency_Stop + 1e-9, Frequency_Step), 2).tolist()
    freq_min = float(np.min(freqs_ghz))
    freq_max = float(np.max(freqs_ghz))
    freq_count = len(freqs_ghz)
    sweep = setup.create_frequency_sweep(
        name=sweep_name,
        unit="GHz",
        start_frequency=freq_min,
        stop_frequency=freq_max,
        num_of_freq_points=freq_count,
        sweep_type="Discrete",
        save_fields=True,
        save_rad_fields=False,
    )
    if sweep is False:
        raise RuntimeError("Sweep 创建失败")
    sweep.props["RangeType"] = "LinearCount"
    sweep.props["RangeCount"] = freq_count
    sweep.props["SaveFields"] = True
    sweep.props["SaveRadFields"] = False
    sweep.update()

    sweeps_name = hfss.get_sweeps(setup_name)
    if DEBUG:
        print(f"HFSS 连接成功 (v{AEDT_VERSION}), sweeps: {sweeps_name} ")
    hfss.modeler.model_units = "mm"
    return hfss, setup, setup_name, sweep_name, sweeps_name


# 启动HFSS仿真
# 返回仿真用时
# out_s2p_path 仿真s2p结果保存路径
def hfss_simulate(hfss, out_s2p_path, setup_name="Setup1", sweep_name="Sweep1", cores=256, tasks=16):
    t0 = time.time()
    hfss.analyze_setup(setup_name, cores=cores, tasks=tasks)
    elapsed = time.time() - t0

    # ----- 导出 -----
    # np.save(os.path.join(folder, f"matrix_{i:04d}.npy"), matrix)
    hfss.export_touchstone(setup=setup_name, sweep=sweep_name, output_file=out_s2p_path)
    # print(f"  仿真耗时: {elapsed:.1f}s  |  已保存: {filepath}")
    return elapsed


# 截取PCB版的俯视图
# out_plot_path 为输出路径
# plot_selections 为选择哪些部分参与截图
def hfss_plot_layout(hfss, out_plot_path, plot_selections):
    hfss.post.export_model_picture(
        full_name=out_plot_path,
        show_axis=False,
        show_ruler=False,
        show_grid=False,
        show_region="False",
        orientation="top",
        selections=plot_selections,
        width=0,
        height=0,
    )


# =====================================================================
# 幂等清场：删除上一轮 filter3_layout_generate 建过的所有几何 / 边界 / 端口
#   让同一个 HFSS 会话可以连续跑不同参数的仿真，不残留脏数据
# =====================================================================
# 白名单：filter3_layout_generate 会用到的所有静态对象名
_STATIC_OBJECT_NAMES = {
    # 基础几何
    "substrate", "gnd_sheet", "air_box",
    # 中心 CPW / 中间叉指
    "cpw_cg", "cpw_cap", "cpw_slot", "cpw_fingers",
    # 4 边螺旋（生成链路里所有中间产物）
    "lrv_sheet", "top_sheet", "spi_sheet",
    # 两侧 CPW 馈线 + 两侧地 + 端口后地
    "cpw_left", "cpw_right",
    "cpw_left_g1", "cpw_left_g2", "cpw_right_g1", "cpw_right_g2",
    "port_left_g", "port_right_g",
    # 集总端口 sheet
    "port_1", "port_2",
}

# 前缀：覆盖 duplicate_along_line / mirror 复制体（_1, _2, ...）与 finger_ 索引
_STATIC_OBJECT_PREFIXES = (
    "substrate_", "gnd_sheet", "air_box_",
    "cpw_", "port_", "finger_",
    "lrv_", "top_", "spi_",
)


def _cleanup_previous_geometry(hfss):
    """删除上一轮建过的所有几何 / 边界 / 端口，用于会话复用时的幂等重跑。"""
    _refresh(hfss)

    # 1) 删所有 lumped port（P1 / P2 等）
    try:
        for pname in list(getattr(hfss, "port_names", []) or []):
            _safe_delete_port(hfss, pname)
    except Exception:
        pass

    # 2) 删所有几何对象（白名单 + 前缀双管齐下）
    try:
        obj_names = list(hfss.modeler.object_names)
    except Exception:
        obj_names = []
    to_delete = set()
    for name in obj_names:
        if name in _STATIC_OBJECT_NAMES:
            to_delete.add(name)
            continue
        for prefix in _STATIC_OBJECT_PREFIXES:
            if name.startswith(prefix):
                to_delete.add(name)
                break
    for name in to_delete:
        _safe_delete_object(hfss, name)

    # 3) 删残留的辐射边界（air_box 若重建，Rad_airbox 也要跟着重建）
    _safe_delete_boundary(hfss, "Rad_airbox")

    # 4) 删残留的 PerfE_* 边界（对应上一步删掉的 sheet），
    #    同时清理旧代码留下的共享边界 "PerfE_GND"
    try:
        for b in list(hfss.boundaries):
            bn = getattr(b, "name", str(b))
            if bn.startswith("PerfE_") or bn == "PerfE_GND":
                try:
                    b.delete()
                except Exception:
                    pass
    except Exception:
        pass

    _refresh(hfss)


# =====================================================================
# 第三版的版图生成
# =====================================================================
# 自建参数说明 P 单元总宽
def filter3_layout_generate(hfss, pcb_x, pcb_y, pcb_z, lf, n, d, a, b1, b2, d0, g, w0, p0, N, cpw_length, port_width=0.15, port_back=2):
    # 幂等清场：允许同一个 HFSS 会话连续跑多个不同参数的样本，不残留脏数据
    _cleanup_previous_geometry(hfss)

    p = d + 2 * a + p0
    models_list = []
    # 创建介质 包含N个单元和两个端口
    pcb_y = N * p + 2 * port_back + 2 * cpw_length
    sub_name = "substrate"
    if sub_name in hfss.modeler.object_names:
        substrate = hfss.modeler.objects[sub_name]
    else:
        origin = [-pcb_x / 2, -p / 2 - port_back - cpw_length, 0]
        sizes = [pcb_x, pcb_y, pcb_z]
        substrate = hfss.modeler.create_box(origin, sizes, name=sub_name, material="Rogers RO4350 (tm)")
    models_list.append(sub_name)

    # 创建地平面
    gnd_name = "gnd_sheet"
    create_pec_sheet(hfss, gnd_name, "XY", -pcb_x / 2, -p / 2, pcb_x, p, pcb_z)
    hfss.modeler[gnd_name].color = (255, 128, 64)

    # 设置空气盒子
    boundary_name = "air_box"
    if boundary_name not in hfss.modeler.object_names:
        boundary = hfss.modeler.create_box(
            origin=[-pcb_x / 2 - Air_Box_Size, -p / 2 - port_back - cpw_length - Air_Box_Size, -Air_Box_Size],
            sizes=[pcb_x + 2 * Air_Box_Size, pcb_y + 2 * Air_Box_Size, 2 * Air_Box_Size],
            name=boundary_name,
            material="vaccum",
        )
        hfss.modeler[boundary_name].visible = False

    # 把 air_box 的所有外表面设为辐射边界（Radiation）
    boundary_name = "air_box"  # 复用上面的命名

    # 拿到 air_box 的所有 face id（六个外表面）
    face_ids = hfss.modeler.get_object_faces(assignment=boundary_name)

    rad_name = hfss.assign_radiation_boundary_to_faces(
        assignment=face_ids,  # 要赋边界的面，可以是 face id 列表
        name="Rad_airbox",  # 边界名字，不填会自动生成
    )

    # 中心波导线
    create_pec_sheet(hfss, "cpw_cg", "XY", -w0 / 2, -p / 2, w0, p, pcb_z)

    # 中心波导线+两边的缝隙
    create_pec_sheet(hfss, "cpw_cap", "XY", -w0 / 2 - g, -p / 2, w0 + 2 * g, p, pcb_z)

    # 相减
    model_sub(hfss, [gnd_name], ["cpw_cap"], False)

    # 中间间隔线
    create_pec_sheet(hfss, "cpw_slot", "XY", -w0 / 2, -d0 / 2, w0, d0, pcb_z)
    model_sub(hfss, ["cpw_cg"], ["cpw_slot"], False)

    # 生成左边的右上旋线
    lrv = get_rect_spiral_vertices(a, b1, b2, (-w0 / 2 - g, -d / 2 - a + b1 / 2), "right_up")
    create_polygon_pec_sheet(hfss, "lrv_sheet", "XY", lrv, pcb_z)
    model_rotate_around_point(hfss, "lrv_sheet", -w0 / 2 - g, -d / 2 - a + b1 / 2, pcb_z)
    # 对称复制
    # XZ 平面的法向是 Y 方向
    hfss.modeler.mirror(assignment="lrv_sheet", origin=[0, 0, 0], vector=[0, 1, 0], duplicate=True)
    model_unite(hfss, ["lrv_sheet"], ["lrv_sheet_1"], False, "top_sheet")
    # YZ 平面的法向是 X 方向
    hfss.modeler.mirror(assignment="top_sheet", origin=[0, 0, 0], vector=[1, 0, 0], duplicate=True)
    model_unite(hfss, ["top_sheet"], ["top_sheet_1"], False, "spi_sheet")

    # gnd减去
    model_sub(hfss, [gnd_name], ["spi_sheet"], False)

    # 创建中心叉指结构
    wf = w0 / (2 * n - 1)
    fingers = []
    for i in range(n):
        if i % 2 == 0:
            create_pec_sheet(hfss, "finger_" + str(i), "XY", -w0 / 2 + 2 * i * wf, d0 / (-2), wf, lf, pcb_z)
        else:
            create_pec_sheet(hfss, "finger_" + str(i), "XY", -w0 / 2 + 2 * i * wf, d0 / 2, wf, -lf, pcb_z)
        # hfss.modeler["finger_" + str(i)].color = (255, 128, 64)
        fingers.append("finger_" + str(i))
    model_unite(hfss, ["cpw_cg"], fingers, False, "cpw_fingers")

    # 多单元级联
    if N > 1:
        # 增加单元数量 cpw_bt_fingers, gnd_name
        hfss.modeler.duplicate_along_line(
            assignment=["cpw_fingers", gnd_name],
            vector=[0, p, 0],
            clones=N,
            attach=False,  # 关键：False，复制体是独立对象，便于后面单独 unite
        )

        # 将新增的单元全部合并
        new_name = hfss.modeler.unite(["cpw_fingers"] + ["cpw_fingers_" + str(i) for i in range(1, N + 1)], keep_originals=False)
        hfss.modeler.objects[new_name].name = "cpw_fingers"
        new_name = hfss.modeler.unite([gnd_name] + [gnd_name + "_" + str(i) for i in range(1, N + 1)], keep_originals=False)
        hfss.modeler.objects[new_name].name = gnd_name

    ##  创建两侧CPW线
    # 中心线
    create_pec_sheet(hfss, "cpw_left", "XY", -w0 / 2, -p / 2 - cpw_length, w0, cpw_length, pcb_z)
    create_pec_sheet(hfss, "cpw_right", "XY", -w0 / 2, N * p - p / 2, w0, cpw_length, pcb_z)
    # 两侧地
    create_pec_sheet(hfss, "cpw_left_g1", "XY", -pcb_x / 2, -p / 2 - cpw_length, (pcb_x - w0) / 2 - g, cpw_length, pcb_z)
    create_pec_sheet(hfss, "cpw_left_g2", "XY", w0 / 2 + g, -p / 2 - cpw_length, (pcb_x - w0) / 2 - g, cpw_length, pcb_z)
    create_pec_sheet(hfss, "cpw_right_g1", "XY", -pcb_x / 2, N * p - p / 2, (pcb_x - w0) / 2 - g, cpw_length, pcb_z)
    create_pec_sheet(hfss, "cpw_right_g2", "XY", w0 / 2 + g, N * p - p / 2, (pcb_x - w0) / 2 - g, cpw_length, pcb_z)
    # 端口地
    create_pec_sheet(hfss, "port_left_g", "XY", -pcb_x / 2, -p / 2 - cpw_length - port_back, pcb_x, port_back, pcb_z)
    create_pec_sheet(hfss, "port_right_g", "XY", -pcb_x / 2, N * p - p / 2 + cpw_length, pcb_x, port_back, pcb_z)
    # 合并
    model_unite(hfss, ["cpw_fingers"], ["cpw_left", "cpw_right"], False, "cpw_fingers")
    model_unite(
        hfss, [gnd_name], ["cpw_left_g1", "cpw_right_g1", "cpw_left_g2", "cpw_right_g2", "port_left_g", "port_right_g"], False, gnd_name
    )
    hfss.modeler["cpw_fingers"].color = (255, 128, 64)
    models_list.extend([gnd_name, "cpw_fingers"])

    # 创建端口
    create_pec_sheet(hfss, "port_1", "XY", -w0 / 2 - g, -p / 2 - cpw_length - port_width, w0 + 2 * g, port_width, pcb_z)
    create_pec_sheet(hfss, "port_2", "XY", -w0 / 2 - g, N * p - p / 2 + cpw_length, w0 + 2 * g, port_width, pcb_z)
    models_list.extend(["port_1", "port_2"])

    # 挖去端口槽
    result = hfss.modeler.subtract(
        blank_list=[gnd_name],  # 被减体（保留）
        tool_list=["port_1", "port_2"],  # 减去的体
        keep_originals=True,  # 保留结果
    )

    # 把两个 port_sheet 设为集总端口
    hfss.lumped_port(
        assignment="port_1",  # 信号 sheet（名字字符串）
        reference=gnd_name,  # 参考地，可选；不填则自动找最近的导体
        impedance=50,
        name="P1",
        renormalize=True,
    )
    hfss.lumped_port(
        assignment="port_2",
        reference=gnd_name,
        impedance=50,
        name="P2",
        renormalize=True,
    )

    return models_list


if __name__ == "__main__":
    print("HFSS_funcs.py")
