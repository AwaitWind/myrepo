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
Frequency_Stop = 10.0
Frequency_Step = 0.1


# =====================================================================
# 幂等清场机制的底层 helper（多次运行的关键）
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
        if port_name in hfss.port_names:
            hfss.delete_port(port_name)
    except Exception:
        pass


def _refresh(hfss):
    """
    强制刷新 pyaedt 对象缓存，使 object_names 与 AEDT 真实几何一致。
    subtract(keep_originals=False) 会消耗 tool 对象，但 pyaedt 内部缓存
    可能没同步移除；不刷新会导致 create_pec_sheet 误判"已存在"而跳过。
    """
    try:
        hfss.modeler.refresh_all_ids()
    except Exception:
        try:
            hfss.modeler.cleanup_objects()
        except Exception:
            pass


# =====================================================================
# 幂等版 create_pec_sheet
#   - 每个 sheet 使用唯一边界名 PerfE_{sheet_name}
#     （不再统一 "PerfE_GND"，否则第二个 sheet 起会因边界名重复而 assign 失败）
#   - 若同名边界残留（对象被删但边界未清），先删掉再 assign，保证可重复调用
# =====================================================================
def create_pec_sheet(hfss, sheet_name, orientation, x, y, length, width, height):
    if sheet_name in hfss.modeler.object_names:
        return False
    if orientation == "XY":
        gnd_sheet = hfss.modeler.create_rectangle(
            orientation=orientation,
            origin=[x, y, height],
            sizes=[length, width],
            name=sheet_name,
        )
    elif orientation == "XZ":
        gnd_sheet = hfss.modeler.create_rectangle(
            orientation=orientation,
            origin=[x, height, y],
            sizes=[width, length],
            name=sheet_name,
        )
    else:
        return False
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

    # 幂等：若旧 s2p 已存在，先删掉再导出，避免 HFSS 写入时被占用
    try:
        if os.path.exists(out_s2p_path):
            os.remove(out_s2p_path)
    except Exception:
        pass
    hfss.export_touchstone(setup=setup_name, sweep=sweep_name, output_file=out_s2p_path)
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


import math


def get_circle_layout_info(x1: float, y1: float, l: float, hd: float, hs: float) -> tuple[tuple[float, float], int]:
    """
    计算最左侧圆心坐标和镶嵌圆的总数量。

    参数
    ----------
    x1, y1:
        线段左端点坐标。
    l:
        线段长度。
    hd:
        圆的直径。
    hs:
        相邻圆边缘之间的间距。
        相邻圆心间距为 hd + hs。

    返回
    ----------
    min_center:
        最左侧圆心坐标 (x_min, y1)。
    circle_count:
        镶嵌圆的总数量。
    """

    center_spacing = hd + hs
    line_center_x = x1 + l / 2

    # 中心圆左右两侧分别能够放置的圆数量
    side_count = math.floor((l - hd) / (2 * center_spacing))

    # 圆的总数量：左侧 + 中心圆 + 右侧
    circle_count = 2 * side_count + 1

    # 最左侧圆心坐标
    min_center_x = line_center_x - side_count * center_spacing
    min_center = (min_center_x, y1)

    return min_center, circle_count


# =====================================================================
# 幂等清场：把上一轮建过的所有几何 / 边界 / 端口全部删掉
#   多次调用 filter_layout_generate 之前必须先清场，否则同名对象/端口冲突。
# =====================================================================
_STATIC_OBJECT_NAMES = {
    # 基础几何
    "substrate", "gnd_sheet", "air_box",
    # filter 主体（中间产物 + 最终产物）
    "tl_1", "tl_2", "tl_c", "filter_sheet",
    # 3 对矩形谐振器（unite 前的中间对象；unite 后并入 tl_c/filter_sheet）
    "re_1_1", "re_1_2", "re_2_1", "re_2_2", "re_3_1", "re_3_2",
    # via 阵列（unite 前的中间对象 + 最终对象）
    "via_top", "via_bottom", "via",
    # 端口 sheet
    "port_1", "port_2",
}


def _cleanup_previous_geometry(hfss, max_copy_index=64):
    """
    删除上一轮建过的所有几何 / 边界 / 端口。
      - max_copy_index: duplicate_along_line 可能产生的 _1, _2, ... 最大后缀
    """
    _refresh(hfss)

    # 1) 删所有 lumped port（P1/P2）
    try:
        for pname in list(hfss.port_names):
            _safe_delete_port(hfss, pname)
    except Exception:
        pass

    # 2) 删所有几何对象（白名单 + 前缀匹配 + duplicate 后缀）
    obj_names = list(hfss.modeler.object_names)
    to_delete = set()
    for name in obj_names:
        if name in _STATIC_OBJECT_NAMES:
            to_delete.add(name)
            continue
        for prefix in ("tl_", "re_", "via_", "port_",
                       "gnd_sheet_", "substrate_", "air_box_", "filter_sheet_"):
            if name.startswith(prefix):
                to_delete.add(name)
                break
    for name in to_delete:
        _safe_delete_object(hfss, name)

    # 3) 显式删可能的 duplicate 副本（via_1, via_2, ...）
    for i in range(1, max_copy_index + 1):
        for base in ("via", "gnd_sheet", "substrate", "tl_c", "filter_sheet"):
            _safe_delete_object(hfss, f"{base}_{i}")

    # 4) 删残留的辐射边界
    _safe_delete_boundary(hfss, "Rad_airbox")

    # 5) 删残留的 PerfE_* 边界（对应上一步删掉的对象）
    try:
        for b in list(hfss.boundaries):
            bn = getattr(b, "name", str(b))
            if bn.startswith("PerfE_"):
                try:
                    b.delete()
                except Exception:
                    pass
    except Exception:
        pass

    _refresh(hfss)


# 第5版的版图生成
# 自建参数说明 P 单元总宽
def filter_layout_generate(hfss, pcb_x, pcb_y, pcb_z, l1, l2, l3, l4, l5, w1, w2, w3, w4, wc1, wc2, wc3, wc4, wc5, h1, h2, h3, hd, hs):
    # 【幂等清场】把上一轮的几何/边界/端口全部删掉（多次运行的关键）
    _cleanup_previous_geometry(hfss)

    models_list = []
    # 创建介质 包含N个单元和两个端口
    pcb_y = l1 + l2 + l3 + l4 + l5 + w1 + w2 + w3 + w4
    sub_name = "substrate"
    # 幂等：清场后 substrate 已被删；这里直接重建（pcb_y 会随参数变化）
    origin = [-pcb_x / 2, 0, 0]
    sizes = [pcb_x, pcb_y, pcb_z]
    substrate = hfss.modeler.create_box(origin, sizes, name=sub_name, material="Rogers RO4350 (tm)")
    models_list.append(sub_name)

    # 创建底面地平面
    gnd_name = "gnd_sheet"
    create_pec_sheet(hfss, gnd_name, "XY", -pcb_x / 2, 0, pcb_x, pcb_y, 0)
    hfss.modeler[gnd_name].color = (255, 128, 64)

    # 设置空气盒子（幂等：清场后每轮重建，因为 pcb_y 会随参数变化）
    boundary_name = "air_box"
    boundary = hfss.modeler.create_box(
        origin=[-pcb_x / 2 - Air_Box_Size, -Air_Box_Size, -Air_Box_Size],
        sizes=[pcb_x + 2 * Air_Box_Size, pcb_y + 2 * Air_Box_Size, 2 * Air_Box_Size],
        name=boundary_name,
        material="vaccum",
    )
    hfss.modeler[boundary_name].visible = False

    # 把 air_box 的所有外表面设为辐射边界（Radiation）
    #   幂等：_cleanup_previous_geometry 已保证 Rad_airbox 不残留，这里直接建
    face_ids = hfss.modeler.get_object_faces(assignment=boundary_name)
    rad_name = hfss.assign_radiation_boundary_to_faces(
        assignment=face_ids,
        name="Rad_airbox",
    )

    # 中心波导线
    create_pec_sheet(hfss, "tl_1", "XY", -wc1 / 2, 0, wc1, l1, pcb_z)
    create_pec_sheet(hfss, "tl_2", "XY", -wc5 / 2, l1 + w1 + l2 + w2 + l3 + w3 + l4 + w4, wc5, l5, pcb_z)

    create_pec_sheet(hfss, "tl_c", "XY", -pcb_x / 2, l1, pcb_x, w1 + l2 + w2 + l3 + w3 + l4 + w4, pcb_z)

    create_pec_sheet(hfss, "re_1_1", "XY", -wc2 / 2 - h1, l1 + w1, h1, l2, pcb_z)
    create_pec_sheet(hfss, "re_1_2", "XY", wc2 / 2, l1 + w1, h1, l2, pcb_z)

    create_pec_sheet(hfss, "re_2_1", "XY", -wc3 / 2 - h2, l1 + w1 + l2 + w2, h2, l3, pcb_z)
    create_pec_sheet(hfss, "re_2_2", "XY", wc3 / 2, l1 + w1 + l2 + w2, h2, l3, pcb_z)

    create_pec_sheet(hfss, "re_3_1", "XY", -wc4 / 2 - h3, l1 + w1 + l2 + w2 + l3 + w3, h3, l4, pcb_z)
    create_pec_sheet(hfss, "re_3_2", "XY", wc4 / 2, l1 + w1 + l2 + w2 + l3 + w3, h3, l4, pcb_z)

    model_sub(hfss, ["tl_c"], ["re_1_1", "re_1_2", "re_2_1", "re_2_2", "re_3_1", "re_3_2"], False)
    model_unite(hfss, ["tl_c"], ["tl_1", "tl_2"], False, "filter_sheet")
    hfss.modeler["filter_sheet"].color = (255, 128, 64)
    models_list.append("filter_sheet")

    # 打孔

    point, num = get_circle_layout_info(l1, pcb_x / 2 - hd, l2 + l3 + l4 + w1 + w2 + w3 + w4, hd, hs)
    tool = hfss.modeler.create_cylinder(
        orientation="Z",
        origin=[point[1], point[0], 0],
        radius=hd / 2,  # 比 via 大一点 = 焊盘
        height=pcb_z,
        name="via_top",
        material="PEC",
    )
    tool = hfss.modeler.create_cylinder(
        orientation="Z",
        origin=[-point[1], point[0], 0],
        radius=hd / 2,  # 比 via 大一点 = 焊盘
        height=pcb_z,
        name="via_bottom",
        material="PEC",
    )
    model_unite(hfss, ["via_top"], ["via_bottom"], False, "via")
    if num > 1:
        hfss.modeler.duplicate_along_line(
            assignment=["via"],
            vector=[0, hd + hs, 0],
            clones=num,
            attach=False,  # 关键：False，复制体是独立对象，便于后面单独 unite
        )

        # 将新增的单元全部合并
        new_name = hfss.modeler.unite(["via"] + ["via" + "_" + str(i) for i in range(1, num + 1)], keep_originals=False)
        hfss.modeler.objects[new_name].name = "via"
    models_list.append("via")
    # 创建端口
    create_pec_sheet(hfss, "port_1", "XZ", -wc1 / 2, 0, wc1, pcb_z, 0)
    create_pec_sheet(hfss, "port_2", "XZ", -wc5 / 2, 0, wc5, pcb_z, pcb_y)
    models_list.extend(["port_1", "port_2"])

    # 把两个 port_sheet 设为集总端口
    #   幂等要点：create_port_sheet=False（sheet 已由 create_pec_sheet 建好，别让 AEDT 再建同名）
    hfss.lumped_port(
        assignment="port_1",  # 信号 sheet（名字字符串）
        reference=gnd_name,  # 参考地，可选；不填则自动找最近的导体
        create_port_sheet=False,
        impedance=50,
        name="P1",
        renormalize=True,
    )
    hfss.lumped_port(
        assignment="port_2",
        reference=gnd_name,
        create_port_sheet=False,
        impedance=50,
        name="P2",
        renormalize=True,
    )

    return models_list


if __name__ == "__main__":
    print("HFSS_funcs.py")
