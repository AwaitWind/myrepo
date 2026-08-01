import numpy as np
from config import AEDT_VERSION, PROJECT_PATH, DATA_FOLDER, DEBUG
from HFSS_funcs import create_pec_sheet
from ansys.aedt.core import Hfss
from ansys.aedt.core import settings as pyaedt_settings

# 该文件存放之前实现过的 带通滤波器结构


# 第一版 CPW带通滤波器
# 为工程文件切割特定形状的槽
# pcb_x, pcb_y, pcb_z 为 pcb 在x,y,z轴上的长度
# gnd_name 为顶层大地面的名称
def hfss_sub_slot_model1(hfss, gnd_name, pcb_x, pcb_y, pcb_z, w1, cap, wl1, ll1, wl2, ll2, ls, port_width=0.15):
    # 创建中心连续波导线 cpw_cg
    # 起点 (-w1/2, -pcb_y/2)
    # w1 对应 x 方向上的宽度;
    create_pec_sheet(hfss, "cpw_cg", "XY", w1 / (-2), pcb_y / (-2), w1, pcb_y, pcb_z)

    # 创建中间间隔槽 cpw_cs
    # 起点 (-w1/2, -ls/2)
    # w1 对应 x 方向上的宽度; ls 对应 y 方向上的长度
    create_pec_sheet(hfss, "cpw_cs", "XY", w1 / (-2), ls / (-2), w1, ls, pcb_z)

    # cpw_cg - cpw_cs 形成中心波导线
    result = hfss.modeler.subtract(
        blank_list=["cpw_cg"],  # 被减体（保留）
        tool_list=["cpw_cs"],  # 减去的体
        keep_originals=False,  # 减完只保留结果
    )

    # 创建CPW中心波导线和两侧地面的间隔 cpw_cap, 通过大地平面减去这个平面, 然后和 cpw_cg 组合起来构成
    # 起点 (-w1/2 - cap, -pcb_y/2)
    # w1 + 2 * cap 对应 x 方向上的宽度; pcb_y 对应 y 方向上的长度
    create_pec_sheet(hfss, "cpw_cap", "XY", w1 / (-2) - cap, pcb_y / (-2), w1 + 2 * cap, pcb_y, pcb_z)

    # 创建1号侧槽 cpw_ss11 和 cpw_ss12 他们关于y轴对称
    # 起点 (-w1/2 - cap - ll1, -wl1/2)
    # wl1 对应 y 方向上的宽度; ll1 对应 x 方向上的长度
    create_pec_sheet(hfss, "cpw_ss11", "XY", w1 / (-2) - cap - ll1, wl1 / (-2), ll1, wl1, pcb_z)
    create_pec_sheet(hfss, "cpw_ss12", "XY", w1 / (2) + cap, wl1 / (-2), ll1, wl1, pcb_z)

    # 创建2号侧槽 cpw_ss21 和 cpw_ss22 他们关于y轴对称
    # 起点 (-w1/2 - cap - ll1 - ll2, -wl2/2)
    # wl2 对应 y 方向上的宽度; ll2 对应 x 方向上的长度
    create_pec_sheet(hfss, "cpw_ss21", "XY", w1 / (-2) - cap - ll1 - ll2, wl2 / (-2), ll2, wl2, pcb_z)
    create_pec_sheet(hfss, "cpw_ss22", "XY", w1 / (2) + cap + ll1, wl2 / (-2), ll2, wl2, pcb_z)

    # 构建完整的地平面
    result = hfss.modeler.subtract(
        blank_list=[gnd_name],  # 被减体（保留）
        tool_list=["cpw_cap", "cpw_ss11", "cpw_ss12", "cpw_ss21", "cpw_ss22"],  # 减去的体
        keep_originals=False,  # 减完只保留结果
    )

    # 创建端口平面 port_1, port_2 和他关于x轴对称
    # 起点 (-w1/2, -pcb_y/2 - port_width)
    create_pec_sheet(hfss, "port_1", "XY", w1 / (-2) - cap, pcb_y / (-2) - port_width, w1 + 2 * cap, port_width, pcb_z)
    create_pec_sheet(hfss, "port_2", "XY", w1 / (-2) - cap, pcb_y / (2), w1 + 2 * cap, port_width, pcb_z)

    # 删减 保留的
    result = hfss.modeler.subtract(
        blank_list=[gnd_name],  # 被减体（保留）
        tool_list=["port_1", "port_2"],  # 减去的体
        keep_originals=True,  # 减完同时保留被减去项
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
