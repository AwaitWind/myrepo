import os
import sys
import time
import platform
import getpass
import subprocess
import numpy as np

from plot_funcs import plot_s2p, plot_combine
from HFSS_funcs import create_pec_sheet, init_hfss, hfss_simulate, hfss_plot_layout, filter3_layout_generate
from config import AEDT_VERSION, PROJECT_PATH, DATA_FOLDER

from ansys.aedt.core import Hfss
from ansys.aedt.core import settings as pyaedt_settings


# 第三版的主函数
def filter3_main():
    # 用 gRPC 通信（远程连接更稳定、带宽占用小）
    pyaedt_settings.use_grpc_api = True

    # size of pcb
    pcb_x = 30
    pcb_y = 14.5
    pcb_z = 0.762
    hfss, setup, setup_name, sweep_name, sweeps_name = init_hfss(3.5)

    a = 3.1
    b1 = 0.3
    b2 = 0.3
    w1 = 9
    g = 0.2
    d = 1.4
    d0 = 0.6
    la = 7.7
    lb = 8.0
    ld1 = 1.1
    wa = 0.3
    wb = 1.0
    wf = 0.3
    wd = 2.2
    l1 = 3.6
    w0 = 2.2
    p0 = 0.5
    n = 10  # 叉指电容数量
    lf = 0.4  # 叉指电容长度 限制条件 lf < d0
    N = 1  # 单元个数 最小为1
    cpw_length = 2
    models_list = filter3_layout_generate(
        hfss=hfss,
        pcb_x=pcb_x,
        pcb_y=pcb_y,
        pcb_z=pcb_z,
        lf=lf,
        d=d,
        a=a,
        n=n,
        b1=b1,
        b2=b2,
        d0=d0,
        g=g,
        w0=w0,
        p0=p0,
        N=N,
        cpw_length=cpw_length,
    )
    s2p_path = "./test.s2p"
    layout_fig_path = "./test_layout.png"
    s2p_fig_path = "./test_s2p.png"
    combine_fig_path = "./combine.png"
    # exit(0)
    # 仿真
    hfss_simulate(hfss, s2p_path, setup_name=setup_name, sweep_name=sweeps_name[0], cores=8, tasks=2)
    # 单独生成 s参数图 和 版图
    plot_s2p(s2p_path, s2p_fig_path)
    # 生成版图图片
    hfss_plot_layout(hfss, layout_fig_path, models_list)
    # 拼接版图（左） + S 参数（右）为一张图
    plot_combine(layout_fig_path, s2p_fig_path, combine_fig_path)
    # hfss.release_desktop()
    print(">>> 完成并释放 HFSS")
    return 0


if __name__ == "__main__":
    filter3_main()
