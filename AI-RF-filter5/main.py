import os
import sys
import time
import platform
import getpass
import subprocess
import numpy as np

from plot_funcs import plot_s2p, plot_combine
from HFSS_funcs import init_hfss, hfss_simulate, hfss_plot_layout, filter_layout_generate
from config import AEDT_VERSION, PROJECT_PATH, DATA_FOLDER

from ansys.aedt.core import Hfss
from ansys.aedt.core import settings as pyaedt_settings


# 仿真主函数
def filter_main():
    # 用 gRPC 通信（远程连接更稳定、带宽占用小）
    pyaedt_settings.use_grpc_api = True

    # size of pcb
    pcb_x = 17.5
    pcb_y = 14.5
    pcb_z = 0.762
    hfss, setup, setup_name, sweep_name, sweeps_name = init_hfss(3.5)

    l1 = 5.25
    l2 = 5.2
    l3 = 5.2
    l4 = 5.2
    l5 = 5.25
    w1 = 1
    w2 = 2
    w3 = 2
    w4 = 1
    wc1 = 1
    wc2 = 1
    wc3 = 1
    wc4 = 1
    wc5 = 1
    h1 = 5
    h2 = 5
    h3 = 5
    hd = 0.8
    hs = 0.5

    models_list = filter_layout_generate(
        hfss,
        pcb_x=pcb_x,
        pcb_y=pcb_y,
        pcb_z=pcb_z,
        l1=l1,
        l2=l2,
        l3=l3,
        l4=l4,
        l5=l5,
        w1=w1,
        w2=w2,
        w3=w3,
        w4=w4,
        wc1=wc1,
        wc2=wc2,
        wc3=wc3,
        wc4=wc4,
        wc5=wc5,
        h1=h1,
        h2=h2,
        h3=h3,
        hd=hd,
        hs=hs,
    )
    s2p_path = "./data/test.s2p"
    layout_fig_path = "./data/test_layout.png"
    s2p_fig_path = "./data/test_s2p.png"
    combine_fig_path = "./data/combine.png"
    # 仿真
    hfss_simulate(hfss, s2p_path, setup_name=setup_name, sweep_name=sweeps_name[0], cores=8, tasks=2)
    # 单独生成 s参数图 和 版图
    plot_s2p(s2p_path, s2p_fig_path)
    # 生成版图图片
    hfss_plot_layout(hfss, layout_fig_path, models_list)
    # 拼接版图（左） + S 参数（右）为一张图
    plot_combine(layout_fig_path, s2p_fig_path, combine_fig_path)
    hfss.release_desktop()


# 版图2
if __name__ == "__main__":
    filter_main()
