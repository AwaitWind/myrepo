"""
inverse_design —— PC25 第三版 CPW 带通滤波器 · 反向设计模块。

底层建模复用 ../HFSS_funcs.filter3_layout_generate（4 边螺旋 + 中心叉指电容），
不修改主目录里的任何"基本配置"。

架构（三层 + 验证）:
  Layer 0  数据生成    sampling → gen_dataset(进程隔离) → dataset.npz
  Layer 1  正向代理    forward_net: MLP  x → S 参数曲线（毫秒级替代 HFSS）
  Layer 2  逆向求解    inverse_optimize: CMA-ES / 梯度  给 y* 求 x̂
           目标定义    spec (固定 target) / passband_spec (自由通带 soft-min)
  验证                validate: 真 HFSS 仿真 x̂，对比 target/代理/HFSS

子模块（各脚本内部使用绝对导入 `import id_config` 等）:
  id_config        参数范围 / 约束 / 频率网格 / 路径（single source of truth）
  sampling         LHS + 约束过滤
  dataio           s2p ↔ y 向量、断点续跑 index.csv
  gen_dataset      Layer 0：主进程 orchestrator + worker 子进程
  forward_net      Layer 1：MLP 正向代理
  spec             高层带通规格 → 目标曲线 y*
  passband_spec    位置无关自由通带 loss（numpy + torch 可微）
  inverse_optimize Layer 2A：CMA-ES / 梯度求 x̂
  validate         用真 HFSS 验证 x̂，出对比图
"""
