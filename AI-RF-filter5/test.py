import math


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

    return vertices


# import matplotlib.pyplot as plt

# vertices = get_rect_spiral_vertices(
#     a=10,
#     b1=1,
#     b2=1,
#     start=(0, 0),
#     direction="左下",
# )

# xs = [p[0] for p in vertices] + [vertices[0][0]]
# ys = [p[1] for p in vertices] + [vertices[0][1]]

# plt.figure()
# plt.plot(xs, ys, "-o")
# plt.axis("equal")
# plt.grid(True)
# plt.show()
