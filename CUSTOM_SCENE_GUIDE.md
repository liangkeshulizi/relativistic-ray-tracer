# 如何配置自定义场景

本文档将详细指导你如何使用 `relativistic-ray-tracer` 项目来创建从简单的静态图像到复杂的、包含多个运动物体的相对论动画。

## 核心概念

在开始编码之前，我们先理解三个核心类：

1.  **`Shape` (形状)**: 这是物体的几何定义和基础颜色（纹理）的结合。例如，一个半径为 0.5 的球体，或者一个无限大的平面。它只关心物体的“长相”，不关心它在哪里或如何运动。
2.  **`MovingObject` (运动物体)**: 这是 `Shape` 的一个“实例”。它将一个 `Shape` 与其在时空中的**位置 (offset)** 和**速度 (beta)** 绑定在一起，构成一个完整的场景物体。
3.  **`Scene` (场景)**: 这是所有 `MovingObject` 的集合。它还包含了相机、光照等环境参数。最终由 `Scene` 对象负责渲染出图像或动画。

---

## 第 1 部分：创建静态图像

这是最基础的用法，适合生成一张静止的快照。

### 第 1 步：创建 Python 文件并导入模块

首先，在项目根目录下创建一个新的 Python 文件，例如 `my_static_scene.py`。

在文件的开头，导入所有需要的模块：
```python
from my_raytracer import *
```

### 第 2 步：定义物体形状 (`Shape`)

定义你场景中需要的几何形状。

*   **`Sphere(radius, diffuse_color_function)`**: 创建一个球心在原点的球体。
*   **`Plane(center, norm, diffuse_color_function)`**: 创建一个无限大的平面。
*   **`Cube(width, height, depth, diffuse_color_function)`**: 创建一个中心在原点的长方体。

`diffuse_color_function` 是一个函数，它接收一个三维向量 `inter` (光线与物体的交点坐标)，返回一个 `rgb` 颜色。你可以使用 `util.py` 中预定义的纯色（如 `RED`, `WHITE`），或使用棋盘格函数（如 `get_checkerboard_color_func`）。

```python
# 示例：创建一个蓝白棋盘格的球和一个灰白棋盘格的地面
shape_sphere = Sphere(radius=0.5, diffuse_color_function=get_checkerboard_color_func(BILIBILIBLUE, WHITE))
shape_ground = Plane(vec3(0,0,0), vec3(0,1,0), diffuse_color_function=get_cubical_checkerboard_color_func(GREY, WHITE, 0.3))
```

### 第 3 步：设置物体的位置和速度

*   **速度 (`beta`)**: 一个三维元组 `(vx, vy, vz)`，表示以光速 `c` 为单位的速度。`(0, 0, 0)` 表示静止。
*   **位置 (`offset`)**: 一个 `vec4(t, x, y, z)` 时空坐标。对于静态图像，`t` 通常为 0。

```python
# 两个物体都静止
beta_static = (0, 0, 0)

# 定义它们的初始位置
offset_sphere = vec4(0, 0, 0, 2)
offset_ground = vec4(0, 0, -0.5, 0)
```

### 第 4 步：组合成 `MovingObject`

```python
moving_sphere = MovingObject(shape=shape_sphere, beta=beta_static, offset=offset_sphere)
moving_ground = MovingObject(shape=shape_ground, beta=beta_static, offset=offset_ground)
```

### 第 5 步：创建场景并渲染单张图像

将所有 `MovingObject` 放入一个列表中，创建 `Scene` 对象，然后调用 `generate_image()`。

```python
# my_static_scene.py
from my_raytracer import *

# 1. 定义形状
shape_sphere = Sphere(radius=0.5, diffuse_color_function=get_checkerboard_color_func(BILIBILIBLUE, WHITE))
shape_ground = Plane(vec3(0,0,0), vec3(0,1,0), diffuse_color_function=get_cubical_checkerboard_color_func(GREY, WHITE, 0.3))

# 2. 定义速度和位置
beta_static = (0, 0, 0)
offset_sphere = vec4(0, 0, 0, 2)
offset_ground = vec4(0, 0, -0.5, 0)

# 3. 创建运动物体
moving_sphere = MovingObject(shape=shape_sphere, beta=beta_static, offset=offset_sphere)
moving_ground = MovingObject(shape=shape_ground, beta=beta_static, offset=offset_ground)

# 4. 创建场景
# 可以自定义相机位置、朝向和点光源位置
scene = Scene(
    moving_objects=[moving_sphere, moving_ground],
    camera_pos=vec4(0, 0, 0, -5),
    camera_dir=vec3(0, 0, 1),
    light_pos=vec3(5, 5, -5)
)

# 5. 渲染并保存图像
if __name__ == "__main__":
    file_name = scene.generate_image(width=1280, height=720, file_name="static_scene.png")
    print(f"场景已渲染并保存为 {file_name}")
```

---

## 第 2 部分：创建动画

现在，我们让物体动起来，生成一段视频。

### 核心函数：`set_render_properties()` 和 `render()`

*   `generate_image()` 用于单张图片。
*   `render()` 用于生成动画序列帧。

你需要使用 `set_render_properties()` 来配置动画参数：

*   `t_start`, `t_end`: 动画的开始和结束**物理时间**。光线追踪器会模拟从 `t_start` 到 `t_end` 的场景演化。
*   `duration`: 生成视频的总时长（秒）。
*   `updaters`: 一个函数列表，用于在动画过程中动态修改场景，这是实现复杂动画的关键（详见下一部分）。

### 示例：恒定速度的动画

在这个例子中，球体将以 0.5c 的速度沿 x 轴正方向运动。

```python
# my_animation_scene.py
from my_raytracer import *

# 1. 定义形状 (与之前相同)
shape_sphere = Sphere(.5, get_checkerboard_color_func(GREEN_C, YELLOW_C))
shape_plane = Plane(vec3(0,0,0), vec3(0,1,0), diffuse_color_function= get_cubical_checkerboard_color_func(GREY, WHITE, .15))

# 2. 定义物体的初始状态
# 球体以 0.5c 速度运动
object1 = MovingObject(shape_sphere, (0.5, 0, 0), vec4(0, 0, 0, 1.5))
# 地面静止
object2 = MovingObject(shape_plane, (0, 0, 0), vec4(0, 0, -0.5, 0))

# 3. 创建场景
scene = Scene([object1, object2])

# 4. 设置动画属性
# 模拟从 t=-3 到 t=3 的时间，生成一个 6 秒的视频
scene.set_render_properties(
    t_start=-3,
    t_end=3,
    duration=6,
)

# 5. 渲染动画
if __name__ == "__main__":
    # render(0) 会开始渲染并保存所有帧到 render/ 目录下
    scene.render(0)
    print("动画帧已生成完毕！")
```

---

## 第 3 部分：高级动画与动态更新

`updater` 函数是本项目的精髓。它允许你在动画的每一帧渲染前，根据时间动态地修改场景中的任何属性。

### `updater` 函数的结构

一个 `updater` 函数接收两个参数：
*   `scene`: 当前的 `Scene` 对象。
*   `t`: 一个从 **0.0** 到 **1.0** 线性变化的时间参数，代表当前动画的进度。

你可以通过修改 `scene` 对象（例如 `scene.movingobjects[0].set_beta(...)`）来更新场景状态。

### 示例：变速运动和动态文字

我们将创建一个场景，其中：
1.  一个方块的速度会随着时间从 -0.99c 平滑地变化到 +0.99c。
2.  屏幕上的文字会实时显示当前的速度。

```python
# my_advanced_animation.py
from my_raytracer import *

# 1. 定义形状
shape_cube = Cube(1, 1, 1, get_cubical_checkerboard_color_func(BILIBILIPINK, WHITE))
shape_plane = Plane(vec3(0,0,0), vec3(0,1,0), diffuse_color_function= get_cubical_checkerboard_color_func(GREY, WHITE, .15))

# 2. 定义物体 (初始速度为0)
object1 = MovingObject(shape_cube, (0, 0, 0), vec4(0, 0, 0, 2))
object2 = MovingObject(shape_plane, (0, 0, 0), vec4(0, 0, -0.5, 0))

# 3. 定义 Updater 函数
def velocity_updater(scene, t):
    """根据动画进度 t (-1 到 1 的映射) 来更新方块速度"""
    # t 从 0 -> 1, v 从 -0.99c -> 0.99c
    v = (t * 2 - 1) * 0.99
    scene.movingobjects[0].set_beta((v, 0, 0))

def text_updater(scene, t):
    """根据当前速度更新屏幕上的文字"""
    current_v = scene.movingobjects[0].beta[0]
    text = f"v = {round(current_v, 3)}c"
    # get_my_compositor 用于创建文字水印
    scene.compositors = [get_my_compositor(text, 0.05, fill=BILIBILIPINK._to_standard_color())]

# 4. 创建场景
# 注意：compositors 可以在这里初始化，也可以在 updater 中动态修改
scene = Scene([object1, object2])

# 5. 设置动画和 Updaters
# 这里的 t_start/t_end 只是为了满足函数要求，实际行为由 updater 控制
scene.set_render_properties(
    t_start=0, 
    t_end=0, 
    duration=10, 
    updaters=[velocity_updater, text_updater] # 注册我们的 updater
)

# 6. 渲染
if __name__ == "__main__":
    scene.render(0)
    print("高级动画已渲染完成！")
```

通过这个例子，你可以看到 `updater` 的强大之处：
*   可以实现**加速度、变速度**等复杂运动。
*   可以动态修改**文字、颜色、位置**等任何你能通过代码访问的场景属性。
*   可以将多个 `updater` 组合使用，实现更复杂的效果。
