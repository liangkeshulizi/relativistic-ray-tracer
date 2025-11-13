# 相对论光追追踪渲染器 Relativistic Ray Tracer

### 简介
首先可以看一下我使用这个渲染器制作的[简单科普 Terrell效应 的视频](https://www.bilibili.com/video/BV1JY411L7xm?spm_id_from=333.999.0.0)。

我们都知道，高速速度运动的物体在运动方向上会发生收缩（长度收缩效应）。然而，由于光速时有限的，在光传播过程中，物体的位置也会发生明显的变化。在同时到达相机底片的所有光子中，光程越远，对拍摄的时刻来说，光映射在底片上体现的就是物体越早的状态。这会导致物体外观发生变化。这种效应被称为Terrell旋转。

事实证明，上述两种效应，相对论长度收缩和视觉失真，实际上是相互抵消的。高速运动的球体总是呈圆形轮廓。在Penrose和Rindler的《in Spinors and Spacetime》一书中提到，洛伦兹变换在天球上充当保角变换；梁灿彬教授的《从零学相对论》一书中也对相对论视觉效应做了完整的阐述；另外推荐入门读物 —— Weisskopf(1960)。

计算机模拟是研究高速运动会物体视觉形象的有力工具，也是通俗直观地理解Terrell旋转的最佳方案，这一工作始于20世纪80年代末期，现在国际上已有多个小组进行研究。

我写的这个简单的“相对论光线追踪渲染器”可以模拟符合相对论的光线。该模型中的球体相对于相机具有速度和方向。与普通光线跟踪器一样，该程序对来自观测者的光线进行建模，并计算它们击中球体的位置。然而，这个程序中的光线实际上是四维闵氏时空中的光线。对每条光线使用洛伦兹变换，转换到物体的参考系下，再进行球体的碰撞计算。

#### 特性和优势
+ python的语法简洁直观。只需要简单的代码就可以渲染出高质量的图像；
+ 轻量。仅使用CPU渲染；
+ 高分辨率。支持1080p、4k分辨率图像的渲染；
+ 高效。单独的1080p图形渲染（包括碰撞检测、上色、光影）耗时不到1s，为高分辨率和帧率的视频制作提供了可能；
+ 高质量。结合Shadowing、Lambert shading、Blinn-Phong shading等算法，成品精美而符合物理事实。

numpy数组运算的高效性使得上述渲染成为可能。

### 依赖
+ Python 3+
+ pillow
+ numpy
+ tqdm
+ pygame

# 安装使用教程

## 安装

1. **克隆项目**

   ```bash
   git clone https://github.com/liangkeshulizi/relativistic_ray_tracer.git
   cd relativistic_ray_tracer
   ```

2. **安装依赖**

   ```bash
   pip install -r requirements.txt
   ```

3. **运行测试**
   执行一个基础的示例脚本来验证安装是否成功：

   ```bash
   python example_image.py
   ```

   如果一切正常，程序将耗时约 3 秒，并在项目根目录生成一张名为 `image.png` 的图片。

## 如何使用

我们提供了非常详细的自定义场景指南，它将一步步教你如何创建从静态图像到复杂动画的所有内容。

> **👉 [点击这里，阅读自定义场景指南 (CUSTOM_SCENE_GUIDE.md)](./CUSTOM_SCENE_GUIDE.md)**

## 查看示例

项目内置了多个示例脚本，展示了不同的功能。你可以直接运行它们来查看效果：

*   `example_image.py`: 生成一张包含静态球体、立方体和棋盘格平面的基础图像。
*   `example_cube.py`: 渲染一个以 0.7c 速度运动的立方体。
*   `example_animation.py`: 创建一个球体以 0.5c 速度飞过的动画。
*   `example_updater.py`: 演示如何使用 `updater` 函数让立方体从 -0.99c 加速到 0.99c。
*   `example_updater_ball.py`: 演示如何让球体在动画中来回运动。
*   `example_ball_and_cube_moving.py`: 创建一个更复杂的场景，其中多个物体一起运动，并且速度会动态变化。

运行任何一个动画示例 (例如 `python example_animation.py`) 后，渲染出的所有帧图片将保存在 `render/` 目录下。
