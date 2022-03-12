# 相对论光追追踪渲染器 Relativistic Ray Tracer
---
（更新中）

### 简介 Introduction
首先可以看一下我使用这个渲染器制作的[简单科普 Terrell效应 的视频](https://www.bilibili.com/video/BV1JY411L7xm?spm_id_from=333.999.0.0)。

我们都知道，高速速度运动的物体在运动方向上会发生收缩（长度收缩效应）。然而，由于光速时有限的，在光传播过程中，物体的位置也会发生明显的变化。在同时到达相机底片的所有光子中，光程越远，对拍摄的时刻来说，光映射在底片上体现的就是物体越早的状态。这会导致物体外观发生变化。这种效应被称为Terrell旋转。

事实证明，上述两种效应，相对论长度收缩和视觉失真，实际上是相互抵消的。高速运动的球体总是呈圆形轮廓。在Penrose和Rindler的《in Spinors and Spacetime》一书中提到，洛伦兹变换在天球上充当保角变换；梁灿彬教授的《从零学相对论》一书中也对相对论视觉效应做了完整的阐述；另外推荐入门读物 —— Weisskopf(1960)。

计算机模拟是研究高速运动会物体视觉形象的有力工具，也是通俗直观地理解Terrell旋转的最佳方案，这一工作始于20世纪80年代末期，现在国际上已有多个小组进行研究。

我写的这个简单的“相对论光线追踪渲染器”可以模拟符合相对论的光线。该模型中的球体相对于相机具有速度和方向。与普通光线跟踪器一样，该程序对来自观测者的光线进行建模，并计算它们击中球体的位置。然而，这个程序中的光线实际上是四维闵氏时空中的光线。对每条光线使用洛伦兹变换，转换到物体的参考系下，再进行球体的碰撞计算。

#### 特性和优势 Features & Advantages
+ python的语法简洁直观。只需要简单的代码就可以渲染出高质量的图像；
+ 轻量。仅使用CPU渲染；
+ 高分辨率。支持1080p、4k分辨率图像的渲染；
+ 高效。单独的1080p图形渲染（包括碰撞检测、上色、光影）耗时不到1s，为高分辨率和帧率的视频制作提供了可能；
+ 高质量。结合Shadowing、Lambert shading、Blinn-Phong shading等算法，成品精美而符合物理事实。

numpy数组运算的高效性使得上述渲染成为可能。

### 依赖 Dependence
+ Python 3+
+ pillow
+ numpy

### 安装 Installation

1. 安装python(略), pillow, numpy
```
C:\> pip install pillow
C:\> pip install numpy
```
2. 克隆源代码
```
C:\> git clone https://github.com/liangkeshulizi/Relativistic_ray_tracer.git
```
```
Cloning into 'Relativistic_ray_tracer'...
remote: Enumerating objects: 32, done.
remote: Counting objects: 100% (32/32), done.
remote: Compressing objects: 100% (32/32), done.
remote: Total 32 (delta 22), reused 0 (delta 0), pack-reused 0
Receiving objects: 100% (32/32), 15.54 KiB | 3.88 MiB/s, done.
Resolving deltas: 100% (22/22), done.
```
3. 测试
```
C:\> cd Relativistic_ray_tracer
C:\Relativistic_ray_tracer> python example_scene.py
```
```
耗时3.0623767375946045s...
```
如果正常运行，渲染大约耗时3秒，会在同级文件夹下输出名为`image.png`的文件:

![输出结果](./image/image.png)


> 这是我第一次写文档，我决定分成两个部分：教程和原理。**教程部分**具有比较强的可读性，先给出示例代码，以实例为中心，逐行解析，在实践中逐步学习，等到迫不得已的位置再水到渠成地补充深入的内容；**原理部分**则重视逻辑性，可以向字典一样找到对应的代码的含义。

### 教程部分

下面，以`example_scene.py`为例详细介绍如何利用`Relativistic Ray Tracer`渲染相对论图像。
```py
from my_raytracer import *

shape1= Sphere(.5, get_checkerboard_color_func(BILIBILIBLUE, WHITE))
shape2= Plane(vec3(0,0,0), vec3(0,1,0), diffuse_color_function= get_cubical_checkerboard_color_func(GREY, WHITE, .3))
beta1 = (0, 0, 0)
beta2 = (0, 0, 0)
offset1 = vec4(0, 0, 0, 2)
offset2 = vec4(0, 0, -.5, 0)

object1= MovingObject(shape1, beta1, offset1)
object2= MovingObject(shape2, beta2, offset2)
movingobjects= [object1, object2]

scene= Scene(movingobjects)
file_name= scene.generate_image(0)
```
`example_scene.py`的渲染效果非常好看，却只需要大约10行代码，这是因为它使用了很多默认的参数，实际上有很大的拓展空间。

这段代码大致可以分成四个部分，下面逐段讲解：

1. 导入
```py
from my_raytracer import *
```
从`my_raytracer.py`中导入需要的定义，主要有`Shape`类，`MovingObject`类和`Scene`类。

1. 形状、位置 和 速度
```py
shape1= Sphere(.5, get_checkerboard_color_func(BILIBILIBLUE, WHITE))
shape2= Plane(vec3(0,0,0), vec3(0,1,0), diffuse_color_function= get_cubical_checkerboard_color_func(GREY, WHITE, .3))
beta1 = (0, 0, 0)
beta2 = (0, 0, 0)
offset1 = vec4(0, 0, 0, 2)
offset2 = vec4(0, 0, -.5, 0)
```
前两行代码创建的两个实例都的父类都是`Shape`。`Shape`用于储存一种三维形状和它的基础色（可以理解为物体本身具有的颜色）。不同的`Shape`需要不同的参数。

以`Sphere`为例，它创建的是一个以原点为球心，半径自定义的球体：
```py
class Sphere(Shape):
    def __init__(self, radius, diffuse_color_function= lambda p: DEFAULT_OBJ_COLOR):
        ...
```
+ `radius`：球的半径。`int`、`float`类型皆可。
+ `diffuse_color_function`：球的基础色函数。这是所有的`Shape`都有的参数。接受一个`vec3`类型的，表示交点位置的参数；返回`rgb`类型的，表示颜色的数据。示例中的`get_checkerboard_color_func(color1, color2)`接受两种颜色，返回一个球状网格图案。
> *tip*  常用的颜色在`util.py`中设置为了常量，可以直接使用，例如`PINK`,`WHITE`

再如`Plane`，表示一个平面：
```py
class Plane(Shape):
    def __init__(self, center: vec3, norm: vec3, range_func= lambda inter: True, diffuse_color_function= lambda p: DEFAULT_OBJ_COLOR):
        ...
```
+ `center`：平面中心。
+ `norm`：平面法向量。
+ `range_func`：只有`range_func(inter) == True`的交点才是有效交点，用于限制平面的大小
  
3 - 6 行代码分别表示：

+ `beta1` `beta2`：物体运动的速度，三维向量。
+ `offset1` `offset2`：物体的位置偏移。

3. 放置物体
```py
object1= MovingObject(shape1, beta1, offset1)
object2= MovingObject(shape2, beta2, offset2)
movingobjects= [object1, object2]
```

4. 拍摄
```py
scene= Scene(movingobjects)
file_name= scene.generate_image(0)
```

未完待续
---
`vec3`可以看成一个三维数组，它支持`+n, -n, -, *n, .dot(n), abs()`运算，但和`numpy`中的`ndarray`有不同之处。
```py
class vec3():
    def __init__(self, x, y, z):
        (self.x, self.y, self.z) = (x, y, z)    
    ...

rgb= vec3
```

下面不加解释地给出一个`diffuse_color_function`的示例：
```py
def color_func(inter: vec3):
    return inter.y * rgb(1,1,1)
```
最终


### 实现原理

#### 相对论光线追踪算法

#### 代码解读
##### 文件结构:
`util.py`

### TODO