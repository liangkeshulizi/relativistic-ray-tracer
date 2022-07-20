'''MIT License

Copyright (c) 2022 LIYIZHOU

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.'''

# Ideas From:
# Relativistic Ray-Tracing: Simulating the Visual Appearance of Rapidly Moving Objects - Sandy Dance
# James Terrell, "Invisibility of the lorentz contraction", Phys. Rev. 116, 1041-1045 (1959).
# https://www.youtube.com/watch?v=oFaSLIsJELY
# https://excamera.com/sphinx/article-ray.html

from util import *
from abc import ABC, abstractmethod
from tqdm import tqdm
from typing import Callable, Iterable
import subprocess, os, pygame, inspect, argparse, shutil

class Material:
    def __init__(self, gloss= False, mirror= 0.5, ambient= rgb(0.08, 0.08, 0.08), shadow= .1, diffuse_combination= .1):
        self.gloss= gloss
        self.mirror= mirror
        self.ambient= ambient
        self.diffuse_combination= diffuse_combination
        self.shadow= shadow
    def smoothen(self): # TODO
        pass
    def roughen(self): # TODO
        pass

# 定义一个形状，给出 求交点 和 求法向量 的方法就行了
class Shape(ABC):
    '''储存形状、基础色'''
    @abstractmethod
    def get_norm(self, Intersections: vec3):
        '''法向量'''
    @abstractmethod
    def get_intersection(self, starts: vec4, directions: vec4, inverted_trace):
        '''交点。注意规范：没有交点的射线要返回 vec4( [ ..., nan, ... ], [ ..., nan, ... ], [ ..., nan, ... ], [ ..., nan, ... ]) ！'''
    def get_diffuse_color(self, Intersections: vec4):
        '''散射颜色'''
        color= self.diffuse_color_function(Intersections.vec3())#.extract(hit))
        return color

class Sphere(Shape):
    def __init__(self, radius, diffuse_color_function= lambda p: DEFAULT_OBJ_COLOR):
        self.radius = radius
        self.diffuse_color_function = diffuse_color_function
        self._radius_sq = radius ** 2

    def get_norm(self, intersections: vec3):
        return intersections * (1 / self.radius)

    def get_intersection(self, starts: vec4, directions: vec4, inverted_trace):
        x0 = starts.vec3()
        d= directions.vec3()

        a = d.dot(d)
        b = 2 * x0.dot(d)
        c = x0.dot(x0) - self._radius_sq

        root1, root2 = quadratic_eqn_roots(a, b, c)
        root1= np.where(root1 >= 0, root1, np.nan)
        root2= np.where(root2 >= 0, root2, np.nan)
        root= np.fmin(root1, root2)
        intersection= x0 +  d * root

        if inverted_trace:
            return intersection.vec4(starts.t - (intersection - starts).norm())
        else:
            return intersection.vec4(starts.t + (intersection - starts).norm())

class Cylinder(Shape):
    # 由轴线段，半径和颜色定义的圆柱
    def __init__(self, start: vec3, end: vec3, radius, diffuse_color_function= lambda p:DEFAULT_OBJ_COLOR):
        self.start = start
        self.end = end
        self.radius = radius
        self.diffuse_color_function = diffuse_color_function
        self._radius_sq = radius ** 2
        self._axis = self.end - self.start
        self._axis_sq = self._axis.dot(self._axis)

    def get_intersection(self, starts: vec4, directions: vec4, inverted_trace):
        x0 = starts.vec3()
        d = directions.vec3()
        d_proj = d - self._axis * (d.dot(self._axis) / self._axis_sq)

        q = x0 - self.start
        q_proj = q - self._axis * (q.dot(self._axis) / self._axis_sq)

        a = d_proj.dot(d_proj)
        np.where(a == 0, np.nan, a)
        b = 2 * d_proj.dot(q_proj)
        c = q_proj.dot(q_proj) - self._radius_sq

        def in_domain(root, x):
            s= (1 / self._axis_sq) * (x - self.start).dot(self._axis)
            return np.logical_and(root>=0, np.logical_and(s>=0, s<=1))

        root1, root2= quadratic_eqn_roots(a, b, c)
        x1= x0 + d * root1 # parameter for the cylinder axis line segment
        root1= np.where(in_domain(root1, x1), root1, np.nan)
        x2= x0 + d * root2
        root2= np.where(in_domain(root2, x2), root2, np.nan)
        root= np.fmin(root1, root2)
        x= x0 + d * root 

        if inverted_trace:
            return x.vec4(starts.t - (x - starts).norm())
        else:
            return x.vec4(starts.t + (x - starts).norm())
    
    def get_norm(self, intersections: vec3):
        a= intersections - self.start
        b= intersections - self.end
        t= b.dot( b - a ) / ( (a - b).dot( a - b ) )
        axis_to_in= a * t +  b * (1 - t)
        return axis_to_in.normalize()

class Plane(Shape):

    def __init__(self, center: vec3= vec3(0, -.5, 0), norm: vec3= vec3(0, 1, 0), range_func= lambda inter: True, diffuse_color_function= lambda p: DEFAULT_OBJ_COLOR):
        self.center= center
        self.norm= norm.normalize()
        self.range_func= range_func
        self.diffuse_color_function= diffuse_color_function

    def get_norm(self, Intersections: vec3):
        return self.norm

    # TODO: improve
    def get_intersection(self, starts: vec4, directions: vec4, inverted_trace):
        x0= starts.vec3()
        d= directions.vec3()
        n= self.norm
        c= self.center

        t= n.dot(c - x0) / n.dot(d)
        t= np.where(np.logical_and(t >= 0, np.logical_not(np.isinf(t))), t, np.nan)
        
        pre_intersection= x0 + d * t
        t= np.where(self.range_func(pre_intersection), t, np.nan)
        intersection= x0 + d * t    # vec3

        if inverted_trace:
            return intersection.vec4(starts.t - (intersection - starts).norm())
        else:
            return intersection.vec4(starts.t + (intersection - starts).norm())

class CompositeShape(Shape):
    def __init__(self, shapes):
        self.shapes = shapes

    def get_intersection(self, starts, directions, inverted_trace):
        intersections= [obj.get_intersection(starts, directions, inverted_trace) for obj in self.shapes]
        distances= [(intersection - starts).vec3().norm() for intersection in intersections]
        distances= [np.where(np.isnan(distance), FARAWAY, distance) for distance in distances]
        nearest= reduce(np.minimum, distances)

        # 注意啦！intersections是用来可见性竞争的！没有交点一定要返回 vec4(nan, nan, nan, nan) ！（diatance会被自动记为FARAWAY） 惨痛教训 --2022.2.6

        # TODO: 虽然修复了bug但出现了color和norm的大量冗余运算
        color= rgb(0,0,0)
        norm= vec3(0,0,0)
        nearest_intersection= vec4(0,0,0,0)
        for shape, distance, intersection in zip(self.shapes, distances, intersections):
            hit= nearest == distance # 错误代码：(nearest != FARAWAY) & (nearest == distance)
            if np.any(hit):
                hit_intersection= intersection.extract(hit)
                nearest_intersection+= hit_intersection.place(hit) # np.nan + np.nan
                
                hit_color= shape.get_diffuse_color(hit_intersection)
                color+= hit_color.place(hit)
            
                hit_norm= shape.get_norm(hit_intersection.vec3())
                norm+= hit_norm.place(hit)
        
        return nearest_intersection, color, norm
    
    # TODO
    def get_norm(self, *args, **kwargs):
        raise Exception('Please use get_intersection to get intersection, dissuse color, and norm. ')

class RectangularPrism(CompositeShape):

    def __init__(self, width, height, depth, segment_radius, diffuse_color_function= lambda p:DEFAULT_OBJ_COLOR):
        self.width = width
        self.height = height
        self.depth = depth
        self.segment_radius = segment_radius
        self.diffuse_color_function = diffuse_color_function
        CompositeShape.__init__(self, self._get_cylinders())

    def _get_cylinders(self):
        x = self.width / 2.0 + self.segment_radius
        y = self.height / 2.0 + self.segment_radius
        z = self.depth / 2.0 + self.segment_radius

        endpoints = [
            ((+x, +y, +z), (+x, -y, +z)),
            ((+x, -y, +z), (-x, -y, +z)),
            ((-x, -y, +z), (-x, +y, +z)),
            ((-x, +y, +z), (+x, +y, +z)),
            ((+x, +y, -z), (+x, -y, -z)),
            ((+x, -y, -z), (-x, -y, -z)),
            ((-x, -y, -z), (-x, +y, -z)),
            ((-x, +y, -z), (+x, +y, -z)),
            ((+x, +y, +z), (+x, +y, -z)),
            ((+x, -y, +z), (+x, -y, -z)),
            ((-x, -y, +z), (-x, -y, -z)),
            ((-x, +y, +z), (-x, +y, -z)),
        ]
        return [Cylinder(vec3(*start), vec3(*end), self.segment_radius, self.diffuse_color_function) for start, end in endpoints]

class Cube(CompositeShape):

    def __init__(self, width, height, depth, diffuse_color_function: Callable[[vec3],rgb]= lambda p:DEFAULT_OBJ_COLOR):
        self.width = width
        self.height = height
        self.depth = depth
        self.diffuse_color_function = diffuse_color_function
        CompositeShape.__init__(self, self._get_surfaces())

    def _get_surfaces(self):
        x = self.width / 2.0
        y = self.height / 2.0
        z = self.depth / 2.0

        def range_func_x(inter: vec3):
            _y= np.logical_and(inter.y <= y, inter.y >= -y)
            _z= np.logical_and(inter.z <= z, inter.z >= -z)
            return np.logical_and(_y, _z)
        def range_func_y(inter: vec3):
            _x= np.logical_and(inter.x <= x, inter.x >= -x)
            _z= np.logical_and(inter.z <= z, inter.z >= -z)
            return np.logical_and(_x, _z)
        def range_func_z(inter: vec3):
            _x= np.logical_and(inter.x <= x, inter.x >= -x)
            _y= np.logical_and(inter.y <= y, inter.y >= -y)
            return np.logical_and(_x, _y)

        #前后左右上下 共6个面  计算量远远少于RectangularPrism
        ctns=    [(0,0,z),(0,0,-z),(0,y,0),(0,-y,0),(x,0,0),(-x,0,0)]
        range_funcs= [range_func_z, range_func_z, range_func_y, range_func_y, range_func_x, range_func_x]

        return [Plane(vec3(*ctn), vec3(*ctn), range_func, self.diffuse_color_function) for ctn, range_func in zip(ctns, range_funcs)]

class MovingObject:
    '''存储变换、速度、形状、材质信息，可以直接获取颜色'''
    def __init__(self, shape: Shape, beta= (0, 0, 0), offset: vec4= vec4(0, 0, 0, 0), material= Material()):
        self.shape = shape
        self.beta = np.array(beta)
        self.v= vec3(*self.beta)
        self.u= np.sqrt(self.beta.dot(self.beta))# 速率
        self.offset = offset
        self.material= material
    def set_beta(self, beta):
        self.beta = np.array(beta)
        self.v= vec3(*self.beta)
        self.u= np.sqrt(self.beta.dot(self.beta))# 速率
    def transform_ray_from_ether(self, start_ether: vec4, direction_ether: vec4):
        boost_matrix= lorentz_boost(self.beta)
        start_obj= (start_ether - self.offset).apply_matrix(boost_matrix)
        direction_obj= direction_ether.apply_matrix(boost_matrix)
        return start_obj, direction_obj
    def transform_point_from_ether(self, point: vec4):
        boost_matrix= lorentz_boost(self.beta)
        return (point - self.offset).apply_matrix(boost_matrix) 
    def transform_point_from_obj(self, point: vec4):
        boost_matrix= lorentz_boost(-self.beta)
        return point.apply_matrix(boost_matrix) + self.offset
    # TODO, make CompositeShape standard
    def get_intersection_and_color(self, starts_ether, directions_ether, inverted_trace= True):
        starts_obj, directions_obj= self.transform_ray_from_ether(starts_ether, directions_ether) # vec4
        if not isinstance(self.shape, CompositeShape):
            intersection_obj= self.shape.get_intersection(starts_obj, directions_obj, inverted_trace)
            diffuse_color= self.shape.get_diffuse_color(intersection_obj)
            intersection_ether= self.transform_point_from_obj(intersection_obj)
            return intersection_ether, diffuse_color
        else:
            intersection_obj, diffuse_color, norm= self.shape.get_intersection(starts_obj, directions_obj, inverted_trace)
            intersection_ether= self.transform_point_from_obj(intersection_obj)
            return intersection_ether, (diffuse_color, norm)
    def get_color(self, objs, intersection_ether, diffuse_color, Origin, light_pos, Norm= None):#starts_obj, directions_obj, travel_times_obj):
        # M 交点  N 法向量  物体参考系下
        M= intersection_ether
        L= light_pos.vec4(intersection_ether.t - (light_pos - intersection_ether).norm())
        ML= L - M
        OM= intersection_ether - Origin
        #N= self.shape.get_norm(self.transform_point_from_ether(M).vec3())
        #nudged = M + N * .0001

        if self.material:
            
            # Shadowing: 确认从光源发出，打中这一点的光线是否经过其他物体
            intersections_and_color= [obj.get_intersection_and_color(L, - ML, inverted_trace= False) for obj in objs] # 交点（以太坐标）, 漫反射颜色
            distances= [(inter_ether - L).t for inter_ether, color in intersections_and_color]
            distances= [np.where(np.isnan(d), FARAWAY, d) for d in distances]# 交点距离
            nearest = reduce(np.minimum, distances)
            #seelight= abs(nearest - (light_pos - intersection_ether).norm()) < 0.001
            seelight= nearest == distances[objs.index(self)]

            # Ambient
            color = self.material.ambient

            # Lambert shading (diffuse)
            N_obj= self.shape.get_norm(self.transform_point_from_ether(intersection_ether).vec3()) if (Norm is None) else Norm
            ML_obj= self.transform_point_from_ether(ML).vec3().normalize()
            lv = np.maximum(N_obj.dot(ML_obj), 0)
            color+= diffuse_color * lv * np.where(seelight,1,self.material.shadow)
            
            # Blinn-Phong shading (specular)
            if self.material.gloss:
                OM_obj= self.transform_point_from_ether(OM).vec3().normalize()
                phong = N_obj.dot((ML_obj + OM_obj).normalize())
                color += rgb(1, 1, 1) * np.power(np.clip(phong, 0, 1), self.material.gloss/4) * seelight

            # Combination  纯个人审美，我觉得整体提高亮度不至于太黑会更好看
            color = color * (1 - self.material.diffuse_combination) + diffuse_color * seelight * self.material.diffuse_combination
        else:
            color= diffuse_color# * np.where(seelight,1,0.2)
        
        # HeadLight Effect & Doppler Effect
        x01=(OM-L).vec3()
        vcosθ1= self.v.dot(x01)/np.sqrt(x01.dot(x01))
        Doppler_factor1= (1 - vcosθ1 )/np.sqrt(1-self.v.dot(self.v))

        x02= OM.vec3()
        vcosθ2= x02.dot(self.v)/np.sqrt(x02.dot(x02))
        Doppler_factor2= np.sqrt(1-self.v.dot(self.v))/(1 + vcosθ2 )
        
        Doppler_factor= Doppler_factor1 * Doppler_factor2
        Headlight_factor= np.power(Doppler_factor, 3)

        color= color.Doppler(Doppler_factor)
        color= color * Headlight_factor
        
        return color

# TODO
class ForegroundObject:
    pass

# 弄清参考系!
# 1. 以太参考系；(为了方便建立，目前相机不能运动，故和相机参考系同)
# 2. 相机参考系；
# 3. 各个物体的参考系；
def raytrace(starts_ether: vec4, directions_ether: vec4, objs, light_pos):
    '''
    starts_ether是光源的以太参考系时空坐标        starts_obj是光源的物体参考系时空坐标
    directions_ether是光方向的以太时空坐标        directions_ether是光方向的物体参考系时空坐标
    '''
    intersections_and_color= [obj.get_intersection_and_color(starts_ether, directions_ether) for obj in objs] # 交点（以太坐标）, 漫反射颜色
    distances= [-(intersection_ether - starts_ether).t for intersection_ether, color in intersections_and_color]
    distances= [np.where(np.isnan(d), FARAWAY, d) for d in distances]
    nearest = reduce(np.minimum, distances)

    color= rgb(0,0,0)

    #hit = (nearest != FARAWAY) & (distances[1] == nearest)
    #color= rgb(1,1,1) * hit

    for (obj, intersection_and_color, distance) in zip(objs, intersections_and_color, distances):
        hit = (nearest != FARAWAY) & (distance == nearest)  # 获胜区域
        if np.any(hit):
            if isinstance(obj.shape, CompositeShape):
                intersection_ether, diffuse_color_and_norm= intersection_and_color
                diffuse_color, norm= diffuse_color_and_norm

                hit_intersection_ether= intersection_ether.extract(hit)
                hit_diffuse_color= diffuse_color.extract(hit)
                hit_norm= norm.extract(hit)
        
                hit_color= obj.get_color(objs, hit_intersection_ether, hit_diffuse_color, starts_ether, light_pos, hit_norm)
            else:
                intersection_ether, diffuse_color= intersection_and_color

                hit_intersection_ether= intersection_ether.extract(hit)
                hit_diffuse_color= diffuse_color.extract(hit)
        
                hit_color= obj.get_color(objs, hit_intersection_ether, hit_diffuse_color, starts_ether, light_pos)
            color += hit_color.place(hit)

    return color

class Camera:
    def __init__(self, definition= DEFAUT_DEFINITION, center= ORIGIN, camera_height= DEFAUT_CAMERA_HEIGHT, focal_length= DEFAUT_FOCAL_LENGTH, fps= 60):
        width, height= definition # 这是像素为单位，相机镜头的分辨率
        resolution= height/width
        camera_width= camera_height/resolution  # 这是相机在场景中的高度
        
        self.center= center
        self.height= height
        self.width= width
        self.camera_width= camera_width
        self.camera_height= camera_height
        self.focal_length= focal_length
        self.fps= fps

        self.bg= rgb(0,0,0)*np.repeat(0,width*height)
        
        self.direction= self._get_directions()

    def _get_directions(self):
        S = (-self.camera_width, self.camera_height, self.camera_width, -self.camera_height)
        x= np.tile(np.linspace(S[0], S[2], self.width), self.height)
        y= np.repeat(np.linspace(S[1], S[3], self.height), self.width)
        z= self.focal_length
        origin_to_image_times= abs(vec4(0,x,y,z))
        # 光线在四维闵可夫斯基时空中的方向
        direction= vec4(-origin_to_image_times, x, y, z) 
        
        return direction

    def get_rays(self, shot_time):
        # 光线在四维闵可夫斯基时空中的端点
        start= self.center + vec4(shot_time, 0, 0, 0)

        return start, self.direction
    
    def __add__(self, other):
        definition= ( (self.width + other.width )/2, ( self.height + other.height )/2 )
        center= ( self.center + other.center )/2
        camera_height= ( self.camera_height + other.camera_height )/2
        focal_length= ( self.focal_length + other.focal_length )/2
        fps= ( self.fps + other.fps )/2
        return Camera(definition, center, camera_height, focal_length, fps)
PR= Camera(LOW_DEFINITION, fps= 10)
HD= Camera(HIGH_DEFINITION, fps= 60)

class Scene:
    def __init__(
        self,
        movingobjects: Iterable[MovingObject],
        light_pos:vec3= DEFAUT_LIGHT_POS,
        foregroundobjects= None,
        compositors= None,
    ):
        # static information of a Scene is initialized here

        if foregroundobjects is None:
            foregroundobjects= []
        if compositors is None:
            compositors= []
        self.movingobjects= movingobjects
        self.light_pos= light_pos
        self.foregroundobjects= foregroundobjects
        self.compositors= compositors
        self.render_kwargs= {}
        self.set_render_properties(t_start= 0, t_end= 10, duration= 10, frames= 300, save_path= ".\render", file_path= None, file_name= "image.png", camera= None, open_path= True, open_file= True, updaters= [], show_window= True, window_width= 700)
    
    def set_render_properties(self, **kwargs):
        """t_start, t_end, frames, save_path, file_name, camera, open_path, frames"""
        # dynamic information of a Scene is initialized here

        for key in kwargs.keys():
            self.render_kwargs[key]= kwargs[key]
        return self

    def _get_scene_name(self):
        module_locals= inspect.currentframe().f_back.f_back.f_back.f_locals
        for local_name in module_locals.keys():
            if module_locals[local_name] is self:
                return local_name

    def render(self, mode= 0): # camera to 
        """
        mode:  
        0 for render to a movie with PR
        1 for render to a movie with Camera()
        2 for render to a movie with self.camera provided through set_render_properties
        3 for render to a image at the moment of self.t_start with Camera()
        4 for render to a sequence of images
        """
        kwargs= {
            "t_start": self.render_kwargs["t_start"],
            "t_end": self.render_kwargs["t_end"],
            "duration": self.render_kwargs["duration"],
            "file_path": self.render_kwargs["file_path"],
            "updaters": self.render_kwargs["updaters"],
            "open_file": self.render_kwargs["open_file"],
            "show_window": self.render_kwargs["show_window"],
            "window_width": self.render_kwargs["window_width"],
        }
        if mode == 0:
            kwargs["camera"] = PR
            self.generate_animation(**kwargs)
        elif mode == 1:
            kwargs["camera"] = Camera()
            self.generate_animation(**kwargs)
        elif mode == 2:
            kwargs["camera"] = self.camera
            self.generate_animation(**kwargs)
        elif mode == 3:
            self.generate_image(
                shot_time= kwargs["t_start"],
                camera= Camera(),
                file_name= self.render_kwargs["file_name"],
                open_file= kwargs["open_file"]
            )
        elif mode == 4:
            self.generate_sequence(
                shot_time= kwargs["t_start"],
                camera= Camera(),
                frames= self.render_kwargs["frames"],
                save_path= self.render_kwargs["save_path"],
                open_file= kwargs["open_file"]
            )
        else:
            raise ValueError("mode provided didn't exist.")

    def add_movingobject(self, movingobject: MovingObject):
        self.movingobjects.append(movingobject)

    def clear_objects(self):
        self.movingobjects= []
    
    def _generate_pixel_array(self, shot_time= 0, camera= None):
        image= self._generate_image().convert("RGBA")
        return np.array(image)

    def _generate_image(self, shot_time= 0, camera= None):
        if camera is None:
            camera= Camera()

        start, direction= camera.get_rays(shot_time)
        color= camera.bg + raytrace(start, direction, self.movingobjects, self.light_pos)

        file_color = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((camera.height, camera.width))).astype(np.uint8), "L") for c in color.components()]
        image= Image.merge("RGB", file_color)
        
        for compositor in self.compositors:
            image= compositor(image)

        return image

    @timeit
    def generate_image(self, shot_time= 0, camera= None, file_name= 'image.png', open_file= True):
        self._generate_image(shot_time, camera).save(file_name)
        print("files' prepared at ", file_name)
        if open_file:
            os.startfile(file_name)
        return file_name
    
    def generate_sequence(self, t_start, t_end, frames= 300, save_path= './render', camera= None, open_path= True): # 输出png序列
        if camera is None:
            camera= Camera()
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path= os.path.realpath(save_path)

        t00= time.time()
        for shot_time, frame_count in tqdm(np.linspace([t_start,1],[t_end,frames], frames)):
            self.generate_image(shot_time= shot_time, camera= camera, file_name= os.path.join(save_path, f"{int(frame_count)}.png"), open_file= False)
            t1= time.time()
            print("%i minute left" % ((t1-t00)/frame_count*(frames-frame_count)/60) )
        print("files' prepared at ", save_path)
    
        if open_path:
            os.startfile(save_path)
    
    def generate_animation(self, t_start, t_end, duration= 5, file_path= None, camera= None, updaters= [], open_file= True, show_window= True, window_width= 700):
        if camera is None:
            camera= Camera()
        if file_path is None:
            file_path= f"./render/{camera.height}p{camera.fps}"    
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        # TODO, predict which frame is under the scene file
        proper_filename= os.path.splitext(os.path.basename(inspect.currentframe().f_back.f_back.f_code.co_filename))[0] + '_' + self._get_scene_name()
        file_path= os.path.realpath(file_path)
        file_name_temporary= os.path.join(file_path, proper_filename + "_临时.mp4")
        file_name_finally= os.path.join(file_path, proper_filename + ".mp4")

        print("initiating ffmpeg pipe")
        command = [
            FFMPEG_BIN,
            '-y',  # overwrite output file if it exists
            '-f', 'rawvideo',
            '-s', '%dx%d' % (camera.width, camera.height),  # size of one frame
            '-pix_fmt', 'rgba',
            '-r', str(camera.fps),  # frames per second
            '-i', '-',  # The imput comes from a pipe
            '-an',  # Tells FFMPEG not to expect any audio
            '-loglevel',
            'error',
        ]
        command += [
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
        ]
        command += [file_name_temporary]
        writing_process = subprocess.Popen(command, stdin=subprocess.PIPE)

        if show_window:
            pygame.init()
            size =  int(window_width), int(window_width/camera.width*camera.height)
            screen= pygame.display.set_mode(size)
            pygame.event.set_blocked(None)
            pygame.event.set_allowed(pygame.QUIT)

        frames= int(duration)*camera.fps
        for shot_time, frame_count in tqdm(np.linspace([t_start,1], [t_end, frames], frames)):
            
            alpha= (frame_count-1)/frames
            for updater in updaters:
                updater(self, alpha)
            
            image= self._generate_image(shot_time, camera)
            frame_ffmpeg= np.array(image.convert("RGBA"))
            writing_process.stdin.write(frame_ffmpeg.tobytes())

            if show_window:
                frame_pygame= np.array(image.resize(size)).transpose((1, 0, 2))
                pygame.surfarray.blit_array(screen, frame_pygame)
                pygame.display.flip()
                pygame.event.get()
    
        if show_window:
            pygame.quit()
    
        writing_process.stdin.close()
        writing_process.wait()

        
        shutil.move(file_name_temporary, file_name_finally)
        print("file's prepared at", file_name_finally)

        if open_file:
            os.startfile(file_name_finally)

    def add_foregroundobject(self, foregroundobject: ForegroundObject):
        self.foregroundobjects.append(foregroundobject)
    
    def add(self, *objects):
        for any_object in objects:
            if isinstance(any_object, MovingObject):
                self.add_movingobject(any_object)
            elif isinstance(any_object, ForegroundObject):
                self.add_foregroundobject(any_object)

class Generator:
    def __init__(self, method, *args, **kwargs):
        self.method= method
        self.args= args
        self.kwargs= kwargs
    def generate(self):
        self.method(*self.args, **self.kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", help="path to file holding the python code for the scene")
    parser.add_argument(
            "-p", "--preview",
            action="store_true",
            help="Automatically open the saved file once its done",
        )
    parser.add_argument(
            "-l", "--low_quality",
            action="store_true",
            help="Render at a low quality (for faster rendering)",
        )
    parser.add_argument(
            "-s", "--save_last_frame",
            action="store_true",
            help="Save the last frame",
        )
    args= parser.parse_args()
    file_name= args.file_name
    module= get_module(file_name)
    scene_names= []
    scene_objects= []
    for (name, scene) in inspect.getmembers(module, lambda x: isinstance(x, module.Scene)):
        scene_names.append(name)
        scene_objects.append(scene)
    print("\n".join(f"{num}: {name}" for num, name in enumerate(scene_names)))
    scene_to_render= scene_objects[int(input("choose form the scenes:"))]
    
    mode= 0 if args.preview and args.low_quality else 1 if args.preview else int(input("mode select (from 0,1,2,3,4):")) # 懒得写完了
    scene_to_render.render(mode)