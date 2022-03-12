# Idea From:
# [1] Relativistic Ray-Tracing: Simulating the Visual Appearance of Rapidly Moving Objects - Sandy Dance
# [2] James Terrell, "Invisibility of the lorentz contraction", Phys. Rev. 116, 1041-1045 (1959).
# [3] https://www.youtube.com/watch?v=oFaSLIsJELY
# [4] https://excamera.com/sphinx/article-ray.html

# Using Libraries: pillow, numpy

from util import *

class Material:
    def __init__(self, gloss= 700, mirror= 0.5, ambient= rgb(0.08, 0.08, 0.08), shadow= .2, diffuse_combination= 0):
        self.gloss= gloss
        self.mirror= mirror
        self.ambient= ambient
        self.diffuse_combination= diffuse_combination
        self.shadow= shadow
    def smoothen(self):
        pass
    def roughen(self):
        pass

# 定义一个形状，给出 求交点 和 求法向量 的方法就行了
class Shape(ABC):
    '''储存形状、基础色'''
    @abstractmethod
    def get_norm(self, Intersections: vec3):
        '''法向量'''
    @abstractmethod
    def get_intersection(self, starts: vec4, directions: vec4, inverted_trace):
        '''交点。注意规范：没有交点的射线要返回 vec4(nan, nan, nan, nan) 鸭！'''
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

    def __init__(self, center: vec3, norm: vec3, range_func= lambda inter: True, diffuse_color_function= lambda p: DEFAULT_OBJ_COLOR):
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

    def __init__(self, width, height, depth, diffuse_color_function= lambda p:DEFAULT_OBJ_COLOR):
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
    def __init__(self, shape: Shape, beta, offset: vec4, material= Material()):
        self.shape = shape
        self.beta = np.array(beta)
        self.v= vec3(*self.beta)
        self.u= np.sqrt(self.beta.dot(self.beta))# 速率
        self.offset = offset
        self.material= material
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
    # TODO
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
        light= light_pos.vec4(intersection_ether.t - (light_pos - intersection_ether).norm())
        MtoL= light - M
        MtoO= intersection_ether - Origin
        #N= self.shape.get_norm(self.transform_point_from_ether(M).vec3())
        #nudged = M + N * .0001

        if self.material:
            
            # Shadowing: 确认从光源发出，打中这一点的光线是否经过其他物体
            intersections_and_color= [obj.get_intersection_and_color(light, - MtoL, inverted_trace= False) for obj in objs] # 交点（以太坐标）, 漫反射颜色
            distances= [(inter_ether - light).t for inter_ether, color in intersections_and_color]
            distances= [np.where(np.isnan(d), FARAWAY, d) for d in distances]# 交点距离
            nearest = reduce(np.minimum, distances)
            #seelight= abs(nearest - (light_pos - intersection_ether).norm()) < 0.001
            seelight= nearest == distances[objs.index(self)]

            # Ambient
            color = self.material.ambient

            # Lambert shading (diffuse)
            N_obj= self.shape.get_norm(self.transform_point_from_ether(intersection_ether).vec3()) if (Norm is None) else Norm
            MtoL_obj= self.transform_point_from_ether(MtoL).vec3().normalize()
            lv = np.maximum(N_obj.dot(MtoL_obj), 0)
            color += diffuse_color * lv * np.where(seelight,1,self.material.shadow)
            
            # Blinn-Phong shading (specular)
            if self.material.gloss:
                MtoO_obj= self.transform_point_from_ether(MtoO).vec3().normalize()
                phong = N_obj.dot((MtoL_obj + MtoO_obj).normalize())
                color += rgb(1, 1, 1) * np.power(np.clip(phong, 0, 1), self.material.gloss/4) * seelight
            
            # HeadLight Effect
            x0=(-MtoO-light).vec3()   # 在物体系下光发出点的四维坐标
            cosθ1= self.v.dot(x0)/np.sqrt((self.v.dot(self.v)) * (x0.dot(x0)))
            headlight_factor1= np.sqrt(1-self.v.dot(self.v))/(1 - self.u * cosθ1 )
            color= color * headlight_factor1

            # Combination  纯个人审美，我觉得整体提高亮度不至于太黑会更好看
            #color = color * (1 - self.material.diffuse_combination) + diffuse_color * seelight * self.material.diffuse_combination
        
        else:
            color= diffuse_color * np.where(seelight,1,0.2)

        # HeadLight Effect
        x00= MtoO.vec3()
        cosθ2= x00.dot(self.v)/np.sqrt(self.v.dot(self.v)*x00.dot(x00))
        headlight_factor2= np.sqrt(1-self.v.dot(self.v))/(1 + self.u * cosθ2 )
        color= color * headlight_factor2
        
        return color

# 弄清参考系十分重要：
# 1. 以太参考系；(为了方便建立，和相机参考系相同)
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
    def __init__(self, center= ORIGIN, definition= DEFAUT_DEFINITION, camera_height= DEFAUT_CAMERA_HEIGHT, focal_length= DEFAUT_FOCAL_LENGTH, fps= 60):
        width, height= definition
        resolution= height/width
        camera_width= camera_height/resolution
        
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

class Scene:
    def __init__(self, movingobjects, camera= None, light_pos= DEFAUT_LIGHT_POS):
        if camera is None:
            camera= Camera()
        self.movingobjects= movingobjects
        self.camera= camera
        self.light_pos= light_pos

    def add_object(self, movingobject: MovingObject):
        self.movingobjects.append(movingobject)

    def clear_objects(self):
        self.movingobjects= []
    
    @timeit
    def generate_image(self, shot_time= 0, file_name= 'image.png'):

        start, direction= self.camera.get_rays(shot_time)

        color= self.camera.bg + raytrace(start, direction, self.movingobjects, self.light_pos)

        file_color = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((self.camera.height, self.camera.width))).astype(np.uint8), "L") for c in color.components()]
        Image.merge("RGB", file_color).save(file_name)

        return file_name
    
    def generate_animation(self, t_start, t_end, frames= 300, Dir= './render', video_name= 'render.avi'): # 输出png序列
        t00= time.time()
        if not os.path.exists(Dir):
            os.mkdir(Dir)
        for shot_time, frame_count in np.linspace([t_start,1],[t_end,frames], frames):
            print('开始渲染第%s帧...' % int(frame_count), end= '')
            self.generate_image(shot_time, os.path.join(Dir, f"{int(frame_count)}.png"))
            t1= time.time()
            print("预计剩余%i分钟" % ((t1-t00)/frame_count*(frames-frame_count)/60) )

if __name__ == '__main__':
    (width, height) = (1920, 1080)      # 屏幕尺寸
    resolution= height/width
    light_pos = vec3(-2, 2, -2)          # 点光源位置
    center = vec4(0, 0, 0, 0)         # 摄像机位置
    focal_length= 200
    shape1= Cylinder(vec3(0, .5, .5), vec3(0, -.5, .5), .5, lambda p: rgb(0.8,0,0.5))
    shape2 = Sphere(.5, get_checkerboard_color_func(rgb(0,0.5,0), rgb(1,1,1)))
    shape3 = Sphere(999999999, lambda p: rgb(0.5,0.5,0.5))#util.get_checkerboard_color_func(rgb(0,.05,.05),rgb(.5,.5,.5), 9999999999))
    shape4 = RectangularPrism(1, 1, 1, .01)
    shape5 = CompositeShape([shape1, shape2])
    beta = (0.1, 0, 0)
    offset1 = vec4(0, -.5, 0, 2)
    offset2 = vec4(0, .4, 0, 1.6)
    offset3 = vec4(0, 0, -999999999.5, 2)

    movingobjects= [MovingObject(shape4, beta, vec4(0, 0, 0, 2), None)]#[MovingObject(shape2, beta, offset2, Material(500)), MovingObject(shape3, (0,0,0), offset3, None)]

    camera_height= 200
    camera_width= camera_height/resolution
    S = (-camera_width, camera_height, camera_width, -camera_height)
    x= np.tile(np.linspace(S[0], S[2], width), height)     # [1,2,3,4,1,2,3,4,1,2,3,4]
    y= np.repeat(np.linspace(S[1], S[3], height), width)   # [1,1,1,1,2,2,2,2,3,3,3,3]
    z= focal_length
    origin_to_image_times= abs(vec4(0,x,y,z))

    t_start= time.time()
    frames= 250
    for shot_time, frame_count in np.linspace([-30,1],[30,frames], frames):
        start= center + vec4(shot_time, 0, 0, 0)# 光线的端点
        directions= vec4(-origin_to_image_times, x, y, z) # 仅仅表示光线的方向
        print('开始渲染第%s帧' % int(frame_count))
        t0 = time.time()
        color = rgb(0,0,0)*np.repeat(0,width*height) + raytrace(start, directions, movingobjects, light_pos)
        print("  耗时%f，预计剩余%i分钟"%(time.time()-t0, (time.time()-t_start)/frame_count*(frames-frame_count)/60))
        filename= ".\\render2\\%i.png" % frame_count
        file_color = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((height, width))).astype(np.uint8), "L") for c in color.components()]
        Image.merge("RGB", file_color).save(filename)