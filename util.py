import numpy as np
import numbers, time, os
from PIL import Image
from functools import reduce
from abc import ABC, abstractmethod

def extract(cond, x):
    if isinstance(x, numbers.Number):
        return x
    else:
        return np.extract(cond, x)

class vec4():
    def __init__(self, t, x, y, z):
        self.t, self.x, self.y, self.z = (t, x, y, z)
    def __mul__(self, other):
        return vec4(self.t * other, self.x * other, self.y * other, self.z * other)
    def __add__(self, other):
        return vec4(self.t + other.t, self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other):
        return vec4(self.t - other.t, self.x - other.x, self.y - other.y, self.z - other.z)
    def __neg__(self):
        return vec4(-self.t, -self.x, -self.y, -self.z)
    def dot(self, other):
        return (self.t * other.t) + (self.x * other.x) + (self.y * other.y) + (self.z * other.z)
    def apply_matrix(self, matrix):
        vec= np.array([self.t, self.x, self.y, self.z])
        t,x,y,z= np.dot(matrix, vec)
        return vec4(t,x,y,z)
    def __abs__(self):  #注意，该方法改为向量求模
        return np.sqrt(self.dot(self))
    def __str__(self):
        return 'vec4(%s, %s, %s, %s)'%(self.t, self.x,self.y,self.z)
    def components(self):
        return (self.t, self.x, self.y, self.z)
    def space_info(self):
        return np.array([self.x, self.y, self.z])
    def extract(self, cond):
        return vec4(extract(cond, self.t),
                    extract(cond, self.x),
                    extract(cond, self.y),
                    extract(cond, self.z))
    def place(self, cond):
        r = vec4(np.zeros(cond.shape), np.zeros(cond.shape), np.zeros(cond.shape), np.zeros(cond.shape))
        np.place(r.t, cond, self.t)
        np.place(r.x, cond, self.x)
        np.place(r.y, cond, self.y)
        np.place(r.z, cond, self.z)
        return r
    def points(self):
        arr= np.array([self.x, self.y, self.z])
        return np.transpose(arr)
    def vec3(self):
        return vec3(self.x, self.y, self.z)

class vec3():
    '''这是一个三维坐标。例如当计算交点时，用不到四维坐标，可以暂时转换为三维减少计算量。'''
    def __init__(self, x, y, z):
        (self.x, self.y, self.z) = (x, y, z)
    def __mul__(self, other):
        return vec3(self.x * other, self.y * other, self.z * other)
    def __truediv__(self, other): #
        return vec3(self.x / other, self.y / other, self.z / other)
    def __rtruediv__(self, other):
        return vec3(other.x / self.x, other.y / self.y, other.z / self.z)
    def __add__(self, other):
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other):
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    def __neg__(self):
        return vec3(-self.x, -self.y, -self.z)
    def dot(self, other):
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)
    def __abs__(self):
        return self.dot(self)
    def __str__(self):
        return 'vec3(%s,%s,%s)'%(self.x,self.y,self.z)
    def norm(self):
        return np.sqrt(abs(self))
    def normalize(self):
        return self * (1/self.norm())
    def components(self):
        return (self.x, self.y, self.z)
    def points(self):
        arr= np.array([self.x, self.y, self.z])
        return np.transpose(arr)
    def extract(self, cond):
        return vec3(extract(cond, self.x),
                    extract(cond, self.y),
                    extract(cond, self.z))
    def place(self, cond):
        r = vec3(np.zeros(cond.shape), np.zeros(cond.shape), np.zeros(cond.shape))
        np.place(r.x, cond, self.x)
        np.place(r.y, cond, self.y)
        np.place(r.z, cond, self.z)
        return r
    def vec4(self, t):
        return vec4(t, self.x, self.y, self.z)
rgb=vec3

def quadratic_eqn_roots(a, b, c):

    np.seterr(all='ignore')

    discriminant = b ** 2 - 4 * a * c
    sqrt_discriminant = np.sqrt(discriminant)

    root1= (-b - sqrt_discriminant) / (2 * a)
    root2= (-b + sqrt_discriminant) / (2 * a)
    
    return root1, root2

def lorentz_boost(beta):
    beta = np.asarray(beta)
    beta_squared = np.inner(beta, beta)
    if beta_squared >= 1:
        raise ValueError(f"beta²= {beta_squared} 超光速")
    if beta_squared == 0:
        return np.identity(4)
    gamma = 1 / np.sqrt(1 - beta_squared)
    lambda_00 = np.matrix([[gamma]])
    lambda_0j = -gamma * np.matrix(beta)
    lambda_i0 = lambda_0j.transpose()
    lambda_ij = np.identity(3) + (gamma - 1) * np.outer(beta, beta) / beta_squared
    return np.asarray(np.bmat([[lambda_00, lambda_0j], [lambda_i0, lambda_ij]]))

def spherical_angles(point: vec3):
    x, y, z = point.components()
    radius = point.norm()
    theta = np.arccos(z / radius)
    phi = np.arctan2(y, x) + np.pi
    return theta, phi

def checkerboard(point, color1= rgb(0.5,0.5,0.5), color2= rgb(1,1,1), ranks= 12):
    theta, phi = spherical_angles(point)
    n_theta = np.floor((theta / np.pi) * ranks)
    n_phi = np.floor((phi / (2 * np.pi)) * ranks)

    return  color1 + (color2 - color1) * ((n_theta + n_phi) % 2)

def get_checkerboard_color_func(color1, color2, ranks= 12):
    return lambda p: checkerboard(p,color1, color2, ranks)

def checkerboard(point:vec3, color1= rgb(0.5,0.5,0.5), color2= rgb(1,1,1), ranks= 12):
    theta, phi = spherical_angles(point)
    n_theta = np.floor((theta / np.pi) * ranks)
    n_phi = np.floor((phi / (2 * np.pi)) * ranks)

    return  color1 + (color2 - color1) * ((n_theta + n_phi) % 2)

def get_cubical_checkerboard_color_func(color1, color2, width= .1, offset: vec3 = vec3(.001,.001,.001)):
    return lambda p: cubical_checkerboard(p + offset ,color1, color2, width)

def cubical_checkerboard(point: vec3, color1= rgb(0.5,0.5,0.5), color2= rgb(1,1,1), width= .1):
    x, y, z = point.components()
    cond_x= np.floor(x/width)
    cond_y= np.floor(y/width)
    cond_z= np.floor(z/width)
    cond= np.mod(cond_x + cond_y + cond_z, 2)
    return color1 + (color2 - color1) * cond

def range_func_from_image(filename, resize= (1.920, 1.080), offset= (0, 0), black= True):
    image = Image.open(filename).convert('1')
    image_width, image_height= image.size
    real_width, real_height= resize
    
    pixels= black ^ np.array(image)
    
    def x_transform_to_image(x): # array
        return  np.int32(np.floor(x * (image_width/real_width) + offset[0] + image_width/2))
    
    def y_transform_to_image(y):
        return np.int32(np.floor(image_height/2 - (y * (image_height/real_height) + offset[1])))

    def range_func(inter: vec3):
        transformed_inter= vec3(x_transform_to_image(inter.x), y_transform_to_image(inter.y), 0)

        hit_image= reduce(np.logical_and, (
                                            transformed_inter.x >= 0,
                                            transformed_inter.x < image_width,
                                            transformed_inter.y >= 0,
                                            transformed_inter.y < image_height
                                            )
                        )
        hit_image_inter= transformed_inter.extract(hit_image)
        hit_image_inter_hit_white_pixel= pixels[hit_image_inter.y, hit_image_inter.x]

        hit= np.zeros(len(inter.x))
        np.place(hit, hit_image, hit_image_inter_hit_white_pixel)

        return hit
    
    return range_func

def timeit(func):
    def time_func(*args, **kwargs):
        t0= time.time()
        output= func(*args, **kwargs)
        print(f'耗时{time.time() - t0}s...', end= '')
        return output
    return time_func

FARAWAY= 1.0e39
DEFAUT_CAMERA_HEIGHT= 200
DEFAUT_FOCAL_LENGTH= 200
ORIGIN= vec4(0, 0, 0, 0)
DEFAUT_LIGHT_POS= vec3(2, 2, -2)# 默认光源位置
LOW_DEFINITION= (533.3, 300)
DEFAUT_DEFINITION= (1920, 1080)
HIGH_DEFINITION= (4096, 3112)
DEFAULT_OBJ_COLOR = rgb(1,1,1)
BILIBILIPINK= rgb(1.0, 0.44140625, 0.62109375) # B站粉
BILIBILIBLUE_A= rgb(0.41796875, 0.7578125, 0.92578125) # B站青
BILIBILIBLUE= rgb(0.00390625, 0.640625, 0.93359375) # B站蓝
DARK_BLUE= rgb(0.140625, 0.421875, 0.55859375)
DARK_BROWN= rgb(0.546875, 0.2734375, 0.078125)
LIGHT_BROWN= rgb(0.8046875, 0.5234375, 0.25)
BLUE_E= rgb(0.11328125, 0.4609375, 0.54296875)
BLUE_D= rgb(0.1640625, 0.671875, 0.79296875)
BLUE_C= rgb(0.34765625, 0.76953125, 0.8671875)
BLUE_B= rgb(0.61328125, 0.86328125, 0.921875)
BLUE_A= rgb(0.78125, 0.9140625, 0.9453125)
TEAL_E= rgb(0.2890625, 0.66015625, 0.5625)
TEAL_D= rgb(0.3359375, 0.7578125, 0.65625)
TEAL_C= rgb(0.36328125, 0.81640625, 0.703125)
TEAL_B= rgb(0.46484375, 0.8671875, 0.75390625)
TEAL_A= rgb(0.67578125, 0.91796875, 0.84375)
GREEN_E= rgb(0.4140625, 0.61328125, 0.32421875)
GREEN_D= rgb(0.46875, 0.69140625, 0.3671875)
GREEN_C= rgb(0.515625, 0.7578125, 0.40625)
GREEN_B= rgb(0.65234375, 0.8125, 0.55078125)
GREEN_A= rgb(0.7890625, 0.88671875, 0.68359375)
YELLOW_E= rgb(0.91015625, 0.7578125, 0.11328125)
YELLOW_D= rgb(0.95703125, 0.828125, 0.2734375)
YELLOW_C= rgb(1.0, 1.0, 0.00390625)
YELLOW_B= rgb(1.0, 0.91796875, 0.58203125)
YELLOW_A= rgb(1.0, 0.9453125, 0.71484375)
GOLD_E= rgb(0.78125, 0.5546875, 0.27734375)
GOLD_D= rgb(0.8828125, 0.6328125, 0.34765625)
GOLD_C= rgb(0.94140625, 0.67578125, 0.375)
GOLD_B= rgb(0.9765625, 0.71875, 0.4609375)
GOLD_A= rgb(0.96875, 0.78125, 0.59375)
RED_E= rgb(0.8125, 0.31640625, 0.26953125)
RED_D= rgb(0.90234375, 0.35546875, 0.30078125)
RED_C= rgb(0.98828125, 0.38671875, 0.3359375)
RED_B= rgb(1.0, 0.50390625, 0.50390625)
RED_A= rgb(0.96875, 0.6328125, 0.640625)
MAROON_E= rgb(0.58203125, 0.26171875, 0.3125)
MAROON_D= rgb(0.63671875, 0.3046875, 0.3828125)
MAROON_C= rgb(0.7734375, 0.375, 0.453125)
MAROON_B= rgb(0.92578125, 0.57421875, 0.671875)
MAROON_A= rgb(0.92578125, 0.671875, 0.7578125)
PURPLE_E= rgb(0.39453125, 0.2578125, 0.44921875)
PURPLE_D= rgb(0.4453125, 0.3359375, 0.51171875)
PURPLE_C= rgb(0.60546875, 0.44921875, 0.67578125)
PURPLE_B= rgb(0.6953125, 0.5390625, 0.77734375)
PURPLE_A= rgb(0.79296875, 0.640625, 0.91015625)
WHITE= rgb(1.0, 1.0, 1.0)
BLACK= rgb(0.00390625, 0.00390625, 0.00390625)
LIGHT_GRAY= rgb(0.734375, 0.734375, 0.734375)
LIGHT_GREY= rgb(0.734375, 0.734375, 0.734375)
GRAY= rgb(0.53515625, 0.53515625, 0.53515625)
GREY= rgb(0.53515625, 0.53515625, 0.53515625)
DARK_GREY= rgb(0.26953125, 0.26953125, 0.26953125)
DARK_GRAY= rgb(0.26953125, 0.26953125, 0.26953125)
DARKER_GREY= rgb(0.13671875, 0.13671875, 0.13671875)
DARKER_GRAY= rgb(0.13671875, 0.13671875, 0.13671875)
GREY_BROWN= rgb(0.453125, 0.390625, 0.34375)
PINK= rgb(0.8203125, 0.28125, 0.7421875)
LIGHT_PINK= rgb(0.86328125, 0.4609375, 0.8046875)
GREEN_SCREEN= rgb(0.00390625, 1.0, 0.00390625)
ORANGE= rgb(1.0, 0.52734375, 0.1875)