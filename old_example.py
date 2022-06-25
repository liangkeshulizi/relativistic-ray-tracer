import os
from my_raytracer import *

if __name__ == '__main__':
    # 基础设置
    (width, height) = (1920, 1080)      # 屏幕尺寸
    resolution= height/width
    light_pos = vec3(2, 2, -2)          # 点光源位置
    center = vec4(0, 0, 0, 0)         # 摄像机位置
    focal_length= 200
    shot_time= 0

    # 搭建场景
    shape= Sphere(.5, get_checkerboard_color_func(BILIBILIBLUE, WHITE))
    beta = (0, 0, 0)
    offset = vec4(0, 0, 0, 2)
    movingobjects= [MovingObject(shape, beta, offset)]

    # 创建屏幕光线
    camera_height= 200
    camera_width= camera_height/resolution
    S = (-camera_width, camera_height, camera_width, -camera_height)
    x= np.tile(np.linspace(S[0], S[2], width), height)     # [1,2,3,4,1,2,3,4,1,2,3,4]
    y= np.repeat(np.linspace(S[1], S[3], height), width)   # [1,1,1,1,2,2,2,2,3,3,3,3]
    z= focal_length
    origin_to_image_times= abs(vec4(0,x,y,z))
    start= center + vec4(shot_time, 0, 0, 0)# 光线的端点
    directions= vec4(-origin_to_image_times, x, y, z) # 仅仅表示光线的方向

    # 渲染和保存
    t0 = time.time()
    color= raytrace(start, directions, movingobjects, light_pos)
    print(f"耗时{time.time()-t0}")
    filename= ".\\image.png"
    file_color = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((height, width))).astype(np.uint8), "L") for c in color.components()]
    Image.merge("RGB", file_color).save(filename)
    os.startfile(filename)
    input('Enter to exit...')