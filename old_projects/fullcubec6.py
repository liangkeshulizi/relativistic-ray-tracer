from my_raytracer import *

(width, height) = (1920, 1080)      # 屏幕尺寸
resolution= height/width
light_pos = vec3(-1, 2, -2)          # 点光源位置
center = vec4(0, 0, 0, 0)         # 摄像机位置
focal_length= 200

shape_ground = Plane(vec3(0, -0.5, 0), vec3(0, 1, 0), diffuse_color_function=lambda p:WHITE)
shape_cube= Cube(1, 1, 1, diffuse_color_function= get_cubical_checkerboard_color_func(BILIBILIBLUE, WHITE))
shape_ball= Sphere(.5)
shape_prism= RectangularPrism(1, 1, 1, 0.02, diffuse_color_function= lambda p: BILIBILIBLUE)
movingobjects= [
                
                #MovingObject(shape_prism, (0, 0, 0), vec4(0, 0, 0, 2)),
                #MovingObject(shape_ball, (0,0,0), vec4(0,-1,0,2)),
                MovingObject(shape_cube, (0.6,0,0), vec4(0,0,0,1.5), Material(False)),
                MovingObject(shape_ground, (0,0,0), vec4(0,0,0,0), Material(False))
                ]

camera_height= 200
camera_width= camera_height/resolution
S = (-camera_width, camera_height, camera_width, -camera_height)
x= np.tile(np.linspace(S[0], S[2], width), height)
y= np.repeat(np.linspace(S[1], S[3], height), width)
z= focal_length
origin_to_image_times= vec3(x,y,z).norm()

t_start= time.time()
frames= 250
for shot_time, frame_count in np.linspace([-2.5,1],[6,frames], frames):
    start= center + vec4(shot_time, 0, 0, 0)# 光线的端点
    directions= vec4(-origin_to_image_times, x, y, z) # 仅仅表示光线的方向
    print('开始渲染第%s帧' % int(frame_count))
    t0 = time.time()
    color = rgb(0,0,0)*np.repeat(0,width*height) + raytrace(start, directions, movingobjects, light_pos)
    print("  耗时%f，预计剩余%i分钟"%(time.time()-t0, (time.time()-t_start)/frame_count*(frames-frame_count)/60))
    filename= ".\\render9\\%i.png" % frame_count
    file_color = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((height, width))).astype(np.uint8), "L") for c in color.components()]
    Image.merge("RGB", file_color).save(filename)

input('渲染完成')