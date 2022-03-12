from my_raytracer import *

(width, height) = (1920, 1080)      # 屏幕尺寸
resolution= height/width
light_pos = vec3(2, 2, -2)          # 点光源位置
center = vec4(0, 0, 0, -1)         # 摄像机位置
focal_length= 200

shape_ground = Plane(vec3(0, -1, 0), vec3(0, -1, 0), diffuse_color_function=lambda p:WHITE)
shape_cube= Cube(1, 1, 1, diffuse_color_function= get_cubical_checkerboard_color_func(BILIBILIBLUE, WHITE))
shape_ball= Sphere(1, diffuse_color_function= get_checkerboard_color_func(BILIBILIBLUE, WHITE))

camera_height= 200
camera_width= camera_height/resolution
S = (-camera_width, camera_height, camera_width, -camera_height)
x= np.tile(np.linspace(S[0], S[2], width), height)
y= np.repeat(np.linspace(S[1], S[3], height), width)
z= focal_length
origin_to_image_times= vec3(x,y,z).norm()

shot_time= 1

groud= MovingObject(shape_ground, (0,0,0), vec4(0,0,0,0), Material(False))

t_start= time.time()
frames= 250
for beta, frame_count in np.linspace([0.01,1],[0.99,frames], frames):
    movingobjects= [
                MovingObject(shape_ball, (beta,0,0), vec4(0,0,0,1.5), Material(200)),
                groud
                ]
    start= center + vec4(shot_time, 0, 0, 0)# 光线的端点
    directions= vec4(-origin_to_image_times, x, y, z) # 仅仅表示光线的方向
    print('开始渲染第%s帧' % int(frame_count))
    t0 = time.time()
    color = rgb(0,0,0)*np.repeat(0,width*height) + raytrace(start, directions, movingobjects, light_pos)
    print("  耗时%f，预计剩余%i分钟"%(time.time()-t0, (time.time()-t_start)/frame_count*(frames-frame_count)/60))
    filename= ".\\render1\\%i.png" % frame_count
    file_color = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((height, width))).astype(np.uint8), "L") for c in color.components()]
    Image.merge("RGB", file_color).save(filename)

input('渲染完成')