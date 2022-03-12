from my_raytracer import *

if __name__ == '__main__':
    (width, height) = (1920, 1080)      # 屏幕尺寸
    resolution= height/width
    light_pos = vec3(-2, 2, -2)          # 点光源位置
    center = vec4(0, 0, 0, 0)         # 摄像机位置
    focal_length= 200

    shape4 = RectangularPrism(1, 1, 1, .04, diffuse_color_function= lambda p: BILIBILIBLUE_A)
    beta = (0, 0, 0.1)
    movingobjects= [MovingObject(shape4, beta, vec4(0, 0, 0, 0))]#[MovingObject(shape2, beta, offset2, Material(500)), MovingObject(shape3, (0,0,0), offset3, None)]

    # Screen coordinates: x0, y0, x1, y1
    camera_height= 200
    camera_width= camera_height/resolution
    S = (-camera_width, camera_height, camera_width, -camera_height)
    x= np.tile(np.linspace(S[0], S[2], width), height)
    y= np.repeat(np.linspace(S[1], S[3], height), width)
    z= focal_length
    origin_to_image_times= abs(vec4(0,x,y,z))

    t_start= time.time()
    frames= 250
    for shot_time, frame_count in np.linspace([-2,1],[16,frames], frames):
        start= center + vec4(shot_time, 0, 0, 0)                # 光线的端点
        directions= vec4(-origin_to_image_times, x, y, z)       # 仅仅表示光线的方向
        print('开始渲染第%s帧' % int(frame_count))
        t0 = time.time()
        color = rgb(0,0,0)*np.repeat(0,width*height) + raytrace(start, directions, movingobjects, light_pos)
        print("  耗时%f，预计剩余%i分钟"%(time.time()-t0, (time.time()-t_start)/frame_count*(frames-frame_count)/60))
        filename= ".\\render1\\%i.png" % frame_count
        file_color = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((height, width))).astype(np.uint8), "L") for c in color.components()]
        Image.merge("RGB", file_color).save(filename)