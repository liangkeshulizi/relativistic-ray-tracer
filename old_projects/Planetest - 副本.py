from my_raytracer import *

def SqueezeMovingobjects(front_shape, bg_shape, beta, offset, depth_in_z, layers):
    return [MovingObject(front_shape, beta, offset, None)] + [MovingObject(bg_shape, beta, offset + vec4(0,0,0,1)*dz, None) for dz in np.arange(depth_in_z/(layers-1), depth_in_z, depth_in_z/(layers-1))]

if __name__ == '__main__':
    (width, height) = (1920, 1080)      # 屏幕尺寸
    resolution= height/width
    light_pos = vec3(0, 2, -2)          # 点光源位置
    center = vec4(0, 0, 0, 0)         # 摄像机位置
    focal_length= 200

    shape = Plane(vec3(0,0,1), vec3(0,0,-1), range_func_from_image(r"E:\视频\特勒尔效应\素材\9HF~}9D2P@0Y(GNTSR$E1O4\hh_00000.jpg", (2,2)), lambda p: WHITE)

    movingobjects= [MovingObject(shape, (0.9,0,0), vec4(0,0,0,0), None)]

    # Screen coordinates: x0, y0, x1, y1
    camera_height= 200
    camera_width= camera_height/resolution
    S = (-camera_width, camera_height, camera_width, -camera_height)
    x= np.tile(np.linspace(S[0], S[2], width), height)
    y= np.repeat(np.linspace(S[1], S[3], height), width)
    z= focal_length
    origin_to_image_times= abs(vec4(0,x,y,z))

    t_start= time.time()
    frames= 1
    for shot_time, frame_count in np.linspace([1,1],[2,frames], frames):
        start= center + vec4(shot_time, 0, 0, 0)                # 光线的端点
        directions= vec4(-origin_to_image_times, x, y, z)       # 仅仅表示光线的方向
        print('开始渲染第%s帧' % int(frame_count))
        t0 = time.time()
        color = rgb(0,0,0) * np.repeat(0,width*height) + raytrace(start, directions, movingobjects, light_pos)
        print("  耗时%f，预计剩余%i分钟"%(time.time()-t0, (time.time()-t_start)/frame_count*(frames-frame_count)/60))
        filename= ".\\render9\\%i.png" % frame_count
        file_color = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((height, width))).astype(np.uint8), "L") for c in color.components()]
        Image.merge("RGB", file_color).save(filename)