from my_raytracer import *

shape1= Sphere(.5, get_checkerboard_color_func(WHITE, GRAY))
#shape1= Cube(1,1,1, get_cubical_checkerboard_color_func(RED_E, WHITE, .1))
shape2= Plane(vec3(0,0,0), vec3(0,1,0), diffuse_color_function= get_cubical_checkerboard_color_func(GREY, WHITE, .15))
beta1 = (0.5, 0, 0)
beta2 = (0, 0, 0)
offset1 = vec4(0, 0, 0, 1.5)
offset2 = vec4(0, 0, -.5, 0)

object1= MovingObject(shape1, beta1, offset1)
object2= MovingObject(shape2, beta2, offset2)
movingobjects= [object1, object2]

scene= Scene(movingobjects)
scene.generate_animation(-4, 9, 100, 'render2', PR)

#frames= 10
#t00= time.time()
#for u, frame_count in np.linspace([0,1],[0.5,frames], frames):
#    print('开始渲染第%s帧...' % int(frame_count), end= '')
#    object1.set_beta((0, 0, u))
#    scene.generate_image(1, os.path.join('render', f"{int(frame_count)}.png"))
#    t1= time.time()
#    print("预计剩余%i分钟" % ((t1-t00)/frame_count*(frames-frame_count)/60))
print('渲染完成')
