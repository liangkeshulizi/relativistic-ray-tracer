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