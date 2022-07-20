from my_raytracer import *

shape0= Cube(1, 1, 1, get_cubical_checkerboard_color_func(BILIBILIPINK, WHITE))
shape1= Sphere(.5, get_checkerboard_color_func(BILIBILIBLUE, WHITE))

shape2= Plane(vec3(0,0,0), vec3(0,1,0), diffuse_color_function= get_cubical_checkerboard_color_func(GREY, WHITE, .3))
beta1 = (0, 0, 0)
beta2 = (0, 0, 0)
offset1 = vec4(0, 0, -1, 2)
offset2 = vec4(0, 0, -1.5, 0)

object1= MovingObject(shape0, beta1, offset1)
object2= MovingObject(shape2, beta2, offset2, Material(False))
object3= MovingObject(shape1, beta2, vec4(0, 0, 0, 2))
movingobjects= [object1, object2, object3]

scene= Scene(movingobjects, light_pos= vec3(0,0,0))

def velocity_updater(scene, t):
    v= t*(0.99)
    scene.movingobjects[0].set_beta((v, 0, 0))
    scene.movingobjects[-1].set_beta((v, 0, 0))

def text_updater(scene, t):
    text= f"v= {round(scene.movingobjects[0].beta[0], 3)}c"
    scene.compositors= [get_my_compositor(text, 0.05, fill= BILIBILIPINK._to_standard_color()), ]

scene.set_render_properties(t_start= 2, t_end= 2, duration= 10, updaters= [velocity_updater, text_updater])

if __name__ == "__main__":
    scene.render(0)