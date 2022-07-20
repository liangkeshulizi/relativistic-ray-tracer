from my_raytracer import *

shape0= Cube(1, 1, 1, get_cubical_checkerboard_color_func(BILIBILIPINK, WHITE))
shape1= Sphere(.5, get_checkerboard_color_func(BILIBILIBLUE, WHITE))
shape2= Plane(vec3(0,0,0), vec3(0,1,0), diffuse_color_function= get_cubical_checkerboard_color_func(GREY, WHITE, .3))
beta = (0, 0, 0.2)

object1= MovingObject(shape0, beta, vec4(0, -1, -1, 2))
object2= MovingObject(shape2, beta, vec4(0, 0, -1.5, 0), Material(False))
object3= MovingObject(shape1, beta, vec4(0, -1, 0, 2))
movingobjects= [object1, object2, object3]

def velocity_updater(scene, t):
    v= t*(-0.5)+(1-t)*0.5
    scene.movingobjects[0].set_beta((0, 0, v))
    scene.movingobjects[1].set_beta((0, 0, v))
    scene.movingobjects[2].set_beta((0, 0, v))

def text_updater(scene, t):
    text= f"v= {round(scene.movingobjects[0].beta[2], 3)}c"
    scene.compositors= [get_my_compositor(text, 0.05, fill= BILIBILIPINK._to_standard_color()), ]

scene_speed_change= Scene(movingobjects, light_pos= ORIGIN.vec3())
scene_speed_change.set_render_properties(t_start= 0, t_end= 0, duration= 10, updaters= ([velocity_updater, text_updater]))

####
scene_backward= Scene(movingobjects, light_pos= ORIGIN.vec3())
scene_backward.set_render_properties(t_start= -10, t_end= 10, duration= 10)

###
beta2= (0, 0, -0.2)
object4= MovingObject(shape0, beta2, vec4(0, -1, -1, 2))
object5= MovingObject(shape2, beta2, vec4(0, 0, -1.5, 0), Material(False))
object6= MovingObject(shape1, beta2, vec4(0, -1, 0, 2))
movingobjects= [object4, object5, object6]
scene_foreward= Scene(movingobjects, light_pos= ORIGIN.vec3())
scene_foreward.set_render_properties(t_start= -10, t_end= 10, duration= 10)


if __name__ == "__main__":
    scene_backward.render(1)
    scene_foreward.render(1)