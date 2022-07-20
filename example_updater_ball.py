from my_raytracer import *

shape_cube= Sphere(0.5, get_checkerboard_color_func(YELLOW_C, GREEN_C))
shape_plane= Plane(vec3(0,0,0), vec3(0,1,0), diffuse_color_function= get_cubical_checkerboard_color_func(GREY, WHITE, .15))

object1= MovingObject( shape_cube, (0, 0, 0), vec4(0, 0, 0, 1.2) )
object2= MovingObject( shape_plane, (0, 0, 0),    vec4(0, 0, -.5, 0) )

movingobjects= [object1, object2]
scene= Scene(movingobjects, light_pos= ORIGIN.vec3())

def velocity_updater(scene, t):
    v= (1-t)*(-0.99) + t*(0.99)
    scene.movingobjects[0].set_beta((v, 0, 0))

def text_updater(scene, t):
    text= f"从左向右以{round(scene.movingobjects[0].beta[0], 3)}c速度运动的球，恰好经过摄像机后第1秒，摄像机为点光源"
    scene.compositors= [get_my_compositor(text, 0.02, fill= WHITE._to_standard_color()), ]

scene.set_render_properties(t_start= 1, t_end= 1, duration= 10, updaters= [velocity_updater, text_updater])

if __name__ == "__main__":
    scene.render(0)