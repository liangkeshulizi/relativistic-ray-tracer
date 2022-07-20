from my_raytracer import *

shape_cube= Cube(1, 1, 1, get_cubical_checkerboard_color_func(YELLOW_C, GREEN_C))
shape_plane= Plane(vec3(0,0,0), vec3(0,1,0), diffuse_color_function= get_cubical_checkerboard_color_func(GREY, WHITE, .15))

object1= MovingObject( shape_cube, (0, 0, 0), vec4(0, 0, 0, 2) )
object2= MovingObject( shape_plane, (0, 0, 0),    vec4(0, 0, -.5, 0) )

movingobjects= [object1, object2]
scene= Scene(movingobjects)

def velocity_updater(scene, t):
    v= t*(-0.99)
    scene.movingobjects[0].set_beta((v, 0, 0))

def text_updater(scene, t):
    text= f"v= {round(scene.movingobjects[0].beta[0], 3)}c"
    scene.compositors= [get_my_compositor(text, 0.05, fill= BILIBILIPINK._to_standard_color()), ]

scene.set_render_properties(t_start= 2, t_end= 2, duration= 10, updaters= [velocity_updater, text_updater])

if __name__ == "__main__":
    scene.render(0)