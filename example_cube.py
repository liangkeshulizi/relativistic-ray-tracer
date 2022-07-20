from my_raytracer import *

shape_cube= Cube(1,1,1, get_cubical_checkerboard_color_func(GREEN_C, WHITE))
shape_ground= Plane(vec3(0,0,0), vec3(0,1,0), diffuse_color_function= get_cubical_checkerboard_color_func(GREY, WHITE, .3))

object1= MovingObject( shape_cube, (0.7, 0, 0), vec4(0, 0, 0, 1.5) )
object2= MovingObject( shape_ground, (0, 0, 0), vec4(0, 0, -.5, 0) )

movingobjects= [object1, object2]
scene= Scene(movingobjects, compositors= [get_my_compositor("v= 0.7c", 0.05, fill= YELLOW_C._to_standard_color()), ])

scene.set_render_properties(t_start= -1.5, t_end= 6, duration= 10, camera= None)

if __name__ == "__main__":
    scene.render(0)