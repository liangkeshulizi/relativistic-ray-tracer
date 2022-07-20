from my_raytracer import *

shape_sphere= Sphere(.5, get_checkerboard_color_func(GREEN_C, YELLOW_C))
shape_plane= Plane(vec3(0,0,0), vec3(0,1,0), diffuse_color_function= get_cubical_checkerboard_color_func(GREY, WHITE, .15))

object1= MovingObject( shape_sphere, (0.5, 0, 0), vec4(0, 0, 0, 1.5) )
object2= MovingObject( shape_plane, (0, 0, 0),    vec4(0, 0, -.5, 0) )

movingobjects= [object1, object2]
scene= Scene(movingobjects, light_pos= ORIGIN.vec3(),
    compositors=[get_my_compositor("黄色绿色球体，从左向右v=0.5c，摄像机为白色点光源", 0.03, fill= WHITE._to_standard_color()),]
    )

scene.set_render_properties(
    t_start= -1,
    t_end= 5,
    duration= 3,
)

if __name__ == "__main__":
    scene.render(0)