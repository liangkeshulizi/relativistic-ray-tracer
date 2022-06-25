from my_raytracer import *

shape_sphere= Sphere(.5, get_checkerboard_color_func(WHITE, GRAY))
shape_plane= Plane(vec3(0,0,0), vec3(0,1,0), diffuse_color_function= get_cubical_checkerboard_color_func(GREY, WHITE, .15))

object1= MovingObject( shape_sphere, (0.5, 0, 0), vec4(0, 0, 0, 1.5) )
object2= MovingObject( shape_plane, (0, 0, 0),    vec4(0, 0, -.5, 0) )

movingobjects= [object1, object2]
scene= Scene(movingobjects)

if __name__ == "__main__":
    scene.generate_animation(-1, 7, 10, camera= PR)
