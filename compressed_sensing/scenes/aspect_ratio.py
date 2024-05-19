import numpy as np
from manimlib import always_redraw, linear, PI, VGroup, ShowCreation, Scene, NumberPlane, Tex, ValueTracker
from manimlib import BLUE, RED, GREEN, YELLOW, LEFT, RIGHT, UP
from compressed_sensing.geometry import Ellipse, Point
from compressed_sensing.geometry.utils import get_set_concentric_l2_balls_with_tangent_and_center
from compressed_sensing.geometry.rectangle import Rectangle



def get_axes(t: float) -> Point:
    if t < 1/6:
        t_s, t_e = 0, 1/6
        a_s, b_s = 1, 0.5 
        a_e, b_e = 1, 2
    elif t < 2/6:
        t_s, t_e = 1/6, 2/6
        a_s, b_s = 1, 2
        a_e, b_e = 1, 0.1
    elif t < 3/6:
        t_s, t_e = 2/6, 3/6
        a_s, b_s = 1, 0.1
        a_e, b_e = 1, 0.5
    elif t < 4/6:
        t_s, t_e = 3/6, 4/6
        a_s, b_s = 1, 0.5
        a_e, b_e = 3, 0.5
    elif t < 5/6:
        t_s, t_e = 4/6, 5/6
        a_s, b_s = 3, 0.5
        a_e, b_e = 0.1, 0.5
    else:
        t_s, t_e = 5/6, 1
        a_s, b_s = 0.1, 0.5
        a_e, b_e = 1, 0.5  
    pt = Point(x=a_s + (a_e - a_s) * (t - t_s) / (t_e - t_s), y=b_s + (b_e - b_s) * (t - t_s) / (t_e - t_s))
    print(pt)
    return pt

class AspectRatio(Scene):
    def construct(self):
        plane = NumberPlane()
        self.add(plane)

        # Create an ellipse
        t_start, t_end = 0, 1
        t = ValueTracker(t_start)
        center = Point(x=2, y=2)
        angle = -np.rad2deg(PI/3)

        l = 1
        l1_ball = Rectangle(center=Point(x=0, y=0), axes=Point(x=np.sqrt(2*l), y=np.sqrt(2*l)), angle=45).to_manim(color=BLUE)
        l1_label = Tex(r"\lVert \mathbf{s} \rVert_1 \leq \beta", font_size=40).set_color(BLUE).shift(1.2*LEFT+0.8*UP)
        l1_group = VGroup(l1_ball, l1_label)

        l2_group = always_redraw(
            lambda: get_set_concentric_l2_balls_with_tangent_and_center(
                ellipse=Ellipse(axes=get_axes(t=t.get_value()), center=center, angle=angle), 
                l=l, 
                color_concentric=RED, 
                color_tangent=GREEN, 
                color_center=YELLOW,
            )
        )
        l2_label = Tex(r"\lVert \mathbf{y} - \mathbf{\Theta} \mathbf{s} \rVert_2^2", font_size=40).set_color(RED).shift(4.5*RIGHT+2.5*UP)

        self.play(ShowCreation(l1_group), ShowCreation(l2_label))
        self.wait()
        self.add(l2_group)
        self.play(
            t.animate.set_value(t_end),
            run_time=20, 
            rate_func=linear,
        )
        self.wait()