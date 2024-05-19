import numpy as np
from manimlib import always_redraw, linear, PI, VGroup, ShowCreation, Scene, NumberPlane, Tex, ValueTracker
from manimlib import BLUE, RED, GREEN, YELLOW, LEFT, RIGHT, UP
from compressed_sensing.geometry import Ellipse, Point
from compressed_sensing.geometry.utils import get_set_concentric_l2_balls_with_tangent_and_center
from compressed_sensing.geometry.rectangle import Rectangle



def get_angle(t: float) -> Point:
    angle_s = -np.rad2deg(PI/3)
    angle_e = -np.rad2deg(PI/3 + PI)
    return angle_s + (angle_e - angle_s) * t

class Rotation(Scene):
    def construct(self):
        plane = NumberPlane()
        self.add(plane)

        # Create an ellipse
        t_start, t_end = 0, 1
        t = ValueTracker(t_start)
        center = Point(x=2, y=2)
        axes = Point(x=1, y=0.5)

        l = 1
        l1_ball = Rectangle(center=Point(x=0, y=0), axes=Point(x=np.sqrt(2*l), y=np.sqrt(2*l)), angle=45).to_manim(color=BLUE)
        l1_label = Tex(r"\lVert \mathbf{s} \rVert_1 \leq \beta", font_size=40).set_color(BLUE).shift(1.2*LEFT+0.8*UP)
        l1_group = VGroup(l1_ball, l1_label)

        l2_group = always_redraw(
            lambda: get_set_concentric_l2_balls_with_tangent_and_center(
                ellipse=Ellipse(axes=axes, center=center, angle=get_angle(t=t.get_value())), 
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
            run_time=10, 
            rate_func=linear,
        )
        self.wait()