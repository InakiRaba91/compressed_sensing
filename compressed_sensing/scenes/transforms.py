import numpy as np
from manimlib import always_redraw, linear, PI, VGroup, ShowCreation, FadeOut, Scene, NumberPlane, Tex, ValueTracker
from manimlib import BLUE, RED, GREEN, YELLOW, LEFT, RIGHT, UP
from compressed_sensing.geometry import Ellipse, Point
from compressed_sensing.geometry.utils import get_set_concentric_l2_balls_with_tangent_and_center
from compressed_sensing.geometry.rectangle import Rectangle



def first_shift_get_center(t: float) -> Point:
    """Divide in six parts the interval [0, 1]"""

    if t < 1/4:
        t_s, t_e = 0, 1/4
        x_s, y_s = 2, 2
        x_e, y_e = 1, 1
    elif t < 2/4:
        t_s, t_e = 1/4, 2/4
        x_s, y_s = 1, 1
        x_e, y_e = 3, 3
    elif t < 3/4:
        t_s, t_e = 2/4, 3/4
        x_s, y_s = 3, 3
        x_e, y_e = 2, 2
    else:
        t_s, t_e = 3/4, 1
        x_s, y_s = 2, 2
        x_e, y_e = -2, 2
    return Point(x=x_s + (x_e - x_s) * (t - t_s) / (t_e - t_s), y=y_s + (y_e - y_s) * (t - t_s) / (t_e - t_s))

def second_rotation_get_angle(t: float) -> Point:
    angle_s = -np.rad2deg(PI/3)
    angle_e = -np.rad2deg(PI/3 + PI)
    return angle_s + (angle_e - angle_s) * t

def third_shift_get_center(t: float) -> Point:
    x_s, y_s = -2, 2
    x_e, y_e = -2, -2
    return Point(x=x_s + (x_e - x_s) * t, y=y_s + (y_e - y_s) * t)

def fourth_aspect_ratio_get_axes(t: float) -> Point:
    if t < 1/3:
        t_s, t_e = 0, 1/3
        a_s, b_s = 1, 0.5 
        a_e, b_e = 1, 2
    elif t < 2/3:
        t_s, t_e = 1/3, 2/3
        a_s, b_s = 1, 2
        a_e, b_e = 1, 0.1
    else:
        t_s, t_e = 2/3, 1
        a_s, b_s = 1, 0.1
        a_e, b_e = 1, 0.5
    return Point(x=a_s + (a_e - a_s) * (t - t_s) / (t_e - t_s), y=b_s + (b_e - b_s) * (t - t_s) / (t_e - t_s))

def fifth_shift_get_center(t: float) -> Point:
    x_s, y_s = -2, -2
    x_e, y_e = 2, -2
    return Point(x=x_s + (x_e - x_s) * t, y=y_s + (y_e - y_s) * t)

def sixth_aspect_ratio_get_axes(t: float) -> Point:
    if t < 1/3:
        t_s, t_e = 0, 1/3
        a_s, b_s = 1, 0.5 
        a_e, b_e = 3, 0.5
    elif t < 2/3:
        t_s, t_e = 1/3, 2/3
        a_s, b_s = 3, 0.5
        a_e, b_e = 0.1, 0.5
    else:
        t_s, t_e = 2/3, 1
        a_s, b_s = 0.1, 0.5
        a_e, b_e = 1, 0.5
    return Point(x=a_s + (a_e - a_s) * (t - t_s) / (t_e - t_s), y=b_s + (b_e - b_s) * (t - t_s) / (t_e - t_s))

def seventh_shift_get_center(t: float) -> Point:
    x_s, y_s = 2, -2
    x_e, y_e = 2, 2
    return Point(x=x_s + (x_e - x_s) * t, y=y_s + (y_e - y_s) * t)

class Transforms(Scene):
    norm_type = "l2"

    def __init__(self, **kwargs):
        Scene.__init__(self, **kwargs)

    def construct(self):
        plane = NumberPlane()
        self.add(plane)

        # Create an ellipse
        t_start, t_end = 0, 1
        ts = [ValueTracker(t_start) for _ in range(7)]
        axes = Point(x=1, y=0.5)
        angle = -np.rad2deg(PI/3)

        l = 1
        if self.norm_type == "l1":
            sparsity_ball = Rectangle(center=Point(x=0, y=0), axes=Point(x=np.sqrt(2*l), y=np.sqrt(2*l)), angle=45).to_manim(color=BLUE)
            sparsity_label = Tex(r"\lVert \mathbf{s} \rVert_1 \leq \beta", font_size=40).set_color(BLUE).shift(1.2*LEFT+0.8*UP)
        elif self.norm_type == "l2":
            sparsity_ball = Ellipse(center=Point(x=0, y=0), axes=Point(x=l, y=l), angle=0).to_manim(color=BLUE)
            sparsity_label = Tex(r"\lVert \mathbf{s} \rVert_2 \leq \beta", font_size=40).set_color(BLUE).shift(1.6*LEFT+0.8*UP)
        else:
            raise ValueError(f"Unknown norm type: {self.norm_type}. It must be 'l1' or 'l2'.")
        sparsity_group = VGroup(sparsity_ball, sparsity_label)

        l2_groups = [
            always_redraw(
                lambda: get_set_concentric_l2_balls_with_tangent_and_center(
                    ellipse=Ellipse(axes=axes, center=first_shift_get_center(t=ts[0].get_value()), angle=angle), 
                    l=l, 
                    norm_type=self.norm_type,
                    color_concentric=RED, 
                    color_tangent=GREEN, 
                    color_center=YELLOW,
                )
            ),
            always_redraw(
                lambda: get_set_concentric_l2_balls_with_tangent_and_center(
                    ellipse=Ellipse(axes=axes, center=first_shift_get_center(t=t_end), angle=second_rotation_get_angle(t=ts[1].get_value())), 
                    l=l, 
                    norm_type=self.norm_type,
                    color_concentric=RED, 
                    color_tangent=GREEN, 
                    color_center=YELLOW,
                )
            ),
            always_redraw(
                lambda: get_set_concentric_l2_balls_with_tangent_and_center(
                    ellipse=Ellipse(axes=axes, center=third_shift_get_center(t=ts[2].get_value()), angle=angle), 
                    l=l, 
                    norm_type=self.norm_type,
                    color_concentric=RED, 
                    color_tangent=GREEN, 
                    color_center=YELLOW,
                )
            ),
            always_redraw(
                lambda: get_set_concentric_l2_balls_with_tangent_and_center(
                    ellipse=Ellipse(axes=fourth_aspect_ratio_get_axes(t=ts[3].get_value()), center=third_shift_get_center(t=t_end), angle=angle), 
                    l=l, 
                    norm_type=self.norm_type,
                    color_concentric=RED, 
                    color_tangent=GREEN, 
                    color_center=YELLOW,
                )
            ),
            always_redraw(
                lambda: get_set_concentric_l2_balls_with_tangent_and_center(
                    ellipse=Ellipse(axes=axes, center=fifth_shift_get_center(t=ts[4].get_value()), angle=angle), 
                    l=l, 
                    norm_type=self.norm_type,
                    color_concentric=RED, 
                    color_tangent=GREEN, 
                    color_center=YELLOW,
                )
            ),
            always_redraw(
                lambda: get_set_concentric_l2_balls_with_tangent_and_center(
                    ellipse=Ellipse(axes=sixth_aspect_ratio_get_axes(t=ts[5].get_value()), center=fifth_shift_get_center(t=t_end), angle=angle), 
                    l=l, 
                    norm_type=self.norm_type,
                    color_concentric=RED, 
                    color_tangent=GREEN, 
                    color_center=YELLOW,
                )
            ),
            always_redraw(
                lambda: get_set_concentric_l2_balls_with_tangent_and_center(
                    ellipse=Ellipse(axes=axes, center=seventh_shift_get_center(t=ts[6].get_value()), angle=angle), 
                    l=l, 
                    norm_type=self.norm_type,
                    color_concentric=RED, 
                    color_tangent=GREEN, 
                    color_center=YELLOW,
                )
            ),
        ]
        l2_label = Tex(r"\lVert \mathbf{y} - \mathbf{\Theta} \mathbf{s} \rVert_2^2", font_size=40).set_color(RED).shift(4.5*RIGHT+2.5*UP)

        self.play(ShowCreation(sparsity_group), ShowCreation(l2_label))
        self.wait()
        for i, (t, l2_group) in enumerate(zip(ts, l2_groups)):
            self.add(l2_group)
            if i > 0:
                self.play(FadeOut(l2_groups[i-1]), run_time=0.01)
            self.play(
                t.animate.set_value(t_end),
                run_time=7, 
                rate_func=linear,
            )
            self.wait()