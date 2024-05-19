import numpy as np
from manimlib import (
    BLUE,
    GREEN,
    LEFT,
    PI,
    RED,
    RIGHT,
    UP,
    YELLOW,
    NumberPlane,
    Scene,
    ShowCreation,
    Tex,
    ValueTracker,
    VGroup,
    always_redraw,
    linear,
)

from compressed_sensing.geometry import Ellipse, Point
from compressed_sensing.geometry.rectangle import Rectangle
from compressed_sensing.geometry.utils import (
    get_set_concentric_l2_balls_with_tangent_and_center,
)


def get_center(t: float) -> Point:
    """Divide in six parts the interval [0, 1]"""

    if t < 1 / 6:
        t_s, t_e = 0, 1 / 6
        x_s, y_s = 2, 2
        x_e, y_e = 1, 1
    elif t < 2 / 6:
        t_s, t_e = 1 / 6, 2 / 6
        x_s, y_s = 1, 1
        x_e, y_e = 2, 2
    elif t < 3 / 6:
        t_s, t_e = 2 / 6, 3 / 6
        x_s, y_s = 2, 2
        x_e, y_e = 2, 0
    elif t < 4 / 6:
        t_s, t_e = 3 / 6, 4 / 6
        x_s, y_s = 2, 0
        x_e, y_e = 2, 2
    elif t < 5 / 6:
        t_s, t_e = 4 / 6, 5 / 6
        x_s, y_s = 2, 2
        x_e, y_e = 0, 2
    else:
        t_s, t_e = 5 / 6, 1
        x_s, y_s = 0, 2
        x_e, y_e = 2, 2
    return Point(x=x_s + (x_e - x_s) * (t - t_s) / (t_e - t_s), y=y_s + (y_e - y_s) * (t - t_s) / (t_e - t_s))


class Shift(Scene):
    norm_type = "l2"

    def __init__(self, **kwargs):
        Scene.__init__(self, **kwargs)

    def construct(self):
        plane = NumberPlane()
        self.add(plane)

        # Create an ellipse
        t_start, t_end = 0, 1
        t = ValueTracker(t_start)
        axes = Point(x=1, y=0.5)
        angle = -np.rad2deg(PI / 3)

        l = 1
        if self.norm_type == "l1":
            sparsity_ball = Rectangle(
                center=Point(x=0, y=0), axes=Point(x=np.sqrt(2 * l), y=np.sqrt(2 * l)), angle=45
            ).to_manim(color=BLUE)
            sparsity_label = (
                Tex(r"\lVert \mathbf{s} \rVert_1 \leq \beta", font_size=40).set_color(BLUE).shift(1.2 * LEFT + 0.8 * UP)
            )
        elif self.norm_type == "l2":
            sparsity_ball = Ellipse(center=Point(x=0, y=0), axes=Point(x=l, y=l), angle=0).to_manim(color=BLUE)
            sparsity_label = (
                Tex(r"\lVert \mathbf{s} \rVert_2 \leq \beta", font_size=40).set_color(BLUE).shift(1.6 * LEFT + 0.8 * UP)
            )
        else:
            raise ValueError(f"Unknown norm type: {self.norm_type}. It must be 'l1' or 'l2'.")
        sparsity_group = VGroup(sparsity_ball, sparsity_label)

        l2_group = always_redraw(
            lambda: get_set_concentric_l2_balls_with_tangent_and_center(
                ellipse=Ellipse(axes=axes, center=get_center(t=t.get_value()), angle=angle),
                l=l,
                norm_type=self.norm_type,
                color_concentric=RED,
                color_tangent=GREEN,
                color_center=YELLOW,
            )
        )
        l2_label = (
            Tex(r"\lVert \mathbf{y} - \mathbf{\Theta} \mathbf{s} \rVert_2^2", font_size=40)
            .set_color(RED)
            .shift(4.5 * RIGHT + 2.5 * UP)
        )

        self.play(ShowCreation(sparsity_group), ShowCreation(l2_label))
        self.wait()
        self.add(l2_group)
        self.play(
            t.animate.set_value(t_end),
            run_time=24,
            rate_func=linear,
        )
        self.wait()
