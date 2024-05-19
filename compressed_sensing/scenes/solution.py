import numpy as np
from manimlib import (
    BLUE,
    GREEN,
    LEFT,
    PI,
    RED,
    RIGHT,
    UP,
    NumberPlane,
    Scene,
    ShowCreation,
    Tex,
    VGroup,
)

from compressed_sensing.geometry import Ellipse, Point
from compressed_sensing.geometry.rectangle import Rectangle
from compressed_sensing.geometry.utils import (
    get_l2_ball_from_set_tangent_to_l1_ball,
    get_l2_ball_from_set_tangent_to_l2_ball,
)


class Solution(Scene):
    norm_type = "l1"

    def __init__(self, **kwargs):
        Scene.__init__(self, **kwargs)

    def construct(self):
        plane = NumberPlane()
        self.add(plane)

        # Create an ellipse
        axes = Point(x=1, y=0.5)
        center = Point(x=2, y=2)
        angle = -np.rad2deg(PI / 3)
        l = 1

        if self.norm_type == "l1":
            sparsity_ball = Rectangle(
                center=Point(x=0, y=0), axes=Point(x=np.sqrt(2 * l), y=np.sqrt(2 * l)), angle=45
            ).to_manim(color=BLUE)
            sparsity_label = (
                Tex(r"\lVert \mathbf{s} \rVert_1 \leq \beta", font_size=40).set_color(BLUE).shift(1.2 * LEFT + 0.8 * UP)
            )
            get_l2_ball_from_set_tangent_to_sparsity_ballcallable = get_l2_ball_from_set_tangent_to_l1_ball
        elif self.norm_type == "l2":
            sparsity_ball = Ellipse(center=Point(x=0, y=0), axes=Point(x=l, y=l), angle=0).to_manim(color=BLUE)
            sparsity_label = (
                Tex(r"\lVert \mathbf{s} \rVert_2 \leq \beta", font_size=40).set_color(BLUE).shift(1.6 * LEFT + 0.8 * UP)
            )
            get_l2_ball_from_set_tangent_to_sparsity_ballcallable = get_l2_ball_from_set_tangent_to_l2_ball
        else:
            raise ValueError(f"Unknown norm type: {self.norm_type}. It must be 'l1' or 'l2'.")

        l2_balls = []
        base_ellipse = Ellipse(axes=axes, center=center, angle=angle)
        tangent_ellipse, pt_tangency = get_l2_ball_from_set_tangent_to_sparsity_ballcallable(
            ellipse=base_ellipse, l=l, color_ellipse=RED, color_tangent=GREEN
        )
        a, b = tangent_ellipse.width / 2, tangent_ellipse.height / 2
        l2_balls = [
            Ellipse(axes=Point(x=s * a, y=s * b), center=center, angle=angle).to_manim(color=RED) for s in (0.5, 0.75, 1)
        ]
        l2_balls = VGroup(*l2_balls)
        l2_label = (
            Tex(r"\lVert \mathbf{y} - \mathbf{\Theta} \mathbf{s} \rVert_2^2", font_size=40)
            .set_color(RED)
            .shift(4.5 * RIGHT + 2.5 * UP)
        )
        l2_group = VGroup(l2_balls, l2_label)

        sparsity_group = VGroup(sparsity_ball, sparsity_label)

        # Add the axes and the ellipse to the scene
        self.play(ShowCreation(l2_group), ShowCreation(sparsity_group), ShowCreation(pt_tangency))
        self.wait()
