import numpy as np
from manimlib import PI, VGroup, ShowCreation, Scene, NumberPlane, Tex
from manimlib import BLUE, RED, GREEN, WHITE, LEFT, RIGHT, UP
from compressed_sensing.geometry import Ellipse, Point, get_l2_ball_with_tangent, get_ellipse_scalled_tangent_to_l1_ball
from compressed_sensing.geometry.rectangle import Rectangle


class Solution(Scene):
    def construct(self):
        plane = NumberPlane()
        self.add(plane)

        # Create an ellipse
        axes = Point(x=1, y=0.5)
        center = Point(x=2, y=2)
        angle = -np.rad2deg(PI/3)
        l = 1
        l2_balls = []
        base_ellipse = Ellipse(axes=axes, center=center, angle=angle)
        tangent_ellipse = get_ellipse_scalled_tangent_to_l1_ball(ellipse=base_ellipse, l=l, color=WHITE)
        _, _, pt_tangency = get_l2_ball_with_tangent(ellipse=tangent_ellipse, color_ellipse=RED, color_tangent=GREEN)
        a, b = tangent_ellipse.axes.x, tangent_ellipse.axes.y
        l2_balls = [Ellipse(axes=Point(x=s*a, y=s*b), center=center, angle=angle).to_manim(color=RED) for s in (0.5, 0.75, 1)]
        l2_balls = VGroup(*l2_balls)
        l2_label = Tex(r"\lVert \mathbf{y} - \mathbf{\Theta} \mathbf{s} \rVert_2^2", font_size=40).set_color(RED).shift(4.5*RIGHT+2.5*UP)
        l2_group = VGroup(l2_balls, l2_label)
        l1_ball = Rectangle(center=Point(x=0, y=0), axes=Point(x=np.sqrt(2*l), y=np.sqrt(2*l)), angle=45).to_manim(color=BLUE)
        l1_label = Tex(r"\lVert \mathbf{s} \rVert_1 \leq \beta", font_size=40).set_color(BLUE).shift(1.2*LEFT+0.8*UP)
        l1_group = VGroup(l1_ball, l1_label)

        # Add the axes and the ellipse to the scene
        self.play(ShowCreation(l2_group), ShowCreation(l1_group), ShowCreation(pt_tangency))
        base = Ellipse(axes=Point(x=1, y=0.5), center=Point(x=1, y=1), angle=-np.rad2deg(PI/4))
        self.wait()