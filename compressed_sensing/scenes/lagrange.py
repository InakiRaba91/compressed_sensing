from re import S
from turtle import color
import numpy as np
from manimlib import PI, VGroup, ShowCreation, Scene, NumberPlane, Tex, Vector
from manimlib import BLUE, RED, GREEN, WHITE, LEFT, RIGHT, UP
from compressed_sensing.geometry import Ellipse, Point
from compressed_sensing.geometry.utils import get_l2_ball_from_set_tangent_to_l1_ball, get_l2_ball_from_set_tangent_to_l2_ball
from compressed_sensing.geometry.rectangle import Rectangle


class Lagrange(Scene):
    def construct(self):
        plane = NumberPlane()
        self.add(plane)

        # Create an ellipse
        ellipse_constraint = Ellipse(axes=Point(x=1, y=1), center=Point(x=0, y=0), angle=-np.rad2deg(PI/3)).to_manim(color=BLUE, fill_color=BLUE, fill_opacity=0.2)
        label_constraint = Tex(r" \mathbf{f} ( \mathbf{x} ) \leq \beta", font_size=40).set_color(BLUE).shift(1.6*LEFT+UP)
        base_ellipse = Ellipse(axes=Point(x=1.5, y=1), center=Point(x=1.5, y=1.5), angle=-np.rad2deg(PI/3))
        tangent_ellipse, pt_tangency = get_l2_ball_from_set_tangent_to_l2_ball(ellipse=base_ellipse, l=1, color_ellipse=RED, color_tangent=GREEN)
        a, b = tangent_ellipse.width / 2, tangent_ellipse.height / 2
        ellipses_objective = [Ellipse(axes=Point(x=s*a, y=s*b), center=base_ellipse.center, angle=base_ellipse.angle).to_manim(color=RED) for s in (0.5, 1, 1.5)]
        labels_objective = [
            Tex(r" \mathbf{g} ( \mathbf{x} ) = \alpha_1", font_size=25).set_color(RED).shift(2.1*RIGHT+3.3*UP),
            Tex(r" \mathbf{g} ( \mathbf{x} ) = \alpha_2", font_size=25).set_color(RED).shift(1.9*RIGHT+2.75*UP),
            Tex(r" \mathbf{g} ( \mathbf{x} ) = \alpha_3", font_size=25).set_color(RED).shift(1.7*RIGHT+2.2*UP),
            Tex(r" \mathbf{g} ( \mathbf{x} )", font_size=40).set_color(RED).shift(3*RIGHT+0.5*UP),
        ]
        ellipses = VGroup(*[ellipse_constraint, *ellipses_objective])
        x, y = pt_tangency.get_center()[0], pt_tangency.get_center()[1]
        gradients = VGroup(
            Vector([x, y], color=BLUE).shift(x*RIGHT+y*UP),
            Vector([-x*0.75, -y*0.75], color=RED).shift(x*RIGHT+y*UP),
        )
        labels = VGroup(*[label_constraint, *labels_objective])

        
        # Add the axes and the ellipse to the scene
        self.play(ShowCreation(ellipses), ShowCreation(labels), ShowCreation(gradients), ShowCreation(pt_tangency))
        self.wait()