from re import S, T
from turtle import color
import numpy as np
from manimlib import VGroup, ShowCreation, Scene, NumberPlane, Tex
from manimlib import BLUE, RED, GREEN, DOWN, LEFT, RIGHT, UP
from compressed_sensing.geometry import Ellipse, Point
from compressed_sensing.geometry.line_segment import LineSegment
from compressed_sensing.geometry.utils import get_l2_ball_from_set_tangent_to_l2_ball
from compressed_sensing.geometry.rectangle import Rectangle


class Tangents(Scene):
    def construct(self):
        plane = NumberPlane()
        self.add(plane)

        # Create an ellipse
        m = 1
        ellipse = Ellipse(axes=Point(x=2, y=1), center=Point(x=0, y=0), angle=0)
        ((_, pt_tangency1), (_, pt_tangency2)) = ellipse.tangent(slope=m)
        pts_tangency = VGroup(pt_tangency1.to_manim(color=GREEN), pt_tangency2.to_manim(color=GREEN))
        tangent1 = LineSegment.from_pt_length_and_slope(pt=pt_tangency1, length=20, slope=m)
        tangent2 = LineSegment.from_pt_length_and_slope(pt=pt_tangency2, length=20, slope=m)
        tangents = VGroup(tangent1.to_manim(color=RED), tangent2.to_manim(color=RED))
        labels = VGroup(
            Tex(r" \frac{x^2}{a^2} + \frac{y^2}{b^2} = 1", font_size=40).set_color(BLUE).shift(1.2*RIGHT+1.5*UP),
            Tex(r" \left( -\frac{a^2 m^2}{\sqrt{a^2 m^2 + b^2}}, \frac{b^2}{\sqrt{a^2 m^2 + b^2}} \right)", font_size=25).set_color(GREEN).shift(3.25*LEFT+UP),
            Tex(r" \left( \frac{a^2 m^2}{\sqrt{a^2 m^2 + b^2}}, -\frac{b^2}{\sqrt{a^2 m^2 + b^2}} \right)", font_size=25).set_color(GREEN).shift(3.25*RIGHT+DOWN),
            Tex(r" y = mx + \sqrt{a^2 m^2 + b^2}", font_size=40).set_color(RED).shift(2.5*LEFT+2*UP),
            Tex(r" y = mx - \sqrt{a^2 m^2 + b^2}", font_size=40).set_color(RED).shift(2.8*RIGHT+2*DOWN),
        )

        
        # Add the axes and the ellipse to the scene
        self.play(ShowCreation(ellipse.to_manim(color=BLUE)), ShowCreation(pts_tangency), ShowCreation(tangents), ShowCreation(labels))
        self.wait()