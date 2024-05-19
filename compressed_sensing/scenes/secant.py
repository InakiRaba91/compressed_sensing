from re import S, T
from turtle import color
import numpy as np
from manimlib import VGroup, ShowCreation, Scene, NumberPlane, Tex
from manimlib import BLUE, RED, GREEN, ORANGE, DOWN, LEFT, RIGHT, UP
from compressed_sensing.geometry import Ellipse, Point
from compressed_sensing.geometry.line_segment import LineSegment
from compressed_sensing.geometry.utils import get_l2_ball_from_set_tangent_to_l2_ball
from compressed_sensing.geometry.rectangle import Rectangle


class Secant(Scene):
    def construct(self):
        plane = NumberPlane()
        self.add(plane)

        # Create an ellipse
        m = 1
        a, b = 2, 1
        ellipses, pts_tangency, tangents = [], [], []
        for s in (0.5, 1, 1.5):
            ellipse = Ellipse(axes=Point(x=a*s, y=b*s), center=Point(x=0, y=0), angle=0)
            ellipses.append(ellipse.to_manim(color=BLUE))
            ((_, pt_tangency1), (_, pt_tangency2)) = ellipse.tangent(slope=m)
            pts_tangency += [pt_tangency1.to_manim(color=GREEN), pt_tangency2.to_manim(color=GREEN)]
            tangent1 = LineSegment.from_pt_length_and_slope(pt=pt_tangency1, length=20, slope=m)
            tangent2 = LineSegment.from_pt_length_and_slope(pt=pt_tangency2, length=20, slope=m)
            tangents += [tangent1.to_manim(color=RED), tangent2.to_manim(color=RED)]
        ellipses = VGroup(*ellipses)
        pts_tangency = VGroup(*pts_tangency)
        tangents = VGroup(*tangents)
        secant = tangent2 = LineSegment.from_pt_length_and_slope(pt=Point(x=0, y=0), length=20, slope=-(b/(a*m))**2).to_manim(color=ORANGE)
        label = VGroup(
            Tex(r" y = \frac{b^2}{a^2 m^2} x", font_size=40).set_color(ORANGE).shift(3.4*LEFT+1.6*UP),
        )

        
        # Add the axes and the ellipse to the scene
        self.play(ShowCreation(ellipses), ShowCreation(pts_tangency), ShowCreation(tangents), ShowCreation(label), ShowCreation(secant))
        self.wait()