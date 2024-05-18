import numpy as np
from manimlib import Line as LineM
from manimlib import PI, VGroup, ShowCreation, FadeOut, Scene, NumberPlane, ValueTracker
from manimlib import always_redraw, linear, BLUE, RED, GREEN, ORANGE, LEFT, RIGHT, UP, DOWN
from compressed_sensing.geometry import Ellipse, Point, get_l2_ball_with_tangent

class Tangency(Scene):
    def construct(self):
        plane = NumberPlane()
        self.add(plane)

        # Create an ellipse
        s_start, s_end = 0.5, 2
        s = ValueTracker(s_start)
        a, b = 1, 1.5
        ct = Point(x=3, y=2)
        angle = 30
        l2_group = always_redraw(
            lambda: VGroup(*get_l2_ball_with_tangent(
                ellipse=Ellipse(axes=Point(x=s.get_value()*a, y=s.get_value()*b), center=ct, angle=angle), 
                color_ellipse=RED, 
                color_tangent=GREEN
            ))
        )

        ct_3d = np.array([ct.x, ct.y, 0])
        x_mark = VGroup(
            LineM(ct_3d + LEFT/8 + UP/8, ct_3d + RIGHT/8 + DOWN/8, color=ORANGE),
            LineM(ct_3d + LEFT/8 + DOWN/8, ct_3d + RIGHT/8 + UP/8, color=ORANGE),
        )


        start_ellipse = Ellipse(axes=Point(x=s_start*a, y=s_start*b), center=ct, angle=angle)
        end_ellipse = Ellipse(axes=Point(x=s_end*a, y=s_end*b), center=ct, angle=angle)
        start_l2_ball = start_ellipse.to_manim(color=RED)
        _, _, start = get_l2_ball_with_tangent(ellipse=start_ellipse, color_tangent=BLUE)
        _, _, end = get_l2_ball_with_tangent(ellipse=end_ellipse)
        secant = LineM(start=start.get_center(), end=end.get_center(), color=BLUE)

        self.play(ShowCreation(x_mark), ShowCreation(start_l2_ball), ShowCreation(start))
        self.wait()
        self.add(l2_group)
        self.play(FadeOut(start_l2_ball), run_time=0.1)
        self.play(
            s.animate.set_value(s_end),
            ShowCreation(secant),
            run_time=3, 
            rate_func=linear,
        )
        self.wait()