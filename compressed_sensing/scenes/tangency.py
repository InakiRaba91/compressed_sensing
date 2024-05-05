import numpy as np
from manimlib import Ellipse as EllipseM, Dot as DotM, DashedLine as DashedLineM, Rectangle as RectangleM
from manimlib import PI, VGroup, ShowCreation, Scene, NumberPlane, BLUE, RED, GREEN, YELLOW
from compressed_sensing.geometry import Ellipse, Line, Point

def get_l2_ball(ellipse: Ellipse, color: str) -> EllipseM:
    a, b = ellipse.axes.x, ellipse.axes.y
    ct = np.array([ellipse.center.x, ellipse.center.y, 0])
    return EllipseM(width=2*a, height=2*b, color=color).rotate(np.deg2rad(-ellipse.angle)).shift(ct)

def get_centered_segment(line: Line, ct: Point, length: float, color: str) -> DashedLineM:
    ct_3d = np.array([ct.x, ct.y, 0])
    m = - line.a / line.b
    delta_x = length / np.sqrt(1 + m ** 2)
    delta_y = m * delta_x
    start = ct_3d - np.array([delta_x, delta_y, 0])
    end = ct_3d + np.array([delta_x, delta_y, 0])
    return DashedLineM(start=start, end=end, color=color)

def get_l2_ball_with_tangent(ellipse: Ellipse, color_ellipse: str, color_tangent: str) -> VGroup:
    l2_ball = get_l2_ball(ellipse=ellipse, color=color_ellipse)
    if (ellipse.center.y == 0) or (ellipse.center.x == 0):
        raise ValueError("The ellipse is in an axis, the tangent is not defined.")
    else:
        slope = - np.sign(ellipse.center.y / ellipse.center.x)
        (line1, pt1), (line2, pt2) = ellipse.tangent(slope=slope)
        # grab the one closest to the origin
        line, pt = (line1, pt1) if pt1.length() < pt2.length() else (line2, pt2)
        tangent = get_centered_segment(line=line, ct=pt, length=1, color=color_tangent)
        pt_tangency = DotM(np.array([pt.x, pt.y, 0]), color=color_tangent)
        return VGroup(l2_ball, tangent, pt_tangency)

def get_l1_ball(l: float, color: str):
    return RectangleM(width=l, height=l, color=color).rotate(PI/4)

class Tangency(Scene):
    def construct(self):
        plane = NumberPlane()
        self.add(plane)

        # Create an ellipse
        a, b = 1, 0.5
        cts = [
            Point(x=1, y=1),
            Point(x=-2, y=2),
            Point(x=-2, y=-2),
            Point(x=2, y=-2),
        ]
        angle = 15
        l = 1
        l2_balls_and_tangents = []
        for s in np.linspace(1, 2, 1):
            for ct in cts:
                ellipse = Ellipse(axes=Point(x=s*a, y=s*b), center=ct, angle=angle)
                l2_balls_and_tangents.append(get_l2_ball_with_tangent(ellipse=ellipse, color_ellipse=RED, color_tangent=GREEN))
        l2_balls_and_tangents = VGroup(*l2_balls_and_tangents)
        l1_ball = get_l1_ball(l=np.sqrt(2*l), color=BLUE)

        # Add the axes and the ellipse to the scene
        self.play(ShowCreation(l2_balls_and_tangents), ShowCreation(l1_ball))
        self.wait()