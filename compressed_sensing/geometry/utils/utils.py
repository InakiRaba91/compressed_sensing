from typing import Tuple
import numpy as np
from manimlib import VGroup, Ellipse as EllipseM, Dot as DotM, Line as LineM, DashedLine as DashedLineM, Rectangle as RectangleM
from manimlib import WHITE, PI, LEFT, RIGHT, UP, DOWN
from compressed_sensing.geometry import Ellipse, Line, Point

def get_centered_segment(line: Line, ct: Point, length: float, color: str = WHITE) -> DashedLineM:
    ct_3d = np.array([ct.x, ct.y, 0])
    m = - line.a / line.b
    delta_x = length / np.sqrt(1 + m ** 2)
    delta_y = m * delta_x
    start = ct_3d - np.array([delta_x, delta_y, 0])
    end = ct_3d + np.array([delta_x, delta_y, 0])
    return DashedLineM(start=start, end=end, color=color)

def get_l2_ball_with_tangent(ellipse: Ellipse, color_ellipse: str = WHITE, color_tangent: str = WHITE, tol: float = 1e-4) -> Tuple[EllipseM, DashedLineM, DotM]:
    vertex_axis = None
    if (abs(ellipse.center.x) <= tol) or (abs(ellipse.center.y) <= tol):
        if (abs(ellipse.center.x) <= tol) and (abs(ellipse.center.y) <= tol):
            raise ValueError("The ellipse is in the origin, the tangent is not defined.")
        slopes = [-1, 1]
        vertex_axis = Point(x=np.sign(ellipse.center.x), y=np.sign(ellipse.center.y))
    else:
        slopes = [- np.sign(ellipse.center.y / ellipse.center.x)]
    if (abs(ellipse.axes.y) <= tol) or (abs(ellipse.axes.y) <= tol): # if ellipse collapsed to a pt
        line = Line(a=-slopes[0], b=1, c=ellipse.center.x - slopes[0] * ellipse.center.y)
        return (
            EllipseM(width=0.01, height=0.01, color=color_ellipse).shift(np.array([ellipse.center.x, ellipse.center.y, 0])),
            get_centered_segment(line=line, ct=ellipse.center, length=1, color=color_tangent),
            DotM(np.array([ellipse.center.x, ellipse.center.y, 0]), color=color_tangent)
        )
    pts_tangency = []
    for slope in slopes:
        (line1, pt1), (line2, pt2) = ellipse.tangent(slope=slope)
        # grab the one closest to the origin
        line, pt = (line1, pt1) if pt1.length() < pt2.length() else (line2, pt2)
        tangent = get_centered_segment(line=line, ct=pt, length=1, color=color_tangent)
        pts_tangency.append(pt)
    pt_tangency = None
    for pt in pts_tangency:
        if (0 <= abs(pt.x) <= 1) and (0 <= abs(pt.y) <= 1):
            pt_tangency = DotM(np.array([pt.x, pt.y, 0]), color=color_tangent)
            break
    if pt_tangency is None:  
        if vertex_axis is not None:  # tangency is at vertex in the axis
            pt = vertex_axis
        else:  # closest vertex to single point of tangency
            vertices = [Point(x=1, y=0), Point(x=0, y=1), Point(x=-1, y=0), Point(x=0, y=-1)]
            pt = min(vertices, key=lambda v: (v - pts_tangency[0]).length())
        pt_tangency = DotM(np.array([pt.x, pt.y, 0]), color=color_tangent)
        print("Got here!")
        ellipse = ellipse.get_scaled_version_through_point(pt=pt)
    l2_ball = ellipse.to_manim(color=color_ellipse)
    return l2_ball, tangent, pt_tangency

def get_ellipse_scalled_tangent_to_l1_ball(ellipse: Ellipse, l: float, color: str) -> Ellipse:
    a, b = ellipse.axes.x, ellipse.axes.y
    ct = np.array([ellipse.center.x, ellipse.center.y, 0])
    angle = np.deg2rad(-ellipse.angle)
    n1 = - np.sign(ct[0] * ct[1])
    n2 = np.sign(ct[1])
    m = (n1 * np.cos(angle) - np.sin(angle)) / (np.cos(angle) + n1 * np.sin(angle))
    den = np.sqrt((m * a) ** 2 + b ** 2)
    num = (n2 * l - ct[1] + n1 * ct[0]) * (np.cos(angle) - m * np.sin(angle))
    s = num / den
    return Ellipse(
        axes=Point(x=a*s, y=b*s), 
        center=Point(x=ct[0], y=ct[1]),
        angle=-np.rad2deg(angle),
    )

def get_set_concentric_l2_balls(base_ellipse: Ellipse, scales: Tuple[float] = (0.5, 0.75, 1), color: str = WHITE) -> Tuple[EllipseM, ...]:
    l2_balls = []
    a, b = base_ellipse.axes.x, base_ellipse.axes.y
    for s in scales:
        ellipse = Ellipse(axes=Point(x=s*a, y=s*b), center=base_ellipse.center, angle=base_ellipse.angle)
        l2_balls.append(ellipse.to_manim(color=color))
    return l2_balls

def get_set_concentric_l2_balls_with_tangent(base_ellipse: Ellipse, l: float, color_concentric: str = WHITE, color_tangent: str = WHITE) -> VGroup:
    concentric_l2_balls = get_set_concentric_l2_balls(base_ellipse=base_ellipse, color=color_concentric)
    tangent_ellipse = get_ellipse_scalled_tangent_to_l1_ball(ellipse=base_ellipse, l=l, color=color_tangent)
    tangent_l2_ball, _, pt_tangency = get_l2_ball_with_tangent(ellipse=tangent_ellipse, color_ellipse=color_tangent, color_tangent=color_tangent)
    return VGroup(*concentric_l2_balls, tangent_l2_ball, pt_tangency)

def get_x_mark(pt: Point, color: str = WHITE) -> VGroup:
    pt_3d = np.array([pt.x, pt.y, 0])
    return VGroup(
        LineM(pt_3d + LEFT/8 + UP/8, pt_3d + RIGHT/8 + DOWN/8, color=color),
        LineM(pt_3d + LEFT/8 + DOWN/8, pt_3d + RIGHT/8 + UP/8, color=color),
    )