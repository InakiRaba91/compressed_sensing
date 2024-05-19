from typing import Tuple
import numpy as np
from manimlib import VGroup, Ellipse as EllipseM, Dot as DotM, Line as LineM, DashedLine as DashedLineM, Rectangle as RectangleM
from manimlib import WHITE, YELLOW, LEFT, RIGHT, UP, DOWN
from compressed_sensing.geometry import Ellipse, Line, Point
from compressed_sensing.geometry.line_segment import LineSegment

def get_l2_ball_with_tangent_closest_to_origin(ellipse: Ellipse, slope: float, color_ellipse: str = WHITE, color_tangent: str = WHITE) -> Tuple[EllipseM, DashedLineM, DotM]:
    lines_and_pts = ellipse.tangent(slope=slope)
    # grab the one closest to the origin
    line, pt = min(lines_and_pts, key=lambda line_and_pt: line_and_pt[1].length())
    tangent = LineSegment.from_pt_length_and_slope(pt=pt, length=1, slope=-line.a/line.b)
    return ellipse.to_manim(color=color_ellipse), tangent.to_manim(color=color_tangent), pt.to_manim(color=color_tangent)

def get_ellipse_scalled_tangent_to_l1_ball(ellipse: Ellipse, l: float) -> Ellipse:
    """
    The idea is to shift and rotate the ellipse to center and align it, applying the same transforms
    to the line with slope +-1. Then, we can find the point of tangency by intersecting the ellipse
    and rotate + shift back the point of tangency
    """
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

def get_l2_ball_from_set_tangent_to_l1_ball(ellipse: Ellipse, l: float, color_ellipse: str = WHITE, color_tangent: str = WHITE, tol: float = 1e-4) -> Tuple[EllipseM, DotM]:
    # get candidate ellipse by finding the scaled one tangent to line with slope +-1
    ellipse = get_ellipse_scalled_tangent_to_l1_ball(ellipse=ellipse, l=l)

    # if ellipse center is in an axis, the tangent is the closest vertex to the center
    if (abs(ellipse.center.x) <= tol) or (abs(ellipse.center.y) <= tol):
        if (abs(ellipse.center.x) <= tol) and (abs(ellipse.center.y) <= tol):
            raise ValueError("The ellipse is in the origin, the tangent is not defined.")
        pt_tangency = Point(x=np.sign(ellipse.center.x), y=np.sign(ellipse.center.y))

        # scale ellipse to pass through the point of tangency
        ellipse = ellipse.get_scaled_version_through_point(pt=pt_tangency)
    else:
        # find point of tangency
        slope =  - np.sign(ellipse.center.y / ellipse.center.x)
        lines_and_pts = ellipse.tangent(slope=slope)

        # grab the one closest to the origin
        _, pt_tangency = min(lines_and_pts, key=lambda line_and_pt: line_and_pt[1].length())

        # if not in the diamond edge, get the closest diamond vertex
        if (abs(pt_tangency.x) + abs(pt_tangency.y) - l) > tol:
            vertices = [Point(x=l, y=0), Point(x=0, y=l), Point(x=-l, y=0), Point(x=0, y=-l)]
            pt_tangency = min(vertices, key=lambda v: (v - pt_tangency).length())
            # scale ellipse to pass through the point of tangency
            ellipse = ellipse.get_scaled_version_through_point(pt=pt_tangency)

    return ellipse.to_manim(color=color_ellipse), pt_tangency.to_manim(color=color_tangent)

def get_set_concentric_l2_balls(base_ellipse: Ellipse, scales: Tuple[float] = (0.5, 0.75, 1), color: str = WHITE) -> Tuple[EllipseM, ...]:
    l2_balls = []
    a, b = base_ellipse.axes.x, base_ellipse.axes.y
    for s in scales:
        ellipse = Ellipse(axes=Point(x=s*a, y=s*b), center=base_ellipse.center, angle=base_ellipse.angle)
        l2_balls.append(ellipse.to_manim(color=color))
    return l2_balls

def get_set_concentric_l2_balls_with_tangent(base_ellipse: Ellipse, l: float, color_concentric: str = WHITE, color_tangent: str = WHITE) -> VGroup:
    concentric_l2_balls = get_set_concentric_l2_balls(base_ellipse=base_ellipse, color=color_concentric)
    tangent_l2_ball, pt_tangency = get_l2_ball_from_set_tangent_to_l1_ball(ellipse=base_ellipse, l=l, color_ellipse=color_tangent, color_tangent=color_tangent)
    return VGroup(*concentric_l2_balls, tangent_l2_ball, pt_tangency)

def get_x_mark(pt: Point, color: str = WHITE) -> VGroup:
    pt_3d = np.array([pt.x, pt.y, 0])
    return VGroup(
        LineM(pt_3d + LEFT/8 + UP/8, pt_3d + RIGHT/8 + DOWN/8, color=color),
        LineM(pt_3d + LEFT/8 + DOWN/8, pt_3d + RIGHT/8 + UP/8, color=color),
    )

def get_set_concentric_l2_balls_with_tangent_and_center(ellipse: Ellipse, l: float, color_concentric: str = WHITE, color_tangent: str = WHITE, color_center: str = YELLOW) -> VGroup:
    tangency_group = get_set_concentric_l2_balls_with_tangent(base_ellipse=ellipse, l=l, color_concentric=color_concentric, color_tangent=color_tangent)
    center_x_mark = get_x_mark(pt=ellipse.center, color=color_center)
    return VGroup(*tangency_group, center_x_mark)