from typing import Tuple

import numpy as np
from manimlib import DOWN, LEFT, RIGHT, UP, WHITE, YELLOW
from manimlib import DashedLine as DashedLineM
from manimlib import Dot as DotM
from manimlib import Ellipse as EllipseM
from manimlib import Line as LineM
from manimlib import VGroup
from scipy.optimize import minimize

from compressed_sensing.geometry import Ellipse, Point
from compressed_sensing.geometry.line_segment import LineSegment


def get_l2_ball_with_tangent_closest_to_origin(
    ellipse: Ellipse, slope: float, color_ellipse: str = WHITE, color_tangent: str = WHITE
) -> Tuple[EllipseM, DashedLineM, DotM]:
    lines_and_pts = ellipse.tangent(slope=slope)
    # grab the one closest to the origin
    line, pt = min(lines_and_pts, key=lambda line_and_pt: line_and_pt[1].length())
    tangent = LineSegment.from_pt_length_and_slope(pt=pt, length=1, slope=-line.a / line.b)
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
    n1 = -np.sign(ct[0] * ct[1])
    n2 = np.sign(ct[1])
    m = (n1 * np.cos(angle) - np.sin(angle)) / (np.cos(angle) + n1 * np.sin(angle))
    den = np.sqrt((m * a) ** 2 + b**2)
    num = (n2 * l - ct[1] + n1 * ct[0]) * (np.cos(angle) - m * np.sin(angle))
    s = num / den
    return Ellipse(
        axes=s * ellipse.axes,
        center=ellipse.center,
        angle=ellipse.angle,
    )


def get_l2_ball_from_set_tangent_to_l1_ball(
    ellipse: Ellipse, l: float, color_ellipse: str = WHITE, color_tangent: str = WHITE, tol: float = 1e-4
) -> Tuple[EllipseM, DotM]:
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
        slope = -np.sign(ellipse.center.y / ellipse.center.x)
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


def get_l2_ball_from_set_tangent_to_l2_ball(
    ellipse: Ellipse, l: float, color_ellipse: str = WHITE, color_tangent: str = WHITE
) -> Tuple[EllipseM, DotM]:
    """
    We need to first shift and rotate the whole system so the ellipse is centered and aligned.

    Then, the l2-term that gives rise to the ellipse isocontour is given by
    (x - cx)^2 / a^2 + (y - cy)^2 / b^2 = s^2

    In matrix form, we can define
    A = [[1/a, 0], [0, 1/b]]
    so our term is
    ||Ax||^2

    The l2-term that gives rise to the circle isocontour is given by
    x^2 + y^2 = l^2

    In matrix form, we can define
    v = [cx, cy]
    so our term is
    ||x - v||^2

    That gives us the following optimization problem
    x* = argmin ||Ax||^2 subject to ||x - v||^2 = l^2
    """
    a, b = ellipse.axes.x, ellipse.axes.y
    A = np.array([[1 / a, 0], [0, 1 / b]])
    v = -ellipse.center.rotate(angle=ellipse.angle).to_array()  # shift and rotate to center and align the ellipse

    # Objective function: ||Ax||^2
    objective = lambda x: np.linalg.norm(A @ x) ** 2

    # Constraint: ||x - l||^2 <= epsilon
    constraint = lambda x: l**2 - np.linalg.norm(x - v) ** 2

    # Solve the problem
    result = minimize(objective, v, constraints=[{"type": "ineq", "fun": constraint}], method="trust-constr")
    pt_tangency_centered_alinged = Point(x=result.x[0], y=result.x[1])
    # find scaling factor
    s = np.sqrt(pt_tangency_centered_alinged.x**2 / a**2 + pt_tangency_centered_alinged.y**2 / b**2)
    ellipse = Ellipse(axes=s * ellipse.axes, center=ellipse.center, angle=ellipse.angle)

    # rotate and shift back the ellipse to the original position
    pt_tangency = pt_tangency_centered_alinged.rotate(angle=-ellipse.angle) + ellipse.center

    return ellipse.to_manim(color=color_ellipse), pt_tangency.to_manim(color=color_tangent)


def get_set_concentric_l2_balls(
    base_ellipse: Ellipse, scales: Tuple[float] = (0.5, 0.75, 1), color: str = WHITE
) -> Tuple[EllipseM, ...]:
    l2_balls = []
    a, b = base_ellipse.axes.x, base_ellipse.axes.y
    for s in scales:
        ellipse = Ellipse(axes=Point(x=s * a, y=s * b), center=base_ellipse.center, angle=base_ellipse.angle)
        l2_balls.append(ellipse.to_manim(color=color))
    return l2_balls


def get_set_concentric_l2_balls_with_tangent_to_l_ball(
    base_ellipse: Ellipse, l: float, norm_type: str, color_concentric: str = WHITE, color_tangent: str = WHITE
) -> VGroup:
    callable = get_l2_ball_from_set_tangent_to_l1_ball if norm_type == "l1" else get_l2_ball_from_set_tangent_to_l2_ball
    concentric_l2_balls = get_set_concentric_l2_balls(base_ellipse=base_ellipse, color=color_concentric)
    tangent_l2_ball, pt_tangency = callable(ellipse=base_ellipse, l=l, color_ellipse=color_tangent, color_tangent=color_tangent)
    return VGroup(*concentric_l2_balls, tangent_l2_ball, pt_tangency)


def get_x_mark(pt: Point, color: str = WHITE) -> VGroup:
    pt_3d = np.array([pt.x, pt.y, 0])
    return VGroup(
        LineM(pt_3d + LEFT / 8 + UP / 8, pt_3d + RIGHT / 8 + DOWN / 8, color=color),
        LineM(pt_3d + LEFT / 8 + DOWN / 8, pt_3d + RIGHT / 8 + UP / 8, color=color),
    )


def get_set_concentric_l2_balls_with_tangent_and_center(
    ellipse: Ellipse,
    l: float,
    norm_type: str,
    color_concentric: str = WHITE,
    color_tangent: str = WHITE,
    color_center: str = YELLOW,
) -> VGroup:
    tangency_group = get_set_concentric_l2_balls_with_tangent_to_l_ball(
        base_ellipse=ellipse, l=l, norm_type=norm_type, color_concentric=color_concentric, color_tangent=color_tangent
    )
    center_x_mark = get_x_mark(pt=ellipse.center, color=color_center)
    return VGroup(*tangency_group, center_x_mark)
