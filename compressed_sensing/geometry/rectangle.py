from typing import List, Tuple

import numpy as np

from .conic import check_symmetric_and_non_degenerate
from .exceptions import InvalidConicMatrixEllipseException, PointNotInEllipseException
from .line import Line
from .point import Point
from manimlib import Rectangle as RectangleM, WHITE


class Rectangle:
    def __init__(self, center: Point, axes: Point, angle: float):
        """
        Initializes the rectangle with the given parameters.

        Args:
            center: center of the rectangle
            axes: length of the axes of the rectangle
            angle: angle of rotation of the rectangle w.r.t. x-axis counter-clock-wise
        """
        self._center = center
        self._axes = axes
        self._angle = angle

    @property
    def center(self) -> Point:
        """Returns the center of the rectangle"""
        return self._center

    @property
    def axes(self) -> Point:
        """Returns the axes of the rectangle"""
        return self._axes

    @property
    def angle(self) -> float:
        """Returns the angle of the rectangle"""
        return self._angle

    def __add__(self, pt: Point) -> "Rectangle":  # type: ignore
        """Adds a point to rectangle, which simply shifts it

        Args:
            pt: Point to add

        Returns:
            Rectangle resulting from sum
        """
        return Rectangle(center=self._center + pt, axes=self._axes, angle=self._angle)

    def to_manim(self, color: str = WHITE):
        a, b = self.axes.x, self.axes.y
        ct = np.array([self.center.x, self.center.y, 0])
        angle = np.deg2rad(-self.angle)
        return RectangleM(width=a, height=b, color=color).rotate(angle).shift(ct)
    
    def __repr__(self):
        return f"Ellipse(center={self.center}, axes={self.axes}, angle={self.angle})"