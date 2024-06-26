from typing import Any, Union

import numpy as np
from manimlib import WHITE
from manimlib import DashedLine as DashedLineM

from .exceptions import LineFromEqualPointsException
from .point import Point


class LineSegment:
    """2D line segment defined by its two endpoints

    Args:
        pt1: first endpoint
        pt2: second endpoint
    """

    tol = 1e-6

    def __init__(self, pt1: Point, pt2: Point):
        if (pt1 - pt2).length() <= self.tol:
            raise LineFromEqualPointsException("Both points are equal.")
        self._pt1 = pt1
        self._pt2 = pt2

    @property
    def pt1(self) -> Point:
        """Returns 1st point"""
        return self._pt1

    @property
    def pt2(self) -> Point:
        """Returns 2nd point"""
        return self._pt2

    def __eq__(self, other: Any, tol: float = 1e-6):
        """Performs the equality comparison between current object and passed one.

        Args:
            other: object to compare against
            tol: float error tolerance for considering two cameras equal

        Returns:
            boolean indicating if two objects are equal
        """
        if isinstance(other, LineSegment):
            pt1_equal = (self.pt1 - other.pt1).length() < tol
            pt2_equal = (self.pt2 - other.pt2).length() < tol
            return pt1_equal and pt2_equal
        return False

    def __neg__(self) -> "LineSegment":
        """Flips a line segment180º w.r.t. the origin of coordinates

        Args: None

        Returns:
            flipped LineSegment
        """
        return LineSegment(pt1=-self.pt1, pt2=-self.pt2)

    def __add__(self, other: Union["LineSegment", Point]) -> "LineSegment":  # type: ignore
        """Adds a point or line segment to line segment

        Args:
            other: Point to add

        Returns:
            LineSegment resulting from the sum
        """
        if isinstance(other, Point):
            return LineSegment(pt1=self.pt1 + other, pt2=self.pt2 + other)
        elif isinstance(other, LineSegment):
            return LineSegment(pt1=self.pt1 + other.pt1, pt2=self.pt2 + other.pt2)

    def length(self) -> float:
        """Computes the length of the line segment

        Args: None

        Returns:
            float length of the line segment
        """
        return (self.pt1 - self.pt2).length()

    @classmethod
    def from_pt_length_and_slope(cls, pt: Point, length: float, slope: float) -> "LineSegment":
        """Computes the sub segment of the line centered at the given point
        and having the given length with given slope

        Args:
            pt: Point where the sub segment starts
            length: float indicating the length of the sub segment
            slope: float indicating slope

        Returns:
            LineSegment starting at the given point and having the given length with given slope
        """
        delta_x = length / np.sqrt(1 + slope**2)
        delta_y = slope * delta_x
        delta = Point(x=delta_x, y=delta_y)
        return cls(pt1=pt - delta, pt2=pt + delta)

    def scale(self, pt: Point) -> "LineSegment":
        """Provides the 2D segment after applying a scaling of the 2D space with
        the scaling given in each coordinate of point

        The 2D x-y space is scaled by
        x' = x * pt.x
        y' = y * pt.y

        Args:
            pt: Point defining the scaling of each axis in the 2D space

        Returns:
            LineSegment resulting from scaling the 2D space
        """
        scale_x = pt.x
        scale_y = pt.y
        return LineSegment(
            pt1=Point(x=self.pt1.x * scale_x, y=self.pt1.y * scale_y),
            pt2=Point(x=self.pt2.x * scale_x, y=self.pt2.y * scale_y),
        )

    def to_manim(self, color: str = WHITE, type_line=DashedLineM):
        pt1 = np.array([self.pt1.x, self.pt1.y, 0])
        pt2 = np.array([self.pt2.x, self.pt2.y, 0])
        return type_line(start=pt1, end=pt2, color=color)
