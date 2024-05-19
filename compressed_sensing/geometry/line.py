from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .point import Point


class Line:
    """2D line parametrized in the general form ax+by+c=0"""

    def __init__(self, a: float, b: float, c: float):
        """Initializes a line with the given coefficients

        Args:
            a: x-weight in equation Ax + by + c.
            b: y-weight in equation ax + By + c.
            c: (float): constant weight in equation ax + by + c.
        """

        self._a = a
        self._b = b
        self._c = c

        self._check_valid_parametrization()

    @property
    def a(self) -> float:
        """Returns the x-weight in equation Ax + by + c"""
        return self._a

    @property
    def b(self) -> float:
        """Returns the y-weight in equation ax + By + c"""
        return self._b

    @property
    def c(self) -> float:
        """Returns the constant weight in equation ax + by + C"""
        return self._c

    def _check_valid_parametrization(self) -> None:
        """Checks the line parametrization is not invalid (a=b=0)

        Raises:
            ValueError: if invalid parametrization is provided (a=b=0)
        """
        tol = 1e-6
        if (np.abs(self.a - self.b) <= tol) and (np.abs(self.a) <= tol):
            raise InvalidLineException("Invalid line parametrization (if a=b=0 -> c=0).")

    def to_array(self) -> np.ndarray:
        """Converts to numpy array
        Returns:
            ndarray  [a, b, c]
        """
        return np.array([self.a, self.b, self.c])

    @classmethod
    def from_points(cls, pt1: Point, pt2: Point, tol: float = 1e-6) -> "Line":
        """Computes the line that passes through two given points

        Args:
            pt1: first Point the line passes through
            pt2: second Point the line passes through
            tol: float error tolerance for considering two points are equal

        Returns:
            line parametrization (ax+by+c) passing through the provided points
        """
        if (pt1 - pt2).length() <= tol:
            raise LineFromEqualPointsException("Both points are equal.")
        a, b, c = np.cross(pt1.to_homogeneous(), pt2.to_homogeneous())
        return cls(a=a, b=b, c=c)

    def __eq__(self, other: Any, tol: float = 1e-6):
        """Performs the equality comparison between current object and passed one.

        Args:
            other: object to compare against
            tol: float error tolerance for considering two cameras equal

        Returns:
            boolean indicating if two objects are equal
        """
        if isinstance(other, Line):
            # we normalize both line arrays
            normalized_self_line = self.to_array() / np.max(np.abs(self.to_array()))
            normalized_other_line = other.to_array() / np.max(np.abs(other.to_array()))
            # they need to be equal except maybe for the sign
            proportional_lines = (np.abs(np.abs(normalized_self_line) - np.abs(normalized_other_line)) < tol).all()
            return proportional_lines
        return False

    def __neg__(self) -> Line:
        """Flips a line 180º w.r.t. the origin of coordinates

        Returns:
            flipped Line
        """
        return Line(a=self._a, b=self._b, c=-self._c)

    def __add__(self, pt: Point) -> Line:  # type: ignore
        """Adds a point to line

        When adding a point, the resulting line is parallel to the original one.
        We can compute a point in the original line, i.e. (-c/a, 0) or (0, -c/b),
        add the given point and force the new line to pass through this point (x', y')
        The original line equation evaluated at the point is
        a*x'+b*y'+c=0
        If we do c' = -(a*x'+b*y')
        the new line
        ax+by+c'=0
        fulfills the definition

        Args:
            pt: Point to add

        Returns:
            Line resulting from the sum
        """
        if np.abs(self._a) > 1e-8:
            pt_line = Point(x=-self._c / self._a, y=0)
        else:
            pt_line = Point(x=0, y=-self._c / self._b)
        pt_new_line = pt_line + pt
        return Line(a=self._a, b=self._b, c=-(self._a * pt_new_line.x + self._b * pt_new_line.y))

    def __repr__(self):
        return f"Line(a={self.a}, b={self.b}, c={self.c})"

    def scale(self, pt: Point) -> Line:
        """Provides the 2D line after applying a scaling of the 2D space with
        the scaling given in each coordinate of point
        The original line is given by equation ax+by+c=0, so a point in homogenous
        coordinates satisfies it. The scaled point (x' y') = (x*pt.x, y*pt.y) should
        then satisfy:
        (a/pt.x)x'+(b/pt.y)y'+c=0

        Args:
            pt: Point defining the scaling of each axis in the 2D space

        Returns:
            Line resulting from scaling the 2D space
        """
        return Line(a=self._a / pt.x, b=self._b / pt.y, c=self._c)

    def intersection_line(self, other: Line, tol: float = 1e-6) -> Optional[Point]:
        """Find the point of intersection between two lines
        Explanation: https://www.cuemath.com/geometry/intersection-of-two-lines/

        Args:
            other: second Line to find the intersection with
            tol: float error tolerance for considering a point belongs to the line

        Returns:
            None if they don't intersect or are the same line, Point of intersection otherwise
        """
        if np.abs((self._a * other.b) - (other.a * self._b)) > tol:
            x = (other.c * self._b - self._c * other.b) / (self._a * other.b - other.a * self._b)
            y = (self._c * other.a - other.c * self._a) / (self._a * other.b - other.a * self._b)
            return Point(x=x, y=y)
        else:
            return None

    def rotate(self, angle: float) -> Line:
        """Rotates a line by the degrees given in angle counter clock-wise
        The line intersects with the axes at (0, -c/a) and (-c/b, 0), whose
        normal vector (pointing to the half-plane given by the greater
        inequality) is thus:
        [   0  ]   [-(c/b)]
        [-(c/a)] x [   0  ]
        [   1  ]   [   1  ]
        Rotating the line is equivalent to rotating these two points,
        which turns them into:
        (-c/a, 0) => (-(c/a)*cos(angle),  -(c/a)*sin(angle))
        (0, -c/b) => ( (c/b)*sin(angle), -(c/b)*cos(angle))
        The new line is then given by the cross product of these two points in
        homogeneous coordinates (the order is relevant, so we don't switch half planes):
        [-(c/a)*cos(angle)]   [ (c/b)*sin(angle)]   [-(c/a)*sin(angle)+(c/b)*cos(angle)]
        [-(c/a)*sin(angle)] x [-(c/b)*cos(angle)] = [ (c/b)*sin(angle)+(c/a)*cos(angle)]
        [       1         ]   [       1         ]   [c²/(a*b)*(cos²(angle)+sin²(angle))]
        Thus:
        -> a_rot =-(c/a)*sin(angle)+(c/b)*cos(angle)
        -> b_rot = (c/b)*sin(angle)+(c/a)*cos(angle)
        -> c_rot = c²/(a*b)
        and if we scale everything by (a*b)/c:
        -> a_rot = a*cos(angle)-b*sin(angle)
        -> b_rot = a*sin(angle)+b*cos(angle)
        -> c_rot = c
        Args:
            angle: float indicating rotation w.r.t. x-axis counter clock-wise in degrees
        Returns:
            rotated Line
        """
        rads = np.deg2rad(angle)
        a_rot = self._a * np.cos(rads) - self._b * np.sin(rads)
        b_rot = self._a * np.sin(rads) + self._b * np.cos(rads)
        return Line(a=a_rot, b=b_rot, c=self._c)

    def shift(self, pt_shift: Point) -> Line:
        """Shift point in the direction of the vector from the origin of
        coordinates to the given point

        Args:
            pt_shift: Point determining the shift of the line
        Returns:
            Line after applying rigid transform
        """
        # addition will verify both point and line are in the same domain
        return self + pt_shift
