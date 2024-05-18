from typing import List, Tuple

import numpy as np

from .conic import check_symmetric_and_non_degenerate
from .exceptions import InvalidConicMatrixEllipseException, PointNotInEllipseException
from .line import Line
from .point import Point
from manimlib import Ellipse as EllipseM, WHITE


class Ellipse:
    def __init__(self, center: Point, axes: Point, angle: float):
        """
        Initializes the ellipse with the given parameters.
        2D ellipse parametrized as in OpenCV. It is built rotating first, shifting then
        https://docs.opencv.org/3.4.13/d6/d6e/group__imgproc__draw.html#ga28b2267d35786f5f890ca167236cbc69

        Args:
            center: center of the ellipse
            axes: length of the axes of the ellipse
            angle: angle of rotation of the ellipse w.r.t. x-axis counter-clock-wise
        """
        self._center = center
        self._axes = axes
        self._angle = angle

    @property
    def center(self) -> Point:
        """Returns the center of the ellipse"""
        return self._center

    @property
    def axes(self) -> Point:
        """Returns the axes of the ellipse"""
        return self._axes

    @property
    def angle(self) -> float:
        """Returns the angle of the ellipse"""
        return self._angle

    @classmethod
    def is_valid_matrix_representation(cls, M: np.ndarray, tol: float = 1e-10) -> bool:
        """Verify if a matrix is a valid representation of an ellipse

        That implies that:
        -> Symmetric 3x3 matrix
        -> Matrix is non-degenerate (non-null determinant)
        -> Sub matrix M[0:2, 0:2] is positive
        Source: https://en.wikipedia.org/wiki/Matrix_representation_of_conic_sections

        Args:
            M: 3x3 ndarray with the matrix representation of the ellipse
            tol: float error tolerance for considering two points are equal

        Returns: boolean indicating if the matrix is a valid representation of an ellipse
        """
        ellipse_matrix = np.linalg.det(M[0:2, 0:2]) > tol
        return ellipse_matrix and check_symmetric_and_non_degenerate(mat=M, tol=tol, ndim=3)

    @classmethod
    def from_matrix(cls, M: np.ndarray, tol: float = 1e-6) -> "Ellipse":
        """Computes center, axes length and angle w.r.t. x-axis from matrix representation
        https://en.wikipedia.org/wiki/Ellipse#General_ellipse

        Note:
            Sources:
              https://en.wikipedia.org/wiki/Ellipse#General_ellipse
              https://en.wikipedia.org/wiki/Matrix_representation_of_conic_sections
        Args:
            M: 3x3 ndarray with the matrix representation of the ellipse
            units: DistanceUnits outlining the units of the 2D space where the ellipse matrix is given
            tol: float error tolerance for considering a point belongs to the ellipse

        Returns: Ellipse corresponding to the matrix representation
        """
        if not cls.is_valid_matrix_representation(M=M):
            raise InvalidConicMatrixEllipseException("Matrix does not represent an ellipse")
        # We follow the naming in https://en.wikipedia.org/wiki/Ellipse#General_ellipse
        A = M[0, 0]
        B = 2 * M[0, 1]
        D = 2 * M[0, 2]
        C = M[1, 1]
        E = 2 * M[1, 2]
        F = M[2, 2]
        den = B * B - 4 * A * C
        num0 = 2 * (A * (E**2) + C * (D**2) - B * D * E + den * F)
        num_a = A + C - np.sqrt((A - C) ** 2 + (B**2))
        num_b = A + C + np.sqrt((A - C) ** 2 + (B**2))
        a = -np.sqrt(num0 * num_a) / den
        b = -np.sqrt(num0 * num_b) / den
        x0 = (2 * C * D - B * E) / den
        y0 = (2 * A * E - B * D) / den
        # I think there is an error in Wikipedia and angle needs to be increased by 90º
        #  Otherwise the returned angle after going back and fort is shifted by 90º
        if abs(B) > tol:
            rads = np.arctan((C - A - np.sqrt((A - C) ** 2 + (B**2))) / B)
            angle = 90 + np.rad2deg(rads)
        elif A < C:
            angle = 90
        else:
            angle = 180
        return Ellipse(center=Point(x=x0, y=y0), axes=Point(x=a, y=b), angle=angle)

    def to_matrix(self) -> np.ndarray:
        """Computes the matrix representation of the ellipse as defined in
        https://en.wikipedia.org/wiki/Ellipse#General_ellipse

        Note:
            Sources:
              https://en.wikipedia.org/wiki/Ellipse#General_ellipse
              https://en.wikipedia.org/wiki/Matrix_representation_of_conic_sections
        Args: None

        Returns: 3x3 ndarray with the matrix representation of the ellipse
        """
        # We follow the naming in https://en.wikipedia.org/wiki/Ellipse#General_ellipse
        x0: float = self._center.x
        y0: float = self._center.y
        a: float = self._axes.x
        b: float = self._axes.y
        theta: float = np.deg2rad(self._angle)
        A = (a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2
        B = 2 * ((b**2) - (a**2)) * np.sin(theta) * np.cos(theta)
        C = (a * np.cos(theta)) ** 2 + (b * np.sin(theta)) ** 2
        D = -2 * A * x0 - B * y0
        E = -B * x0 - 2 * C * y0
        F = A * (x0**2) + B * x0 * y0 + C * (y0**2) - (a * b) ** 2
        return np.array([[A, B / 2, D / 2], [B / 2, C, E / 2], [D / 2, E / 2, F]])

    def __add__(self, pt: Point) -> "Ellipse":  # type: ignore
        """Adds a point to ellipse, which simply shifts it

        Args:
            pt: Point to add

        Returns:
            Ellipse resulting from sum
        """
        return Ellipse(center=self._center + pt, axes=self._axes, angle=self._angle)

    def scale(self, pt: Point) -> "Ellipse":
        """Provides the ellipse after applying a scaling of the 2D space with
        the scaling given in each coordinate of point

        If a transform can be characterized as an homography, a conic defined by
        its matrix representation M will undergo the following transform
        M' = H^-T * M * H^-1

        A scaling transform can be characterized as
            | pt.x   0    0 |
        H = |  0   pt.y   0 |
            |  0     0    1 |

        Note:
            Source: http://www.macs.hw.ac.uk/bmvc2006/papers/306.pdf (eq. 11)

        Args:
            pt: Point defining the scaling of each axis in the 2D space

        Returns:
            Ellipse resulting from scaling the 2D space
        """
        M = self.to_matrix()
        Hinv = np.array([[1 / pt.x, 0, 0], [0, 1 / pt.y, 0], [0, 0, 1]])  # symmetric
        M_scaled = Hinv.dot(M).dot(Hinv)
        return Ellipse.from_matrix(M=M_scaled)

    def intersection_line(self, line: Line, tol: float = 1e-6) -> Tuple[Point, ...]:
        """Find the point of intersection between the ellipse and a given line
        A rigid transform is applied to both the ellipse and the line in order
        to center the ellipse at the origin of coordinates and align its axes
        with the xy-axes. This implies applying the two operations sequantially
        in reverse order w.r.t. the way the ellipse is built: shift first, rotate
        then.
        Afterwards, we can apply the method defined in:
        https://www.emathzone.com/tutorials/geometry/intersection-of-line-and-ellipse.html
        The line is defined by y=m*x+c, whereas the ellipse is defined by (x²/a²)+(y²/b²)=1
        The points of intersection satisfy both, thus
        (x²/a²)+(m*x+c)²/b²=1
        ->  b²*x²+a²*(m*x+c)²=a²b²
        -> (a²*m²+b²)*x² + (2a²*m*c)*x + a²(c²–b²)
        which is a quadratic equation of the form Ax²+Bx+C=0
        Args:
            line: Line to find the intersection with
            tol: float error tolerance for considering a point belongs to the line
        Returns:
            None if they don't intersect, Point in case of tangency, or tuple with two points of intersection
        """
        line_rigid = line.rigid_transform(pt_shift=-self._center, angle=-self._angle)
        pts_intersection: List[Point] = []
        if np.abs(line_rigid.b) > tol:
            # The method in the reference is only valid if the line is not vertical
            # we'll follow the notation in there
            a, b = self._axes.x, self._axes.y
            m = -line_rigid.a / line_rigid.b
            c = -line_rigid.c / line_rigid.b

            # Params of the quadratic equation Ax²+Bx+C
            A = (a**2) * (m**2) + b**2
            B = 2 * (a**2) * m * c
            C = (a**2) * ((c**2) - (b**2))
            # compute unique solutions
            roots = np.roots([A, B, C])
            if np.abs(roots[1] - roots[0]) <= tol:
                unique_roots: np.ndarray = roots[:1]
            else:
                unique_roots = roots
            if np.isreal(unique_roots).all():
                # append point if the solution(s) are not complex,
                # which means there's no intersection
                for root in unique_roots:
                    x = root
                    y = m * x + c
                    pts_intersection.append(Point(x=x, y=y))
            else:
                pass
        # Vertical line
        else:
            # for a vertical line, the x coordinate of the intersection points is given by
            x = -line_rigid.c / line_rigid.a

            # Following the notation in the reference, the ellipse is (x²/a²)+(y²/b²)=1
            # Thus y = sqrt(1-y²/b²)*a, which can have 0, 1 or 2 solutions depending
            # on whether the discriminant 1-y²/b² is negative, null or positive
            discriminant = 1.0 - (x**2) / (self._axes.x**2)
            if np.abs(discriminant) <= tol:
                pts_intersection.append(Point(x=x, y=0))
            elif discriminant > 0:
                y = self._axes.y * np.sqrt(discriminant)
                pts_intersection.append(Point(x=x, y=y))
                pts_intersection.append(Point(x=x, y=-y))
            else:
                pass

        # we need to undo the rigid transform in reverse order
        # 1. Rotate point
        # 2. Shift point
        pts_intersection_rigid: List[Point] = []
        for pt in pts_intersection:
            rotated_pt = pt.rotate(angle=self._angle)
            shifted_pt = rotated_pt + self._center
            pts_intersection_rigid.append(shifted_pt)
        return tuple(pts_intersection_rigid)

    def tangent(self, slope: float) -> Tuple[Tuple[Line, Point], ...]:
        """
        Computes the tangent line to the ellipse at a given slope
        The equation for the tangents to a centered non-rotated ellipse is
        y = m*x +- sqrt(a²*m²+b²) = mx +- n
        where n1 = sqrt(a²*m²+b²)

        We need to apply two transforms to it:

        1. Rotate the line by the angle of the ellipse
        y = x * (m * cos(angle) + sin(angle)) / (cos(angle) - m * sin(angle)) +- n1 / (cos(angle) - m * sin(angle))
        y = x * m2 +- n2
        where 
        m2 = (m * cos(angle) + sin(angle)) / (cos(angle) - m * sin(angle))
        n2 = n1 / (cos(angle) - m* sin(angle))

        2. Shift the line by the center of the ellipse
        y = x * m2 +- n2 + ct_y - m2 * ct_x 
        y = x * m2 + n3
        where
        n3 = +- n2 + ct_y - m2 * ct_x

        We are looking for the line such that m2 = slope
        slope = (m * cos(angle) + sin(angle)) / (cos(angle) - m * sin(angle))
        m = (slope * cos(angle) - sin(angle)) / (cos(angle) + slope * sin(angle))

        We end up with two lines, one for each sign of n3

        The point of tangency (x0, y0) for a centered non-rotated ellipse with slope m is
        x0 = +-a² * m / sqrt(a² * m² + b²)
        y0 = +-b² / sqrt(a² * m² + b²)

        If m is negative, both x0 and y0 have the same sign. Otherwise, they have opposite signs
        y0 = - b² / sqrt(a² * m² + b²) * sign(x0) * sign(m)

        If we revert the transforms for the lines, we can compute this coordinates for both
        points w.r.t. to the centered non-rotated ellipse


        Args:
            slope: float slope of the tangent line

        Returns:
            Tuple with two lines and the corresponding points of tangency
        """
        a, b = self._axes.x, self._axes.y
        angle = - np.deg2rad(self._angle)
        
        m = (slope * np.cos(angle) - np.sin(angle)) / (np.cos(angle) + slope * np.sin(angle))
        n1 = np.sqrt((a**2) * (m**2) + (b**2))
        m2 = (m * np.cos(angle) + np.sin(angle)) / (np.cos(angle) - m * np.sin(angle))
        n2 = n1 / (np.cos(angle) - m * np.sin(angle))
        n3_1 = n2 + self._center.y - m2 * self._center.x
        n3_2 = -n2 + self._center.y - m2 * self._center.x

        line1 = Line(a=m2, b=-1, c=n3_1)
        line2 = Line(a=m2, b=-1, c=n3_2)
        
        line_centered_aligned = line1.shift(pt_shift=-self._center).rotate(angle=self._angle)
        # We write it in form y = mx + c
        m = -line_centered_aligned.a / line_centered_aligned.b
        x0 = a**2 * m / np.sqrt(a**2 * m**2 + b**2)
        y0 = - b**2 / np.sqrt(a**2 * m**2 + b**2) * np.sign(x0) * np.sign(m)
        pt1 = Point(x=x0, y=y0)
        pt2 = Point(x=-x0, y=-y0)

        # Now we rotate and shift back
        pt1 = pt1.rotate(angle=-self._angle) + self._center
        pt2 = pt2.rotate(angle=-self._angle) + self._center

        # group them by distance to the origin
        pt1, pt2 = (pt1, pt2) if pt1.length() < pt2.length() else (pt2, pt1)
        line1, line2 = (line1, line2) if abs(line1.c/line1.a) < abs(line2.c/line2.a) else (line2, line1)
    
        return ((line1, pt1), (line2, pt2))

    def get_scaled_version_through_point(self, pt: Point) -> "Ellipse":
        a, b = self._axes.x, self._axes.y
        # apply rigid transform to center the ellipse and align it with xy-axes
        pt = (pt - self._center).rotate(angle=self._angle)
        s = np.sqrt((pt.x**2) / (a**2) + (pt.y**2) / (b**2))
        return Ellipse(center=self._center, axes=Point(x=s*a, y=s*b), angle=self._angle)
    
    def contains_pt(self, pt: Point, tol: float = 1e-6) -> bool:
        """Determines whether a point belongs to an ellipse or not
        The points that belong to the centered&aligned ellipse satisfies
         (x/a)² + (y/b)² = 1
        Args:
            pt: Point to check whether or not it belongs to the ellipse
            tol: float error tolerance for considering a point belongs to the ellipse
        Returns:
            boolean indicating if the point belongs to the ellipse
        """
        # apply rigid transform to center the ellipse and align it with xy-axes
        centered_pt = pt - self._center
        aligned_pt = centered_pt.rotate(angle=-self._angle)
        root_augend = aligned_pt.x / self._axes.x
        root_addend = aligned_pt.y / self._axes.y
        value = root_augend**2 + root_addend**2
        return np.abs(value - 1) <= tol

    def circle_angle_from_ellipse_point(self, pt: Point, tol: float = 1e-6) -> float:
        """Computes the angle in degrees of a point in the w.r.t. the
        inner/outer circle associated to the ellipse.
        Note:
            See ai_sb_toolbox/geometry/README.md file for detailed diagram
            Method will raise an Exception if point does not belong to the ellipse with the given tol
        Args:
            pt: Point in the ellipse to compute the angle for
            tol: float error tolerance for considering a point belongs to the ellipse
        Returns:
            float angle in degrees in the corresponding inner/outer circle
        """
        # the point needs to belong to the ellipse
        if not self.contains_pt(pt=pt, tol=tol):
            raise PointNotInEllipseException("Ellipse and line do not intersect.")
        # apply rigid transform to center the ellipse and align it with xy-axes
        centered_pt = pt - self._center
        aligned_pt = centered_pt.rotate(angle=-self._angle)

        # get the point in the inner/outer circle given by the x-axis width of the ellipse
        # we need to preserve the sign of the y coordinate and compute the angle
        x = aligned_pt.x
        y = np.sign(aligned_pt.y) * np.sqrt((self._axes.x**2) - x**2)
        rads = np.arctan2(y, x)
        # They are returned in the range [-pi, pi]. We map them to [0, 360]
        return (np.rad2deg(rads) + 360) % 360

    def ellipse_point_from_circle_angle(self, gamma):
        """Computes the point in ellipse at a given circle angle gamme in degrees. The angle is
        given w.r.t. the centered and xy-aligned ellipse
        Note:
            See ai_sb_toolbox/geometry/README.md file for detailed diagram
        Args:
            gamma: float angle in degrees to evaluate the ellipse at. Given w.r.t. the centered and xy-aligned ellipse
        Returns:
            Point in the ellipse at given angle
        """
        # compute point in inner/outer circle given by ellipse x-axis width,
        # preserving the sign of the y coordinate
        rads = np.deg2rad(gamma)
        pt_circle = Point(x=self._axes.x * np.cos(rads), y=self._axes.x * np.sin(rads))
        x = pt_circle.x
        sign = np.sign(pt_circle.y)
        root_subtrahend = pt_circle.x / self._axes.x
        y = sign * self._axes.y * np.sqrt(1 - root_subtrahend**2)
        pt_rigid_ellipse = Point(x=x, y=y)

        # apply reverted rigid transform
        rotated_pt = pt_rigid_ellipse.rotate(angle=self._angle)
        return rotated_pt + self._center

    def to_manim(self, color: str = WHITE):
        a, b = self.axes.x, self.axes.y
        ct = np.array([self.center.x, self.center.y, 0])
        angle = np.deg2rad(-self.angle)
        return EllipseM(width=2*a, height=2*b, color=color).rotate(angle).shift(ct)
    
    def __repr__(self):
        return f"Ellipse(center={self.center}, axes={self.axes}, angle={self.angle})"