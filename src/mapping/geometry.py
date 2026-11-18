"""Spatial geometry engine for 2D indoor coordinates, line segments, and polygons.

Adheres strictly to Task 24's standard coordinate conventions:
- Metric Cartesian 2D coordinate system (x, y) in meters (m).
- Building-local X-axis: Longitudinal / East axis (+X).
- Building-local Y-axis: Transverse / North axis (+Y).
- Standard BUILDING_LOCAL_CARTESIAN_ENU reference frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class Point2D:
    """A 2D point in the building-local metric coordinate frame."""
    x: float
    y: float

    def distance_to(self, other: "Point2D") -> float:
        """Calculates Euclidean distance to another 2D point in meters."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def to_tuple(self) -> Tuple[float, float]:
        return (round(self.x, 3), round(self.y, 3))

    def to_dict(self) -> Dict[str, float]:
        return {"x": round(self.x, 3), "y": round(self.y, 3)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Point2D":
        return cls(x=float(data["x"]), y=float(data["y"]))

    @classmethod
    def from_tuple(cls, coords: Tuple[float, float]) -> "Point2D":
        return cls(x=float(coords[0]), y=float(coords[1]))


@dataclass
class LineSegment2D:
    """A directed line segment between two 2D points in meters."""
    p1: Point2D
    p2: Point2D

    @property
    def length(self) -> float:
        return self.p1.distance_to(self.p2)

    @property
    def midpoint(self) -> Point2D:
        return Point2D((self.p1.x + self.p2.x) / 2.0, (self.p1.y + self.p2.y) / 2.0)

    def distance_to_point(self, pt: Point2D) -> float:
        """Calculates minimum distance from a point to this line segment."""
        px = self.p2.x - self.p1.x
        py = self.p2.y - self.p1.y
        norm_sq = px * px + py * py
        if norm_sq < 1e-9:
            return self.p1.distance_to(pt)

        # Projection factor u on segment [0, 1]
        u = ((pt.x - self.p1.x) * px + (pt.y - self.p1.y) * py) / float(norm_sq)
        u = max(0.0, min(1.0, u))
        proj_x = self.p1.x + u * px
        proj_y = self.p1.y + u * py
        return math.hypot(pt.x - proj_x, pt.y - proj_y)

    def intersects(self, other: "LineSegment2D") -> bool:
        """Determines whether two line segments cross each other (proper intersection)."""
        def ccw(a: Point2D, b: Point2D, c: Point2D) -> bool:
            return (c.y - a.y) * (b.x - a.x) > (b.y - a.y) * (c.x - a.x)

        a, b = self.p1, self.p2
        c, d = other.p1, other.p2
        return (ccw(a, c, d) != ccw(b, c, d)) and (ccw(a, b, c) != ccw(a, b, d))

    def intersects_with(self, other: "LineSegment2D") -> bool:
        """Alias for intersects()."""
        return self.intersects(other)

    def to_dict(self) -> Dict[str, Any]:
        return {"p1": self.p1.to_dict(), "p2": self.p2.to_dict(), "length_m": round(self.length, 3)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LineSegment2D":
        return cls(p1=Point2D.from_dict(data["p1"]), p2=Point2D.from_dict(data["p2"]))


@dataclass
class BoundingBox2D:
    """Axis-aligned 2D bounding box in metric coordinates."""
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return max(0.0, self.max_x - self.min_x)

    @property
    def height(self) -> float:
        return max(0.0, self.max_y - self.min_y)

    @property
    def area(self) -> float:
        return self.width * self.height

    def contains_point(self, pt: Point2D) -> bool:
        return (self.min_x <= pt.x <= self.max_x) and (self.min_y <= pt.y <= self.max_y)

    def intersects(self, other: "BoundingBox2D") -> bool:
        return not (
            self.max_x < other.min_x
            or self.min_x > other.max_x
            or self.max_y < other.min_y
            or self.min_y > other.max_y
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "min_x": round(self.min_x, 3),
            "min_y": round(self.min_y, 3),
            "max_x": round(self.max_x, 3),
            "max_y": round(self.max_y, 3),
            "width": round(self.width, 3),
            "height": round(self.height, 3),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BoundingBox2D":
        return cls(
            min_x=float(data["min_x"]),
            min_y=float(data["min_y"]),
            max_x=float(data["max_x"]),
            max_y=float(data["max_y"]),
        )


@dataclass
class Polygon2D:
    """A 2D planar polygon defined by an ordered list of vertices in meters."""
    vertices: List[Point2D]

    def __post_init__(self):
        # Auto-remove duplicate consecutive closure vertex if provided
        if len(self.vertices) > 1 and self.vertices[0] == self.vertices[-1]:
            self.vertices = self.vertices[:-1]

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def is_valid_polygon(self) -> bool:
        """Validates polygon has at least 3 vertices, non-zero area, and no self-intersections."""
        if len(self.vertices) < 3:
            return False
        if abs(self.area) < 1e-4:
            return False
        return not self.has_self_intersections()

    @property
    def area(self) -> float:
        """Calculates polygon area using the Shoelace formula (positive for CCW/CW)."""
        n = len(self.vertices)
        if n < 3:
            return 0.0
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i].x * self.vertices[j].y
            area -= self.vertices[j].x * self.vertices[i].y
        return abs(area) / 2.0

    @property
    def perimeter(self) -> float:
        """Calculates total boundary perimeter length in meters."""
        n = len(self.vertices)
        if n < 2:
            return 0.0
        perim = 0.0
        for i in range(n):
            j = (i + 1) % n
            perim += self.vertices[i].distance_to(self.vertices[j])
        return perim

    @property
    def centroid(self) -> Point2D:
        """Calculates geometric centroid coordinates."""
        n = len(self.vertices)
        if n == 0:
            return Point2D(0.0, 0.0)
        if n < 3:
            return Point2D(
                sum(v.x for v in self.vertices) / n,
                sum(v.y for v in self.vertices) / n,
            )

        cx, cy = 0.0, 0.0
        signed_area = 0.0
        for i in range(n):
            j = (i + 1) % n
            factor = self.vertices[i].x * self.vertices[j].y - self.vertices[j].x * self.vertices[i].y
            signed_area += factor
            cx += (self.vertices[i].x + self.vertices[j].x) * factor
            cy += (self.vertices[i].y + self.vertices[j].y) * factor

        signed_area *= 0.5
        if abs(signed_area) < 1e-9:
            return Point2D(sum(v.x for v in self.vertices) / n, sum(v.y for v in self.vertices) / n)

        cx /= (6.0 * signed_area)
        cy /= (6.0 * signed_area)
        return Point2D(round(cx, 3), round(cy, 3))

    @property
    def bounding_box(self) -> BoundingBox2D:
        """Computes outer axis-aligned bounding box."""
        if not self.vertices:
            return BoundingBox2D(0, 0, 0, 0)
        xs = [v.x for v in self.vertices]
        ys = [v.y for v in self.vertices]
        return BoundingBox2D(min(xs), min(ys), max(xs), max(ys))

    def contains_point(self, pt: Point2D) -> bool:
        """Ray-casting algorithm to test whether a 2D point lies inside or on the boundary of the polygon."""
        if not self.bounding_box.contains_point(pt):
            return False

        # Check if point lies directly on any boundary edge
        for edge in self.edges():
            if edge.distance_to_point(pt) < 1e-6:
                return True

        n = len(self.vertices)
        inside = False
        p1 = self.vertices[0]
        for i in range(1, n + 1):
            p2 = self.vertices[i % n]
            if pt.y > min(p1.y, p2.y):
                if pt.y <= max(p1.y, p2.y):
                    if pt.x <= max(p1.x, p2.x):
                        if p1.y != p2.y:
                            x_inters = (pt.y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) + p1.x
                        if p1.x == p2.x or pt.x <= x_inters:
                            inside = not inside
            p1 = p2
        return inside

    def edges(self) -> List[LineSegment2D]:
        """Returns list of bounding line segments forming the polygon contour."""
        n = len(self.vertices)
        if n < 2:
            return []
        return [LineSegment2D(self.vertices[i], self.vertices[(i + 1) % n]) for i in range(n)]

    def has_self_intersections(self) -> bool:
        """Checks for non-adjacent edge intersections."""
        edges = self.edges()
        m = len(edges)
        for i in range(m):
            for j in range(i + 2, m):
                # Skip adjacent edges that share a vertex
                if i == 0 and j == m - 1:
                    continue
                if edges[i].intersects(edges[j]):
                    return True
        return False

    def distance_to_point(self, pt: Point2D) -> float:
        """Calculates minimum distance from a point to the polygon boundary."""
        if self.contains_point(pt):
            return 0.0
        min_d = float("inf")
        for edge in self.edges():
            d = edge.distance_to_point(pt)
            if d < min_d:
                min_d = d
        return min_d if min_d != float("inf") else 0.0

    def to_coordinates_list(self, closed: bool = True) -> List[List[float]]:
        """Returns coordinate pairs as [[x, y], ...]."""
        coords = [[round(v.x, 3), round(v.y, 3)] for v in self.vertices]
        if closed and coords and coords[0] != coords[-1]:
            coords.append(coords[0])
        return coords

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vertices": [v.to_dict() for v in self.vertices],
            "area_sq_m": round(self.area, 3),
            "perimeter_m": round(self.perimeter, 3),
            "centroid": self.centroid.to_dict(),
            "bounding_box": self.bounding_box.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Polygon2D":
        raw_verts = data.get("vertices", [])
        return cls(vertices=[Point2D.from_dict(v) for v in raw_verts])

    @classmethod
    def from_tuples(cls, coords: Sequence[Tuple[float, float]]) -> "Polygon2D":
        return cls(vertices=[Point2D.from_tuple(c) for c in coords])

    @classmethod
    def create_rectangle(
        cls,
        origin_x: Optional[float] = None,
        origin_y: Optional[float] = None,
        width_m: Optional[float] = None,
        height_m: Optional[float] = None,
        min_x: Optional[float] = None,
        min_y: Optional[float] = None,
        max_x: Optional[float] = None,
        max_y: Optional[float] = None,
    ) -> "Polygon2D":
        """Convenience constructor for a metric axis-aligned rectangular room.

        Supports either (origin_x, origin_y, width_m, height_m) or (min_x, min_y, max_x, max_y).
        """
        if min_x is not None and max_x is not None and min_y is not None and max_y is not None:
            ox = float(min_x)
            oy = float(min_y)
            w = float(max_x - min_x)
            h = float(max_y - min_y)
        elif origin_x is not None and origin_y is not None and width_m is not None and height_m is not None:
            ox = float(origin_x)
            oy = float(origin_y)
            w = float(width_m)
            h = float(height_m)
        else:
            raise ValueError("Must provide either (min_x, min_y, max_x, max_y) or (origin_x, origin_y, width_m, height_m)")

        return cls(
            vertices=[
                Point2D(ox, oy),
                Point2D(ox + w, oy),
                Point2D(ox + w, oy + h),
                Point2D(ox, oy + h),
            ]
        )
