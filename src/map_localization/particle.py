"""Particle state entity representation for indoor particle filtering."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple


def wrap_heading_rad(theta: float) -> float:
    """Wraps heading angle in radians to [-pi, pi]."""
    return math.atan2(math.sin(theta), math.cos(theta))


class Particle:
    """Represents a single hypothesis of 2D location, heading, and floor within a building."""

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        theta: float = 0.0,
        floor_id: int = 0,
        weight: float = 1.0,
        log_weight: float = 0.0,
        building_id: str = "BLDG_01",
        is_alive: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if "x_m" in kwargs:
            x = float(kwargs["x_m"])
        if "y_m" in kwargs:
            y = float(kwargs["y_m"])
        if "theta_rad" in kwargs:
            theta = float(kwargs["theta_rad"])

        self.x = float(x)
        self.y = float(y)
        self.theta = wrap_heading_rad(float(theta))
        self.floor_id = int(floor_id)
        self.weight = float(weight)
        self.log_weight = float(log_weight)
        self.building_id = str(building_id)
        self.is_alive = bool(is_alive)
        self.metadata: Dict[str, Any] = dict(metadata or {})

    @property
    def x_m(self) -> float:
        return self.x

    @x_m.setter
    def x_m(self, val: float) -> None:
        self.x = float(val)

    @property
    def y_m(self) -> float:
        return self.y

    @y_m.setter
    def y_m(self, val: float) -> None:
        self.y = float(val)

    @property
    def theta_rad(self) -> float:
        return self.theta

    @theta_rad.setter
    def theta_rad(self, val: float) -> None:
        self.theta = wrap_heading_rad(float(val))

    def copy(self) -> "Particle":
        """Creates a deep copy of this particle."""
        return Particle(
            x=self.x,
            y=self.y,
            theta=self.theta,
            floor_id=self.floor_id,
            weight=self.weight,
            log_weight=self.log_weight,
            building_id=self.building_id,
            is_alive=self.is_alive,
            metadata=dict(self.metadata),
        )

    def distance_to(self, target_x: float, target_y: float) -> float:
        """Euclidean 2D planar distance to target coordinates."""
        dx = self.x - target_x
        dy = self.y - target_y
        return math.hypot(dx, dy)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "theta_rad": round(self.theta, 4),
            "floor_id": self.floor_id,
            "weight": self.weight,
            "log_weight": self.log_weight,
            "building_id": self.building_id,
            "is_alive": self.is_alive,
        }

    def __repr__(self) -> str:
        return f"Particle(x={self.x:.2f}, y={self.y:.2f}, theta={self.theta:.2f}, floor={self.floor_id}, w={self.weight:.4f})"
