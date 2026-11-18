"""Building-local coordinate conventions, transformations, and angular utilities for sensor fusion (Task 24).

Defines standard coordinate conventions:
- Metric Cartesian 2D coordinate system (x, y) in meters (m).
- Building-local X-axis: Longitudinal / East axis (+X).
- Building-local Y-axis: Transverse / North axis (+Y).
- Heading angle theta: In radians [rad] in [-pi, pi], measured counter-clockwise from positive X-axis.
- Zero heading: Facing +X (East).
- Rotation direction: Positive counter-clockwise (+Y is +pi/2, North).
- Compass Azimuth conversion: True/Magnetic compass azimuth psi in [0, 360) deg clockwise from North
  maps to local theta via: theta = (pi/2 - psi_rad).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def wrap_angle_rad(angle_rad: float) -> float:
    """Normalize an angle in radians into the principal interval [-pi, pi]."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def wrap_angle_deg(angle_deg: float) -> float:
    """Normalize an angle in degrees into the principal interval [-180.0, 180.0]."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def compass_azimuth_to_local_heading_rad(azimuth_deg: float) -> float:
    """Convert clockwise compass azimuth (0 deg = North, 90 deg = East) to local Cartesian heading.

    Local Cartesian convention: 0 rad = +X (East), +pi/2 rad = +Y (North).
    """
    psi_rad = math.radians(azimuth_deg)
    # North (psi=0) -> +Y (pi/2)
    # East (psi=pi/2) -> +X (0)
    theta = math.pi / 2.0 - psi_rad
    return wrap_angle_rad(theta)


def local_heading_rad_to_compass_azimuth(theta_rad: float) -> float:
    """Convert local Cartesian heading (0 rad = +X East, +pi/2 = +Y North) to compass azimuth."""
    psi_rad = math.pi / 2.0 - theta_rad
    psi_deg = math.degrees(psi_rad) % 360.0
    return round(psi_deg, 2)


@dataclass
class CoordinateFrameMetadata:
    """Documents and validates the spatial reference frame used for sensor fusion."""
    frame_id: str = "BUILDING_LOCAL_CARTESIAN_ENU"
    units: str = "meters"
    angle_units: str = "radians"
    x_axis_description: str = "Building Local East (+X)"
    y_axis_description: str = "Building Local North (+Y)"
    heading_zero_description: str = "Facing +X (East), counter-clockwise positive"
    building_id: Optional[str] = None
    floor_id: Optional[int] = None
    origin_description: Optional[str] = None

    def validate_compatibility(self, other: CoordinateFrameMetadata) -> Tuple[bool, Optional[str]]:
        """Validate whether two observations share compatible coordinate frames."""
        if self.units != other.units:
            return False, f"Incompatible distance units: '{self.units}' vs '{other.units}'"
        if self.angle_units != other.angle_units:
            return False, f"Incompatible angle units: '{self.angle_units}' vs '{other.angle_units}'"
        if self.building_id and other.building_id and self.building_id != other.building_id:
            return False, f"Building mismatch: '{self.building_id}' vs '{other.building_id}'"
        if self.floor_id is not None and other.floor_id is not None and self.floor_id != other.floor_id:
            return False, f"Floor mismatch: Floor {self.floor_id} vs Floor {other.floor_id}"
        return True, None


@dataclass
class Pose2D:
    """Represents a 2D pose (x, y, theta) with building/floor context."""
    x_m: float
    y_m: float
    theta_rad: float
    building_id: str = "DEFAULT_BUILDING"
    floor_id: int = 0

    def __post_init__(self):
        self.theta_rad = wrap_angle_rad(float(self.theta_rad))
        self.x_m = float(self.x_m)
        self.y_m = float(self.y_m)

    @property
    def theta_deg(self) -> float:
        return round(float(np.degrees(self.theta_rad)), 2)

    def to_array(self) -> np.ndarray:
        return np.array([self.x_m, self.y_m, self.theta_rad], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray, building_id: str = "DEFAULT_BUILDING", floor_id: int = 0) -> Pose2D:
        return cls(
            x_m=float(arr[0]),
            y_m=float(arr[1]),
            theta_rad=float(arr[2]),
            building_id=building_id,
            floor_id=floor_id,
        )
