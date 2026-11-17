"""Coordinate frame transformations and attitude representation for Pedestrian Dead Reckoning (Task 21).

Coordinate Systems Supported:
1. Body Frame (B): Smartphone reference (X=right, Y=top, Z=out of screen).
2. Navigation Frame (W): East-North-Up (ENU) geodetic/local reference.
3. Gravity-Aligned Frame (G): Z-axis aligned with true vertical (opposite gravity), X-Y horizontal plane.
4. Horizontal Movement Plane (H): 2D Cartesian plane (X=East/Right, Y=North/Forward).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def quaternion_from_android_rotation_vector(x: float, y: float, z: float, w: float | None = None) -> np.ndarray:
    """Normalizes an Android rotation vector into a unit quaternion [w, x, y, z]."""
    if w is None or math.isnan(w):
        sum_sq = x * x + y * y + z * z
        w_val = math.sqrt(max(0.0, 1.0 - sum_sq)) if sum_sq <= 1.0 else 0.0
    else:
        w_val = float(w)

    q = np.array([w_val, float(x), float(y), float(z)], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm > 1e-9:
        q /= norm
    else:
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Converts a unit quaternion [w, x, y, z] to a 3x3 rotation matrix R (body to navigation)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def quaternion_to_euler_angles(q: np.ndarray) -> Tuple[float, float, float]:
    """Extracts roll, pitch, and yaw (in radians) from quaternion [w, x, y, z].

    Returns:
        (roll, pitch, yaw) in radians.
        roll: rotation around X axis [-pi, pi]
        pitch: rotation around Y axis [-pi/2, pi/2]
        yaw: rotation around Z axis (azimuth) [-pi, pi]
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation / azimuth)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def rotate_body_to_world(v_body: np.ndarray, R_body_to_world: np.ndarray) -> np.ndarray:
    """Rotates a 3D vector from body frame B to world/navigation frame W."""
    return R_body_to_world @ v_body


def rotate_world_to_body(v_world: np.ndarray, R_body_to_world: np.ndarray) -> np.ndarray:
    """Rotates a 3D vector from world/navigation frame W to body frame B."""
    return R_body_to_world.T @ v_world


def project_to_horizontal_plane(v: np.ndarray, gravity_unit_vector: np.ndarray) -> Tuple[np.ndarray, float]:
    """Decomposes a 3D vector into horizontal and vertical components relative to gravity.

    Args:
        v: 3D vector in body or world coordinates.
        gravity_unit_vector: Normalized unit vector pointing in the direction of gravity.

    Returns:
        (v_horizontal, v_vertical_scalar)
        where v_horizontal is orthogonal to gravity, and v_vertical_scalar is component along -gravity (upward).
    """
    g_unit = gravity_unit_vector / np.linalg.norm(gravity_unit_vector)
    # Component along gravity vector
    v_along_g = np.dot(v, g_unit)
    v_vertical_proj = v_along_g * g_unit
    v_horizontal = v - v_vertical_proj
    # Upward vertical scalar (positive upward against gravity)
    v_vertical_upward = -v_along_g
    return v_horizontal, float(v_vertical_upward)
