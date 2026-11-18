"""Rigorous 2D spatial uncertainty auditor, Chi-square confidence ellipse computation, and ground-truth coverage testing (Task 22)."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from src.core.logging import setup_logger
from src.pdr.calibration.models import ConfidenceEllipse, UncertaintyEvaluationMetrics

logger = setup_logger("UncertaintyAuditor")

# Chi-Square 95% Quantile for 2 Degrees of Freedom: -2 * ln(1 - 0.95)
CHI2_2DOF_95 = 5.9914645471


def compute_confidence_ellipse_2d(cov_matrix_2x2: np.ndarray, quantile: float = CHI2_2DOF_95) -> ConfidenceEllipse:
    """Computes exact 2D Gaussian confidence ellipse semi-axes and orientation angle from 2x2 spatial covariance.

    Uses eigenvalue decomposition and Chi-square quantile for 2 degrees of freedom (5.991 for 95%).
    """
    p_xx = float(cov_matrix_2x2[0, 0])
    p_yy = float(cov_matrix_2x2[1, 1])
    p_xy = float(cov_matrix_2x2[0, 1])

    # Analytical eigenvalue calculation for 2x2 symmetric matrix
    mean_diag = (p_xx + p_yy) / 2.0
    half_diff = (p_xx - p_yy) / 2.0
    term = math.sqrt(max(0.0, half_diff**2 + p_xy**2))

    lambda_1 = max(0.0, mean_diag + term)
    lambda_2 = max(0.0, mean_diag - term)

    semi_major = math.sqrt(quantile * lambda_1)
    semi_minor = math.sqrt(quantile * lambda_2)

    # Orientation angle of semi-major axis relative to X-axis
    angle_rad = 0.5 * math.atan2(2.0 * p_xy, p_xx - p_yy)
    angle_deg = math.degrees(angle_rad)

    return ConfidenceEllipse(
        semi_major_axis_m=round(semi_major, 3),
        semi_minor_axis_m=round(semi_minor, 3),
        orientation_deg=round(angle_deg, 2),
        chi2_quantile=round(quantile, 3),
        conservative_radius_m=round(semi_major, 3)
    )


def is_point_inside_ellipse(
    point: Tuple[float, float],
    center: Tuple[float, float],
    cov_matrix_2x2: np.ndarray,
    quantile: float = CHI2_2DOF_95
) -> bool:
    """Checks whether point (x, y) falls within the 2D confidence ellipse centered at (cx, cy).

    Computes Mahalanobis distance squared: (p - c)^T * P^(-1) * (p - c) <= quantile.
    """
    dx = point[0] - center[0]
    dy = point[1] - center[1]

    det = float(cov_matrix_2x2[0, 0] * cov_matrix_2x2[1, 1] - cov_matrix_2x2[0, 1]**2)
    if det <= 1e-12:
        # Singular covariance: check distance against major axis
        dist = math.sqrt(dx**2 + dy**2)
        major = math.sqrt(quantile * max(float(cov_matrix_2x2[0, 0]), float(cov_matrix_2x2[1, 1])))
        return dist <= major

    p_xx = float(cov_matrix_2x2[0, 0])
    p_yy = float(cov_matrix_2x2[1, 1])
    p_xy = float(cov_matrix_2x2[0, 1])

    # Inverse of 2x2 matrix
    inv_xx = p_yy / det
    inv_yy = p_xx / det
    inv_xy = -p_xy / det

    mahalanobis_sq = dx * (inv_xx * dx + inv_xy * dy) + dy * (inv_xy * dx + inv_yy * dy)
    return mahalanobis_sq <= quantile


class UncertaintyAuditor:
    """Audits spatial covariance growth and validates empirical coverage against ground truth."""

    def audit_trajectory_uncertainty(
        self,
        final_covariance: np.ndarray,
        estimated_endpoint: Tuple[float, float],
        ground_truth_endpoint: Tuple[float, float],
        intermediate_points: Optional[List[Tuple[float, float]]] = None,
        intermediate_covariances: Optional[List[np.ndarray]] = None,
        intermediate_truth: Optional[List[Tuple[float, float]]] = None
    ) -> UncertaintyEvaluationMetrics:
        """Evaluates final covariance, confidence ellipse, and empirical containment ratio."""
        ellipse = compute_confidence_ellipse_2d(final_covariance)
        within_ellipse = is_point_inside_ellipse(
            point=ground_truth_endpoint,
            center=estimated_endpoint,
            cov_matrix_2x2=final_covariance
        )

        coverage_ratio = None
        if intermediate_points and intermediate_covariances and intermediate_truth:
            n = min(len(intermediate_points), len(intermediate_covariances), len(intermediate_truth))
            if n > 0:
                inside_count = 0
                for i in range(n):
                    if is_point_inside_ellipse(
                        point=intermediate_truth[i],
                        center=intermediate_points[i],
                        cov_matrix_2x2=intermediate_covariances[i]
                    ):
                        inside_count += 1
                coverage_ratio = round(float(inside_count / n), 3)

        trace_val = float(final_covariance[0, 0] + final_covariance[1, 1])

        return UncertaintyEvaluationMetrics(
            final_covariance_trace=round(trace_val, 4),
            confidence_ellipse=ellipse,
            ground_truth_within_95_ellipse=within_ellipse,
            empirical_coverage_ratio=coverage_ratio
        )
