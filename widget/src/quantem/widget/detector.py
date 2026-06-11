"""Detector utilities for 4D-STEM virtual imaging.

Provides BF disk detection and BF/ADF/HAADF virtual image computation.
The detection algorithm matches Show4DSTEM.auto_detect_center (sum all
diffraction patterns, threshold at mean+std, centroid, radius from area).
"""

from typing import Any

import numpy as np

from quantem.widget.array_utils import to_numpy


def detect_bf_disk(data: Any) -> tuple[float, float, float]:
    """Detect BF disk center and radius from a 4D-STEM dataset.

    Uses the same algorithm as Show4DSTEM.auto_detect_center:
    sum all diffraction patterns, threshold at mean+std, compute centroid,
    estimate radius from disk area (A = pi * r^2).

    Parameters
    ----------
    data : array-like
        4D array ``(scan_rows, scan_cols, det_rows, det_cols)`` or
        5D array ``(n_frames, scan_rows, scan_cols, det_rows, det_cols)``.
        Accepts NumPy, PyTorch, or CuPy.

    Returns
    -------
    center_row, center_col, bf_radius : float
        Detected center position and estimated BF disk radius in pixels.
    """
    arr = to_numpy(data)
    det_rows, det_cols = arr.shape[-2], arr.shape[-1]
    summed = arr.reshape(-1, det_rows, det_cols).sum(axis=0)
    threshold = summed.mean() + summed.std()
    disk = summed > threshold
    total = disk.sum()
    if total > 0:
        row_coords = np.arange(det_rows, dtype=np.float64).reshape(-1, 1)
        col_coords = np.arange(det_cols, dtype=np.float64).reshape(1, -1)
        center_row = float((row_coords * disk).sum() / total)
        center_col = float((col_coords * disk).sum() / total)
        bf_radius = float(np.sqrt(total / np.pi))
    else:
        center_row = det_rows / 2.0
        center_col = det_cols / 2.0
        bf_radius = det_rows / 6.0
    return center_row, center_col, bf_radius


def make_virtual_masks(
    det_rows: int,
    det_cols: int,
    center_row: float,
    center_col: float,
    bf_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create BF, ADF, HAADF annular masks for virtual imaging.

    - BF: r < bf_radius
    - ADF: bf_radius <= r < 2 * bf_radius
    - HAADF: r >= 2 * bf_radius

    Parameters
    ----------
    det_rows, det_cols : int
        Detector dimensions.
    center_row, center_col : float
        BF disk center in (row, col) coordinates.
    bf_radius : float
        Estimated BF disk radius.

    Returns
    -------
    bf_mask, adf_mask, haadf_mask : np.ndarray
        Float32 masks with shape ``(det_rows, det_cols)``.
    """
    row_coords = np.arange(det_rows, dtype=np.float64).reshape(-1, 1)
    col_coords = np.arange(det_cols, dtype=np.float64).reshape(1, -1)
    rad = np.sqrt((row_coords - center_row) ** 2 + (col_coords - center_col) ** 2)
    bf_mask = (rad < bf_radius).astype(np.float32)
    adf_mask = ((rad >= bf_radius) & (rad < bf_radius * 2)).astype(np.float32)
    haadf_mask = (rad >= bf_radius * 2).astype(np.float32)
    return bf_mask, adf_mask, haadf_mask


def virtual_images(
    data: Any,
    center: tuple[float, float] | None = None,
    bf_radius: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute BF, ADF, HAADF virtual images from 4D/5D-STEM data.

    Auto-detects the BF disk center and radius if not provided.

    Parameters
    ----------
    data : array-like
        4D ``(scan_r, scan_c, det_r, det_c)`` or
        5D ``(n_frames, scan_r, scan_c, det_r, det_c)``.
    center : (row, col), optional
        BF disk center. Auto-detected if None.
    bf_radius : float, optional
        BF disk radius. Auto-detected if None.

    Returns
    -------
    bf, adf, haadf : np.ndarray
        Virtual images. Same leading dimensions as input minus the
        last two (detector) axes.

    Examples
    --------
    >>> from quantem.widget import virtual_images
    >>> bf, adf, haadf = virtual_images(data_4d)
    >>> Show2D(bf, title="BF")
    """
    arr = to_numpy(data, dtype=np.float32)
    if center is None or bf_radius is None:
        cr, cc, br = detect_bf_disk(arr)
        if center is not None:
            cr, cc = center
        if bf_radius is not None:
            br = bf_radius
    else:
        cr, cc = center
        br = bf_radius
    bf_mask, adf_mask, haadf_mask = make_virtual_masks(
        arr.shape[-2], arr.shape[-1], cr, cc, br,
    )
    bf = (arr * bf_mask).mean(axis=(-2, -1))
    adf = (arr * adf_mask).mean(axis=(-2, -1))
    haadf = (arr * haadf_mask).mean(axis=(-2, -1))
    return bf, adf, haadf
