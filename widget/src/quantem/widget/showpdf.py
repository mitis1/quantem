import base64
import json
import pathlib
import time
from typing import Self

import anywidget
import numpy as np
import traitlets

from quantem.core.datastructures import Dataset2d, Dataset4dstem
from quantem.diffraction import PairDistributionFunction
from quantem.widget.array_utils import to_numpy
from quantem.widget.detector import virtual_images
from quantem.widget.state import (
    resolve_widget_version,
    save_state_file,
    unwrap_state_payload,
)


class ShowPDF(anywidget.AnyWidget):
    """Interactive pair distribution function (PDF) analysis widget for 4D-STEM data.

    Wraps ``PairDistributionFunction`` from ``quantem.diffraction``. This widget provides:
    - A real-space navigation image with masking function (left panel)
    - Real-time 1D curve plots for I(k)+B(k), windowed F(k), and G(r) (right panel)
    - Tunable PDF parameters (k-range, window, damping) with live feedback
    """

    _esm = pathlib.Path(__file__).parent / "static" / "showpdf.js"

    # =========================================================================
    # Version
    # =========================================================================
    widget_version = traitlets.Unicode("unknown").tag(sync=True)

    # =========================================================================
    # Shape / data (read-only from JS)
    # =========================================================================
    title = traitlets.Unicode("PDF").tag(sync=True)
    scan_rows = traitlets.Int(1).tag(sync=True)
    scan_cols = traitlets.Int(1).tag(sync=True)
    nav_image_bytes = traitlets.Bytes(b"").tag(sync=True)
    nav_data_min = traitlets.Float(0.0).tag(sync=True)
    nav_data_max = traitlets.Float(1.0).tag(sync=True)
    # Real-space scan calibration for the nav panel scalebar. Set from
    # input_data.sampling[0] / units[0] when the dataset is loaded; defaults
    # to "px" with size 1.0 when no calibration is present.
    nav_pixel_size = traitlets.Float(1.0).tag(sync=True)
    nav_unit = traitlets.Unicode("px").tag(sync=True)

    # =========================================================================
    # Mask (bidirectional — JS paints, Python reads)
    # Convention: 1 = include, 0 = exclude (matches calculate_radial_mean)
    # =========================================================================
    mask_bytes = traitlets.Bytes(b"").tag(sync=True)  # Python→JS only
    mask_b64 = traitlets.Unicode("").tag(sync=True)  # JS→Python (base64-encoded mask)
    mask_version = traitlets.Int(0).tag(sync=True)   # JS increments to trigger recompute
    mask_tool = traitlets.Unicode("rectangle").tag(sync=True)
    mask_brush_size = traitlets.Int(5).tag(sync=True)
    mask_pixel_count = traitlets.Int(0).tag(sync=True)
    mask_fraction = traitlets.Float(0.0).tag(sync=True)

    # =========================================================================
    # Analysis mode: "mask" (existing painted mask) or "probe" (single moveable
    # square region of side 2*probe_size-1 centered on the scan grid).
    # In probe mode, mask_b64 is ignored and a synthetic mask is built from
    # probe_row/probe_col/probe_size.
    # =========================================================================
    analysis_mode = traitlets.Unicode("mask").tag(sync=True)
    probe_row = traitlets.Int(0).tag(sync=True)
    probe_col = traitlets.Int(0).tag(sync=True)
    probe_size = traitlets.Int(1).tag(sync=True)

    # =========================================================================
    # Line mode: a straight selection band across the scan. The two endpoints
    # (row0,col0)-(row1,col1) and line_width (band thickness in scan px) define
    # a boolean mask of every scan position within line_width/2 of the segment.
    # line_active is False until a line is drawn. line_mode selects "averaged"
    # (one PDF from the whole band) or "linescan" (per-position PDFs — Phase 2).
    # JS bumps line_version after editing endpoints to trigger a single recompute.
    # =========================================================================
    line_row0 = traitlets.Float(0.0).tag(sync=True)
    line_col0 = traitlets.Float(0.0).tag(sync=True)
    line_row1 = traitlets.Float(0.0).tag(sync=True)
    line_col1 = traitlets.Float(0.0).tag(sync=True)
    line_width = traitlets.Int(3).tag(sync=True)
    line_active = traitlets.Bool(False).tag(sync=True)
    line_mode = traitlets.Unicode("averaged").tag(sync=True)
    line_version = traitlets.Int(0).tag(sync=True)
    # When True the drawn line is a reference (e.g. traced along a feature) and
    # the band/line-scan is sampled along its perpendicular bisector instead —
    # so the scan is perpendicular to the traced feature regardless of frame.
    line_perpendicular = traitlets.Bool(False).tag(sync=True)

    # Line-scan output (line_mode="linescan"): a per-position PDF stack along
    # the line. linescan_bytes is float32 (n_linescan × linescan_n_points),
    # row-major by bin; linescan_axis_bytes is the shared r/k axis (the curve
    # selected by plot_mode). Empty until a line is drawn in linescan mode.
    linescan_bytes = traitlets.Bytes(b"").tag(sync=True)
    n_linescan = traitlets.Int(0).tag(sync=True)
    linescan_n_points = traitlets.Int(0).tag(sync=True)
    linescan_axis_bytes = traitlets.Bytes(b"").tag(sync=True)
    linescan_max_bins = traitlets.Int(256).tag(sync=True)

    # =========================================================================
    # 1D curve data (synced to JS, all updated on every recompute)
    # =========================================================================
    # I(k) radial mean + background fit
    ik_x_bytes = traitlets.Bytes(b"").tag(sync=True)
    ik_y_bytes = traitlets.Bytes(b"").tag(sync=True)
    ik_bg_y_bytes = traitlets.Bytes(b"").tag(sync=True)
    n_points_ik = traitlets.Int(0).tag(sync=True)
    # F(k) windowed (Lorch)
    fk_x_bytes = traitlets.Bytes(b"").tag(sync=True)
    fk_y_bytes = traitlets.Bytes(b"").tag(sync=True)
    n_points_fk = traitlets.Int(0).tag(sync=True)
    # G(r) reduced PDF
    gr_x_bytes = traitlets.Bytes(b"").tag(sync=True)
    gr_y_bytes = traitlets.Bytes(b"").tag(sync=True)
    n_points_gr = traitlets.Int(0).tag(sync=True)
    # g(r) pair distribution function
    pdf_x_bytes = traitlets.Bytes(b"").tag(sync=True)
    pdf_y_bytes = traitlets.Bytes(b"").tag(sync=True)
    n_points_pdf = traitlets.Int(0).tag(sync=True)

    # =========================================================================
    # PDF parameters (user-tunable, trigger recompute)
    # 0.0 sentinel means "auto" or "disabled"
    # =========================================================================
    k_min_fit = traitlets.Float(0.0).tag(sync=True)
    k_max_fit = traitlets.Float(0.0).tag(sync=True)
    k_min_window = traitlets.Float(0.0).tag(sync=True)
    k_max_window = traitlets.Float(0.0).tag(sync=True)
    k_lowpass = traitlets.Float(0.0).tag(sync=True)
    k_highpass = traitlets.Float(0.0).tag(sync=True)
    r_min = traitlets.Float(0.0).tag(sync=True)
    r_max = traitlets.Float(20.0).tag(sync=True)
    r_step = traitlets.Float(0.02).tag(sync=True)
    damp_origin_oscillations = traitlets.Bool(False).tag(sync=True)
    r_cut = traitlets.Float(1.0).tag(sync=True)

    # Density for g(r) computation. mode: "estimated" (auto via estimate_density,
    # shares r_cut with damping) or "manual" (user-supplied density_value).
    density_mode = traitlets.Unicode("estimated").tag(sync=True)
    density_value = traitlets.Float(0.05).tag(sync=True)

    # K-range metadata (read-only, set once from data, used for slider bounds)
    k_min_available = traitlets.Float(0.0).tag(sync=True)
    k_max_available = traitlets.Float(10.0).tag(sync=True)

    # =========================================================================
    # Display
    # =========================================================================
    plot_mode = traitlets.Unicode("Gr").tag(sync=True)
    show_background = traitlets.Bool(True).tag(sync=True)
    cmap = traitlets.Unicode("gray").tag(sync=True)
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(True).tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)

    # Status
    computing = traitlets.Bool(False).tag(sync=True)
    status_message = traitlets.Unicode("").tag(sync=True)

    # =========================================================================
    # Constructor
    # =========================================================================
    def __init__(
        self,
        data,
        *,
        nav_image=None,
        title="",
        # PDF parameters
        k_min_fit=0.0,
        k_max_fit=0.0,
        k_min_window=0.0,
        k_max_window=0.0,
        k_lowpass=0.0,
        k_highpass=0.0,
        r_min=0.0,
        r_max=20.0,
        r_step=0.02,
        damp_origin_oscillations=False,
        r_cut=1.0,
        # Density (for g(r))
        density_mode="estimated",
        density_value=0.05,
        # Analysis mode + probe geometry
        analysis_mode="mask",
        probe_row=None,
        probe_col=None,
        probe_size=1,
        # Line geometry (analysis_mode="line")
        line_row0=0.0,
        line_col0=0.0,
        line_row1=0.0,
        line_col1=0.0,
        line_width=3,
        line_mode="averaged",
        line_perpendicular=False,
        line_max_bins=256,
        # Display
        plot_mode="Gr",
        show_background=True,
        cmap="gray",
        log_scale=False,
        auto_contrast=True,
        show_stats=True,
        show_controls=True,
        # PDF construction params (only used when data is not already a PDF)
        find_origin=True,
        origin_row=None,
        origin_col=None,
        num_annular_bins=180,
        radial_min=0.0,
        radial_max=None,
        radial_step=1.0,
        two_fold_rotation_symmetry=False,
        device=None,
        # Real-space scale bar (overrides auto-detect from ds.sampling/units)
        pixel_size=None,
        # State
        state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._initializing = True
        self.widget_version = resolve_widget_version()
        _t0 = time.perf_counter()

        # --- Input dispatch ---
        _extracted_title = ""
        if isinstance(data, PairDistributionFunction):
            self._pdf = data
        else:
            # Duck-type IOResult / Dataset
            if hasattr(data, "title") and hasattr(data, "data"):
                _extracted_title = data.title or ""
                data = data.data
            if hasattr(data, "array") and hasattr(data, "name"):
                _extracted_title = _extracted_title or (data.name or "")

            # Wrap raw NumPy/Torch/CuPy arrays in a Dataset so PairDistributionFunction
            # can consume them. Datasets (which carry .array) pass through unchanged.
            if not hasattr(data, "array"):
                arr = to_numpy(data)
                if arr.ndim == 4:
                    data = Dataset4dstem.from_array(arr)
                elif arr.ndim == 2:
                    data = Dataset2d.from_array(arr)
                else:
                    raise ValueError(
                        f"ShowPDF expects 4D (4D-STEM) or 2D input; got {arr.ndim}D array."
                    )

            # Resolve device. PairDistributionFunction uses float64 tensors
            # internally, which MPS does not support and which AMD/embedded
            # GPU drivers handle inconsistently. The compute is light (1D
            # FFTs, radial mean, iterative density solver), so default to CPU
            # for cross-platform reliability. Users can still pass
            # device="cuda" explicitly if they have a confirmed-working GPU.
            if device is None:
                device_str = "cpu"
            else:
                device_str = device

            self._pdf = PairDistributionFunction.from_data(
                data,
                find_origin=find_origin,
                origin_row=origin_row,
                origin_col=origin_col,
                num_annular_bins=num_annular_bins,
                radial_min=radial_min,
                radial_max=radial_max,
                radial_step=radial_step,
                two_fold_rotation_symmetry=two_fold_rotation_symmetry,
                device=device_str,
            )

        # Clear any cached intermediates carried in from preprocessing or a
        # serialized cache. calculate_Gr reuses a cached background/radial mean
        # when no mask is passed, so stale caches would make the initial compute
        # ignore this widget's k-fit/window settings. Clearing guarantees a fresh
        # recompute with the widget's own parameters (a loaded rdf then behaves
        # identically to a freshly constructed one).
        for _attr in ("Ik", "bg", "f", "Sk", "Fk", "Fk_masked", "rho0",
                      "_reduced_pdf", "_r", "_pdf", "reduced_pdf_damped"):
            if hasattr(self._pdf, _attr):
                setattr(self._pdf, _attr, None)

        # --- Extract shape and k-range from polar data ---
        polar_shape = self._pdf.polar.shape
        self.scan_rows = int(polar_shape[0])
        self.scan_cols = int(polar_shape[1])

        qq = np.asarray(self._pdf.qq, dtype=np.float32)
        self.k_min_available = float(qq[0])
        self.k_max_available = float(qq[-1])

        # --- Default k fit/window range to the full data range if not set ---
        if k_min_fit == 0.0:
            k_min_fit = float(qq[0])
        if k_max_fit == 0.0:
            k_max_fit = float(qq[-1])
        if k_min_window == 0.0:
            k_min_window = float(qq[0])
        if k_max_window == 0.0:
            k_max_window = float(qq[-1])

        # --- Set parameter traits ---
        self.title = title or _extracted_title or "PDF"
        self.k_min_fit = k_min_fit
        self.k_max_fit = k_max_fit
        self.k_min_window = k_min_window
        self.k_max_window = k_max_window
        self.k_lowpass = k_lowpass
        self.k_highpass = k_highpass
        self.r_min = r_min
        self.r_max = r_max
        self.r_step = r_step
        self.damp_origin_oscillations = damp_origin_oscillations
        self.r_cut = r_cut
        self.density_mode = density_mode
        self.density_value = float(density_value)
        self.analysis_mode = analysis_mode
        self.probe_row = int(probe_row) if probe_row is not None else self.scan_rows // 2
        self.probe_col = int(probe_col) if probe_col is not None else self.scan_cols // 2
        self.probe_size = int(probe_size)
        self.line_row0 = float(line_row0)
        self.line_col0 = float(line_col0)
        self.line_row1 = float(line_row1)
        self.line_col1 = float(line_col1)
        self.line_width = int(line_width)
        self.line_mode = line_mode
        self.line_perpendicular = bool(line_perpendicular)
        self.linescan_max_bins = int(line_max_bins)
        # A line is "active" once its endpoints span a nonzero length.
        self.line_active = (line_row0, line_col0) != (line_row1, line_col1)

        # --- Set display traits ---
        self.plot_mode = plot_mode
        self.show_background = show_background
        self.cmap = cmap
        self.log_scale = log_scale
        self.auto_contrast = auto_contrast
        self.show_stats = show_stats
        self.show_controls = show_controls

        # --- Navigation image ---
        if nav_image is not None:
            nav_img = to_numpy(nav_image).astype(np.float32)
        else:
            nav_img = self._compute_nav_image()
        self._nav_image = nav_img
        self.nav_data_min = float(nav_img.min())
        self.nav_data_max = float(nav_img.max())
        self.nav_image_bytes = nav_img.tobytes()
        self.nav_pixel_size, self.nav_unit = self._resolve_nav_calibration()
        if pixel_size is not None:
            self.nav_pixel_size = float(pixel_size)
            self.nav_unit = "Å"

        # --- Register observers ---
        self.observe(self._on_mask_change, names=["mask_version"])
        self.observe(
            self._on_fit_params_change,
            names=[
                "k_min_fit",
                "k_max_fit",
                "k_min_window",
                "k_max_window",
                "k_lowpass",
                "k_highpass",
            ],
        )
        self.observe(
            self._on_output_params_change,
            names=[
                "r_min",
                "r_max",
                "r_step",
                "damp_origin_oscillations",
                "r_cut",
            ],
        )
        self.observe(
            self._on_density_change,
            names=["density_mode", "density_value"],
        )
        self.observe(
            self._on_probe_change,
            names=["analysis_mode", "probe_row", "probe_col", "probe_size"],
        )
        # Endpoints sync without observers; JS bumps line_version once per edit
        # so a drag triggers a single recompute (not one per endpoint trait).
        self.observe(
            self._on_line_change,
            names=["line_version", "line_width", "line_mode", "line_perpendicular"],
        )
        # In line-scan mode the displayed curve drives which PDF the stack holds,
        # so a plot_mode change must recompute the stack.
        self.observe(self._on_plot_mode_change, names=["plot_mode"])

        # --- Initial computation ---
        self._initializing = False
        self._update_mask_stats()
        self._recompute_full()

        # Preserve any error message _recompute_full set instead of clobbering it.
        _elapsed = time.perf_counter() - _t0
        if not self.status_message.startswith("Error"):
            self.status_message = f"Ready ({_elapsed:.1f}s)"

        # --- Restore state ---
        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                )
            else:
                state = unwrap_state_payload(state)
            self.load_state_dict(state)

    # =========================================================================
    # Observers
    # =========================================================================
    def _compute_nav_image(self) -> np.ndarray:
        """Compute a BF virtual image for the nav panel.

        Uses the original 4D-STEM data if available (via input_data),
        otherwise falls back to summing the polar data over angle and radius.
        """
        input_data = getattr(self._pdf, "input_data", None)
        if input_data is not None:
            arr = getattr(input_data, "array", None)
            if arr is not None and arr.ndim == 4:
                bf, _, _ = virtual_images(arr)
                return bf.astype(np.float32)
        # Fallback: sum polar data over phi and r
        return self._pdf.polar.numpy().sum(axis=(-2, -1)).astype(np.float32)

    def _resolve_nav_calibration(self) -> tuple[float, str]:
        """Resolve real-space scan calibration from the input dataset.

        Returns (pixel_size, unit) where unit is "Å" if the dataset's first
        axis is in Å/A/angstrom/nm (with nm→Å conversion), else "px".
        """
        input_data = getattr(self._pdf, "input_data", None)
        sampling = getattr(input_data, "sampling", None)
        units = getattr(input_data, "units", None)
        if sampling is None or units is None or len(units) == 0:
            return 1.0, "px"
        unit0 = str(units[0]).lower()
        size0 = float(sampling[0])
        if unit0 in ("å", "a", "angstrom", "angstroms"):
            return size0, "Å"
        if unit0 == "nm":
            return size0 * 10.0, "Å"
        return 1.0, "px"

    def _on_mask_change(self, change=None):
        if self._initializing:
            return
        self._pdf.Ik = None
        self._pdf.bg = None
        self._update_mask_stats()
        self._recompute_full()

    def _on_fit_params_change(self, change=None):
        if self._initializing:
            return
        self._pdf.bg = None
        self._recompute_full()

    def _on_output_params_change(self, change=None):
        if self._initializing:
            return
        self._pdf.bg = None
        self._recompute_full()

    def _on_density_change(self, change=None):
        if self._initializing:
            return
        # In manual mode the user-supplied value is used directly; in estimated
        # mode we invalidate the cached rho0 so estimate_density runs again
        # against the current G(r).
        if self.density_mode == "estimated":
            self._pdf.rho0 = None
        self._compute_gr()

    def _on_probe_change(self, change=None):
        if self._initializing:
            return
        # Probe defines a synthetic mask; treat as a mask change and recompute.
        self._pdf.Ik = None
        self._pdf.bg = None
        self._update_mask_stats()
        self._recompute_full()

    def _on_line_change(self, change=None):
        if self._initializing:
            return
        # The line defines a synthetic band mask; treat as a mask change.
        self._pdf.Ik = None
        self._pdf.bg = None
        self._update_mask_stats()
        self._recompute_full()

    def _on_plot_mode_change(self, change=None):
        if self._initializing:
            return
        # Only line-scan mode depends on plot_mode for its compute; the 1D
        # paths just re-draw the already-synced curves on the JS side.
        if self.analysis_mode == "line" and self.line_mode == "linescan":
            self._compute_linescan()

    # =========================================================================
    # Core computation
    # =========================================================================
    def _effective_line(self):
        """Endpoints actually sampled: the perpendicular bisector of the drawn
        line when line_perpendicular is set, else the drawn line itself.

        The bisector keeps the drawn line's length and is centered on its
        midpoint, rotated 90° — so tracing along a feature samples across it.
        """
        r0, c0 = self.line_row0, self.line_col0
        r1, c1 = self.line_row1, self.line_col1
        if not self.line_perpendicular:
            return r0, c0, r1, c1
        dr, dc = r1 - r0, c1 - c0
        length = float(np.hypot(dr, dc))
        if length == 0:
            return r0, c0, r1, c1
        mr, mc = (r0 + r1) / 2.0, (c0 + c1) / 2.0
        pr, pc = -dc / length, dr / length  # unit vector ⟂ to the drawn line
        half = length / 2.0
        return mr - pr * half, mc - pc * half, mr + pr * half, mc + pc * half

    def _line_band_mask(self):
        """Boolean mask of scan positions within line_width/2 of the line segment.

        Returns None when no line is active or the segment has zero length.
        """
        if not self.line_active:
            return None
        r0, c0, r1, c1 = self._effective_line()
        dr, dc = r1 - r0, c1 - c0
        seg_len_sq = dr * dr + dc * dc
        if seg_len_sq <= 0:
            return None
        length = np.sqrt(seg_len_sq)
        rows = np.arange(self.scan_rows, dtype=np.float64)[:, None]
        cols = np.arange(self.scan_cols, dtype=np.float64)[None, :]
        pr = rows - r0
        pc = cols - c0
        # Position along the line (0 = start, 1 = end) and perpendicular distance
        # to the *infinite* line. Requiring t in [0, 1] gives flat ends (no
        # caps); the perpendicular test gives a constant width → a rectangle.
        t = (pr * dr + pc * dc) / seg_len_sq
        perp = np.abs(pr * dc - pc * dr) / length
        half_w = max(0.5, self.line_width / 2.0)
        mask = (t >= 0.0) & (t <= 1.0) & (perp <= half_w)
        if not mask.any():
            return None
        return mask

    def _get_mask(self):
        # Line mode: synthesize a band mask along the drawn segment.
        if self.analysis_mode == "line":
            return self._line_band_mask()
        # Probe mode: synthesize a square mask centered at (probe_row, probe_col)
        # with side 2*probe_size-1 (matching the freeform brush sizing convention).
        if self.analysis_mode == "probe":
            half = max(0, int(self.probe_size) - 1)
            r0 = max(0, int(self.probe_row) - half)
            r1 = min(self.scan_rows, int(self.probe_row) + half + 1)
            c0 = max(0, int(self.probe_col) - half)
            c1 = min(self.scan_cols, int(self.probe_col) + half + 1)
            if r1 <= r0 or c1 <= c0:
                return None
            mask = np.zeros((self.scan_rows, self.scan_cols), dtype=bool)
            mask[r0:r1, c0:c1] = True
            return mask
        # Mask mode: decode painted mask
        if not self.mask_b64:
            return None
        raw = base64.b64decode(self.mask_b64)
        mask = np.frombuffer(raw, dtype=np.uint8).reshape(
            self.scan_rows, self.scan_cols
        )
        # All-ones mask is equivalent to no mask
        if mask.all():
            return None
        bool_mask = mask > 0
        if not bool_mask.any():
            return None
        return bool_mask

    def _update_mask_stats(self):
        mask = self._get_mask()
        total = self.scan_rows * self.scan_cols
        if mask is not None:
            count = int(mask.sum())
            self.mask_pixel_count = count
            self.mask_fraction = count / total
        elif self.analysis_mode in ("probe", "line"):
            # Probe out of bounds / no line drawn — nothing selected
            self.mask_pixel_count = 0
            self.mask_fraction = 0.0
        elif not self.mask_b64:
            self.mask_pixel_count = total
            self.mask_fraction = 1.0
        else:
            raw = base64.b64decode(self.mask_b64)
            count = int(np.frombuffer(raw, dtype=np.uint8).sum())
            self.mask_pixel_count = count if count > 0 else total
            self.mask_fraction = (count / total) if count > 0 else 1.0

    def _curve_params(self):
        """k-range sentinels (0.0 → None) shared by every calculate_Gr call."""
        return dict(
            k_min_fit=self.k_min_fit if self.k_min_fit > 0 else None,
            k_max_fit=self.k_max_fit if self.k_max_fit > 0 else None,
            k_min_window=self.k_min_window if self.k_min_window > 0 else None,
            k_max_window=self.k_max_window if self.k_max_window > 0 else None,
            k_lowpass=self.k_lowpass if self.k_lowpass > 0 else None,
            k_highpass=self.k_highpass if self.k_highpass > 0 else None,
            r_min=self.r_min,
            r_max=self.r_max,
            r_step=self.r_step,
            damp_origin_oscillations=self.damp_origin_oscillations,
            r_cut=self.r_cut,
        )

    def _compute_curve_for_mask(self, mask):
        """Compute the curve selected by plot_mode for one real-space mask.

        Returns (y, x) float32 arrays, or (None, None) if unavailable.
        """
        self._pdf.Ik = None
        self._pdf.bg = None
        self._pdf.calculate_Gr(mask_realspace=mask, **self._curve_params())
        return self._extract_plot_curve()

    def _extract_plot_curve(self):
        """Extract (y, x) for the current plot_mode from the already-computed
        PDF state (Ik / Fk_masked / reduced_pdf). Returns (None, None) if the
        needed quantity has not been computed."""
        mode = self.plot_mode
        if mode == "Ik":
            x = np.asarray(self._pdf.qq, dtype=np.float32)
            y = to_numpy(self._pdf.Ik).astype(np.float32)
        elif mode == "Fk":
            if self._pdf.Fk_masked is None:
                return None, None
            qq = np.asarray(self._pdf.qq, dtype=np.float32)
            y = to_numpy(self._pdf.Fk_masked).astype(np.float32)
            x = qq[: len(y)]
        elif mode == "gr":
            if self._pdf._reduced_pdf is None or self._pdf._r is None:
                return None, None
            density = float(self.density_value) if self.density_mode == "manual" else None
            self._pdf.calculate_gr(density=density, r_cut=self.r_cut)
            x = to_numpy(self._pdf._r).astype(np.float32)
            y = to_numpy(self._pdf._pdf).astype(np.float32)
        else:  # "Gr"
            if self._pdf._r is None or self._pdf._reduced_pdf is None:
                return None, None
            Gr = (
                self._pdf.reduced_pdf_damped
                if self._pdf.reduced_pdf_damped is not None
                else self._pdf._reduced_pdf
            )
            x = to_numpy(self._pdf._r).astype(np.float32)
            y = to_numpy(Gr).astype(np.float32)
        return y, x

    def _clear_linescan(self):
        self.n_linescan = 0
        self.linescan_n_points = 0
        self.linescan_bytes = b""
        self.linescan_axis_bytes = b""

    def _linescan_bin_radial_means(self, band, bin_idx, n_bins):
        """All bins' radial means I(k) in a single pass over the polar data.

        Returns (Ik_all, valid_bins): Ik_all is an (n_valid, Nk) torch tensor on
        the PDF device, valid_bins the matching non-empty bin indices. This is
        equivalent to calling calculate_radial_mean once per bin mask, but reads
        the 4D polar array only once (chunked) instead of n_bins times.
        """
        import torch

        polar = self._pdf.polar.numpy()  # (Sr, Sc, phi, Nk)
        scan_row, scan_col, _n_phi, n_k = polar.shape
        device = self._pdf.device
        # Per-position bin label, -1 outside the band.
        labels = np.where(band, bin_idx, -1).astype(np.int64)
        sums = torch.zeros(n_bins, n_k, dtype=torch.float64, device=device)
        counts = torch.zeros(n_bins, dtype=torch.float64, device=device)
        chunk = 16
        for row0 in range(0, scan_row, chunk):
            row1 = min(row0 + chunk, scan_row)
            arr = torch.as_tensor(
                np.ascontiguousarray(polar[row0:row1]), device=device
            )
            # mean over phi, flatten scan positions, accumulate into bins.
            rad = arr.mean(dim=2).reshape(-1, n_k).double()
            lbl = torch.as_tensor(labels[row0:row1].reshape(-1), device=device)
            valid = lbl >= 0
            if bool(valid.any()):
                v = lbl[valid]
                sums.index_add_(0, v, rad[valid])
                counts.index_add_(
                    0, v, torch.ones(int(valid.sum()), dtype=torch.float64, device=device)
                )
        nonempty = counts > 0
        valid_bins = torch.nonzero(nonempty, as_tuple=False).flatten().tolist()
        Ik_all = sums[nonempty] / counts[nonempty][:, None]
        return Ik_all, valid_bins

    def _compute_linescan(self):
        """Compute a per-position PDF stack along the drawn line.

        The band is split into n_linescan bins along the line direction; each
        bin averages the scan positions whose projection falls in that bin and
        whose perpendicular distance is within line_width/2. All bins' radial
        means are computed in one pass and all backgrounds are fit at once
        (batched LM); only the cheap F(k) -> windowing -> sine transform runs
        per bin. The curve selected by plot_mode is stacked into linescan_bytes
        (n_bins × n_points).
        """
        self.computing = True
        self.status_message = "Computing line-scan..."
        try:
            band = self._line_band_mask()
            if band is None:
                self._clear_linescan()
                self.status_message = ""
                return
            r0, c0, r1, c1 = self._effective_line()
            dr, dc = r1 - r0, c1 - c0
            seg_len_sq = dr * dr + dc * dc
            length = float(np.sqrt(seg_len_sq))
            n_bins = max(1, min(int(self.linescan_max_bins), int(round(length)) + 1))

            rows = np.arange(self.scan_rows, dtype=np.float64)[:, None]
            cols = np.arange(self.scan_cols, dtype=np.float64)[None, :]
            t = np.clip(((rows - r0) * dr + (cols - c0) * dc) / seg_len_sq, 0.0, 1.0)
            bin_idx = np.minimum((t * n_bins).astype(int), n_bins - 1)

            # One pass for every bin's I(k), then fit all backgrounds at once.
            Ik_all, valid_bins = self._linescan_bin_radial_means(band, bin_idx, n_bins)
            if not valid_bins:
                self._clear_linescan()
                self.status_message = ""
                return
            params = self._curve_params()
            bg_all, f_all = self._pdf.fit_bg_batched(
                Ik_all, kmin=params["k_min_fit"], kmax=params["k_max_fit"]
            )

            # Per-bin downstream reuses the precomputed I(k)/bg/f (mask=None), so
            # calculate_Gr skips the radial mean and the background refit and only
            # does F(k) -> windowing -> sine transform.
            stack = None
            axis = None
            for j, bin_i in enumerate(valid_bins):
                self._pdf.Ik = Ik_all[j]
                self._pdf.bg = bg_all[j]
                self._pdf.f = f_all[j]
                self._pdf.calculate_Gr(mask_realspace=None, **params)
                y, x = self._extract_plot_curve()
                if y is None:
                    continue
                if stack is None:
                    axis = x
                    stack = np.full((n_bins, len(y)), np.nan, dtype=np.float32)
                n = min(len(y), stack.shape[1])
                stack[bin_i, :n] = y[:n]

            if stack is None:
                self._clear_linescan()
                self.status_message = ""
                return
            self.n_linescan = int(n_bins)
            self.linescan_n_points = int(stack.shape[1])
            self.linescan_bytes = stack.tobytes()
            self.linescan_axis_bytes = np.asarray(axis, dtype=np.float32).tobytes()
            self.status_message = ""
        except Exception as e:
            self.status_message = f"Line-scan error: {e}"
            self._clear_linescan()
        finally:
            self.computing = False

    def _recompute_full(self):
        # Line-scan: compute a per-position PDF stack instead of one curve set.
        if self.analysis_mode == "line" and self.line_mode == "linescan":
            self._compute_linescan()
            return
        self.computing = True
        self.status_message = "Computing..."
        try:
            mask = self._get_mask()

            # Convert 0.0 sentinels to None for the PDF API
            k_min_fit = self.k_min_fit if self.k_min_fit > 0 else None
            k_max_fit = self.k_max_fit if self.k_max_fit > 0 else None
            k_min_window = self.k_min_window if self.k_min_window > 0 else None
            k_max_window = self.k_max_window if self.k_max_window > 0 else None
            k_lowpass = self.k_lowpass if self.k_lowpass > 0 else None
            k_highpass = self.k_highpass if self.k_highpass > 0 else None

            self._pdf.calculate_Gr(
                k_min_fit=k_min_fit,
                k_max_fit=k_max_fit,
                k_min_window=k_min_window,
                k_max_window=k_max_window,
                k_lowpass=k_lowpass,
                k_highpass=k_highpass,
                r_min=self.r_min,
                r_max=self.r_max,
                r_step=self.r_step,
                mask_realspace=mask,
                damp_origin_oscillations=self.damp_origin_oscillations,
                r_cut=self.r_cut,
            )
            self._sync_curves_to_js()
            self._compute_gr()
            self.status_message = ""
        except Exception as e:
            self.status_message = f"Error: {e}"
        finally:
            self.computing = False

    def _compute_gr(self):
        """Compute g(r) via PairDistributionFunction.calculate_gr and sync to JS.

        Density choice follows ``self.density_mode``:
        - "manual": pass ``self.density_value`` to calculate_gr.
        - "estimated": density=None → calculate_gr uses cached rho0 or runs
          estimate_density (with the shared ``r_cut``). Result is written back
          to ``self.density_value`` so the JS readout reflects the estimate.
        """
        if self._pdf._reduced_pdf is None or self._pdf._r is None:
            self.n_points_pdf = 0
            self.pdf_x_bytes = b""
            self.pdf_y_bytes = b""
            return
        try:
            if self.density_mode == "manual":
                self._pdf.calculate_gr(
                    density=float(self.density_value),
                    r_cut=self.r_cut,
                )
            else:
                self._pdf.calculate_gr(density=None, r_cut=self.r_cut)
                if self._pdf.rho0 is not None:
                    rho_est = float(self._pdf.rho0)
                    if rho_est != self.density_value:
                        # Update the displayed value without re-triggering the
                        # density observer.
                        self._initializing = True
                        self.density_value = rho_est
                        self._initializing = False
        except Exception as e:
            self.status_message = f"g(r) error: {e}"
            self.n_points_pdf = 0
            self.pdf_x_bytes = b""
            self.pdf_y_bytes = b""
            return

        r = to_numpy(self._pdf._r).astype(np.float32)
        gr_arr = to_numpy(self._pdf._pdf).astype(np.float32)
        self.n_points_pdf = len(r)
        self.pdf_x_bytes = r.tobytes()
        self.pdf_y_bytes = gr_arr.tobytes()

    def _sync_curves_to_js(self):
        qq = np.asarray(self._pdf.qq, dtype=np.float32)
        self.n_points_ik = len(qq)
        self.ik_x_bytes = qq.tobytes()
        self.ik_y_bytes = (
            to_numpy(self._pdf.Ik).astype(np.float32).tobytes()
        )

        # Background fit B(k)
        if self._pdf.bg is not None:
            self.ik_bg_y_bytes = (
                to_numpy(self._pdf.bg).astype(np.float32).tobytes()
            )
        else:
            self.ik_bg_y_bytes = b""

        # F(k) windowed (Lorch window applied)
        if self._pdf.Fk_masked is not None:
            fk = to_numpy(self._pdf.Fk_masked).astype(np.float32)
            self.n_points_fk = len(fk)
            self.fk_x_bytes = qq[: len(fk)].tobytes()
            self.fk_y_bytes = fk.tobytes()
        else:
            self.n_points_fk = 0
            self.fk_x_bytes = b""
            self.fk_y_bytes = b""

        # G(r)
        if self._pdf._r is not None and self._pdf._reduced_pdf is not None:
            Gr = (
                self._pdf.reduced_pdf_damped
                if self._pdf.reduced_pdf_damped is not None
                else self._pdf._reduced_pdf
            )
            r = to_numpy(self._pdf._r).astype(np.float32)
            gr = to_numpy(Gr).astype(np.float32)
            self.n_points_gr = len(r)
            self.gr_x_bytes = r.tobytes()
            self.gr_y_bytes = gr.tobytes()
        else:
            self.n_points_gr = 0
            self.gr_x_bytes = b""
            self.gr_y_bytes = b""

    # =========================================================================
    # Public API
    # =========================================================================
    @property
    def pdf(self) -> PairDistributionFunction:
        return self._pdf

    @property
    def mask(self) -> np.ndarray:
        m = self._get_mask()
        if m is None:
            return np.ones((self.scan_rows, self.scan_cols), dtype=bool)
        return m

    def set_mask(self, mask) -> Self:
        mask_np = to_numpy(mask).astype(bool)
        if mask_np.shape != (self.scan_rows, self.scan_cols):
            raise ValueError(
                f"Mask shape {mask_np.shape} does not match scan shape "
                f"({self.scan_rows}, {self.scan_cols})"
            )
        raw = mask_np.astype(np.uint8).tobytes()
        self.mask_bytes = raw
        # Mirror to mask_b64 so _get_mask() (analysis read path) sees it.
        self.mask_b64 = base64.b64encode(raw).decode("ascii")
        self._on_mask_change()
        return self

    def clear_mask(self) -> Self:
        self.mask_bytes = b""
        self.mask_b64 = ""
        self._on_mask_change()
        return self

    def set_line(
        self,
        row0: float,
        col0: float,
        row1: float,
        col1: float,
        width: int | None = None,
    ) -> Self:
        """Set the line-band selection and switch to line mode.

        Endpoints are in scan (row, col) coordinates. ``width`` is the band
        thickness in scan pixels; when omitted the current ``line_width`` is kept.
        """
        self._initializing = True
        self.line_row0 = float(row0)
        self.line_col0 = float(col0)
        self.line_row1 = float(row1)
        self.line_col1 = float(col1)
        if width is not None:
            self.line_width = int(width)
        self.line_active = True
        self.analysis_mode = "line"
        self._initializing = False
        self._on_line_change()
        return self

    def clear_line(self) -> Self:
        self._initializing = True
        self.line_active = False
        self.line_row0 = self.line_col0 = self.line_row1 = self.line_col1 = 0.0
        self._initializing = False
        self._on_line_change()
        return self

    def set_data(self, data, *, nav_image=None, find_origin=True, **pdf_kwargs) -> Self:
        if isinstance(data, PairDistributionFunction):
            self._pdf = data
        else:
            if hasattr(data, "data") and hasattr(data, "title"):
                data = data.data
            if not hasattr(data, "array"):
                arr = to_numpy(data)
                if arr.ndim == 4:
                    data = Dataset4dstem.from_array(arr)
                elif arr.ndim == 2:
                    data = Dataset2d.from_array(arr)
                else:
                    raise ValueError(
                        f"ShowPDF expects 4D (4D-STEM) or 2D input; got {arr.ndim}D array."
                    )
            self._pdf = PairDistributionFunction.from_data(
                data, find_origin=find_origin, **pdf_kwargs
            )

        polar_shape = self._pdf.polar.shape
        self.scan_rows = int(polar_shape[0])
        self.scan_cols = int(polar_shape[1])

        qq = np.asarray(self._pdf.qq, dtype=np.float32)
        self.k_min_available = float(qq[0])
        self.k_max_available = float(qq[-1])

        # Nav image
        if nav_image is not None:
            nav_img = to_numpy(nav_image).astype(np.float32)
        else:
            nav_img = self._compute_nav_image()
        self._nav_image = nav_img
        self.nav_data_min = float(nav_img.min())
        self.nav_data_max = float(nav_img.max())
        self.nav_image_bytes = nav_img.tobytes()
        self.nav_pixel_size, self.nav_unit = self._resolve_nav_calibration()

        # Reset mask and recompute
        self.mask_bytes = b""
        self._update_mask_stats()
        self._pdf.Ik = None
        self._pdf.bg = None
        self._recompute_full()
        return self

    def save_image(
        self,
        path: str | pathlib.Path,
        *,
        plot_mode: str | None = None,
        format: str | None = None,
        dpi: int = 150,
    ) -> pathlib.Path:
        """Save the current PDF curve as PNG, PDF, or TIFF.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file path.
        plot_mode : {"Ik", "Fk", "Gr", "gr"}, optional
            Curve to render. Defaults to the widget's current ``plot_mode``.
        format : {"png", "pdf", "tiff"}, optional
            Inferred from the file extension when omitted.
        dpi : int, default 150
            Output resolution.

        Returns
        -------
        pathlib.Path
            The written file path.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path = pathlib.Path(path)
        fmt = (format or path.suffix.lstrip(".").lower() or "png").lower()
        if fmt in ("tif",):
            fmt = "tiff"
        if fmt not in ("png", "pdf", "tiff"):
            raise ValueError(f"Unsupported format: {fmt!r}. Use 'png', 'pdf', or 'tiff'.")

        mode = plot_mode if plot_mode is not None else self.plot_mode
        if mode not in ("Ik", "Fk", "Gr", "gr"):
            raise ValueError(f"Unknown plot_mode: {mode!r}. Use 'Ik', 'Fk', 'Gr', or 'gr'.")

        bg = None
        if mode == "Ik":
            if self._pdf.Ik is None:
                raise ValueError("I(k) is not computed yet — set a mask or call _recompute_full().")
            x = np.asarray(self._pdf.qq, dtype=np.float32)
            y = to_numpy(self._pdf.Ik).astype(np.float32)
            if self._pdf.bg is not None and self.show_background:
                bg = to_numpy(self._pdf.bg).astype(np.float32)
            xlabel, ylabel = "k (1/Å)", "I(k)"
        elif mode == "Fk":
            if self._pdf.Fk_masked is None:
                raise ValueError("F(k) is not computed yet.")
            qq = np.asarray(self._pdf.qq, dtype=np.float32)
            fk = to_numpy(self._pdf.Fk_masked).astype(np.float32)
            x = qq[: len(fk)]
            y = fk
            xlabel, ylabel = "k (1/Å)", "F(k)"
        elif mode == "Gr":
            if self._pdf._r is None or self._pdf._reduced_pdf is None:
                raise ValueError("G(r) is not computed yet.")
            Gr = (
                self._pdf.reduced_pdf_damped
                if self._pdf.reduced_pdf_damped is not None
                else self._pdf._reduced_pdf
            )
            x = to_numpy(self._pdf._r).astype(np.float32)
            y = to_numpy(Gr).astype(np.float32)
            xlabel, ylabel = "r (Å)", "G(r)"
        else:  # "gr"
            if not self.pdf_x_bytes or not self.pdf_y_bytes:
                raise ValueError("g(r) is not computed yet.")
            x = np.frombuffer(self.pdf_x_bytes, dtype=np.float32)
            y = np.frombuffer(self.pdf_y_bytes, dtype=np.float32)
            xlabel, ylabel = "r (Å)", "g(r)"

        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=dpi)
        ax.plot(x, y, color="steelblue", linewidth=1.0, label=ylabel)
        if bg is not None:
            ax.plot(x, bg, color="orange", linewidth=1.0, linestyle="--", label="B(k)")
            ax.legend(fontsize=8, framealpha=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if self.title:
            ax.set_title(self.title, fontsize=11)
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), format=fmt, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    # =========================================================================
    # State persistence
    # =========================================================================
    def state_dict(self) -> dict:
        return {
            "title": self.title,
            "k_min_fit": self.k_min_fit,
            "k_max_fit": self.k_max_fit,
            "k_min_window": self.k_min_window,
            "k_max_window": self.k_max_window,
            "k_lowpass": self.k_lowpass,
            "k_highpass": self.k_highpass,
            "r_min": self.r_min,
            "r_max": self.r_max,
            "r_step": self.r_step,
            "damp_origin_oscillations": self.damp_origin_oscillations,
            "r_cut": self.r_cut,
            "density_mode": self.density_mode,
            "density_value": self.density_value,
            "analysis_mode": self.analysis_mode,
            "probe_row": self.probe_row,
            "probe_col": self.probe_col,
            "probe_size": self.probe_size,
            "line_row0": self.line_row0,
            "line_col0": self.line_col0,
            "line_row1": self.line_row1,
            "line_col1": self.line_col1,
            "line_width": self.line_width,
            "line_active": self.line_active,
            "line_mode": self.line_mode,
            "line_perpendicular": self.line_perpendicular,
            "linescan_max_bins": self.linescan_max_bins,
            # Painted mask at save time (base64 uint8, scan_rows*scan_cols).
            # Empty string means "no mask / full scan".
            "mask_b64": self.mask_b64,
            "plot_mode": self.plot_mode,
            "show_background": self.show_background,
            "cmap": self.cmap,
            "log_scale": self.log_scale,
            "auto_contrast": self.auto_contrast,
            "show_stats": self.show_stats,
            "show_controls": self.show_controls,
        }

    def save(self, path: str) -> None:
        save_state_file(path, "ShowPDF", self.state_dict())

    def load_state_dict(self, state: dict) -> None:
        self._initializing = True
        allowed_keys = set(self.state_dict().keys())
        for key, val in state.items():
            if key in allowed_keys and hasattr(self, key):
                setattr(self, key, val)
        # Restore the painted mask to the frontend canvas. mask_b64 is the
        # JS→Python read path; mask_bytes is the Python→JS path the canvas
        # reads on mount, so both must be set for the mask to repaint and to
        # feed _get_mask(). Drop a saved mask whose size no longer matches the
        # current scan shape (e.g. restoring onto a different dataset).
        if self.mask_b64:
            try:
                raw = base64.b64decode(self.mask_b64)
            except Exception:
                raw = b""
            if len(raw) == self.scan_rows * self.scan_cols:
                self.mask_bytes = raw
            else:
                self.mask_b64 = ""
                self.mask_bytes = b""
        self._initializing = False
        # Recompute with all restored parameters
        self._update_mask_stats()
        self._pdf.Ik = None
        self._pdf.bg = None
        self._recompute_full()

    def summary(self) -> None:
        name = self.title or "ShowPDF"
        lines = [name, "═" * 32]
        lines.append(f"Scan:     {self.scan_rows} × {self.scan_cols}")
        lines.append(
            f"k range:  [{self.k_min_available:.2f}, {self.k_max_available:.2f}] Å⁻¹"
        )
        lines.append(f"Fit:      k=[{self.k_min_fit:.2f}, {self.k_max_fit:.2f}]")
        lines.append(
            f"Output:   r=[{self.r_min:.2f}, {self.r_max:.2f}], step={self.r_step}"
        )
        lines.append(f"Plot:     {self.plot_mode}")
        if self.analysis_mode == "line":
            if self.line_active:
                length = float(
                    np.hypot(
                        self.line_row1 - self.line_row0,
                        self.line_col1 - self.line_col0,
                    )
                )
                lines.append(
                    f"Line:     ({self.line_row0:.1f}, {self.line_col0:.1f}) → "
                    f"({self.line_row1:.1f}, {self.line_col1:.1f})"
                )
                lines.append(
                    f"          len={length:.1f} px, width={self.line_width} px, "
                    f"{self.mask_pixel_count} px ({self.mask_fraction * 100:.1f}%)"
                )
            else:
                lines.append("Line:     none drawn")
        elif self.mask_bytes:
            lines.append(
                f"Mask:     {self.mask_pixel_count} px ({self.mask_fraction * 100:.1f}%)"
            )
        else:
            lines.append("Mask:     full scan (no mask)")
        if self.damp_origin_oscillations:
            lines.append(f"Damping:  ON (r_cut={self.r_cut})")
        print("\n".join(lines))

    def __repr__(self) -> str:
        mask_info = (
            f", mask={self.mask_pixel_count}px" if self.mask_bytes else ""
        )
        return (
            f"ShowPDF(scan=({self.scan_rows}, {self.scan_cols}), "
            f"k=[{self.k_min_fit:.1f}, {self.k_max_fit:.1f}]{mask_info})"
        )
