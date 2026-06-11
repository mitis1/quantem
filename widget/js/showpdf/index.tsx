import * as React from "react";
import { createRender, useModelState, useModel } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Button from "@mui/material/Button";
import Switch from "@mui/material/Switch";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Slider from "@mui/material/Slider";
import CircularProgress from "@mui/material/CircularProgress";
import { useTheme } from "../theme";
import { COLORMAPS, renderToOffscreen, renderToOffscreenReuse } from "../colormaps";

const CMAP_OPTIONS = ["gray", "inferno", "viridis", "magma"] as const;
import {
  findDataRange,
  percentileClip,
  applyLogScale,
} from "../stats";
import { extractFloat32, formatNumber } from "../format";
import { roundToNiceValue, drawScaleBarHiDPI } from "../figure";

// ============================================================================
// Constants
// ============================================================================
const DPR = Math.max(1, window.devicePixelRatio || 1);
const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 };
const FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
const MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, monospace";
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 32;

const TRACE_COLORS = {
  ik: "#4fc3f7",
  bg: "#ef5350",
  fk: "#66bb6a",
  gr: "#4fc3f7",
  pdf: "#ba68c8",
};

const MARGIN_TOP = 12;
const MARGIN_RIGHT = 16;
const MARGIN_BOTTOM = 56;
const MARGIN_LEFT_MIN = 72;
const AXIS_TICK_PX = 4;
const TICK_LABEL_W = 50;

// ============================================================================
// Helpers
// ============================================================================
function snap(v: number): number {
  return Math.round(v) + 0.5;
}

function computeTicks(min: number, max: number, maxTicks = 8): number[] {
  const range = max - min;
  if (range <= 0 || !isFinite(range)) return [min, max];
  const step = roundToNiceValue(range / maxTicks);
  if (step <= 0) return [min, max];
  const start = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= max + step * 0.001; v += step) {
    if (v >= min - step * 0.001) ticks.push(v);
  }
  return ticks.length > 0 ? ticks : [min, max];
}

function fillRectMask(
  mask: Uint8Array, w: number, h: number,
  r0: number, c0: number, r1: number, c1: number, value: number,
) {
  const minR = Math.max(0, Math.min(r0, r1));
  const maxR = Math.min(h - 1, Math.max(r0, r1));
  const minC = Math.max(0, Math.min(c0, c1));
  const maxC = Math.min(w - 1, Math.max(c0, c1));
  for (let r = minR; r <= maxR; r++)
    for (let c = minC; c <= maxC; c++)
      mask[r * w + c] = value;
}

function fillCircleMask(
  mask: Uint8Array, w: number, h: number,
  cr: number, cc: number, radius: number, value: number,
) {
  const r2 = radius * radius;
  const rMin = Math.max(0, Math.floor(cr - radius));
  const rMax = Math.min(h - 1, Math.ceil(cr + radius));
  const cMin = Math.max(0, Math.floor(cc - radius));
  const cMax = Math.min(w - 1, Math.ceil(cc + radius));
  for (let r = rMin; r <= rMax; r++)
    for (let c = cMin; c <= cMax; c++)
      if ((r - cr) ** 2 + (c - cc) ** 2 <= r2)
        mask[r * w + c] = value;
}

// ============================================================================
// Main component
// ============================================================================
function ShowPDFWidget() {
  const model = useModel();
  const { colors } = useTheme();
  const isDark = colors.bg === "#1e1e1e";

  // --- Model state ---
  const [title] = useModelState<string>("title");
  const [scanRows] = useModelState<number>("scan_rows");
  const [scanCols] = useModelState<number>("scan_cols");
  const [navImageBytes] = useModelState<DataView>("nav_image_bytes");
  const [navPixelSize] = useModelState<number>("nav_pixel_size");
  const [navUnit] = useModelState<string>("nav_unit");
  const [maskPixelCount] = useModelState<number>("mask_pixel_count");
  const [maskFraction] = useModelState<number>("mask_fraction");
  const [maskVersion, setMaskVersion_model] = useModelState<number>("mask_version");
  const [maskTool, setMaskTool] = useModelState<string>("mask_tool");
  const [maskBrushSize, setMaskBrushSize] = useModelState<number>("mask_brush_size");
  const [ikXBytes] = useModelState<DataView>("ik_x_bytes");
  const [ikYBytes] = useModelState<DataView>("ik_y_bytes");
  const [ikBgYBytes] = useModelState<DataView>("ik_bg_y_bytes");
  const [fkXBytes] = useModelState<DataView>("fk_x_bytes");
  const [fkYBytes] = useModelState<DataView>("fk_y_bytes");
  const [grXBytes] = useModelState<DataView>("gr_x_bytes");
  const [grYBytes] = useModelState<DataView>("gr_y_bytes");
  const [pdfXBytes] = useModelState<DataView>("pdf_x_bytes");
  const [pdfYBytes] = useModelState<DataView>("pdf_y_bytes");
  const [kMinFit, setKMinFit] = useModelState<number>("k_min_fit");
  const [kMaxFit, setKMaxFit] = useModelState<number>("k_max_fit");
  const [kMinWindow, setKMinWindow] = useModelState<number>("k_min_window");
  const [kMaxWindow, setKMaxWindow] = useModelState<number>("k_max_window");
  const [rMax, setRMax] = useModelState<number>("r_max");
  const [kLowpass, setKLowpass] = useModelState<number>("k_lowpass");
  const [kHighpass, setKHighpass] = useModelState<number>("k_highpass");
  const [dampOrigin, setDampOrigin] = useModelState<boolean>("damp_origin_oscillations");
  const [rCut, setRCut] = useModelState<number>("r_cut");
  const [densityMode, setDensityMode] = useModelState<string>("density_mode");
  const [densityValue, setDensityValue] = useModelState<number>("density_value");
  const [kMinAvail] = useModelState<number>("k_min_available");
  const [kMaxAvail] = useModelState<number>("k_max_available");
  const [plotMode, setPlotMode] = useModelState<string>("plot_mode");
  const [showBackground, setShowBackground] = useModelState<boolean>("show_background");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [logScale] = useModelState<boolean>("log_scale");
  const [autoContrast] = useModelState<boolean>("auto_contrast");
  const [showStats] = useModelState<boolean>("show_stats");
  const [showControls] = useModelState<boolean>("show_controls");
  const [computing] = useModelState<boolean>("computing");
  const [statusMessage] = useModelState<string>("status_message");
  const [analysisMode, setAnalysisMode] = useModelState<string>("analysis_mode");
  // ShowPDF no longer surfaces registry-driven tool-visibility traits; all
  // UI groups are always visible/enabled. Constants kept as `false` here so
  // the downstream JSX conditionals (`disabled={lockMask}`, etc.) stay valid
  // without rewriting every call site. If per-feature visibility traits are
  // ever wanted, expose them in showpdf.py with the show2d trait style.
  const lockMask = false;
  const hideMask = false;
  const lockDisplay = false;
  const lockParameters = false;
  const hideParameters = false;
  const hideStats = false;
  const [probeRow, setProbeRow] = useModelState<number>("probe_row");
  const [probeCol, setProbeCol] = useModelState<number>("probe_col");
  const [probeSize, setProbeSize] = useModelState<number>("probe_size");
  const [lineRow0] = useModelState<number>("line_row0");
  const [lineCol0] = useModelState<number>("line_col0");
  const [lineRow1] = useModelState<number>("line_row1");
  const [lineCol1] = useModelState<number>("line_col1");
  const [lineWidth, setLineWidth] = useModelState<number>("line_width");
  const [lineActive] = useModelState<boolean>("line_active");
  const [lineMode, setLineMode] = useModelState<string>("line_mode");
  const [linePerp, setLinePerp] = useModelState<boolean>("line_perpendicular");
  const [linescanBytes] = useModelState<DataView>("linescan_bytes");
  const [nLinescan] = useModelState<number>("n_linescan");
  const [linescanNPoints] = useModelState<number>("linescan_n_points");
  const [linescanAxisBytes] = useModelState<DataView>("linescan_axis_bytes");

  // --- Local state ---
  const PLOT_H = 360;
  const NAV_SIZE = PLOT_H;
  const PLOT_W = 520;
  const navH = Math.round(NAV_SIZE * (scanRows / Math.max(scanCols, 1)));
  const [navZoom, setNavZoom] = React.useState(1);
  const [navPanX, setNavPanX] = React.useState(0);
  const [navPanY, setNavPanY] = React.useState(0);
  const [maskAction, setMaskAction] = React.useState<"add" | "subtract">("add");
  const [maskRenderVersion, setMaskRenderVersion] = React.useState(0);
  const [shapePreview, setShapePreview] = React.useState<{ r0: number; c0: number; r1: number; c1: number } | null>(null);
  const [plotXMin, setPlotXMin] = React.useState(0);
  const [plotXMax, setPlotXMax] = React.useState(10);
  const [plotYMin, setPlotYMin] = React.useState(-1);
  const [plotYMax, setPlotYMax] = React.useState(1);
  const [ikLogScale, setIkLogScale] = React.useState(true);
  const [cursorData, setCursorData] = React.useState<{ x: number; y: number } | null>(null);
  // Heatmap hover readout: position along line, r/k, and the stack value there.
  const [hmHover, setHmHover] = React.useState<{ pos: number; r: number; val: number } | null>(null);
  const [localProbe, setLocalProbe] = React.useState<{ row: number; col: number }>({ row: probeRow, col: probeCol });
  const probeGenRef = React.useRef(0);
  const isDraggingProbeRef = React.useRef(false);
  // Line: local endpoints (scan coords) mirror the model for smooth dragging.
  const [localLine, setLocalLine] = React.useState<{ r0: number; c0: number; r1: number; c1: number }>(
    { r0: lineRow0, c0: lineCol0, r1: lineRow1, c1: lineCol1 });
  const localLineRef = React.useRef(localLine);
  const lineGenRef = React.useRef(0);
  const lineVersionRef = React.useRef(0);
  const [localLineWidth, setLocalLineWidth] = React.useState<number>(lineWidth);
  const isDrawingLineRef = React.useRef(false);
  const isDraggingLineBodyRef = React.useRef(false);
  const draggingEndpointRef = React.useRef<0 | 1 | null>(null);
  const lineDrawStartRef = React.useRef<{ row: number; col: number } | null>(null);
  const lineBodyStartRef = React.useRef<{ imgX: number; imgY: number; line: { r0: number; c0: number; r1: number; c1: number } } | null>(null);
  // Heatmap vertical reference guides (constant r/k) + geometry for click→r mapping.
  const [hmGuides, setHmGuides] = React.useState<number[]>([]);
  const heatmapGeomRef = React.useRef<{ rLo: number; rHi: number; pwH: number; lenUnits: number; N: number; M: number; data: Float32Array } | null>(null);
  const draggingGuideRef = React.useRef<{ index: number; moved: boolean; added: boolean } | null>(null);
  const [localKFit, setLocalKFit] = React.useState<[number, number]>([kMinFit, kMaxFit]);
  const [localKWin, setLocalKWin] = React.useState<[number, number]>([kMinWindow, kMaxWindow]);
  const [localRMax, setLocalRMax] = React.useState(rMax);
  const [localKLowpass, setLocalKLowpass] = React.useState(kLowpass);
  const [localKHighpass, setLocalKHighpass] = React.useState(kHighpass);
  const [localRCut, setLocalRCut] = React.useState(rCut);
  const [localDensity, setLocalDensity] = React.useState<string>(String(densityValue));
  const [dataVersion, setDataVersion] = React.useState(0);

  React.useEffect(() => { setLocalKFit([kMinFit, kMaxFit]); }, [kMinFit, kMaxFit]);
  React.useEffect(() => { setLocalKWin([kMinWindow, kMaxWindow]); }, [kMinWindow, kMaxWindow]);
  React.useEffect(() => { setLocalRMax(rMax); }, [rMax]);
  React.useEffect(() => { setLocalKLowpass(kLowpass); }, [kLowpass]);
  React.useEffect(() => { setLocalKHighpass(kHighpass); }, [kHighpass]);
  React.useEffect(() => { setLocalRCut(rCut); }, [rCut]);
  React.useEffect(() => { setLocalDensity(densityValue.toPrecision(4)); }, [densityValue]);
  React.useEffect(() => { setLocalProbe({ row: probeRow, col: probeCol }); }, [probeRow, probeCol]);
  React.useEffect(() => {
    const nl = { r0: lineRow0, c0: lineCol0, r1: lineRow1, c1: lineCol1 };
    localLineRef.current = nl;
    setLocalLine(nl);
  }, [lineRow0, lineCol0, lineRow1, lineCol1]);
  React.useEffect(() => { setLocalLineWidth(lineWidth); }, [lineWidth]);
  // Line length (scan px) and undirected orientation (0°=horizontal, 90°=vertical).
  const lineLenPx = Math.hypot(localLine.r1 - localLine.r0, localLine.c1 - localLine.c0);
  const lineOrientDeg = ((Math.atan2(-(localLine.r1 - localLine.r0), localLine.c1 - localLine.c0) * 180 / Math.PI) % 180 + 180) % 180;
  // The line actually sampled: perpendicular bisector of the drawn line when
  // line_perpendicular is on, else the drawn line itself (mirrors Python).
  function effectiveLine(l: { r0: number; c0: number; r1: number; c1: number }) {
    if (!linePerp) return l;
    const dr = l.r1 - l.r0, dc = l.c1 - l.c0, L = Math.hypot(dr, dc);
    if (L === 0) return l;
    const mr = (l.r0 + l.r1) / 2, mc = (l.c0 + l.c1) / 2, pr = -dc / L, pc = dr / L, half = L / 2;
    return { r0: mr - pr * half, c0: mc - pc * half, r1: mr + pr * half, c1: mc + pc * half };
  }

  const userZoomedRef = React.useRef(false);

  // --- Refs ---
  const navCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const navUiRef = React.useRef<HTMLCanvasElement>(null);  // High-DPI overlay for scale bar
  const navOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const navOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const navImgDataRef = React.useRef<ImageData | null>(null);
  const rawNavRef = React.useRef<Float32Array | null>(null);
  const maskRef = React.useRef<Uint8Array | null>(null);
  const plotCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const ikXRef = React.useRef<Float32Array | null>(null);
  const ikYRef = React.useRef<Float32Array | null>(null);
  const ikBgRef = React.useRef<Float32Array | null>(null);
  const fkXRef = React.useRef<Float32Array | null>(null);
  const fkYRef = React.useRef<Float32Array | null>(null);
  const grXRef = React.useRef<Float32Array | null>(null);
  const grYRef = React.useRef<Float32Array | null>(null);
  const pdfXRef = React.useRef<Float32Array | null>(null);
  const pdfYRef = React.useRef<Float32Array | null>(null);
  const isPanningRef = React.useRef(false);
  const panStartRef = React.useRef<{ x: number; y: number; px: number; py: number } | null>(null);
  const isPaintingRef = React.useRef(false);
  const shapeStartRef = React.useRef<{ row: number; col: number } | null>(null);
  const isDraggingShapeRef = React.useRef(false);
  const isPlotPanRef = React.useRef(false);
  const plotPanStartRef = React.useRef<{ mx: number; my: number; xMin: number; xMax: number; yMin: number; yMax: number } | null>(null);

  // =========================================================================
  // Effect 1: Parse nav image
  // =========================================================================
  React.useEffect(() => {
    if (!navImageBytes || navImageBytes.byteLength < 4) return;
    const raw = extractFloat32(navImageBytes);
    if (!raw) return;
    rawNavRef.current = raw;
    if (!navOffscreenRef.current || navOffscreenRef.current.width !== scanCols || navOffscreenRef.current.height !== scanRows) {
      const oc = document.createElement("canvas");
      oc.width = scanCols;
      oc.height = scanRows;
      navOffscreenRef.current = oc;
      navImgDataRef.current = new ImageData(scanCols, scanRows);
    }
    if (!maskRef.current || maskRef.current.length !== scanRows * scanCols) {
      maskRef.current = new Uint8Array(scanRows * scanCols).fill(1);
    }
    setDataVersion((v) => v + 1);
  }, [navImageBytes, scanRows, scanCols]);

  // Effect 1b: Init mask from Python
  React.useEffect(() => {
    const raw = model.get("mask_bytes") as DataView | undefined;
    if (!raw || raw.byteLength === 0) {
      if (scanRows > 0 && scanCols > 0)
        maskRef.current = new Uint8Array(scanRows * scanCols).fill(1);
    } else {
      maskRef.current = new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength).slice();
    }
    setMaskRenderVersion((v) => v + 1);
  }, [model, scanRows, scanCols]);

  // =========================================================================
  // Effect 2: Render nav to offscreen (expensive)
  // =========================================================================
  React.useEffect(() => {
    const raw = rawNavRef.current;
    const oc = navOffscreenRef.current;
    const imgData = navImgDataRef.current;
    if (!raw || !oc || !imgData) return;
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    const processed = logScale ? applyLogScale(raw) : raw;
    let vmin: number, vmax: number;
    if (autoContrast) {
      ({ vmin, vmax } = percentileClip(processed, 2, 98));
    } else {
      const range = findDataRange(processed);
      vmin = range.min;
      vmax = range.max;
    }
    renderToOffscreenReuse(processed, lut, vmin, vmax, oc, imgData);
    setDataVersion((v) => v + 1);
  }, [cmap, logScale, autoContrast, dataVersion]);

  // =========================================================================
  // Effect 3: Draw nav canvas (cheap)
  // =========================================================================
  React.useLayoutEffect(() => {
    const cvs = navCanvasRef.current;
    const oc = navOffscreenRef.current;
    if (!cvs || !oc || oc.width === 0) return;
    const w = NAV_SIZE * DPR;
    const h = navH * DPR;
    cvs.width = w;
    cvs.height = h;
    const ctx = cvs.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, w, h);
    ctx.save();
    ctx.translate(w / 2 + navPanX * DPR, h / 2 + navPanY * DPR);
    ctx.scale(navZoom, navZoom);
    ctx.translate(-w / 2, -h / 2);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(oc, 0, 0, scanCols, scanRows, 0, 0, w, h);
    ctx.restore();
  }, [dataVersion, navZoom, navPanX, navPanY, navH, scanCols, scanRows]);

  // =========================================================================
  // Effect 3b: Nav scale bar (HiDPI overlay)
  // =========================================================================
  React.useEffect(() => {
    const cvs = navUiRef.current;
    if (!cvs) return;
    cvs.width = NAV_SIZE * DPR;
    cvs.height = navH * DPR;
    const unit = (navUnit === "Å" ? "Å" : "px") as "Å" | "px";
    drawScaleBarHiDPI(cvs, DPR, navZoom, navPixelSize || 1, unit, scanCols);
  }, [navZoom, navPanX, navPanY, navPixelSize, navUnit, scanCols, scanRows, navH]);

  // =========================================================================
  // Effect 4: Render mask overlay
  // =========================================================================
  React.useLayoutEffect(() => {
    const cvs = navOverlayRef.current;
    if (!cvs) return;
    const w = NAV_SIZE * DPR;
    const h = navH * DPR;
    cvs.width = w;
    cvs.height = h;
    const ctx = cvs.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, w, h);
    const scX = (NAV_SIZE * DPR) / Math.max(scanCols, 1);
    const scY = (navH * DPR) / Math.max(scanRows, 1);

    // Probe mode: draw a single circle inscribed in the synthetic square at (localProbe).
    if (analysisMode === "probe") {
      const half = Math.max(0, probeSize - 1);
      const r0 = localProbe.row - half;
      const c0 = localProbe.col - half;
      const sideScan = 2 * half + 1;
      const sx = c0 * scX;
      const sy = r0 * scY;
      const sw = sideScan * scX;
      const sh = sideScan * scY;
      ctx.save();
      ctx.translate(w / 2 + navPanX * DPR, h / 2 + navPanY * DPR);
      ctx.scale(navZoom, navZoom);
      ctx.translate(-w / 2, -h / 2);
      // Translucent fill of the square footprint (data actually used)
      ctx.fillStyle = "rgba(100,200,255,0.18)";
      ctx.fillRect(sx, sy, sw, sh);
      // Inscribed circle outline (visual probe)
      ctx.strokeStyle = "rgba(100,200,255,1.0)";
      ctx.lineWidth = 2 / navZoom;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.ellipse(sx + sw / 2, sy + sh / 2, sw / 2, sh / 2, 0, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
      return;
    }

    // Line mode: draw the reference line (the drawn line — its endpoints are the
    // drag handles) and the scan line (perpendicular bisector when ⟂ is on).
    if (analysisMode === "line") {
      const ll = localLine;
      const degenerate = ll.r0 === ll.r1 && ll.c0 === ll.c1;
      if (!lineActive && !isDrawingLineRef.current && degenerate) return;
      const sl = effectiveLine(ll);
      const refAx = (ll.c0 + 0.5) * scX, refAy = (ll.r0 + 0.5) * scY;
      const refBx = (ll.c1 + 0.5) * scX, refBy = (ll.r1 + 0.5) * scY;
      const ax = (sl.c0 + 0.5) * scX, ay = (sl.r0 + 0.5) * scY;   // scan start (A)
      const bx = (sl.c1 + 0.5) * scX, by = (sl.r1 + 0.5) * scY;   // scan end (B)
      const bandPx = Math.max(0.5, localLineWidth) * (scX + scY) / 2;
      ctx.save();
      ctx.translate(w / 2 + navPanX * DPR, h / 2 + navPanY * DPR);
      ctx.scale(navZoom, navZoom);
      ctx.translate(-w / 2, -h / 2);
      // Reference line (dimmed, dashed) + its drag handles — only when ⟂, since
      // otherwise the reference and scan lines coincide.
      if (linePerp) {
        ctx.strokeStyle = "rgba(180,180,180,0.75)";
        ctx.lineWidth = 1.5 / navZoom;
        ctx.setLineDash([4 / navZoom, 3 / navZoom]);
        ctx.beginPath(); ctx.moveTo(refAx, refAy); ctx.lineTo(refBx, refBy); ctx.stroke();
        ctx.setLineDash([]);
        const rh = 3.5 / navZoom;
        ctx.fillStyle = "rgba(200,200,200,0.95)";
        for (const [ex, ey] of [[refAx, refAy], [refBx, refBy]] as const) {
          ctx.beginPath(); ctx.rect(ex - rh, ey - rh, rh * 2, rh * 2); ctx.fill();
        }
      }
      // Translucent band footprint (selected positions) along the scan line —
      // flat ends (butt) so it's a rectangle of even width, matching the mask.
      ctx.strokeStyle = "rgba(100,200,255,0.20)";
      ctx.lineWidth = bandPx;
      ctx.lineCap = "butt";
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
      // Center segment
      ctx.strokeStyle = "rgba(100,200,255,1.0)";
      ctx.lineWidth = 2 / navZoom;
      ctx.lineCap = "butt";
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
      // Arrowhead at the end (B) pointing start→end, so direction is unambiguous.
      const segLen = Math.hypot(bx - ax, by - ay);
      if (segLen > 0) {
        const ux = (bx - ax) / segLen, uy = (by - ay) / segLen;
        const ah = 9 / navZoom;
        const px = -uy, py = ux;  // perpendicular
        ctx.fillStyle = "rgba(255,110,110,1)";
        ctx.beginPath();
        ctx.moveTo(bx, by);
        ctx.lineTo(bx - ah * ux + ah * 0.5 * px, by - ah * uy + ah * 0.5 * py);
        ctx.lineTo(bx - ah * ux - ah * 0.5 * px, by - ah * uy - ah * 0.5 * py);
        ctx.closePath();
        ctx.fill();
      }
      // Scan endpoints: start (A) green, end (B) red, with labels. When ⟂ these
      // are not the drag handles (those are on the dimmed reference line above).
      const hr = 4 / navZoom;
      const ends = [
        [ax, ay, "rgba(100,225,140,1)", "A"],
        [bx, by, "rgba(255,110,110,1)", "B"],
      ] as const;
      for (const [ex, ey, col, lbl] of ends) {
        ctx.fillStyle = col;
        ctx.beginPath();
        ctx.ellipse(ex, ey, hr, hr, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.font = `${12 / navZoom}px ${FONT}`;
        ctx.textAlign = "left"; ctx.textBaseline = "bottom";
        ctx.fillText(lbl, ex + hr + 1 / navZoom, ey - hr);
      }
      ctx.restore();
      return;
    }

    // Mask mode: paint excluded pixels with a translucent overlay.
    const mask = maskRef.current;
    if (!mask || mask.length === 0) return;
    const mc = document.createElement("canvas");
    mc.width = scanCols;
    mc.height = scanRows;
    const mctx = mc.getContext("2d")!;
    const mimg = mctx.createImageData(scanCols, scanRows);
    for (let i = 0; i < mask.length; i++) {
      if (mask[i] === 0) {
        mimg.data[i * 4] = 255;
        mimg.data[i * 4 + 1] = 0;
        mimg.data[i * 4 + 2] = 0;
        mimg.data[i * 4 + 3] = 110;
      }
    }
    mctx.putImageData(mimg, 0, 0);
    ctx.save();
    ctx.translate(w / 2 + navPanX * DPR, h / 2 + navPanY * DPR);
    ctx.scale(navZoom, navZoom);
    ctx.translate(-w / 2, -h / 2);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(mc, 0, 0, scanCols, scanRows, 0, 0, w, h);
    ctx.restore();
    // Shape preview
    if (shapePreview && isDraggingShapeRef.current) {
      const { r0, c0, r1, c1 } = shapePreview;
      const scX = (NAV_SIZE * DPR) / Math.max(scanCols, 1);
      const scY = (navH * DPR) / Math.max(scanRows, 1);
      ctx.save();
      ctx.translate(w / 2 + navPanX * DPR, h / 2 + navPanY * DPR);
      ctx.scale(navZoom, navZoom);
      ctx.translate(-w / 2, -h / 2);
      ctx.strokeStyle = maskAction === "add" ? "rgba(100,200,255,0.9)" : "rgba(255,100,100,0.9)";
      ctx.lineWidth = 2 / navZoom;
      ctx.setLineDash([4 / navZoom, 4 / navZoom]);
      const sx = Math.min(c0, c1) * scX;
      const sy = Math.min(r0, r1) * scY;
      const sw = Math.abs(c1 - c0) * scX;
      const sh = Math.abs(r1 - r0) * scY;
      if (maskTool === "circle") {
        ctx.beginPath();
        ctx.ellipse((c0 + c1) / 2 * scX, (r0 + r1) / 2 * scY, sw / 2, sh / 2, 0, 0, Math.PI * 2);
        ctx.stroke();
      } else {
        ctx.strokeRect(sx, sy, sw, sh);
      }
      ctx.restore();
    }
  }, [maskRenderVersion, navZoom, navPanX, navPanY, navH, scanRows, scanCols, shapePreview, maskAction, maskTool, analysisMode, localProbe, probeSize, localLine, localLineWidth, lineActive, linePerp]);

  // =========================================================================
  // Effect 5: Parse curve bytes + auto-fit
  // =========================================================================
  React.useEffect(() => {
    ikXRef.current = ikXBytes ? extractFloat32(ikXBytes) : null;
    ikYRef.current = ikYBytes ? extractFloat32(ikYBytes) : null;
    ikBgRef.current = ikBgYBytes ? extractFloat32(ikBgYBytes) : null;
    fkXRef.current = fkXBytes ? extractFloat32(fkXBytes) : null;
    fkYRef.current = fkYBytes ? extractFloat32(fkYBytes) : null;
    grXRef.current = grXBytes ? extractFloat32(grXBytes) : null;
    grYRef.current = grYBytes ? extractFloat32(grYBytes) : null;
    pdfXRef.current = pdfXBytes ? extractFloat32(pdfXBytes) : null;
    pdfYRef.current = pdfYBytes ? extractFloat32(pdfYBytes) : null;
    if (!userZoomedRef.current) autoFitPlot();
  }, [ikXBytes, ikYBytes, ikBgYBytes, fkXBytes, fkYBytes, grXBytes, grYBytes, pdfXBytes, pdfYBytes]);

  React.useEffect(() => { userZoomedRef.current = false; autoFitPlot(); }, [plotMode]);
  // r/k units change with plot mode, so existing guide positions become invalid.
  React.useEffect(() => { setHmGuides([]); }, [plotMode]);

  function autoFitPlot() {
    let xArr: Float32Array | null = null;
    let yArr: Float32Array | null = null;
    if (plotMode === "Ik") { xArr = ikXRef.current; yArr = ikYRef.current; }
    else if (plotMode === "Fk") { xArr = fkXRef.current; yArr = fkYRef.current; }
    else if (plotMode === "gr") { xArr = pdfXRef.current; yArr = pdfYRef.current; }
    else { xArr = grXRef.current; yArr = grYRef.current; }
    if (!xArr || !yArr || xArr.length === 0) return;
    const xR = findDataRange(xArr);
    const useLog = plotMode === "Ik" && ikLogScale;
    const yR = findDataRange(yArr);
    setPlotXMin(xR.min);
    setPlotXMax(xR.max);
    if (useLog) {
      const logMin = yR.min > 0 ? Math.log10(yR.min) : 0;
      const logMax = yR.max > 0 ? Math.log10(yR.max) : 1;
      const logPad = (logMax - logMin) * 0.05 || 0.1;
      setPlotYMin(logMin - logPad);
      setPlotYMax(logMax + logPad);
    } else {
      const yPad = (yR.max - yR.min) * 0.05 || 0.1;
      setPlotYMin(yR.min - yPad);
      setPlotYMax(yR.max + yPad);
    }
  }

  // Value of the currently displayed 1D curve at a given x (linear interp).
  function curveValueAt(x: number): number | null {
    let xa: Float32Array | null, ya: Float32Array | null;
    if (plotMode === "Ik") { xa = ikXRef.current; ya = ikYRef.current; }
    else if (plotMode === "Fk") { xa = fkXRef.current; ya = fkYRef.current; }
    else if (plotMode === "gr") { xa = pdfXRef.current; ya = pdfYRef.current; }
    else { xa = grXRef.current; ya = grYRef.current; }
    if (!xa || !ya || xa.length === 0) return null;
    const n = Math.min(xa.length, ya.length);
    if (x <= xa[0]) return ya[0];
    if (x >= xa[n - 1]) return ya[n - 1];
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xa[m] <= x) lo = m; else hi = m; }
    const span = xa[hi] - xa[lo];
    const t = span !== 0 ? (x - xa[lo]) / span : 0;
    return ya[lo] + t * (ya[hi] - ya[lo]);
  }

  // =========================================================================
  // Effect 6: Render 1D plot
  // =========================================================================
  React.useLayoutEffect(() => {
    const cvs = plotCanvasRef.current;
    if (!cvs) return;
    const cw = PLOT_W * DPR;
    const ch = PLOT_H * DPR;
    cvs.width = cw;
    cvs.height = ch;
    const ctx = cvs.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);
    const mL = MARGIN_LEFT_MIN, mT = MARGIN_TOP, mR = MARGIN_RIGHT, mB = MARGIN_BOTTOM;
    const pw = PLOT_W - mL - mR;
    const ph = PLOT_H - mT - mB;
    if (pw <= 0 || ph <= 0) return;

    // Line-scan heatmap: r/k (x) vs position-along-line (y), color = curve value.
    if (analysisMode === "line" && lineMode === "linescan") {
      const CB_RESERVE = 52;          // right-side room for the colorbar + labels
      const pwH = pw - CB_RESERVE;    // heatmap plot width (x = r/k)
      ctx.fillStyle = isDark ? "#1a1a1a" : "#f8f8f8";
      ctx.fillRect(0, 0, PLOT_W, PLOT_H);
      const N = nLinescan, M = linescanNPoints;
      const ready = N > 0 && M > 0 && !!linescanBytes && linescanBytes.byteLength >= N * M * 4;
      if (!ready) {
        heatmapGeomRef.current = null;
        ctx.fillStyle = isDark ? "#888" : "#777";
        ctx.font = `14px ${FONT}`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(lineActive ? "Computing line-scan…" : "Draw a line to compute the line-scan", mL + pwH / 2, mT + ph / 2);
        return;
      }
      const data = extractFloat32(linescanBytes);
      if (!data) return;
      const axis = linescanAxisBytes && linescanAxisBytes.byteLength >= M * 4 ? extractFloat32(linescanAxisBytes) : null;
      let nf = 0;
      for (let i = 0; i < N * M; i++) if (isFinite(data[i])) nf++;
      const finite = new Float32Array(nf);
      let fi = 0;
      for (let i = 0; i < N * M; i++) { const v = data[i]; if (isFinite(v)) finite[fi++] = v; }
      const { vmin, vmax } = percentileClip(finite, 1, 99);
      const lut = COLORMAPS.viridis || COLORMAPS.inferno;
      // Offscreen M (x = r/k) × N (y = position); col 0 = smallest r/k (left),
      // row 0 = top = end of line (position increases upward, start (A) at bottom).
      const img = new Float32Array(M * N);
      for (let bin = 0; bin < N; bin++)
        for (let j = 0; j < M; j++) { const v = data[bin * M + j]; img[(N - 1 - bin) * M + j] = isFinite(v) ? v : vmin; }
      const off = renderToOffscreen(img, M, N, lut, vmin, vmax);
      if (off) { ctx.imageSmoothingEnabled = true; ctx.drawImage(off, 0, 0, M, N, mL, mT, pwH, ph); }
      const lenScan = Math.hypot(localLine.r1 - localLine.r0, localLine.c1 - localLine.c0);
      const calibrated = navUnit === "Å" && navPixelSize > 0;
      const lenUnits = calibrated ? lenScan * navPixelSize : lenScan;
      const rLo = axis ? axis[0] : 0, rHi = axis ? axis[M - 1] : 1;
      heatmapGeomRef.current = { rLo, rHi, pwH, lenUnits, N, M, data };
      const hx = (rv: number) => mL + ((rv - rLo) / ((rHi - rLo) || 1)) * pwH;       // r/k → x
      const hy = (pos: number) => mT + ph - (pos / (lenUnits || 1)) * ph;           // position → y
      // Vertical guide lines (constant r/k) for checking how straight features are.
      if (hmGuides.length) {
        for (const gv of hmGuides) {
          const cx = snap(hx(gv));
          if (cx < mL || cx > mL + pwH) continue;
          ctx.strokeStyle = isDark ? "rgba(255,255,255,0.85)" : "rgba(0,0,0,0.8)";
          ctx.lineWidth = 1; ctx.setLineDash([5, 4]);
          ctx.beginPath(); ctx.moveTo(cx, mT); ctx.lineTo(cx, mT + ph); ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = isDark ? "#eee" : "#222"; ctx.font = `10px ${MONO}`;
          ctx.textAlign = "center"; ctx.textBaseline = "top";
          ctx.fillText(formatNumber(gv), cx, mT + 2);
        }
      }
      // Axes
      ctx.strokeStyle = isDark ? "#666" : "#999"; ctx.lineWidth = 1; ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(snap(mL), mT); ctx.lineTo(snap(mL), snap(mT + ph)); ctx.lineTo(mL + pwH, snap(mT + ph)); ctx.stroke();
      ctx.fillStyle = isDark ? "#aaa" : "#555"; ctx.font = `13px ${FONT}`;
      ctx.textAlign = "center"; ctx.textBaseline = "top";   // X ticks: r/k
      for (const tv of computeTicks(rLo, rHi, Math.max(3, Math.floor(pwH / TICK_LABEL_W)))) {
        const cx = snap(hx(tv)); if (cx < mL - 1 || cx > mL + pwH + 1) continue;
        ctx.beginPath(); ctx.moveTo(cx, mT + ph); ctx.lineTo(cx, mT + ph + AXIS_TICK_PX); ctx.stroke();
        ctx.fillText(formatNumber(tv), cx, mT + ph + AXIS_TICK_PX + 2);
      }
      ctx.textAlign = "right"; ctx.textBaseline = "middle";  // Y ticks: position
      for (const tv of computeTicks(0, lenUnits, Math.max(3, Math.floor(ph / 40)))) {
        const cy = snap(hy(tv)); if (cy < mT - 1 || cy > mT + ph + 1) continue;
        ctx.beginPath(); ctx.moveTo(mL, cy); ctx.lineTo(mL - AXIS_TICK_PX, cy); ctx.stroke();
        ctx.fillText(formatNumber(tv), mL - AXIS_TICK_PX - 2, cy);
      }
      ctx.font = `14px ${FONT}`; ctx.fillStyle = isDark ? "#ccc" : "#333"; ctx.textAlign = "center"; ctx.textBaseline = "top";
      ctx.fillText((plotMode === "Gr" || plotMode === "gr") ? "r (Å)" : "k (Å⁻¹)", mL + pwH / 2, mT + ph + AXIS_TICK_PX + 26);
      ctx.save(); ctx.translate(14, mT + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.textBaseline = "top";
      ctx.fillText(calibrated ? "position (Å)" : "position (px)", 0, 0); ctx.restore();
      // Start/end markers tying the heatmap to the nav line endpoints.
      ctx.font = `11px ${MONO}`; ctx.textAlign = "left";
      ctx.fillStyle = "rgba(255,110,110,1)"; ctx.textBaseline = "top";
      ctx.fillText("B", mL + 4, mT + 3);
      ctx.fillStyle = "rgba(100,225,140,1)"; ctx.textBaseline = "bottom";
      ctx.fillText("A", mL + 4, mT + ph - 3);
      // Colorbar
      const barX = mL + pwH + 14, barY = mT, barH = ph, barW = 12;
      for (let row = 0; row < barH; row++) {
        const t = 1 - row / Math.max(1, barH - 1); const li = Math.round(t * 255) * 3;
        ctx.fillStyle = `rgb(${lut[li]},${lut[li + 1]},${lut[li + 2]})`;
        ctx.fillRect(barX, barY + row, barW, 1);
      }
      ctx.strokeStyle = isDark ? "#666" : "#999"; ctx.lineWidth = 1; ctx.strokeRect(snap(barX), snap(barY), barW, barH);
      ctx.fillStyle = isDark ? "#aaa" : "#555"; ctx.font = `10px ${MONO}`; ctx.textAlign = "left";
      ctx.textBaseline = "top"; ctx.fillText(formatNumber(vmax), barX + barW + 3, barY);
      ctx.textBaseline = "middle"; ctx.fillText(formatNumber((vmin + vmax) / 2), barX + barW + 3, barY + barH / 2);
      ctx.textBaseline = "bottom"; ctx.fillText(formatNumber(vmin), barX + barW + 3, barY + barH);
      // Hover crosshair + value readout (position, r, stack value).
      if (hmHover) {
        const cx = snap(hx(hmHover.r)), cy = snap(hy(hmHover.pos));
        if (cx >= mL && cx <= mL + pwH && cy >= mT && cy <= mT + ph) {
          ctx.strokeStyle = isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)";
          ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
          ctx.beginPath(); ctx.moveTo(cx, mT); ctx.lineTo(cx, mT + ph); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(mL, cy); ctx.lineTo(mL + pwH, cy); ctx.stroke();
          ctx.setLineDash([]);
          const label = `${formatNumber(hmHover.r)}, ${formatNumber(hmHover.pos)} → ${formatNumber(hmHover.val)}`;
          ctx.font = `12px ${MONO}`;
          const tw = ctx.measureText(label).width + 8;
          let bx = cx + 10; if (bx + tw > mL + pwH) bx = cx - tw - 10;
          let by = Math.max(mT, Math.min(mT + ph - 18, cy - 22));
          ctx.fillStyle = isDark ? "rgba(30,30,30,0.92)" : "rgba(255,255,255,0.92)";
          ctx.fillRect(bx, by, tw, 18);
          ctx.strokeStyle = isDark ? "#555" : "#ccc"; ctx.lineWidth = 1; ctx.strokeRect(bx, by, tw, 18);
          ctx.fillStyle = isDark ? "#eee" : "#333"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
          ctx.fillText(label, bx + 4, by + 9);
        }
      }
      if (computing) { ctx.fillStyle = isDark ? "rgba(0,0,0,0.3)" : "rgba(255,255,255,0.3)"; ctx.fillRect(mL, mT, pwH, ph); }
      return;
    }

    const xMin = plotXMin, xMax = plotXMax, yMin = plotYMin, yMax = plotYMax;
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;
    const d2cx = (dx: number) => mL + ((dx - xMin) / xRange) * pw;
    const useLogY = plotMode === "Ik" && ikLogScale;
    // Maps axis-space value to canvas y (yMin/yMax are already in log10 space when useLogY)
    const d2cy = (dy: number) => mT + ph - ((dy - yMin) / yRange) * ph;
    // Maps raw data value to canvas y, applying log10 when needed
    const d2cyData = (dy: number) => {
      const v = useLogY ? (dy > 0 ? Math.log10(dy) : yMin) : dy;
      return mT + ph - ((v - yMin) / yRange) * ph;
    };

    // Background
    ctx.fillStyle = isDark ? "#1a1a1a" : "#f8f8f8";
    ctx.fillRect(0, 0, PLOT_W, PLOT_H);
    // Grid
    ctx.strokeStyle = isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 3]);
    const xTicks = computeTicks(xMin, xMax, Math.max(3, Math.floor(pw / TICK_LABEL_W)));
    for (const tv of xTicks) { const cx = snap(d2cx(tv)); ctx.beginPath(); ctx.moveTo(cx, mT); ctx.lineTo(cx, mT + ph); ctx.stroke(); }
    const yTicks = computeTicks(yMin, yMax, Math.max(3, Math.floor(ph / 40)));
    for (const tv of yTicks) { const cy = snap(d2cy(tv)); ctx.beginPath(); ctx.moveTo(mL, cy); ctx.lineTo(mL + pw, cy); ctx.stroke(); }
    ctx.setLineDash([]);
    // Axes
    ctx.strokeStyle = isDark ? "#666" : "#999";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(snap(mL), mT); ctx.lineTo(snap(mL), snap(mT + ph)); ctx.lineTo(mL + pw, snap(mT + ph)); ctx.stroke();
    // X ticks
    ctx.fillStyle = isDark ? "#aaa" : "#555";
    ctx.font = `13px ${FONT}`;
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const tv of xTicks) { const cx = snap(d2cx(tv)); ctx.beginPath(); ctx.moveTo(cx, mT + ph); ctx.lineTo(cx, mT + ph + AXIS_TICK_PX); ctx.stroke(); ctx.fillText(formatNumber(tv), cx, mT + ph + AXIS_TICK_PX + 2); }
    // Y ticks
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const tv of yTicks) { const cy = snap(d2cy(tv)); ctx.beginPath(); ctx.moveTo(mL, cy); ctx.lineTo(mL - AXIS_TICK_PX, cy); ctx.stroke(); ctx.fillText(useLogY ? formatNumber(Math.pow(10, tv)) : formatNumber(tv), mL - AXIS_TICK_PX - 2, cy); }
    // Axis labels
    ctx.font = `14px ${FONT}`; ctx.fillStyle = isDark ? "#ccc" : "#333"; ctx.textAlign = "center"; ctx.textBaseline = "top";
    const xAxisLabel = (plotMode === "Gr" || plotMode === "gr") ? "r (Å)" : "k (Å⁻¹)";
    ctx.fillText(xAxisLabel, mL + pw / 2, mT + ph + AXIS_TICK_PX + 26);
    ctx.save(); ctx.translate(14, mT + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.textBaseline = "top";
    const yAxisLabel = plotMode === "Ik" ? "I(k)" : plotMode === "Fk" ? "F(k)" : plotMode === "gr" ? "g(r)" : "G(r)";
    ctx.fillText(yAxisLabel, 0, 0); ctx.restore();
    // Clip
    ctx.save(); ctx.beginPath(); ctx.rect(mL, mT, pw, ph); ctx.clip();
    // Draw traces
    const drawLine = (xD: Float32Array | null, yD: Float32Array | null, color: string, dashed = false, lw = 1.5) => {
      if (!xD || !yD) return;
      ctx.strokeStyle = color; ctx.lineWidth = lw;
      if (dashed) ctx.setLineDash([4, 3]); else ctx.setLineDash([]);
      ctx.beginPath();
      let started = false;
      const len = Math.min(xD.length, yD.length);
      for (let i = 0; i < len; i++) {
        if (!isFinite(yD[i])) continue;
        const cx = d2cx(xD[i]), cy = d2cyData(yD[i]);
        if (!started) { ctx.moveTo(cx, cy); started = true; } else ctx.lineTo(cx, cy);
      }
      ctx.stroke(); ctx.setLineDash([]);
    };
    const drawKRange = (k0: number, k1: number, color: string, label0: string, label1: string) => {
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.setLineDash([14, 6]);
      ctx.fillStyle = color;
      ctx.font = `11px ${MONO}`;
      ctx.textBaseline = "top";
      for (const [kv, lbl, align] of [[k0, label0, "left"], [k1, label1, "right"]] as const) {
        if (!isFinite(kv)) continue;
        const cx = snap(d2cx(kv));
        if (cx < mL || cx > mL + pw) continue;
        ctx.beginPath();
        ctx.moveTo(cx, mT);
        ctx.lineTo(cx, mT + ph);
        ctx.stroke();
        ctx.textAlign = align;
        const tx = align === "left" ? cx + 3 : cx - 3;
        ctx.fillText(lbl, tx, mT + 2);
      }
      ctx.restore();
    };
    if (plotMode === "Ik") {
      drawLine(ikXRef.current, ikYRef.current, TRACE_COLORS.ik);
      if (showBackground) drawLine(ikXRef.current, ikBgRef.current, TRACE_COLORS.bg, true);
      drawKRange(kMinFit, kMaxFit, "#ffb300", "k fit", "");
    } else if (plotMode === "Fk") {
      drawLine(fkXRef.current, fkYRef.current, TRACE_COLORS.fk);
      drawKRange(kMinWindow, kMaxWindow, "#ffb300", "k window", "");
    } else if (plotMode === "gr") {
      ctx.strokeStyle = isDark ? "rgba(255,255,255,0.2)" : "rgba(0,0,0,0.2)";
      ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      const oneY = d2cy(1); ctx.beginPath(); ctx.moveTo(mL, oneY); ctx.lineTo(mL + pw, oneY); ctx.stroke(); ctx.setLineDash([]);
      drawLine(pdfXRef.current, pdfYRef.current, TRACE_COLORS.pdf);
    } else {
      ctx.strokeStyle = isDark ? "rgba(255,255,255,0.2)" : "rgba(0,0,0,0.2)";
      ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      const zy = d2cy(0); ctx.beginPath(); ctx.moveTo(mL, zy); ctx.lineTo(mL + pw, zy); ctx.stroke(); ctx.setLineDash([]);
      drawLine(grXRef.current, grYRef.current, TRACE_COLORS.gr);
    }
    ctx.restore();
    // Crosshair — the readout snaps to the displayed curve's value at cursor x.
    if (cursorData) {
      const cval = curveValueAt(cursorData.x);
      const cx = d2cx(cursorData.x);
      if (cval !== null && isFinite(cval) && cx >= mL && cx <= mL + pw) {
        const cy = d2cyData(cval);
        ctx.strokeStyle = isDark ? "rgba(255,255,255,0.25)" : "rgba(0,0,0,0.25)";
        ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(snap(cx), mT); ctx.lineTo(snap(cx), mT + ph); ctx.stroke();
        ctx.setLineDash([]);
        if (cy >= mT && cy <= mT + ph) {
          ctx.fillStyle = isDark ? "#eee" : "#333";
          ctx.beginPath(); ctx.ellipse(cx, cy, 3, 3, 0, 0, Math.PI * 2); ctx.fill();
        }
        const label = `${formatNumber(cursorData.x)}, ${formatNumber(cval)}`;
        ctx.font = `12px ${MONO}`;
        const tw = ctx.measureText(label).width + 8;
        let bx = cx + 10; if (bx + tw > mL + pw) bx = cx - tw - 10;
        let by = Math.max(mT, Math.min(mT + ph - 18, cy - 22));
        ctx.fillStyle = isDark ? "rgba(30,30,30,0.9)" : "rgba(255,255,255,0.9)";
        ctx.fillRect(bx, by, tw, 18);
        ctx.strokeStyle = isDark ? "#555" : "#ccc"; ctx.lineWidth = 1; ctx.strokeRect(bx, by, tw, 18);
        ctx.fillStyle = isDark ? "#eee" : "#333"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
        ctx.fillText(label, bx + 4, by + 9);
      }
    }
    if (computing) {
      ctx.fillStyle = isDark ? "rgba(0,0,0,0.3)" : "rgba(255,255,255,0.3)";
      ctx.fillRect(mL, mT, pw, ph);
    }
  }, [PLOT_W, PLOT_H, plotXMin, plotXMax, plotYMin, plotYMax, plotMode, showBackground, ikLogScale, cursorData, computing, isDark,
      ikXBytes, ikYBytes, ikBgYBytes, fkXBytes, fkYBytes, grXBytes, grYBytes, pdfXBytes, pdfYBytes,
      kMinFit, kMaxFit, kMinWindow, kMaxWindow,
      analysisMode, lineMode, nLinescan, linescanNPoints, linescanBytes, linescanAxisBytes, localLine, navPixelSize, navUnit, lineActive, hmGuides, hmHover]);

  // =========================================================================
  // Nav mouse handlers
  // =========================================================================
  function screenToImage(e: React.MouseEvent) {
    const cvs = navCanvasRef.current;
    if (!cvs) return { row: 0, col: 0 };
    const rect = cvs.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    const cx = NAV_SIZE / 2, cy = navH / 2;
    const imgX = (sx - cx - navPanX) / navZoom + cx;
    const imgY = (sy - cy - navPanY) / navZoom + cy;
    return { row: Math.round((imgY / navH) * scanRows), col: Math.round((imgX / NAV_SIZE) * scanCols) };
  }

  const maskVersionRef = React.useRef(maskVersion ?? 0);
  function syncMaskToPython() {
    if (!maskRef.current) return;
    const copy = maskRef.current.slice();
    // Encode mask as base64 string (Bytes traits don't sync JS→Python in anywidget)
    let binary = "";
    for (let i = 0; i < copy.length; i++) binary += String.fromCharCode(copy[i]);
    model.set("mask_b64", btoa(binary));
    model.save_changes();
    // Increment mask_version to trigger Python observer
    maskVersionRef.current += 1;
    setMaskVersion_model(maskVersionRef.current);
  }

  // Live probe move with rAF coalescing — accumulates rapid mousemoves and
  // sends at most one model.set per display frame.
  function syncProbeRAF(row: number, col: number) {
    const r = Math.max(0, Math.min(scanRows - 1, row));
    const c = Math.max(0, Math.min(scanCols - 1, col));
    setLocalProbe({ row: r, col: c });
    const gen = ++probeGenRef.current;
    requestAnimationFrame(() => {
      if (gen !== probeGenRef.current) return;
      if (r !== probeRow) setProbeRow(r);
      if (c !== probeCol) setProbeCol(c);
    });
  }

  // Mouse position in un-zoomed canvas (image) coordinates — used for
  // hit-testing line endpoints/body in the same space the overlay draws in.
  function screenToImgXY(e: React.MouseEvent) {
    const cvs = navCanvasRef.current;
    if (!cvs) return { imgX: 0, imgY: 0 };
    const rect = cvs.getBoundingClientRect();
    const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
    const cx = NAV_SIZE / 2, cy = navH / 2;
    return { imgX: (sx - cx - navPanX) / navZoom + cx, imgY: (sy - cy - navPanY) / navZoom + cy };
  }
  // Scan (row,col) → image-space pixel center (matches the overlay transform).
  function scanToImgXY(row: number, col: number) {
    return { imgX: (col + 0.5) / Math.max(scanCols, 1) * NAV_SIZE, imgY: (row + 0.5) / Math.max(scanRows, 1) * navH };
  }
  function distToSegmentImg(px: number, py: number, ax: number, ay: number, bx: number, by: number) {
    const dx = bx - ax, dy = by - ay;
    const lenSq = dx * dx + dy * dy;
    let t = lenSq > 0 ? ((px - ax) * dx + (py - ay) * dy) / lenSq : 0;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
  }

  function applyLineLocal(line: { r0: number; c0: number; r1: number; c1: number }) {
    const clampR = (v: number) => Math.max(0, Math.min(scanRows - 1, v));
    const clampC = (v: number) => Math.max(0, Math.min(scanCols - 1, v));
    const nl = { r0: clampR(line.r0), c0: clampC(line.c0), r1: clampR(line.r1), c1: clampC(line.c1) };
    localLineRef.current = nl;
    setLocalLine(nl);
    return nl;
  }
  // Push endpoints + a single line_version bump → one Python recompute.
  function commitLine(nl: { r0: number; c0: number; r1: number; c1: number }) {
    model.set("line_row0", nl.r0); model.set("line_col0", nl.c0);
    model.set("line_row1", nl.r1); model.set("line_col1", nl.c1);
    if (!lineActive) model.set("line_active", true);
    lineVersionRef.current = (model.get("line_version") as number ?? lineVersionRef.current) + 1;
    model.set("line_version", lineVersionRef.current);
    model.save_changes();
  }
  // During a drag: update the local overlay every frame. In averaged mode also
  // commit live (rAF-coalesced, one recompute per frame). In line-scan mode the
  // N-PDF compute is heavy, so commit is deferred to mouse-up.
  function dragLine(line: { r0: number; c0: number; r1: number; c1: number }) {
    const nl = applyLineLocal(line);
    if (lineMode !== "linescan") {
      const gen = ++lineGenRef.current;
      requestAnimationFrame(() => { if (gen === lineGenRef.current) commitLine(nl); });
    }
  }

  function clearLine() {
    isDrawingLineRef.current = false;
    isDraggingLineBodyRef.current = false;
    draggingEndpointRef.current = null;
    const nl = { r0: 0, c0: 0, r1: 0, c1: 0 };
    localLineRef.current = nl;
    setLocalLine(nl);
    model.set("line_row0", 0); model.set("line_col0", 0);
    model.set("line_row1", 0); model.set("line_col1", 0);
    model.set("line_active", false);
    lineVersionRef.current = (model.get("line_version") as number ?? 0) + 1;
    model.set("line_version", lineVersionRef.current);
    model.save_changes();
  }

  const handleNavMouseDown = (e: React.MouseEvent) => {
    if (!maskRef.current) return;
    e.preventDefault();
    if (e.button === 1 || (e.button === 0 && e.altKey)) {
      if (lockDisplay) return;  // pan locked
      isPanningRef.current = true;
      panStartRef.current = { x: e.clientX, y: e.clientY, px: navPanX, py: navPanY };
      return;
    }
    if (lockMask) return;  // mask group locked: blocks paint AND probe drag
    const { row, col } = screenToImage(e);
    if (analysisMode === "line") {
      const hit = 8 / navZoom;  // endpoint/body hit radius in image px
      const { imgX, imgY } = screenToImgXY(e);
      if (lineActive) {
        const e0 = scanToImgXY(localLineRef.current.r0, localLineRef.current.c0);
        const e1 = scanToImgXY(localLineRef.current.r1, localLineRef.current.c1);
        const d0 = Math.hypot(imgX - e0.imgX, imgY - e0.imgY);
        const d1 = Math.hypot(imgX - e1.imgX, imgY - e1.imgY);
        if (d0 < hit || d1 < hit) {
          draggingEndpointRef.current = d0 <= d1 ? 0 : 1;
          return;
        }
        if (distToSegmentImg(imgX, imgY, e0.imgX, e0.imgY, e1.imgX, e1.imgY) < hit) {
          isDraggingLineBodyRef.current = true;
          lineBodyStartRef.current = { imgX, imgY, line: { ...localLineRef.current } };
          return;
        }
      }
      // Otherwise start drawing a fresh line from this point.
      isDrawingLineRef.current = true;
      lineDrawStartRef.current = { row, col };
      dragLine({ r0: row, c0: col, r1: row, c1: col });
      return;
    }
    if (analysisMode === "probe") {
      isDraggingProbeRef.current = true;
      syncProbeRAF(row, col);
      return;
    }
    // "+" adds to the mask (excludes the area from analysis = 0);
    // "-" removes from the mask (re-includes the area = 1).
    const value = maskAction === "add" ? 0 : 1;
    if (maskTool === "freeform") {
      isPaintingRef.current = true;
      const half = maskBrushSize - 1;
      fillRectMask(maskRef.current, scanCols, scanRows, row - half, col - half, row + half, col + half, value);
      setMaskRenderVersion((v) => v + 1);
    } else {
      isDraggingShapeRef.current = true;
      shapeStartRef.current = { row, col };
      setShapePreview({ r0: row, c0: col, r1: row, c1: col });
    }
  };

  const handleNavMouseMove = (e: React.MouseEvent) => {
    if (isPanningRef.current && panStartRef.current) {
      setNavPanX(panStartRef.current.px + e.clientX - panStartRef.current.x);
      setNavPanY(panStartRef.current.py + e.clientY - panStartRef.current.y);
      return;
    }
    const { row, col } = screenToImage(e);
    if (isDraggingProbeRef.current) {
      syncProbeRAF(row, col);
      return;
    }
    if (isDrawingLineRef.current && lineDrawStartRef.current) {
      const s = lineDrawStartRef.current;
      dragLine({ r0: s.row, c0: s.col, r1: row, c1: col });
      return;
    }
    if (draggingEndpointRef.current !== null) {
      const upd = draggingEndpointRef.current === 0 ? { r0: row, c0: col } : { r1: row, c1: col };
      dragLine({ ...localLineRef.current, ...upd });
      return;
    }
    if (isDraggingLineBodyRef.current && lineBodyStartRef.current) {
      const { imgX, imgY } = screenToImgXY(e);
      const start = lineBodyStartRef.current;
      const dRow = (imgY - start.imgY) / navH * scanRows;
      const dCol = (imgX - start.imgX) / NAV_SIZE * scanCols;
      dragLine({
        r0: start.line.r0 + dRow, c0: start.line.c0 + dCol,
        r1: start.line.r1 + dRow, c1: start.line.c1 + dCol,
      });
      return;
    }
    if (isPaintingRef.current && maskRef.current) {
      const half = maskBrushSize - 1;
      fillRectMask(maskRef.current, scanCols, scanRows, row - half, col - half, row + half, col + half, maskAction === "add" ? 0 : 1);
      setMaskRenderVersion((v) => v + 1);
    }
    if (isDraggingShapeRef.current && shapeStartRef.current)
      setShapePreview({ r0: shapeStartRef.current.row, c0: shapeStartRef.current.col, r1: row, c1: col });
  };

  const handleNavMouseUp = () => {
    if (isPanningRef.current) { isPanningRef.current = false; panStartRef.current = null; return; }
    if (isDraggingProbeRef.current) { isDraggingProbeRef.current = false; return; }
    if (isDrawingLineRef.current) { isDrawingLineRef.current = false; lineDrawStartRef.current = null; commitLine(localLineRef.current); return; }
    if (draggingEndpointRef.current !== null) { draggingEndpointRef.current = null; commitLine(localLineRef.current); return; }
    if (isDraggingLineBodyRef.current) { isDraggingLineBodyRef.current = false; lineBodyStartRef.current = null; commitLine(localLineRef.current); return; }
    if (isPaintingRef.current) { isPaintingRef.current = false; syncMaskToPython(); return; }
    if (isDraggingShapeRef.current && shapeStartRef.current && maskRef.current && shapePreview) {
      const value = maskAction === "add" ? 0 : 1;
      const { r0, c0, r1, c1 } = shapePreview;
      if (maskTool === "circle") {
        fillCircleMask(maskRef.current, scanCols, scanRows, (r0 + r1) / 2, (c0 + c1) / 2, Math.max(Math.abs(c1 - c0), Math.abs(r1 - r0)) / 2, value);
      } else {
        fillRectMask(maskRef.current, scanCols, scanRows, r0, c0, r1, c1, value);
      }
      setMaskRenderVersion((v) => v + 1);
      syncMaskToPython();
    }
    isDraggingShapeRef.current = false; shapeStartRef.current = null; setShapePreview(null);
  };

  const handleNavWheel = (e: React.WheelEvent) => {
    e.preventDefault(); e.stopPropagation();
    if (lockDisplay) return;  // zoom locked
    setNavZoom((z) => Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z * (e.deltaY > 0 ? 0.9 : 1.1))));
  };

  // =========================================================================
  // Plot mouse handlers
  // =========================================================================
  function plotScreenToData(e: React.MouseEvent) {
    const cvs = plotCanvasRef.current;
    if (!cvs) return null;
    const rect = cvs.getBoundingClientRect();
    const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
    const pw = PLOT_W - MARGIN_LEFT_MIN - MARGIN_RIGHT;
    const ph = PLOT_H - MARGIN_TOP - MARGIN_BOTTOM;
    return {
      x: plotXMin + ((sx - MARGIN_LEFT_MIN) / pw) * (plotXMax - plotXMin),
      y: plotYMin + ((MARGIN_TOP + ph - sy) / ph) * (plotYMax - plotYMin),
    };
  }

  const handlePlotMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    // Line-scan heatmap: click empty space adds a vertical guide; click on an
    // existing guide removes it; drag a guide to fine-tune its r/k position.
    if (analysisMode === "line" && lineMode === "linescan") {
      const g = heatmapGeomRef.current;
      const cvs = plotCanvasRef.current;
      if (g && cvs) {
        const rect = cvs.getBoundingClientRect();
        const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
        const mL = MARGIN_LEFT_MIN, mT = MARGIN_TOP;
        const ph = PLOT_H - mT - MARGIN_BOTTOM;
        if (sx >= mL && sx <= mL + g.pwH && sy >= mT && sy <= mT + ph) {
          const r = g.rLo + ((sx - mL) / g.pwH) * (g.rHi - g.rLo);
          const tolR = ((g.rHi - g.rLo) / g.pwH) * 6;
          const near = hmGuides.findIndex((v) => Math.abs(v - r) <= tolR);
          if (near >= 0) {
            draggingGuideRef.current = { index: near, moved: false, added: false };
          } else {
            draggingGuideRef.current = { index: hmGuides.length, moved: false, added: true };
            setHmGuides((prev) => [...prev, r]);
          }
        }
      }
      return;
    }
    if (lockDisplay) return;  // pan locked
    isPlotPanRef.current = true;
    plotPanStartRef.current = { mx: e.clientX, my: e.clientY, xMin: plotXMin, xMax: plotXMax, yMin: plotYMin, yMax: plotYMax };
  };

  const handlePlotMouseMove = (e: React.MouseEvent) => {
    // Dragging a heatmap guide line to fine-tune its r/k position.
    if (draggingGuideRef.current) {
      const g = heatmapGeomRef.current;
      const cvs = plotCanvasRef.current;
      if (g && cvs) {
        const rect = cvs.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const mL = MARGIN_LEFT_MIN;
        const r = Math.max(g.rLo, Math.min(g.rHi, g.rLo + ((sx - mL) / g.pwH) * (g.rHi - g.rLo)));
        const idx = draggingGuideRef.current.index;
        draggingGuideRef.current.moved = true;
        setHmGuides((prev) => { const c = [...prev]; if (idx < c.length) c[idx] = r; return c; });
      }
      return;
    }
    // Heatmap hover: report position, r/k, and the stack value under the cursor.
    if (analysisMode === "line" && lineMode === "linescan") {
      const g = heatmapGeomRef.current;
      const cvs = plotCanvasRef.current;
      if (g && cvs) {
        const rect = cvs.getBoundingClientRect();
        const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
        const mL = MARGIN_LEFT_MIN, mT = MARGIN_TOP, ph = PLOT_H - mT - MARGIN_BOTTOM;
        if (sx >= mL && sx <= mL + g.pwH && sy >= mT && sy <= mT + ph) {
          const r = g.rLo + ((sx - mL) / g.pwH) * (g.rHi - g.rLo);
          const pos = (1 - (sy - mT) / ph) * g.lenUnits;        // 0 at bottom
          const j = Math.max(0, Math.min(g.M - 1, Math.round(((r - g.rLo) / ((g.rHi - g.rLo) || 1)) * (g.M - 1))));
          const bin = Math.max(0, Math.min(g.N - 1, Math.floor((pos / (g.lenUnits || 1)) * g.N)));
          const val = g.data[bin * g.M + j];
          setHmHover({ pos, r, val });
        } else {
          setHmHover(null);
        }
      }
      return;
    }
    if (isPlotPanRef.current && plotPanStartRef.current) {
      const dx = e.clientX - plotPanStartRef.current.mx;
      const dy = e.clientY - plotPanStartRef.current.my;
      const pw = PLOT_W - MARGIN_LEFT_MIN - MARGIN_RIGHT;
      const ph = PLOT_H - MARGIN_TOP - MARGIN_BOTTOM;
      const dxD = -(dx / pw) * (plotPanStartRef.current.xMax - plotPanStartRef.current.xMin);
      const dyD = (dy / ph) * (plotPanStartRef.current.yMax - plotPanStartRef.current.yMin);
      setPlotXMin(plotPanStartRef.current.xMin + dxD); setPlotXMax(plotPanStartRef.current.xMax + dxD);
      setPlotYMin(plotPanStartRef.current.yMin + dyD); setPlotYMax(plotPanStartRef.current.yMax + dyD);
      userZoomedRef.current = true;
      return;
    }
    const d = plotScreenToData(e);
    if (d) setCursorData(d);
  };

  const handlePlotMouseUp = () => {
    const d = draggingGuideRef.current;
    if (d) {
      draggingGuideRef.current = null;
      // A click (no drag) on an existing guide removes it; a click on empty
      // space already added one, so leave it.
      if (!d.moved && !d.added) setHmGuides((prev) => prev.filter((_, i) => i !== d.index));
      return;
    }
    isPlotPanRef.current = false; plotPanStartRef.current = null;
  };

  const handlePlotWheel = (e: React.WheelEvent) => {
    e.preventDefault(); e.stopPropagation();
    if (lockDisplay) return;  // zoom locked
    const d = plotScreenToData(e);
    if (!d) return;
    const f = e.deltaY > 0 ? 1.1 : 1 / 1.1;
    setPlotXMin((p) => d.x - (d.x - p) * f); setPlotXMax((p) => d.x + (p - d.x) * f);
    setPlotYMin((p) => d.y - (d.y - p) * f); setPlotYMax((p) => d.y + (p - d.y) * f);
    userZoomedRef.current = true;
  };

  // =========================================================================
  // Keyboard
  // =========================================================================
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
    switch (e.key.toLowerCase()) {
      case "r": if (!lockDisplay) { setNavZoom(1); setNavPanX(0); setNavPanY(0); userZoomedRef.current = false; autoFitPlot(); } break;
      case "x": if (!lockMask) setMaskAction((a) => {
          if (maskRef.current) {
            maskRef.current.fill(a === "add" ? 0 : 1);
            setMaskRenderVersion((v) => v + 1);
            syncMaskToPython();
          }
          return a === "add" ? "subtract" : "add";
        }); break;
      case "c":
        if (!lockMask && maskRef.current) { maskRef.current.fill(1); setMaskRenderVersion((v) => v + 1); syncMaskToPython(); }
        break;
      case "i":
        if (!lockMask && maskRef.current) { for (let i = 0; i < maskRef.current.length; i++) maskRef.current[i] = maskRef.current[i] ? 0 : 1; setMaskRenderVersion((v) => v + 1); syncMaskToPython(); }
        break;
      case "1": setPlotMode("Ik"); break;
      case "2": setPlotMode("Fk"); break;
      case "3": setPlotMode("Gr"); break;
    }
  };

  // Prevent scroll
  React.useEffect(() => {
    const h = (e: WheelEvent) => e.preventDefault();
    const opts: AddEventListenerOptions = { passive: false };
    const n1 = navCanvasRef.current?.parentElement;
    const n2 = plotCanvasRef.current?.parentElement;
    n1?.addEventListener("wheel", h, opts);
    n2?.addEventListener("wheel", h, opts);
    return () => { n1?.removeEventListener("wheel", h); n2?.removeEventListener("wheel", h); };
  }, []);

  // =========================================================================
  // Styles
  // =========================================================================
  const typo = {
    title: { fontSize: 15, fontWeight: 600, color: colors.accent, fontFamily: FONT },
    label: { fontSize: 13, color: colors.text, fontFamily: FONT },
    labelSmall: { fontSize: 12, fontWeight: 600, color: colors.textMuted, fontFamily: FONT },
    value: { fontSize: 12, fontFamily: MONO, color: colors.text },
  };
  const switchSmall = { "& .MuiSwitch-thumb": { width: 12, height: 12 }, "& .MuiSwitch-switchBase": { padding: "4px" } };
  const compactBtn = { fontSize: 12, minWidth: 36, px: 1, py: 0.25, textTransform: "none" as const };
  const imageBox = { position: "relative" as const, border: `1px solid ${colors.border}`, overflow: "hidden", bgcolor: isDark ? "#111" : "#eee" };

  // =========================================================================
  // JSX
  // =========================================================================
  return (
    <Box className="showpdf-root" tabIndex={0} onKeyDown={handleKeyDown} sx={{ fontFamily: FONT, outline: "none", border: "none", p: 1, bgcolor: colors.bg, color: colors.text, "&:focus": { outline: "none" }, "&:focus-visible": { outline: "none" } }}>
      <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: `${SPACING.SM}px` }}>
        <Typography sx={typo.title}>{title || "PDF"}</Typography>
        <Stack direction="row" alignItems="center" gap={1}>
          {computing && <CircularProgress size={14} sx={{ color: colors.accent }} />}
          {statusMessage && <Typography sx={{ ...typo.labelSmall, color: computing ? colors.accent : colors.textMuted }}>{statusMessage}</Typography>}
        </Stack>
      </Stack>

      <Stack direction="row" spacing={`${SPACING.LG}px`}>
        {/* LEFT: Nav + Mask */}
        <Box sx={{ width: NAV_SIZE }}>
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, minHeight: 32, flexWrap: "wrap", rowGap: `${SPACING.XS}px` }}>
            <span style={{ fontSize: 12, fontFamily: FONT, color: colors.textMuted }}>Scan ({scanRows}×{scanCols})</span>
            <Stack direction="row" alignItems="center" spacing={`2px`}>
              {(["mask", "probe", "line"] as const).map((m) => (
                <Button key={m} size="small" variant={analysisMode === m ? "contained" : "outlined"} disabled={lockMask}
                  sx={{ ...compactBtn, minWidth: 38 }} onClick={() => setAnalysisMode(m)}>
                  {m}
                </Button>
              ))}
            </Stack>
          </Stack>
          {analysisMode === "mask" && !hideMask && (
            <Stack direction="row" spacing={`2px`} alignItems="center" sx={{ mb: `${SPACING.XS}px`, flexWrap: "wrap", rowGap: `${SPACING.XS}px` }}>
              {(["rectangle", "circle", "freeform"] as const).map((tool) => (
                <Button key={tool} size="small" variant={maskTool === tool ? "contained" : "outlined"} disabled={lockMask}
                  sx={{ ...compactBtn, minWidth: 24 }} onClick={() => setMaskTool(tool)}>
                  {tool === "rectangle" ? "▭" : tool === "circle" ? "○" : "✎"}
                </Button>
              ))}
              <Button size="small" variant={maskAction === "add" ? "contained" : "outlined"} disabled={lockMask}
                sx={{ ...compactBtn, minWidth: 24 }} color={maskAction === "add" ? "primary" : "error"}
                onClick={() => setMaskAction((a) => {
                  if (!maskRef.current) return a === "add" ? "subtract" : "add";
                  if (a === "add") {
                    // Switching to subtract: drawing now re-includes — start with everything excluded
                    maskRef.current.fill(0);
                  } else {
                    // Switching to add: drawing now excludes — start with everything included
                    maskRef.current.fill(1);
                  }
                  setMaskRenderVersion((v) => v + 1);
                  syncMaskToPython();
                  return a === "add" ? "subtract" : "add";
                })}>
                {maskAction === "add" ? "+" : "−"}
              </Button>
              <Button size="small" sx={compactBtn} disabled={lockMask} onClick={() => { if (maskRef.current) { maskRef.current.fill(1); setMaskRenderVersion((v) => v + 1); syncMaskToPython(); } }}>Clear</Button>
              <Button size="small" sx={compactBtn} disabled={lockMask} onClick={() => { if (maskRef.current) { for (let i = 0; i < maskRef.current.length; i++) maskRef.current[i] = maskRef.current[i] ? 0 : 1; setMaskRenderVersion((v) => v + 1); syncMaskToPython(); setMaskAction((a) => a === "add" ? "subtract" : "add"); } }}>Invert</Button>
            </Stack>
          )}
          <Box sx={imageBox} style={{ width: NAV_SIZE, height: navH }}>
            <canvas ref={navCanvasRef} style={{ width: NAV_SIZE, height: navH, display: "block" }} />
            <canvas ref={navOverlayRef} style={{ position: "absolute", top: 0, left: 0, width: NAV_SIZE, height: navH, pointerEvents: "none" }} />
            <canvas ref={navUiRef} style={{ position: "absolute", top: 0, left: 0, width: NAV_SIZE, height: navH, pointerEvents: "none" }} />
            <div style={{ position: "absolute", inset: 0, cursor: analysisMode === "probe" || analysisMode === "line" || maskTool === "freeform" ? "crosshair" : "default" }}
              onMouseDown={handleNavMouseDown} onMouseMove={handleNavMouseMove} onMouseUp={handleNavMouseUp}
              onMouseLeave={() => { isPaintingRef.current = false; isDraggingShapeRef.current = false; isDraggingProbeRef.current = false; setShapePreview(null); }}
              onWheel={handleNavWheel} />
          </Box>
          {showStats && !hideStats && <div style={{ fontSize: 12, fontFamily: MONO, color: colors.textMuted, marginTop: SPACING.XS }}>{maskPixelCount} / {scanRows * scanCols} included ({(maskFraction * 100).toFixed(1)}%)</div>}
          {showControls && (
            <Box sx={{ mt: `${SPACING.SM}px` }}>
              {analysisMode === "mask" && maskTool === "freeform" && (
                <Stack direction="row" alignItems="center" gap={1} sx={{ mb: `${SPACING.XS}px` }}>
                  <Typography sx={typo.labelSmall}>Brush:</Typography>
                  <Slider value={maskBrushSize} onChange={(_, v) => setMaskBrushSize(v as number)} disabled={lockMask} min={1} max={20} size="small"
                    sx={{ width: 80, "& .MuiSlider-thumb": { width: 10, height: 10 } }} />
                  <Typography sx={typo.value}>{maskBrushSize}</Typography>
                </Stack>
              )}
              {analysisMode === "probe" && (
                <Stack direction="row" alignItems="center" gap={1} sx={{ mb: `${SPACING.XS}px` }}>
                  <Typography sx={typo.labelSmall}>Probe:</Typography>
                  <Slider value={probeSize} onChange={(_, v) => setProbeSize(v as number)} disabled={lockMask} min={1} max={100} size="small"
                    sx={{ width: 80, "& .MuiSlider-thumb": { width: 10, height: 10 } }} />
                  <Typography sx={typo.value}>{probeSize}</Typography>
                  <Typography sx={{ ...typo.value, color: colors.textMuted, ml: `${SPACING.SM}px` }}>
                    @({localProbe.row}, {localProbe.col})
                  </Typography>
                </Stack>
              )}
              {analysisMode === "line" && (
                <Stack direction="row" alignItems="center" gap={1} sx={{ mb: `${SPACING.XS}px`, flexWrap: "wrap", rowGap: `${SPACING.XS}px` }}>
                  <Typography sx={typo.labelSmall}>Width:</Typography>
                  <Slider value={localLineWidth} onChange={(_, v) => setLocalLineWidth(v as number)}
                    onChangeCommitted={(_, v) => setLineWidth(v as number)} disabled={lockMask} min={1} max={100} size="small"
                    sx={{ width: 80, "& .MuiSlider-thumb": { width: 10, height: 10 } }} />
                  <Typography sx={typo.value}>{localLineWidth}</Typography>
                  <Button size="small" variant={linePerp ? "contained" : "outlined"} disabled={lockMask}
                    sx={{ ...compactBtn, minWidth: 24 }} onClick={() => setLinePerp(!linePerp)}
                    title="Sample perpendicular to the drawn line">⟂</Button>
                  {(["averaged", "linescan"] as const).map((lm) => (
                    <Button key={lm} size="small" variant={lineMode === lm ? "contained" : "outlined"} disabled={lockMask}
                      sx={{ ...compactBtn, minWidth: 30 }} onClick={() => setLineMode(lm)}>
                      {lm === "averaged" ? "avg" : "scan"}
                    </Button>
                  ))}
                  <Typography sx={{ ...typo.value, color: colors.textMuted }}>
                    {lineActive ? `len ${lineLenPx.toFixed(1)} px · θ ${lineOrientDeg.toFixed(1)}°` : "draw a line…"}
                  </Typography>
                  <Button size="small" sx={compactBtn} disabled={lockMask || !lineActive} onClick={clearLine}>Clear</Button>
                </Stack>
              )}
              <Stack direction="row" alignItems="center" gap={1}>
                <Typography sx={typo.labelSmall}>Cmap:</Typography>
                <Select
                  value={cmap}
                  onChange={(e) => setCmap(e.target.value)}
                  size="small"
                  sx={{
                    fontSize: 12,
                    height: 28,
                    minWidth: 80,
                    color: colors.text,
                    bgcolor: colors.controlBg,
                    "& .MuiSvgIcon-root": { color: colors.text },
                    "& .MuiOutlinedInput-notchedOutline": { borderColor: colors.border },
                    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: colors.accent },
                  }}
                  MenuProps={{
                    PaperProps: {
                      sx: {
                        bgcolor: colors.controlBg,
                        color: colors.text,
                        border: `1px solid ${colors.border}`,
                        "& .MuiMenuItem-root": { color: colors.text },
                        "& .MuiMenuItem-root:hover": { bgcolor: colors.bg },
                        "& .MuiMenuItem-root.Mui-selected": { bgcolor: colors.bg },
                      },
                    },
                  }}
                >
                  {CMAP_OPTIONS.map((name) => <MenuItem key={name} value={name} sx={{ fontSize: 12 }}>{name}</MenuItem>)}
                </Select>
              </Stack>
            </Box>
          )}
        </Box>

        {/* RIGHT: Curves */}
        <Box sx={{ width: PLOT_W, flexShrink: 0 }}>
          {/* Spacer matching the height of the left-panel mask-tools row, so
              both canvases sit at the same vertical position in mask mode. */}
          {analysisMode === "mask" && (
            <Stack direction="row" spacing={`2px`} alignItems="center" aria-hidden
              sx={{ mb: `${SPACING.XS}px`, flexWrap: "wrap", rowGap: `${SPACING.XS}px`, visibility: "hidden" }}>
              <Button size="small" sx={{ ...compactBtn, minWidth: 24 }}>▭</Button>
            </Stack>
          )}
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, minHeight: 32, flexWrap: "wrap", rowGap: `${SPACING.XS}px` }}>
            <Stack direction="row" spacing={`${SPACING.XS}px`}>
              {([["Ik", "I(k)"], ["Fk", "F(k)"], ["Gr", "G(r)"], ["gr", "g(r)"]] as const).map(([mode, label]) => (
                <Button key={mode} size="small" variant={plotMode === mode ? "contained" : "outlined"} sx={compactBtn}
                  onClick={() => setPlotMode(mode)}>{label}</Button>
              ))}
            </Stack>
            <Stack direction="row" alignItems="center" gap={1}>
              {plotMode === "Ik" && (<><Typography sx={typo.labelSmall}>Log:</Typography>
                <Switch checked={ikLogScale} onChange={(e) => { setIkLogScale(e.target.checked); userZoomedRef.current = false; setTimeout(autoFitPlot, 0); }} size="small" sx={switchSmall} />
                <Typography sx={typo.labelSmall}>Background:</Typography>
                <Switch checked={showBackground} onChange={(e) => setShowBackground(e.target.checked)} size="small" sx={switchSmall} /></>)}
              <Button size="small" sx={compactBtn} disabled={lockDisplay} onClick={() => { userZoomedRef.current = false; autoFitPlot(); }}>Reset</Button>
            </Stack>
          </Stack>
          <Box sx={imageBox} style={{ width: PLOT_W, height: PLOT_H }}>
            <canvas ref={plotCanvasRef} style={{ width: PLOT_W, height: PLOT_H, display: "block", cursor: analysisMode === "line" && lineMode === "linescan" ? "col-resize" : "default" }}
              onMouseDown={handlePlotMouseDown} onMouseMove={handlePlotMouseMove} onMouseUp={handlePlotMouseUp}
              onMouseLeave={() => { isPlotPanRef.current = false; draggingGuideRef.current = null; setCursorData(null); setHmHover(null); }}
              onDoubleClick={() => {
                if (analysisMode === "line" && lineMode === "linescan") { setHmGuides([]); return; }
                if (!lockDisplay) { userZoomedRef.current = false; autoFitPlot(); }
              }} onWheel={handlePlotWheel} />
          </Box>
          {showStats && !hideStats && (() => {
            const xLab = (plotMode === "Gr" || plotMode === "gr") ? "r" : "k";
            const yLab = plotMode === "Ik" ? "I" : plotMode === "Fk" ? "F" : plotMode === "gr" ? "g" : "G";
            if (analysisMode === "line" && lineMode === "linescan") {
              if (!hmHover) return null;
              return <Typography sx={{ ...typo.value, mt: `${SPACING.XS}px` }}>
                {xLab} = {formatNumber(hmHover.r, 4)}, position = {formatNumber(hmHover.pos, 2)}, {yLab} = {formatNumber(hmHover.val, 4)}
              </Typography>;
            }
            if (!cursorData) return null;
            const v = curveValueAt(cursorData.x);
            if (v === null) return null;
            return <Typography sx={{ ...typo.value, mt: `${SPACING.XS}px` }}>
              {xLab} = {formatNumber(cursorData.x, 4)}, {yLab} = {formatNumber(v, 4)}
            </Typography>;
          })()}
          {showControls && !hideParameters && (
            <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px` }}>
              <Stack direction="row" alignItems="center" gap={1}>
                <Typography sx={{ ...typo.labelSmall, minWidth: 55 }}>k fit:</Typography>
                <Slider value={localKFit} onChange={(_, v) => setLocalKFit(v as [number, number])}
                  onChangeCommitted={(_, v) => { const val = v as [number, number]; setKMinFit(val[0]); setKMaxFit(val[1]); }}
                  disabled={lockParameters} min={kMinAvail} max={kMaxAvail} step={0.01} size="small"
                  sx={{ flex: 1, "& .MuiSlider-thumb": { width: 10, height: 10 } }} />
                <Typography sx={{ ...typo.value, color: colors.textMuted, minWidth: 90 }}>[{localKFit[0].toFixed(2)}, {localKFit[1].toFixed(2)}]</Typography>
              </Stack>
              <Stack direction="row" alignItems="center" gap={1}>
                <Typography sx={{ ...typo.labelSmall, minWidth: 55 }}>k window:</Typography>
                <Slider value={localKWin} onChange={(_, v) => setLocalKWin(v as [number, number])}
                  onChangeCommitted={(_, v) => { const val = v as [number, number]; setKMinWindow(val[0]); setKMaxWindow(val[1]); }}
                  disabled={lockParameters} min={kMinAvail} max={kMaxAvail} step={0.01} size="small"
                  sx={{ flex: 1, "& .MuiSlider-thumb": { width: 10, height: 10 } }} />
                <Typography sx={{ ...typo.value, color: colors.textMuted, minWidth: 90 }}>[{localKWin[0].toFixed(2)}, {localKWin[1].toFixed(2)}]</Typography>
              </Stack>
              <Stack direction="row" alignItems="center" gap={1}>
                <Typography sx={{ ...typo.labelSmall, minWidth: 55 }}>r max:</Typography>
                <Slider value={localRMax} onChange={(_, v) => setLocalRMax(v as number)}
                  onChangeCommitted={(_, v) => setRMax(v as number)} disabled={lockParameters} min={1} max={50} step={0.5} size="small"
                  sx={{ flex: 1, "& .MuiSlider-thumb": { width: 10, height: 10 } }} />
                <Typography sx={{ ...typo.value, color: colors.textMuted, minWidth: 50 }}>{localRMax.toFixed(1)} Å</Typography>
              </Stack>
              <Stack direction="row" alignItems="center" gap={1}>
                <Typography sx={{ ...typo.labelSmall, minWidth: 55 }}>k lowpass:</Typography>
                <Slider value={localKLowpass} onChange={(_, v) => setLocalKLowpass(v as number)}
                  onChangeCommitted={(_, v) => setKLowpass(v as number)} disabled={lockParameters} min={0} max={0.1} step={0.001} size="small"
                  sx={{ flex: 1, "& .MuiSlider-thumb": { width: 10, height: 10 } }} />
                <Typography sx={{ ...typo.value, color: colors.textMuted, minWidth: 50 }}>{localKLowpass > 0 ? localKLowpass.toFixed(3) : "off"}</Typography>
              </Stack>
              <Stack direction="row" alignItems="center" gap={1}>
                <Typography sx={{ ...typo.labelSmall, minWidth: 55 }}>k highpass:</Typography>
                <Slider value={localKHighpass} onChange={(_, v) => setLocalKHighpass(v as number)}
                  onChangeCommitted={(_, v) => setKHighpass(v as number)} disabled={lockParameters} min={0} max={0.1} step={0.001} size="small"
                  sx={{ flex: 1, "& .MuiSlider-thumb": { width: 10, height: 10 } }} />
                <Typography sx={{ ...typo.value, color: colors.textMuted, minWidth: 50 }}>{localKHighpass > 0 ? localKHighpass.toFixed(3) : "off"}</Typography>
              </Stack>
              <Stack direction="row" alignItems="center" gap={1}>
                <Typography sx={typo.labelSmall}>Damp:</Typography>
                <Switch checked={dampOrigin} disabled={lockParameters} onChange={(e) => setDampOrigin(e.target.checked)} size="small" sx={switchSmall} />
                {dampOrigin && (<>
                  <Typography sx={{ ...typo.labelSmall, ml: 1 }}>r_cut:</Typography>
                  <Slider value={localRCut} onChange={(_, v) => setLocalRCut(v as number)}
                    onChangeCommitted={(_, v) => setRCut(v as number)} disabled={lockParameters} min={0.1} max={5} step={0.1} size="small"
                    sx={{ width: 80, "& .MuiSlider-thumb": { width: 10, height: 10 } }} />
                  <Typography sx={{ ...typo.value, color: colors.textMuted }}>{localRCut.toFixed(1)} Å</Typography>
                </>)}
              </Stack>
              {plotMode === "gr" && (
                <Stack direction="row" alignItems="center" gap={0}>
                  <Typography sx={{ ...typo.labelSmall, minWidth: 55 }}>Density:</Typography>
                  <Typography sx={typo.labelSmall}>Estimated</Typography>
                  <Switch
                    checked={densityMode === "manual"}
                    disabled={lockParameters}
                    onChange={(e) => setDensityMode(e.target.checked ? "manual" : "estimated")}
                    size="small"
                    sx={switchSmall}
                  />
                  <Typography sx={typo.labelSmall}>Manual</Typography>
                  <input
                    type="number"
                    step="0.001"
                    min={0}
                    value={localDensity}
                    disabled={lockParameters || densityMode === "estimated"}
                    onChange={(e) => setLocalDensity(e.target.value)}
                    onBlur={() => {
                      const v = parseFloat(localDensity);
                      if (!isNaN(v) && v > 0 && v !== densityValue) setDensityValue(v);
                      else setLocalDensity(densityValue.toPrecision(4));
                    }}
                    onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
                    style={{
                      width: 90, fontSize: 12, padding: "3px 6px", marginLeft: SPACING.LG, fontFamily: MONO,
                      border: `1px solid ${colors.border}`,
                      background: densityMode === "estimated" ? (isDark ? "#222" : "#f0f0f0") : (isDark ? "#1a1a1a" : "#fff"),
                      color: colors.textMuted,
                      outline: "none",
                    }}
                  />
                  <Typography sx={{ ...typo.labelSmall, ml: `${SPACING.XS}px` }}>Å⁻³</Typography>
                </Stack>
              )}
            </Box>
          )}
        </Box>
      </Stack>
    </Box>
  );
}

export const render = createRender(ShowPDFWidget);
