import { useEffect, useMemo, useRef, useState } from "react";
import { Search, ZoomIn, ZoomOut, RotateCcw, Box } from "lucide-react";
import type { EmbeddingPoint, ReductionMethod, RunRecord } from "../types";

interface ScatterPlotProps {
  runs: RunRecord[];
  selectedPointId: string | null;
  query: string;
  is3d: boolean;
  reduction: ReductionMethod;
  neighborhoodPointIds: Set<string> | null;
  onQueryChange: (query: string) => void;
  onPointSelect: (point: EmbeddingPoint) => void;
  onReductionChange: (reduction: ReductionMethod) => void;
  onToggle3d: (enabled: boolean) => void;
}

interface PlotPoint {
  point: EmbeddingPoint;
  color: string;
}

interface ProjectedPoint {
  point: EmbeddingPoint;
  color: string;
  cx: number;
  cy: number;
}

interface ViewState {
  scale: number;
  offsetX: number;
  offsetY: number;
}

const WIDTH = 860;
const HEIGHT = 640;
const PADDING = 58;
const AXIS_LIMIT = 6;
const CENTER_X = WIDTH / 2;
const CENTER_Y = HEIGHT / 2;
const LABEL_LIMIT = 2000;
const PICK_RADIUS = 12;
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 12;
const ZOOM_STEP = 1.35;
const INITIAL_VIEW: ViewState = { scale: 1, offsetX: 0, offsetY: 0 };
const PLOT_LABEL_MAX_CHARS = 32;
const PLOT_LABEL_EDGE_GUTTER = 14;
const REDUCTION_METHODS: ReductionMethod[] = ["PCA", "UMAP", "t-SNE"];

export function ScatterPlot({
  runs,
  selectedPointId,
  query,
  is3d,
  reduction,
  neighborhoodPointIds,
  onQueryChange,
  onPointSelect,
  onReductionChange,
  onToggle3d,
}: ScatterPlotProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragRef = useRef<{ startX: number; startY: number; startView: ViewState; moved: boolean } | null>(null);
  const primaryButtonDownRef = useRef(false);
  const [view, setView] = useState<ViewState>(INITIAL_VIEW);
  const points = useMemo<PlotPoint[]>(
    () => runs.filter((run) => run.visible).flatMap((run) => run.points.map((point) => ({ point, color: run.color }))),
    [runs],
  );
  const hasPoints = points.length > 0;
  const selected = points.find(({ point }) => point.id === selectedPointId) ?? points[0];
  const filtered = useMemo(() => {
    const normalizedQuery = query.toLowerCase();
    return points.filter(({ point }) => {
      if (neighborhoodPointIds && !neighborhoodPointIds.has(point.id)) return false;
      const target = `${point.label} ${point.snippet} ${point.source}`.toLowerCase();
      return target.includes(normalizedQuery);
    });
  }, [neighborhoodPointIds, points, query]);

  const projected = useMemo(
    () =>
      filtered.map(({ point, color }) => {
        const depthOffset = is3d ? point.z * 9 : 0;
        const baseX = mapX(point.x) + depthOffset;
        const baseY = mapY(point.y) - depthOffset * 0.35;
        const transformed = applyView(baseX, baseY, view);
        return {
          point,
          color,
          cx: transformed.x,
          cy: transformed.y,
        };
      }),
    [filtered, is3d, view],
  );

  const selectedProjected = useMemo(() => {
    if (!selected) return null;
    return projected.find(({ point }) => point.id === selected.point.id) ?? null;
  }, [projected, selected]);

  const labelStride = query ? Math.max(Math.ceil(projected.length / 32), 1) : Math.max(Math.ceil(projected.length / 12), 1);
  const showAmbientLabels = projected.length <= LABEL_LIMIT;
  const axisTicks = useMemo(() => buildAxisTicks(view), [view]);
  const emptyMessage =
    points.length === 0
      ? runs.length > 0
        ? "No visible points. Turn a run back on to show embeddings."
        : "Run a projection to render embeddings."
      : projected.length === 0
        ? query
          ? "No points match the current search."
          : "No points in the selected neighborhood."
        : "";

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    renderWebGlPoints(canvas, projected, selected?.point.id ?? null, neighborhoodPointIds);
  }, [neighborhoodPointIds, projected, selected?.point.id]);

  function handleZoomIn() {
    setView((current) => zoomAt(current, ZOOM_STEP, CENTER_X, CENTER_Y));
  }

  function handleZoomOut() {
    setView((current) => zoomAt(current, 1 / ZOOM_STEP, CENTER_X, CENTER_Y));
  }

  function handleResetView() {
    setView(INITIAL_VIEW);
  }

  function handleCanvasWheel(event: React.WheelEvent<HTMLCanvasElement>) {
    if (!primaryButtonDownRef.current) return;

    event.preventDefault();
    const { x, y } = canvasPoint(event);
    const factor = event.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP;
    setView((current) => zoomAt(current, factor, x, y));
  }

  function handleCanvasMouseDown(event: React.MouseEvent<HTMLCanvasElement>) {
    if (event.button !== 0) return;

    primaryButtonDownRef.current = true;
    const { x, y } = canvasPoint(event);
    dragRef.current = { startX: x, startY: y, startView: view, moved: false };
  }

  function handleCanvasMouseMove(event: React.MouseEvent<HTMLCanvasElement>) {
    const drag = dragRef.current;
    if (!drag) return;
    const { x, y } = canvasPoint(event);
    const dx = x - drag.startX;
    const dy = y - drag.startY;
    if (Math.abs(dx) + Math.abs(dy) > 2) {
      drag.moved = true;
    }
    setView({
      ...drag.startView,
      offsetX: drag.startView.offsetX + dx,
      offsetY: drag.startView.offsetY + dy,
    });
  }

  function handleCanvasMouseUp() {
    primaryButtonDownRef.current = false;
    window.setTimeout(() => {
      dragRef.current = null;
    }, 0);
  }

  function handleCanvasClick(event: React.MouseEvent<HTMLCanvasElement>) {
    if (dragRef.current?.moved) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const { x, y } = canvasPoint(event);
    const nearest = nearestPoint(projected, x, y);
    if (nearest) {
      onPointSelect(nearest.point);
    }
  }

  return (
    <main className="plotRegion" aria-label="Embedding projection">
      <div className="plotToolbar">
        <label className="searchBox">
          <Search size={18} />
          <input
            value={query}
            onChange={(event) => onQueryChange(event.target.value)}
            placeholder="Search points..."
            aria-label="Search points"
            disabled={!hasPoints}
          />
        </label>

        <div className="segmented projectionMethods" aria-label="Reduction method">
          {REDUCTION_METHODS.map((method) => (
            <button
              key={method}
              className={reduction === method ? "active" : ""}
              type="button"
              onClick={() => onReductionChange(method)}
              title={`Project with ${method}`}
              data-testid={`reduction-${method}`}
            >
              {method}
            </button>
          ))}
        </div>

        <div className="iconGroup" aria-label="Plot controls">
          <button type="button" title="Zoom in" onClick={handleZoomIn} disabled={!hasPoints}>
            <ZoomIn size={18} />
          </button>
          <button type="button" title="Zoom out" onClick={handleZoomOut} disabled={!hasPoints}>
            <ZoomOut size={18} />
          </button>
          <button type="button" title="Reset view" onClick={handleResetView} disabled={!hasPoints}>
            <RotateCcw size={18} />
          </button>
        </div>

        <div className="segmented small" aria-label="Projection dimension">
          <button className={!is3d ? "active" : ""} type="button" onClick={() => onToggle3d(false)} disabled={!hasPoints}>
            2D
          </button>
          <button className={is3d ? "active" : ""} type="button" onClick={() => onToggle3d(true)} disabled={!hasPoints}>
            3D
          </button>
        </div>

      </div>

      <div className="plotCanvas">
        <canvas
          ref={canvasRef}
          className="pointCloudCanvas"
          width={WIDTH}
          height={HEIGHT}
          aria-label="2D embedding point cloud"
          data-testid="point-cloud-canvas"
          onWheel={handleCanvasWheel}
          onMouseDown={handleCanvasMouseDown}
          onMouseMove={handleCanvasMouseMove}
          onMouseUp={handleCanvasMouseUp}
          onMouseLeave={handleCanvasMouseUp}
          onClick={handleCanvasClick}
        />
        <svg
          className="plotOverlay"
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          preserveAspectRatio="none"
          role="img"
          aria-label="2D embedding scatter plot"
        >
          {axisTicks.x.map(({ value, position }) => (
            <g key={`x-${value}`}>
              <line className="gridLine" x1={position} x2={position} y1={PADDING} y2={HEIGHT - PADDING} />
              <text className="axisTick" x={position} y={HEIGHT - 24} textAnchor="middle" data-testid="x-axis-tick">
                {formatTick(value)}
              </text>
            </g>
          ))}

          {axisTicks.y.map(({ value, position }) => (
            <g key={`y-${value}`}>
              <line className="gridLine" x1={PADDING} x2={WIDTH - PADDING} y1={position} y2={position} />
              <text className="axisTick" x={28} y={position + 4} textAnchor="middle" data-testid="y-axis-tick">
                {formatTick(value)}
              </text>
            </g>
          ))}

          <line className="axisLine" x1={PADDING} x2={WIDTH - PADDING} y1={HEIGHT - PADDING} y2={HEIGHT - PADDING} />
          <line className="axisLine" x1={PADDING} x2={PADDING} y1={PADDING} y2={HEIGHT - PADDING} />
          <text className="axisLabel" x={WIDTH / 2} y={HEIGHT - 6} textAnchor="middle">
            Axis 1
          </text>
          <text className="axisLabel" transform={`translate(14 ${HEIGHT / 2}) rotate(-90)`} textAnchor="middle">
            Axis 2
          </text>

          {projected.map(({ point, cx, cy }, index) => {
            const isSelected = point.id === selected?.point.id;
            const showLabel = isSelected || (query ? index % labelStride === 0 : showAmbientLabels && index % labelStride === 0);
            const labelX = clamp(cx + (cx > WIDTH - 170 ? -12 : 12), PLOT_LABEL_EDGE_GUTTER, WIDTH - PLOT_LABEL_EDGE_GUTTER);
            const labelY = clamp(cy - 8, PADDING - 16, HEIGHT - PADDING + 16);
            const labelAnchor = cx > WIDTH - 170 ? "end" : "start";
            const label = truncatePlotLabel(point.label);
            return showLabel ? (
              <text className="pointLabel" x={labelX} y={labelY} textAnchor={labelAnchor} key={point.id} data-testid="point-label">
                <title>{point.label}</title>
                {label}
              </text>
            ) : null;
          })}

          {selectedProjected ? <SelectedPointMarker selected={selectedProjected} /> : null}
        </svg>

        {emptyMessage ? (
          <div className="emptyPlot">
            <Box size={28} />
            <span>{emptyMessage}</span>
          </div>
        ) : null}
      </div>
    </main>
  );
}

function SelectedPointMarker({ selected }: { selected: ProjectedPoint }) {
  return <circle className="selectedPointRing" cx={selected.cx} cy={selected.cy} r="8" fill={selected.color} data-testid="selected-point-marker" />;
}

function renderWebGlPoints(
  canvas: HTMLCanvasElement,
  points: ProjectedPoint[],
  selectedPointId: string | null,
  neighborhoodPointIds: Set<string> | null,
) {
  const gl = canvas.getContext("webgl", { antialias: true });
  if (!gl) {
    renderCanvasPoints(canvas, points, selectedPointId, neighborhoodPointIds);
    return;
  }

  const program = createProgram(gl);
  if (!program) {
    renderCanvasPoints(canvas, points, selectedPointId, neighborhoodPointIds);
    return;
  }

  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.useProgram(program.program);

  const vertices = new Float32Array(points.length * 6);
  points.forEach(({ point, color, cx, cy }, index) => {
    const rgb = hexToRgb(color);
    const offset = index * 6;
    vertices[offset] = (cx / WIDTH) * 2 - 1;
    vertices[offset + 1] = 1 - (cy / HEIGHT) * 2;
    vertices[offset + 2] = rgb[0];
    vertices[offset + 3] = rgb[1];
    vertices[offset + 4] = rgb[2];
    vertices[offset + 5] = point.id === selectedPointId ? 1 : neighborhoodPointIds?.has(point.id) ? 0.55 : 0;
  });

  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STREAM_DRAW);

  const stride = 6 * Float32Array.BYTES_PER_ELEMENT;
  gl.enableVertexAttribArray(program.position);
  gl.vertexAttribPointer(program.position, 2, gl.FLOAT, false, stride, 0);
  gl.enableVertexAttribArray(program.color);
  gl.vertexAttribPointer(program.color, 3, gl.FLOAT, false, stride, 2 * Float32Array.BYTES_PER_ELEMENT);
  gl.enableVertexAttribArray(program.selected);
  gl.vertexAttribPointer(program.selected, 1, gl.FLOAT, false, stride, 5 * Float32Array.BYTES_PER_ELEMENT);

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.drawArrays(gl.POINTS, 0, points.length);
  gl.deleteBuffer(buffer);
}

function createProgram(gl: WebGLRenderingContext) {
  const vertexShader = compileShader(
    gl,
    gl.VERTEX_SHADER,
    `
      attribute vec2 a_position;
      attribute vec3 a_color;
      attribute float a_focus;
      varying vec3 v_color;
      varying float v_focus;

      void main() {
        v_color = a_color;
        v_focus = a_focus;
        gl_Position = vec4(a_position, 0.0, 1.0);
        gl_PointSize = mix(2.0, 10.0, v_focus);
      }
    `,
  );
  const fragmentShader = compileShader(
    gl,
    gl.FRAGMENT_SHADER,
    `
      precision mediump float;
      varying vec3 v_color;
      varying float v_focus;

      void main() {
        vec2 delta = gl_PointCoord - vec2(0.5);
        float dist = length(delta);
        if (dist > 0.5) discard;
        float alpha = smoothstep(0.5, 0.32, dist);
        vec3 color = mix(v_color, vec3(0.06, 0.09, 0.16), v_focus * 0.35);
        gl_FragColor = vec4(color, alpha * 0.88);
      }
    `,
  );

  if (!vertexShader || !fragmentShader) return null;

  const program = gl.createProgram();
  if (!program) return null;
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    return null;
  }

  return {
    program,
    position: gl.getAttribLocation(program, "a_position"),
    color: gl.getAttribLocation(program, "a_color"),
    selected: gl.getAttribLocation(program, "a_focus"),
  };
}

function compileShader(gl: WebGLRenderingContext, type: number, source: string) {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  return gl.getShaderParameter(shader, gl.COMPILE_STATUS) ? shader : null;
}

function renderCanvasPoints(
  canvas: HTMLCanvasElement,
  points: ProjectedPoint[],
  selectedPointId: string | null,
  neighborhoodPointIds: Set<string> | null,
) {
  const context = canvas.getContext("2d");
  if (!context) return;

  context.clearRect(0, 0, canvas.width, canvas.height);
  for (const { point, color, cx, cy } of points) {
    const selected = point.id === selectedPointId;
    const neighbor = !selected && neighborhoodPointIds?.has(point.id);
    context.beginPath();
    context.arc(cx, cy, selected ? 5 : neighbor ? 3.2 : 1.5, 0, Math.PI * 2);
    context.fillStyle = selected ? "#0f172a" : color;
    context.globalAlpha = selected || neighbor ? 1 : 0.85;
    context.fill();
  }
  context.globalAlpha = 1;
}

function nearestPoint(points: ProjectedPoint[], x: number, y: number) {
  let nearest: ProjectedPoint | null = null;
  let nearestDistance = PICK_RADIUS * PICK_RADIUS;
  for (const point of points) {
    const dx = point.cx - x;
    const dy = point.cy - y;
    const distance = dx * dx + dy * dy;
    if (distance <= nearestDistance) {
      nearest = point;
      nearestDistance = distance;
    }
  }
  return nearest;
}

function truncatePlotLabel(label: string) {
  const normalized = label.replace(/\s+/g, " ").trim();
  return normalized.length > PLOT_LABEL_MAX_CHARS ? `${normalized.slice(0, PLOT_LABEL_MAX_CHARS - 3)}...` : normalized;
}

function hexToRgb(hex: string) {
  const value = Number.parseInt(hex.slice(1), 16);
  return [((value >> 16) & 255) / 255, ((value >> 8) & 255) / 255, (value & 255) / 255] as const;
}

function canvasPoint(event: React.MouseEvent<HTMLCanvasElement> | React.WheelEvent<HTMLCanvasElement>) {
  const rect = event.currentTarget.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * WIDTH,
    y: ((event.clientY - rect.top) / rect.height) * HEIGHT,
  };
}

function applyView(x: number, y: number, view: ViewState) {
  return {
    x: CENTER_X + (x - CENTER_X) * view.scale + view.offsetX,
    y: CENTER_Y + (y - CENTER_Y) * view.scale + view.offsetY,
  };
}

function invertView(x: number, y: number, view: ViewState) {
  return {
    x: CENTER_X + (x - CENTER_X - view.offsetX) / view.scale,
    y: CENTER_Y + (y - CENTER_Y - view.offsetY) / view.scale,
  };
}

function zoomAt(view: ViewState, factor: number, anchorX: number, anchorY: number) {
  const nextScale = clamp(view.scale * factor, MIN_ZOOM, MAX_ZOOM);
  const scaleRatio = nextScale / view.scale;
  return {
    scale: nextScale,
    offsetX: anchorX - CENTER_X - (anchorX - CENTER_X - view.offsetX) * scaleRatio,
    offsetY: anchorY - CENTER_Y - (anchorY - CENTER_Y - view.offsetY) * scaleRatio,
  };
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function mapX(value: number) {
  return PADDING + ((value + AXIS_LIMIT) / (AXIS_LIMIT * 2)) * (WIDTH - PADDING * 2);
}

function mapY(value: number) {
  return PADDING + ((AXIS_LIMIT - value) / (AXIS_LIMIT * 2)) * (HEIGHT - PADDING * 2);
}

function unmapX(position: number) {
  return ((position - PADDING) / (WIDTH - PADDING * 2)) * (AXIS_LIMIT * 2) - AXIS_LIMIT;
}

function unmapY(position: number) {
  return AXIS_LIMIT - ((position - PADDING) / (HEIGHT - PADDING * 2)) * (AXIS_LIMIT * 2);
}

function buildAxisTicks(view: ViewState) {
  const left = unmapX(invertView(PADDING, CENTER_Y, view).x);
  const right = unmapX(invertView(WIDTH - PADDING, CENTER_Y, view).x);
  const bottom = unmapY(invertView(CENTER_X, HEIGHT - PADDING, view).y);
  const top = unmapY(invertView(CENTER_X, PADDING, view).y);

  return {
    x: visibleTicks(left, right).map((value) => ({ value, position: applyView(mapX(value), CENTER_Y, view).x })),
    y: visibleTicks(bottom, top).map((value) => ({ value, position: applyView(CENTER_X, mapY(value), view).y })),
  };
}

function visibleTicks(start: number, end: number) {
  const min = Math.min(start, end);
  const max = Math.max(start, end);
  const step = niceTickStep((max - min) / 6);
  const first = Math.ceil(min / step) * step;
  const ticks: number[] = [];

  for (let value = first; value <= max + step * 0.5; value += step) {
    ticks.push(roundTick(value));
  }

  return ticks;
}

function niceTickStep(rawStep: number) {
  if (!Number.isFinite(rawStep) || rawStep <= 0) return 1;
  const exponent = Math.floor(Math.log10(rawStep));
  const magnitude = 10 ** exponent;
  const normalized = rawStep / magnitude;
  const nice = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  return nice * magnitude;
}

function roundTick(value: number) {
  return Math.abs(value) < 1e-9 ? 0 : Number(value.toPrecision(12));
}

function formatTick(value: number) {
  if (Number.isInteger(value)) return String(value);
  return value.toFixed(Math.abs(value) >= 1 ? 1 : 2).replace(/\.?0+$/, "");
}
