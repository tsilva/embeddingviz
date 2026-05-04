import { useEffect, useMemo, useRef, useState } from "react";
import { Search, ZoomIn, ZoomOut, RotateCcw, Move, Box } from "lucide-react";
import type { EmbeddingPoint, ReductionMethod, RunRecord } from "../types";

interface ScatterPlotProps {
  runs: RunRecord[];
  selectedPointId: string | null;
  query: string;
  is3d: boolean;
  primaryReduction: ReductionMethod;
  onQueryChange: (query: string) => void;
  onPointSelect: (point: EmbeddingPoint) => void;
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
const CENTER_X = WIDTH / 2;
const CENTER_Y = HEIGHT / 2;
const LABEL_LIMIT = 2000;
const PICK_RADIUS = 12;
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 12;
const ZOOM_STEP = 1.35;
const INITIAL_VIEW: ViewState = { scale: 1, offsetX: 0, offsetY: 0 };

export function ScatterPlot({
  runs,
  selectedPointId,
  query,
  is3d,
  primaryReduction,
  onQueryChange,
  onPointSelect,
  onToggle3d,
}: ScatterPlotProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragRef = useRef<{ startX: number; startY: number; startView: ViewState; moved: boolean } | null>(null);
  const [view, setView] = useState<ViewState>(INITIAL_VIEW);
  const points = useMemo<PlotPoint[]>(
    () => runs.filter((run) => run.visible).flatMap((run) => run.points.map((point) => ({ point, color: run.color }))),
    [runs],
  );
  const selected = points.find(({ point }) => point.id === selectedPointId) ?? points[0];
  const filtered = useMemo(() => {
    const normalizedQuery = query.toLowerCase();
    return points.filter(({ point }) => {
      const target = `${point.label} ${point.snippet} ${point.source}`.toLowerCase();
      return target.includes(normalizedQuery);
    });
  }, [points, query]);

  const projected = useMemo(
    () =>
      filtered.map(({ point, color }) => {
        const depthOffset = is3d ? point.z * 9 : 0;
        const baseX = mapAxis(point.x, WIDTH, PADDING) + depthOffset;
        const baseY = mapAxis(-point.y, HEIGHT, PADDING) - depthOffset * 0.35;
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
    const depthOffset = is3d ? selected.point.z * 9 : 0;
    const transformed = applyView(
      mapAxis(selected.point.x, WIDTH, PADDING) + depthOffset,
      mapAxis(-selected.point.y, HEIGHT, PADDING) - depthOffset * 0.35,
      view,
    );
    return {
      point: selected.point,
      color: selected.color,
      cx: transformed.x,
      cy: transformed.y,
    };
  }, [selected, is3d, view]);

  const labelStride = query ? Math.max(Math.ceil(projected.length / 32), 1) : Math.max(Math.ceil(projected.length / 12), 1);
  const showAmbientLabels = projected.length <= LABEL_LIMIT;
  const axisPrefix = primaryReduction === "PCA" ? "PC" : primaryReduction;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    renderWebGlPoints(canvas, projected, selected?.point.id ?? null);
  }, [projected, selected?.point.id]);

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
    event.preventDefault();
    const { x, y } = canvasPoint(event);
    const factor = event.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP;
    setView((current) => zoomAt(current, factor, x, y));
  }

  function handleCanvasMouseDown(event: React.MouseEvent<HTMLCanvasElement>) {
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
          />
        </label>

        <div className="iconGroup" aria-label="Plot controls">
          <button type="button" title="Zoom in" onClick={handleZoomIn}>
            <ZoomIn size={18} />
          </button>
          <button type="button" title="Zoom out" onClick={handleZoomOut}>
            <ZoomOut size={18} />
          </button>
          <button type="button" title="Pan by dragging the plot">
            <Move size={18} />
          </button>
          <button type="button" title="Reset view" onClick={handleResetView}>
            <RotateCcw size={18} />
          </button>
        </div>

        <div className="segmented small" aria-label="Projection dimension">
          <button className={!is3d ? "active" : ""} type="button" onClick={() => onToggle3d(false)}>
            2D
          </button>
          <button className={is3d ? "active" : ""} type="button" onClick={() => onToggle3d(true)}>
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
          aria-label={`2D ${primaryReduction} point cloud`}
          onWheel={handleCanvasWheel}
          onMouseDown={handleCanvasMouseDown}
          onMouseMove={handleCanvasMouseMove}
          onMouseUp={handleCanvasMouseUp}
          onMouseLeave={handleCanvasMouseUp}
          onClick={handleCanvasClick}
        />
        <svg className="plotOverlay" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={`2D ${primaryReduction} scatter plot`}>
          <defs>
            <filter id="tooltipShadow" x="-20%" y="-20%" width="140%" height="140%">
              <feDropShadow dx="0" dy="12" stdDeviation="10" floodOpacity="0.14" />
            </filter>
          </defs>

          {Array.from({ length: 7 }, (_, index) => {
            const x = PADDING + index * ((WIDTH - PADDING * 2) / 6);
            const y = PADDING + index * ((HEIGHT - PADDING * 2) / 6);
            const label = -6 + index * 2;
            return (
              <g key={index}>
                <line className="gridLine" x1={x} x2={x} y1={PADDING} y2={HEIGHT - PADDING} />
                <line className="gridLine" x1={PADDING} x2={WIDTH - PADDING} y1={y} y2={y} />
                <text className="axisTick" x={x} y={HEIGHT - 24} textAnchor="middle">
                  {label}
                </text>
                <text className="axisTick" x={28} y={HEIGHT - y + 4} textAnchor="middle">
                  {label}
                </text>
              </g>
            );
          })}

          <line className="axisLine" x1={PADDING} x2={WIDTH - PADDING} y1={HEIGHT - PADDING} y2={HEIGHT - PADDING} />
          <line className="axisLine" x1={PADDING} x2={PADDING} y1={PADDING} y2={HEIGHT - PADDING} />
          <text className="axisLabel" x={WIDTH / 2} y={HEIGHT - 6} textAnchor="middle">
            {axisPrefix} 1
          </text>
          <text className="axisLabel" transform={`translate(14 ${HEIGHT / 2}) rotate(-90)`} textAnchor="middle">
            {axisPrefix} 2
          </text>

          {projected.map(({ point, cx, cy }, index) => {
            const isSelected = point.id === selected?.point.id;
            const showLabel = isSelected || (query ? index % labelStride === 0 : showAmbientLabels && index % labelStride === 0);
            return showLabel ? (
              <text className="pointLabel" x={cx + 12} y={cy - 8} key={point.id}>
                {point.label}
              </text>
            ) : null;
          })}

          {selectedProjected ? <SelectedPointOverlay selected={selectedProjected} axisPrefix={axisPrefix} /> : null}
        </svg>

        {points.length === 0 ? (
          <div className="emptyPlot">
            <Box size={28} />
            <span>Run a projection to render embeddings.</span>
          </div>
        ) : null}
      </div>
    </main>
  );
}

function SelectedPointOverlay({ selected, axisPrefix }: { selected: ProjectedPoint; axisPrefix: string }) {
  const { point, color, cx, cy } = selected;
  return (
    <>
      <circle className="selectedPointRing" cx={cx} cy={cy} r="8" fill={color} />
      <g transform={`translate(${Math.min(cx + 34, WIDTH - 246)} ${Math.max(cy - 20, 74)})`}>
        <rect className="tooltipPanel" width="218" height="104" rx="8" filter="url(#tooltipShadow)" />
        <circle cx="20" cy="24" r="6" fill={color} />
        <text className="tooltipTitle" x="36" y="29">
          {point.label}
        </text>
        <text className="tooltipKey" x="18" y="58">
          Output
        </text>
        <text className="tooltipValue" x="78" y="58">
          {point.output}
        </text>
        <text className="tooltipKey" x="18" y="84">
          {axisPrefix} 1
        </text>
        <text className="tooltipValue" x="78" y="84">
          {point.x.toFixed(2)}
        </text>
        <text className="tooltipKey" x="132" y="84">
          {axisPrefix} 2
        </text>
        <text className="tooltipValue" x="174" y="84">
          {point.y.toFixed(2)}
        </text>
      </g>
    </>
  );
}

function renderWebGlPoints(canvas: HTMLCanvasElement, points: ProjectedPoint[], selectedPointId: string | null) {
  const gl = canvas.getContext("webgl", { antialias: true });
  if (!gl) {
    renderCanvasPoints(canvas, points, selectedPointId);
    return;
  }

  const program = createProgram(gl);
  if (!program) {
    renderCanvasPoints(canvas, points, selectedPointId);
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
    vertices[offset + 5] = point.id === selectedPointId ? 1 : 0;
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
      attribute float a_selected;
      varying vec3 v_color;
      varying float v_selected;

      void main() {
        v_color = a_color;
        v_selected = a_selected;
        gl_Position = vec4(a_position, 0.0, 1.0);
        gl_PointSize = mix(2.0, 10.0, a_selected);
      }
    `,
  );
  const fragmentShader = compileShader(
    gl,
    gl.FRAGMENT_SHADER,
    `
      precision mediump float;
      varying vec3 v_color;
      varying float v_selected;

      void main() {
        vec2 delta = gl_PointCoord - vec2(0.5);
        float dist = length(delta);
        if (dist > 0.5) discard;
        float alpha = smoothstep(0.5, 0.32, dist);
        vec3 color = mix(v_color, vec3(0.06, 0.09, 0.16), v_selected * 0.35);
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
    selected: gl.getAttribLocation(program, "a_selected"),
  };
}

function compileShader(gl: WebGLRenderingContext, type: number, source: string) {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  return gl.getShaderParameter(shader, gl.COMPILE_STATUS) ? shader : null;
}

function renderCanvasPoints(canvas: HTMLCanvasElement, points: ProjectedPoint[], selectedPointId: string | null) {
  const context = canvas.getContext("2d");
  if (!context) return;

  context.clearRect(0, 0, canvas.width, canvas.height);
  for (const { point, color, cx, cy } of points) {
    const selected = point.id === selectedPointId;
    context.beginPath();
    context.arc(cx, cy, selected ? 5 : 1.5, 0, Math.PI * 2);
    context.fillStyle = selected ? "#0f172a" : color;
    context.globalAlpha = selected ? 1 : 0.85;
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

function mapAxis(value: number, size: number, padding: number) {
  const range = size - padding * 2;
  return padding + ((value + 6) / 12) * range;
}
