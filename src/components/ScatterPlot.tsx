import { useEffect, useMemo, useRef } from "react";
import { Search, ZoomIn, ZoomOut, RotateCcw, Move, Box } from "lucide-react";
import type { EmbeddingPoint, RunRecord } from "../types";
import { colorForGroup } from "../lib/embeddings";

interface ScatterPlotProps {
  runs: RunRecord[];
  selectedPointId: string | null;
  query: string;
  is3d: boolean;
  onQueryChange: (query: string) => void;
  onPointSelect: (point: EmbeddingPoint) => void;
  onToggle3d: (enabled: boolean) => void;
}

interface ProjectedPoint {
  point: EmbeddingPoint;
  cx: number;
  cy: number;
}

const WIDTH = 860;
const HEIGHT = 640;
const PADDING = 58;
const LABEL_LIMIT = 2000;
const PICK_RADIUS = 12;

export function ScatterPlot({
  runs,
  selectedPointId,
  query,
  is3d,
  onQueryChange,
  onPointSelect,
  onToggle3d,
}: ScatterPlotProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const points = useMemo(() => runs.filter((run) => run.visible).flatMap((run) => run.points), [runs]);
  const selected = points.find((point) => point.id === selectedPointId) ?? points[0];
  const filtered = useMemo(() => {
    const normalizedQuery = query.toLowerCase();
    return points.filter((point) => {
      const target = `${point.label} ${point.snippet} ${point.group}`.toLowerCase();
      return target.includes(normalizedQuery);
    });
  }, [points, query]);

  const projected = useMemo(
    () =>
      filtered.map((point) => {
        const depthOffset = is3d ? point.z * 9 : 0;
        return {
          point,
          cx: mapAxis(point.x, WIDTH, PADDING) + depthOffset,
          cy: mapAxis(-point.y, HEIGHT, PADDING) - depthOffset * 0.35,
        };
      }),
    [filtered, is3d],
  );

  const selectedProjected = selected
    ? {
        point: selected,
        cx: mapAxis(selected.x, WIDTH, PADDING) + (is3d ? selected.z * 9 : 0),
        cy: mapAxis(-selected.y, HEIGHT, PADDING) - (is3d ? selected.z * 9 * 0.35 : 0),
      }
    : null;

  const labelStride = query ? Math.max(Math.ceil(projected.length / 32), 1) : Math.max(Math.ceil(projected.length / 12), 1);
  const showAmbientLabels = projected.length <= LABEL_LIMIT;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    renderWebGlPoints(canvas, projected, selected?.id ?? null);
  }, [projected, selected?.id]);

  function handleCanvasClick(event: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * WIDTH;
    const y = ((event.clientY - rect.top) / rect.height) * HEIGHT;
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
          <button type="button" title="Zoom in">
            <ZoomIn size={18} />
          </button>
          <button type="button" title="Zoom out">
            <ZoomOut size={18} />
          </button>
          <button type="button" title="Pan">
            <Move size={18} />
          </button>
          <button type="button" title="Reset view">
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

        <label className="selectControl colorBy">
          <span>Color by</span>
          <select aria-label="Color by">
            <option>Group</option>
            <option>Run</option>
            <option>Output</option>
          </select>
        </label>
      </div>

      <div className="plotCanvas">
        <canvas
          ref={canvasRef}
          className="pointCloudCanvas"
          width={WIDTH}
          height={HEIGHT}
          aria-label="2D PCA point cloud"
          onClick={handleCanvasClick}
        />
        <svg className="plotOverlay" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label="2D PCA scatter plot">
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
            PC 1
          </text>
          <text className="axisLabel" transform={`translate(14 ${HEIGHT / 2}) rotate(-90)`} textAnchor="middle">
            PC 2
          </text>

          {projected.map(({ point, cx, cy }, index) => {
            const isSelected = point.id === selected?.id;
            const showLabel = isSelected || (query ? index % labelStride === 0 : showAmbientLabels && index % labelStride === 0);
            return showLabel ? (
              <text className="pointLabel" x={cx + 12} y={cy - 8} key={point.id}>
                {point.label}
              </text>
            ) : null;
          })}

          {selectedProjected ? <SelectedPointOverlay selected={selectedProjected} /> : null}
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

function SelectedPointOverlay({ selected }: { selected: ProjectedPoint }) {
  const { point, cx, cy } = selected;
  return (
    <>
      <circle className="selectedPointRing" cx={cx} cy={cy} r="8" fill={colorForGroup(point.group)} />
      <g transform={`translate(${Math.min(cx + 34, WIDTH - 246)} ${Math.max(cy - 20, 74)})`}>
        <rect className="tooltipPanel" width="218" height="128" rx="8" filter="url(#tooltipShadow)" />
        <circle cx="20" cy="24" r="6" fill={colorForGroup(point.group)} />
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
          Group
        </text>
        <text className="tooltipBadge" x="78" y="84">
          {point.group}
        </text>
        <text className="tooltipKey" x="18" y="110">
          PC 1
        </text>
        <text className="tooltipValue" x="78" y="110">
          {point.x.toFixed(2)}
        </text>
        <text className="tooltipKey" x="132" y="110">
          PC 2
        </text>
        <text className="tooltipValue" x="174" y="110">
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
  points.forEach(({ point, cx, cy }, index) => {
    const color = hexToRgb(colorForGroup(point.group));
    const offset = index * 6;
    vertices[offset] = (cx / WIDTH) * 2 - 1;
    vertices[offset + 1] = 1 - (cy / HEIGHT) * 2;
    vertices[offset + 2] = color[0];
    vertices[offset + 3] = color[1];
    vertices[offset + 4] = color[2];
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
  for (const { point, cx, cy } of points) {
    const selected = point.id === selectedPointId;
    context.beginPath();
    context.arc(cx, cy, selected ? 5 : 1.5, 0, Math.PI * 2);
    context.fillStyle = selected ? "#0f172a" : colorForGroup(point.group);
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

function mapAxis(value: number, size: number, padding: number) {
  const range = size - padding * 2;
  return padding + ((value + 6) / 12) * range;
}
