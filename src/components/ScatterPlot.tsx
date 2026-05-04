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

const WIDTH = 860;
const HEIGHT = 640;
const PADDING = 58;

export function ScatterPlot({
  runs,
  selectedPointId,
  query,
  is3d,
  onQueryChange,
  onPointSelect,
  onToggle3d,
}: ScatterPlotProps) {
  const points = runs.filter((run) => run.visible).flatMap((run) => run.points);
  const selected = points.find((point) => point.id === selectedPointId) ?? points[0];
  const filtered = points.filter((point) => {
    const target = `${point.label} ${point.snippet} ${point.group}`.toLowerCase();
    return target.includes(query.toLowerCase());
  });

  const projected = filtered.map((point) => {
    const depthOffset = is3d ? point.z * 9 : 0;
    return {
      point,
      cx: mapAxis(point.x, WIDTH, PADDING) + depthOffset,
      cy: mapAxis(-point.y, HEIGHT, PADDING) - depthOffset * 0.35,
    };
  });

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
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label="2D PCA scatter plot">
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
            const showLabel = isSelected || index % Math.max(Math.ceil(projected.length / 9), 1) === 0;
            return (
              <g key={point.id}>
                <circle
                  className={`point ${isSelected ? "selected" : ""}`}
                  cx={cx}
                  cy={cy}
                  r={isSelected ? 7 : 4.5}
                  fill={colorForGroup(point.group)}
                  onClick={() => onPointSelect(point)}
                />
                {showLabel ? (
                  <text className="pointLabel" x={cx + 12} y={cy - 8}>
                    {point.label}
                  </text>
                ) : null}
              </g>
            );
          })}

          {selected ? (
            <g transform={`translate(${Math.min(mapAxis(selected.x, WIDTH, PADDING) + 34, WIDTH - 246)} ${Math.max(mapAxis(-selected.y, HEIGHT, PADDING) - 20, 74)})`}>
              <rect className="tooltipPanel" width="218" height="128" rx="8" filter="url(#tooltipShadow)" />
              <circle cx="20" cy="24" r="6" fill={colorForGroup(selected.group)} />
              <text className="tooltipTitle" x="36" y="29">
                {selected.label}
              </text>
              <text className="tooltipKey" x="18" y="58">
                Output
              </text>
              <text className="tooltipValue" x="78" y="58">
                {selected.output}
              </text>
              <text className="tooltipKey" x="18" y="84">
                Group
              </text>
              <text className="tooltipBadge" x="78" y="84">
                {selected.group}
              </text>
              <text className="tooltipKey" x="18" y="110">
                PC 1
              </text>
              <text className="tooltipValue" x="78" y="110">
                {selected.x.toFixed(2)}
              </text>
              <text className="tooltipKey" x="132" y="110">
                PC 2
              </text>
              <text className="tooltipValue" x="174" y="110">
                {selected.y.toFixed(2)}
              </text>
            </g>
          ) : null}
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

function mapAxis(value: number, size: number, padding: number) {
  const range = size - padding * 2;
  return padding + ((value + 6) / 12) * range;
}
