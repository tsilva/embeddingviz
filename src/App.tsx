import { useEffect, useMemo, useState, type DragEvent } from "react";
import {
  BrainCircuit,
  Check,
  ChevronDown,
  Database,
  FileText,
  Loader2,
  Play,
  Plus,
  Sparkles,
  Type,
  X,
} from "lucide-react";
import { MODEL_PRESETS, SAMPLE_SNIPPETS } from "./data";
import { ScatterPlot } from "./components/ScatterPlot";
import type {
  EmbeddingInputPlan,
  EmbeddingPoint,
  InputPlanItem,
  InputType,
  OutputMode,
  PipelineStatus,
  ReductionMethod,
  RunRecord,
  TextSnippet,
} from "./types";

const initialRuns: RunRecord[] = [];

type EmbeddingServices = Pick<typeof import("./lib/embeddings"), "buildEmbeddingInputPlan" | "createEmbeddingRun" | "runColor">;

declare global {
  interface Window {
    __EMBEDDINGVIZ_TEST__?: Partial<EmbeddingServices>;
  }
}

function App({ embeddingServices }: { embeddingServices?: Partial<EmbeddingServices> } = {}) {
  const [modelId, setModelId] = useState(MODEL_PRESETS[0].id);
  const [outputMode, setOutputMode] = useState<OutputMode>("final");
  const [inputType, setInputType] = useState<InputType>("text");
  const [reduction, setReduction] = useState<ReductionMethod>("PCA");
  const [snippets, setSnippets] = useState<TextSnippet[]>(SAMPLE_SNIPPETS.slice(0, 3));
  const [files, setFiles] = useState<File[]>([]);
  const [fileMessage, setFileMessage] = useState("");
  const [inputPlan, setInputPlan] = useState<EmbeddingInputPlan | null>(null);
  const [inputPlanStatus, setInputPlanStatus] = useState<PipelineStatus>({
    phase: "idle",
    message: "Counting tokens",
    progress: 0,
  });
  const [runs, setRuns] = useState<RunRecord[]>(initialRuns);
  const [selectedPointId, setSelectedPointId] = useState<string | null>(null);
  const [neighborLimit, setNeighborLimit] = useState(5);
  const [showNeighborhood, setShowNeighborhood] = useState(false);
  const [query, setQuery] = useState("");
  const [is3d, setIs3d] = useState(false);
  const [status, setStatus] = useState<PipelineStatus>({
    phase: "idle",
    message: "Ready",
    progress: 0,
  });

  const model = useMemo(() => MODEL_PRESETS.find((preset) => preset.id === modelId) ?? MODEL_PRESETS[0], [modelId]);
  const activeOutputMode = model.outputModes.includes(outputMode) ? outputMode : model.recommendedOutput;
  const selectedPoint = useMemo(
    () => runs.flatMap((run) => run.points).find((point) => point.id === selectedPointId) ?? runs[0]?.points[0] ?? null,
    [runs, selectedPointId],
  );
  const selectedRun = useMemo(
    () => (selectedPoint ? runs.find((run) => run.points.some((point) => point.id === selectedPoint.id)) ?? null : null),
    [runs, selectedPoint],
  );
  const nearestByRun = useMemo(
    () => (selectedPoint ? nearestNeighborsByRun(selectedPoint, runs, neighborLimit) : []),
    [neighborLimit, runs, selectedPoint],
  );
  const neighborhoodPointIds = useMemo(() => {
    if (!showNeighborhood || !selectedPoint) return null;
    return new Set([selectedPoint.id, ...nearestByRun.flatMap((group) => group.neighbors.map((neighbor) => neighbor.point.id))]);
  }, [nearestByRun, selectedPoint, showNeighborhood]);
  const totalVisible = runs.filter((run) => run.visible).reduce((sum, run) => sum + run.count, 0);
  const effectiveInputType = activeOutputMode === "tokens" ? "tokens" : inputType;
  const isWorking = status.phase === "loading" || status.phase === "embedding" || status.phase === "projecting";
  const isPlanning = inputPlanStatus.phase === "loading";
  const candidateInputCount =
    effectiveInputType === "tokens" ? 2 : effectiveInputType === "files" ? files.length : snippets.filter((snippet) => snippet.text.trim()).length;
  const canRun = !isWorking && !isPlanning && candidateInputCount >= 2;

  useEffect(() => {
    setInputPlan(null);
    if (effectiveInputType === "tokens") {
      setInputPlanStatus({ phase: "idle", message: "Token table loads vocabulary on Run", progress: 0 });
      return;
    }

    setInputPlanStatus({ phase: "idle", message: "Token plan will be prepared on Run", progress: 0 });
  }, [effectiveInputType, files, model, snippets]);

  async function handleRun() {
    try {
      setStatus({ phase: "loading", message: "Loading projection pipeline", progress: 0.02 });
      const embeddingModule = await import("./lib/embeddings");
      const services: EmbeddingServices = {
        buildEmbeddingInputPlan: embeddingModule.buildEmbeddingInputPlan,
        createEmbeddingRun: embeddingModule.createEmbeddingRun,
        runColor: embeddingModule.runColor,
        ...devMockEmbeddingServices(),
        ...(typeof window === "undefined" ? {} : window.__EMBEDDINGVIZ_TEST__),
        ...embeddingServices,
      };
      const preparedInputPlan =
        effectiveInputType === "tokens"
          ? null
          : await services.buildEmbeddingInputPlan({
              model,
              inputType: effectiveInputType,
              snippets,
              files,
              onStatus: setInputPlanStatus,
            });

      setInputPlan(preparedInputPlan);
      setStatus({ phase: "loading", message: "Preparing model", progress: 0.04 });
      const result = await services.createEmbeddingRun({
        model,
        outputMode: activeOutputMode,
        inputType: effectiveInputType,
        reduction,
        snippets,
        files,
        inputPlan: preparedInputPlan,
        onStatus: setStatus,
      });

      const runIndex = runs.length;
      const run: RunRecord = {
        id: `${Date.now()}`,
        name: runName(effectiveInputType),
        model: model.label,
        output: outputLabel(activeOutputMode, model.task),
        reduction,
        color: services.runColor(runIndex),
        count: result.points.length,
        current: true,
        visible: true,
        points: result.points,
      };

      setRuns((current) => [run, ...current.map((item) => ({ ...item, current: false }))].slice(0, 4));
      setSelectedPointId(result.points[0]?.id ?? null);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Embedding run failed";
      setStatus({
        phase: "error",
        message,
        progress: 0,
      });
      setInputPlanStatus((current) => (current.phase === "loading" ? { phase: "error", message, progress: 0 } : current));
    }
  }

  function updateSnippet(id: string, value: string) {
    setSnippets((current) => current.map((snippet) => (snippet.id === id ? { ...snippet, text: value } : snippet)));
  }

  function addSnippet() {
    setSnippets((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        text: "",
      },
    ]);
  }

  function removeSnippet(id: string) {
    setSnippets((current) => current.filter((snippet) => snippet.id !== id));
  }

  function toggleRunVisibility(id: string) {
    setRuns((current) => current.map((run) => (run.id === id ? { ...run, visible: !run.visible } : run)));
  }

  function handleModelChange(nextModelId: string) {
    const next = MODEL_PRESETS.find((preset) => preset.id === nextModelId) ?? MODEL_PRESETS[0];
    setModelId(next.id);
    setOutputMode(next.recommendedOutput);
    const compatibleFiles = files.filter((file) => isFileCompatibleWithModel(file, next));
    if (compatibleFiles.length !== files.length) {
      setFiles(compatibleFiles);
      const rejected = files.length - compatibleFiles.length;
      setFileMessage(`${rejected} selected ${rejected === 1 ? "file is" : "files are"} incompatible with ${next.label}.`);
    }
  }

  function chooseInputType(nextInputType: InputType) {
    setInputType(nextInputType);
    if (nextInputType === "tokens") {
      setOutputMode(model.outputModes.includes("tokens") ? "tokens" : model.recommendedOutput);
    } else if (activeOutputMode === "tokens") {
      setOutputMode(model.recommendedOutput === "tokens" ? "final" : model.recommendedOutput);
    }
  }

  function chooseOutputMode(nextOutputMode: OutputMode) {
    if (model.outputModes.includes(nextOutputMode)) {
      setOutputMode(nextOutputMode);
    }
  }

  function handleFilesSelected(nextFiles: File[]) {
    const compatibleFiles = nextFiles.filter((file) => isFileCompatibleWithModel(file, model));
    const rejectedCount = nextFiles.length - compatibleFiles.length;

    setFiles(compatibleFiles);
    if (rejectedCount > 0) {
      setFileMessage(`${rejectedCount} ${rejectedCount === 1 ? "file was" : "files were"} not added because ${model.label} cannot embed that MIME type.`);
    } else {
      setFileMessage("");
    }
  }

  function handleFileDragOver(event: DragEvent<HTMLDivElement>) {
    const canDrop = Array.from(event.dataTransfer.items).some((item) => item.kind === "file" && isMimeCompatibleWithModel(item.type, model));
    event.preventDefault();
    event.dataTransfer.dropEffect = canDrop ? "copy" : "none";
  }

  function handleFileDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    handleFilesSelected(Array.from(event.dataTransfer.files));
  }

  return (
    <div className="appShell">
      <header className="topbar">
        <div className="brand">
          <div className="brandMark" aria-hidden="true">
            {Array.from({ length: 9 }, (_, index) => (
              <span key={index} />
            ))}
          </div>
          <span>EmbeddingViz</span>
        </div>

        <div className="runtimeChip">
          <Database size={16} />
          Hugging Face
        </div>
        <div className="runtimeChip">
          <BrainCircuit size={16} />
          transformers.js local
        </div>
        <div className="topbarStatus" aria-live="polite">
          <div className={`topbarStatusSegment primary ${status.phase}`}>
            {isWorking ? <Loader2 size={16} className="spin" /> : status.phase === "error" ? <X size={16} /> : <Check size={16} />}
            <span>{status.message}</span>
          </div>
          <div className="topbarStatusSegment">{totalVisible.toLocaleString()} visible points</div>
          <div className="topbarStatusSegment">{runs[0]?.reduction ?? reduction} projected</div>
        </div>

        <button className="runButton" type="button" onClick={handleRun} disabled={!canRun} data-testid="run-projection">
          {isWorking ? <Loader2 size={17} className="spin" /> : <Play size={17} fill="currentColor" />}
          Run
        </button>
        <div
          className={`progressTrack ${isWorking ? "active" : ""}`}
          role="progressbar"
          aria-label="Embedding run progress"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(status.progress * 100)}
        >
          <span style={{ width: `${Math.round(status.progress * 100)}%` }} />
        </div>
      </header>

      <div className="workspace">
        <aside className="leftPanel">
          <section className="controlSection">
            <label className="fieldLabel" htmlFor="model">
              Model
            </label>
            <div className="selectShell">
              <select id="model" value={modelId} onChange={(event) => handleModelChange(event.target.value)}>
                {MODEL_PRESETS.map((preset) => (
                  <option key={preset.id} value={preset.id}>
                    {preset.id}
                  </option>
                ))}
              </select>
              <ChevronDown size={16} />
            </div>
            <p className="compatLine">
              <span />
              {model.summary}
            </p>
          </section>

          <section className="controlSection">
            <label className="fieldLabel" htmlFor="output">
              Output
            </label>
            <div className="selectShell">
              <select id="output" value={activeOutputMode} onChange={(event) => chooseOutputMode(event.target.value as OutputMode)}>
                {model.outputModes.map((mode) => (
                  <option key={mode} value={mode}>
                    {outputLabel(mode, model.task)}
                  </option>
                ))}
              </select>
              <ChevronDown size={16} />
            </div>
            <p className="recommend">
              <Sparkles size={13} />
              Recommended
            </p>
          </section>

          {activeOutputMode === "tokens" ? (
            <section className="controlSection">
              <span className="fieldLabel">Token table</span>
              <p className="notice tokenModeNotice">Plots the tokenizer vocabulary with a WebGL point layer.</p>
            </section>
          ) : (
            <section className="controlSection">
              <div className="fieldRow">
                <span className="fieldLabel">Input type</span>
              </div>
              <div className="segmented">
                <button className={inputType === "text" ? "active" : ""} type="button" onClick={() => chooseInputType("text")}>
                  <Type size={15} />
                  Text
                </button>
                <button className={inputType === "files" ? "active" : ""} type="button" onClick={() => chooseInputType("files")}>
                  <FileText size={15} />
                  Files
                </button>
              </div>

              {inputType === "text" ? (
                <div className="snippetList">
                  <div className="fieldRow">
                    <span className="subLabel">Text snippets</span>
                    <span className="counter">{snippets.length} / 100</span>
                  </div>
                  {snippets.map((snippet) => (
                    <div className="snippetItem" key={snippet.id}>
                      <input
                        value={snippet.text}
                        onChange={(event) => updateSnippet(snippet.id, event.target.value)}
                        aria-label="Snippet text"
                        data-testid="snippet-input"
                      />
                      <button type="button" title="Remove snippet" onClick={() => removeSnippet(snippet.id)}>
                        <X size={15} />
                      </button>
                    </div>
                  ))}
                  <button className="addButton" type="button" onClick={addSnippet}>
                    <Plus size={16} />
                    Add snippet
                  </button>
                </div>
              ) : null}

              {inputType === "files" ? (
                <div className="fileDrop" onDragOver={handleFileDragOver} onDrop={handleFileDrop}>
                  <input
                    type="file"
                    multiple
                    accept={fileAcceptValue(model)}
                    onChange={(event) => handleFilesSelected(Array.from(event.target.files ?? []))}
                  />
                  <span>{files.length ? selectedFileLabel(files) : `Drop or choose ${acceptedFileLabel(model)}`}</span>
                  <small>{fileMessage || `Accepts ${acceptedFileLabel(model)} for ${model.label}.`}</small>
                </div>
              ) : null}

              <InputPlanSummary inputPlan={inputPlan} status={inputPlanStatus} maxInputTokens={model.maxInputTokens} />
            </section>
          )}

          <section className="controlSection">
            <span className="fieldLabel">Reduction</span>
            <div className="segmented">
              {(["PCA", "UMAP", "t-SNE"] as ReductionMethod[]).map((method) => (
                <button
                  key={method}
                  className={reduction === method ? "active" : ""}
                  type="button"
                  onClick={() => setReduction(method)}
                  title={`Project with ${method}`}
                  data-testid={`reduction-${method}`}
                >
                  {method}
                </button>
              ))}
            </div>
          </section>
        </aside>

        <ScatterPlot
          runs={runs}
          selectedPointId={selectedPointId}
          query={query}
          is3d={is3d}
          primaryReduction={runs[0]?.reduction ?? reduction}
          neighborhoodPointIds={neighborhoodPointIds}
          onQueryChange={setQuery}
          onPointSelect={(point: EmbeddingPoint) => setSelectedPointId(point.id)}
          onToggle3d={setIs3d}
        />

        <aside className="rightPanel">
          <div className="panelHeader">
            <h2>Runs</h2>
            <button type="button" onClick={handleRun} disabled={!canRun}>
              <Plus size={16} />
              New run
            </button>
          </div>

          <div className="runsList">
            {runs.length === 0 ? (
              <div className="emptyRuns">
                <span>Run comparison appears here after projection.</span>
              </div>
            ) : null}
            {runs.map((run) => (
              <div className={`runCard ${run.current ? "current" : ""}`} key={run.id}>
                <span className="swatch" style={{ backgroundColor: run.color }} />
                <div>
                  <strong>{run.name}</strong>
                  <span>{run.output} · {run.reduction}</span>
                  <small>{run.model} · {run.count} points</small>
                </div>
                <label className="checkbox">
                  <input type="checkbox" checked={run.visible} onChange={() => toggleRunVisibility(run.id)} />
                </label>
              </div>
            ))}
          </div>

          <div className="selectedPanel">
            <h2>Selected point</h2>
            {selectedPoint ? (
              <>
                <span className="metaLabel">{selectedPoint.kind === "token" ? "Subword token" : "Label"}</span>
                <strong data-testid="selected-point-label">{selectedPoint.label}</strong>
                <p>
                  {selectedPoint.kind === "token"
                    ? `Raw token ${selectedPoint.rawToken ?? selectedPoint.snippet}${selectedPoint.tokenId === undefined ? "" : ` · id ${selectedPoint.tokenId}`}`
                    : selectedPoint.snippet}
                </p>
                {selectedPoint.kind !== "token" && selectedPoint.chunkIndex ? (
                  <>
                    <span className="metaLabel">Chunk</span>
                    <p>
                      {selectedPoint.chunkIndex} of {selectedPoint.chunkCount ?? 1}
                      {selectedPoint.tokenStart && selectedPoint.tokenEnd
                        ? ` · tokens ${selectedPoint.tokenStart.toLocaleString()}-${selectedPoint.tokenEnd.toLocaleString()}`
                        : ""}
                      {selectedPoint.tokenCount ? ` · ${selectedPoint.tokenCount.toLocaleString()} tokens` : ""}
                    </p>
                  </>
                ) : null}
                <span className="metaLabel">Source</span>
                <p>{selectedPoint.source} · {selectedPoint.output}</p>
                <span className="metaLabel">Dimensions</span>
                <p>{selectedPoint.vector.length}</p>
                <div className="nearestPanel">
                  <div className="nearestHeader">
                    <div>
                      <h3>Nearest points</h3>
                      <span>{selectedRun ? `Cosine distance from ${selectedRun.name}` : "Cosine distance"}</span>
                    </div>
                    <label className="neighborhoodToggle">
                      <input type="checkbox" checked={showNeighborhood} onChange={(event) => setShowNeighborhood(event.target.checked)} />
                      <span>Show neighborhood</span>
                    </label>
                  </div>
                  <div className="segmented small nearestLimit" aria-label="Nearest neighbor count">
                    {[3, 5, 10].map((limit) => (
                      <button
                        key={limit}
                        className={neighborLimit === limit ? "active" : ""}
                        type="button"
                        onClick={() => setNeighborLimit(limit)}
                      >
                        {limit}
                      </button>
                    ))}
                  </div>
                  <div className="nearestGroups">
                    {nearestByRun.length === 0 ? <p className="muted">No comparable points in visible runs.</p> : null}
                    {nearestByRun.map((group) => (
                      <div className="nearestGroup" key={group.run.id}>
                        <div className="nearestRunHeader">
                          <span className="swatch" style={{ backgroundColor: group.run.color }} />
                          <strong>{group.run.name}</strong>
                          <small>
                            {group.run.current ? "Current" : "Previous"} · {group.neighbors.length} nearest
                          </small>
                        </div>
                        {group.neighbors.map((neighbor) => (
                          <button
                            className="nearestRow"
                            type="button"
                            key={neighbor.point.id}
                            onClick={() => setSelectedPointId(neighbor.point.id)}
                          >
                            <span>
                              <strong>{neighbor.point.label}</strong>
                              <small>{neighbor.point.source}</small>
                            </span>
                            <code>{neighbor.distance.toFixed(4)}</code>
                          </button>
                        ))}
                      </div>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              <p className="muted">No point selected.</p>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}

function InputPlanSummary({
  inputPlan,
  status,
  maxInputTokens,
}: {
  inputPlan: EmbeddingInputPlan | null;
  status: PipelineStatus;
  maxInputTokens: number;
}) {
  if (status.phase === "loading") {
    return (
      <div className="inputPlanPanel">
        <div className="inputPlanHeader">
          <span>Token plan</span>
          <small>Counting...</small>
        </div>
        <div className="inputPlanProgress">
          <span style={{ width: `${Math.round(status.progress * 100)}%` }} />
        </div>
      </div>
    );
  }

  if (status.phase === "error") {
    return (
      <div className="inputPlanPanel warning">
        <div className="inputPlanHeader">
          <span>Token plan</span>
          <small>Error</small>
        </div>
        <p>{status.message}</p>
      </div>
    );
  }

  if (!inputPlan) {
    return null;
  }

  return (
    <div className="inputPlanPanel">
      <div className="inputPlanHeader">
        <span>Token plan</span>
        <small>{inputPlan.totalChunks.toLocaleString()} plot points</small>
      </div>
      <div className="inputPlanStats">
        <span>{inputPlan.totalTokens.toLocaleString()} tokens</span>
        <span>{inputPlan.chunkSize.toLocaleString()} usable / {maxInputTokens.toLocaleString()} max</span>
      </div>
      <div className="inputPlanRows">
        {inputPlan.items.length === 0 ? <p className="muted">No inputs selected.</p> : null}
        {inputPlan.items.map((item) => (
          <InputPlanRow item={item} key={item.id} />
        ))}
      </div>
    </div>
  );
}

function InputPlanRow({ item }: { item: InputPlanItem }) {
  return (
    <div className={`inputPlanRow ${item.status}`}>
      <div>
        <strong title={item.label}>{item.label}</strong>
        <span>{item.message}</span>
      </div>
      <div>
        <span>{item.tokenCount.toLocaleString()}</span>
        <small>{item.chunkCount.toLocaleString()} chunks</small>
      </div>
    </div>
  );
}

function devMockEmbeddingServices(): Partial<EmbeddingServices> {
  if (
    !import.meta.env.DEV ||
    typeof window === "undefined" ||
    !new URLSearchParams(window.location.search).has("mockEmbeddings")
  ) {
    return {};
  }

  return {
    buildEmbeddingInputPlan: async ({ model, inputType, snippets, onStatus }) => {
      const samples = snippets
        .filter((snippet) => snippet.text.trim())
        .map((snippet, index) => {
          const label = snippet.text.trim();
          return {
            id: `mock-snippet-${index}`,
            text: label,
            label,
            parentLabel: label,
            source: "Text snippets",
            kind: "input" as const,
            tokenCount: 3,
            chunkIndex: 1,
            chunkCount: 1,
            tokenStart: 1,
            tokenEnd: 3,
          };
        });

      onStatus({ phase: "ready", message: `${samples.length} chunks`, progress: 1 });
      return {
        inputType,
        modelId: model.id,
        chunkSize: model.maxInputTokens,
        overlapTokens: 24,
        items: [],
        samples,
        totalTokens: samples.length * 3,
        totalChunks: samples.length,
        skippedCount: 0,
      };
    },
    createEmbeddingRun: async ({ snippets, reduction, onStatus }) => {
      const labels = snippets.filter((snippet) => snippet.text.trim()).map((snippet) => snippet.text.trim());
      onStatus({ phase: "embedding", message: "Mock embeddings", progress: 0.6 });
      onStatus({ phase: "projecting", message: `Running ${reduction}`, progress: 0.85 });
      onStatus({ phase: "ready", message: `Mock ${reduction} complete`, progress: 1 });

      return {
        explained: [0.8, 0.15, 0.05],
        points: labels.map((label, index) => ({
          id: `mock-point-${index}`,
          label,
          parentLabel: label,
          snippet: label,
          source: "Text snippets",
          output: "Final embedding",
          vector: [index + 1, index === 1 ? 1 : 0, index === 2 ? 1 : 0],
          x: [-5, 0, 5][index] ?? 0,
          y: [0, 5, -5][index] ?? 0,
          z: 0,
          kind: "input" as const,
          tokenCount: 3,
          chunkIndex: 1,
          chunkCount: 1,
          tokenStart: 1,
          tokenEnd: 3,
        })),
      };
    },
    runColor: () => "#2563eb",
  };
}

interface NearestNeighbor {
  point: EmbeddingPoint;
  distance: number;
}

interface NearestNeighborGroup {
  run: RunRecord;
  neighbors: NearestNeighbor[];
}

function nearestNeighborsByRun(selectedPoint: EmbeddingPoint, runs: RunRecord[], limit: number): NearestNeighborGroup[] {
  return runs
    .filter((run) => run.visible)
    .map((run) => ({
      run,
      neighbors: run.points
        .filter((point) => point.id !== selectedPoint.id)
        .map((point) => ({
          point,
          distance: cosineDistance(selectedPoint.vector, point.vector),
        }))
        .filter((neighbor) => Number.isFinite(neighbor.distance))
        .sort((a, b) => a.distance - b.distance)
        .slice(0, limit),
    }))
    .filter((group) => group.neighbors.length > 0);
}

function cosineDistance(a: ArrayLike<number>, b: ArrayLike<number>) {
  const length = Math.min(a.length, b.length);
  if (length === 0) return Number.POSITIVE_INFINITY;

  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let index = 0; index < length; index += 1) {
    const valueA = a[index];
    const valueB = b[index];
    dot += valueA * valueB;
    normA += valueA * valueA;
    normB += valueB * valueB;
  }

  if (normA === 0 || normB === 0) return Number.POSITIVE_INFINITY;
  const similarity = dot / (Math.sqrt(normA) * Math.sqrt(normB));
  return 1 - Math.min(Math.max(similarity, -1), 1);
}

function runName(inputType: InputType) {
  if (inputType === "tokens") return "Token table";
  if (inputType === "files") return "Files";
  return "Text snippets";
}

function outputLabel(outputMode: OutputMode, task?: typeof MODEL_PRESETS[number]["task"]) {
  if (task === "text-generation") {
    if (outputMode === "final") return "Final LM value state";
    if (outputMode === "tokens") return "Tokenizer subword features";
    return "Layer 4 · LM value state";
  }
  if (outputMode === "hidden-4") return "Layer 4 · hidden state";
  if (outputMode === "tokens") return "Token table · embeddings";
  return "Final embedding";
}

function classifyFileMime(mimeType: string): "text" | "image" | "unsupported" {
  if (mimeType.startsWith("image/")) return "image";
  if (mimeType.startsWith("text/")) return "text";
  if (["application/json", "application/csv", "application/xml", "application/x-ndjson"].includes(mimeType)) return "text";
  return "unsupported";
}

function classifyFile(file: File) {
  const mimeKind = classifyFileMime(file.type);
  if (mimeKind !== "unsupported") return mimeKind;
  if (/\.(txt|md|csv|json|jsonl|ndjson|xml)$/i.test(file.name)) return "text";
  return "unsupported";
}

function isMimeCompatibleWithModel(mimeType: string, model: typeof MODEL_PRESETS[number]) {
  const kind = classifyFileMime(mimeType);
  if (kind === "image") return model.supportsImages;
  if (kind === "text") return model.task !== "image-feature-extraction";
  return false;
}

function isFileCompatibleWithModel(file: File, model: typeof MODEL_PRESETS[number]) {
  const kind = classifyFile(file);
  if (kind === "image") return model.supportsImages;
  if (kind === "text") return model.task !== "image-feature-extraction";
  return false;
}

function fileAcceptValue(model: typeof MODEL_PRESETS[number]) {
  return model.supportsImages ? "image/*" : ".txt,.md,.csv,.json,.jsonl,.ndjson,text/*,application/json,application/csv";
}

function acceptedFileLabel(model: typeof MODEL_PRESETS[number]) {
  return model.supportsImages ? "image files" : "text, Markdown, CSV, or JSON files";
}

function selectedFileLabel(files: File[]) {
  const counts = files.reduce(
    (result, file) => {
      const kind = classifyFile(file);
      if (kind === "image") result.images += 1;
      if (kind === "text") result.text += 1;
      return result;
    },
    { images: 0, text: 0 },
  );

  const parts = [];
  if (counts.text) parts.push(`${counts.text} text`);
  if (counts.images) parts.push(`${counts.images} image`);
  return `${parts.join(" · ")} ${files.length === 1 ? "file" : "files"} selected`;
}

export default App;
