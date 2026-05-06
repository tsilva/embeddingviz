import { useEffect, useMemo, useState, type DragEvent } from "react";
import {
  Check,
  ChevronDown,
  FileText,
  Image as ImageIcon,
  Loader2,
  Play,
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
const NEAREST_NEIGHBOR_LIMIT = 10;

type EmbeddingServices = Pick<typeof import("./lib/embeddings"), "buildEmbeddingInputPlan" | "createEmbeddingRun" | "runColor">;

declare global {
  interface Window {
    __EMBEDDINGVIZ_TEST__?: Partial<EmbeddingServices>;
  }
}

function App({ embeddingServices }: { embeddingServices?: Partial<EmbeddingServices> } = {}) {
  const [modelId, setModelId] = useState(MODEL_PRESETS[0].id);
  const [outputMode, setOutputMode] = useState<OutputMode>("final");
  const [reduction, setReduction] = useState<ReductionMethod>("PCA");
  const [snippets, setSnippets] = useState<TextSnippet[]>(SAMPLE_SNIPPETS.slice(0, 3));
  const [composerText, setComposerText] = useState("");
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
  const draftSnippetText = composerText.trim();
  const visibleRuns = useMemo(() => runs.filter((run) => run.visible), [runs]);
  const selectedPoint = useMemo(
    () => {
      const visiblePoints = visibleRuns.flatMap((run) => run.points);
      if (visiblePoints.length === 0) return null;
      return visiblePoints.find((point) => point.id === selectedPointId) ?? visiblePoints[0];
    },
    [selectedPointId, visibleRuns],
  );
  const selectedRun = useMemo(
    () => (selectedPoint ? visibleRuns.find((run) => run.points.some((point) => point.id === selectedPoint.id)) ?? null : null),
    [selectedPoint, visibleRuns],
  );
  const nearestNeighbors = useMemo(
    () => (selectedPoint ? nearestNeighborsForPoint(selectedPoint, runs, NEAREST_NEIGHBOR_LIMIT) : []),
    [runs, selectedPoint],
  );
  const neighborhoodPointIds = useMemo(() => {
    if (!showNeighborhood || !selectedPoint) return null;
    return new Set([selectedPoint.id, ...nearestNeighbors.map((neighbor) => neighbor.point.id)]);
  }, [nearestNeighbors, selectedPoint, showNeighborhood]);
  const totalVisible = runs.filter((run) => run.visible).reduce((sum, run) => sum + run.count, 0);
  const isImageModel = model.task === "image-feature-extraction";
  const isClipTextModel = model.task === "clip-text";
  const effectiveInputType: InputType = activeOutputMode === "tokens" ? "tokens" : isImageModel ? "files" : "text";
  const isWorking = status.phase === "loading" || status.phase === "embedding" || status.phase === "projecting";
  const isPlanning = inputPlanStatus.phase === "loading";
  const draftInputCount = !isImageModel && draftSnippetText ? 1 : 0;
  const candidateInputCount =
    effectiveInputType === "tokens"
      ? 2
      : isImageModel
        ? files.length
        : snippets.filter((snippet) => snippet.text.trim()).length + files.length + draftInputCount;
  const canRun = !isWorking && !isPlanning && candidateInputCount >= 2 && (!isImageModel || effectiveInputType === "files");
  const inputPlanItemsById = useMemo(() => new Map(inputPlan?.items.map((item) => [item.id, item]) ?? []), [inputPlan]);

  useEffect(() => {
    setInputPlan(null);
    if (effectiveInputType === "tokens") {
      setInputPlanStatus({ phase: "idle", message: "Token table loads vocabulary on Run", progress: 0 });
      return;
    }

    if (isImageModel) {
      setInputPlanStatus({ phase: "idle", message: "Vision processor loads on Run", progress: 0 });
      return;
    }

    if (isClipTextModel) {
      setInputPlanStatus({ phase: "idle", message: "CLIP encoders load in worker on Run", progress: 0 });
      return;
    }

    setInputPlanStatus({ phase: "idle", message: "Token plan will be prepared on Run", progress: 0 });
  }, [effectiveInputType, files, isClipTextModel, isImageModel, model, snippets]);

  async function handleRun() {
    try {
      const runSnippets =
        !isImageModel && draftSnippetText
          ? [
              ...snippets,
              {
                id: crypto.randomUUID(),
                text: draftSnippetText,
              },
            ]
          : snippets;
      if (runSnippets !== snippets) {
        setSnippets(runSnippets);
        setComposerText("");
      }

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
        effectiveInputType === "tokens" || isImageModel || isClipTextModel
          ? null
          : await services.buildEmbeddingInputPlan({
              model,
              inputType: effectiveInputType,
              snippets: runSnippets,
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
        snippets: runSnippets,
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

  function addComposerText() {
    const text = composerText.trim();
    if (!text || isImageModel) return;

    setSnippets((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        text,
      },
    ]);
    setComposerText("");
  }

  function removeSnippet(id: string) {
    setSnippets((current) => current.filter((snippet) => snippet.id !== id));
  }

  function removeFile(id: string) {
    setFiles((current) => current.filter((file) => filePlanItemId(file) !== id));
    setFileMessage("");
  }

  function toggleRunVisibility(id: string) {
    setRuns((current) => current.map((run) => (run.id === id ? { ...run, visible: !run.visible } : run)));
  }

  function handleModelChange(nextModelId: string) {
    const next = MODEL_PRESETS.find((preset) => preset.id === nextModelId) ?? MODEL_PRESETS[0];
    setModelId(next.id);
    setOutputMode(next.recommendedOutput);
    if (next.task === "image-feature-extraction") {
      setComposerText("");
    }
    const compatibleFiles = files.filter((file) => isFileCompatibleWithModel(file, next));
    if (compatibleFiles.length !== files.length) {
      setFiles(compatibleFiles);
      const rejected = files.length - compatibleFiles.length;
      setFileMessage(`${rejected} selected ${rejected === 1 ? "file is" : "files are"} incompatible with ${next.label}.`);
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

    setFiles((current) => {
      const existingIds = new Set(current.map(filePlanItemId));
      const additions = compatibleFiles.filter((file) => !existingIds.has(filePlanItemId(file)));
      return [...current, ...additions];
    });
    if (rejectedCount > 0) {
      setFileMessage(`${rejectedCount} ${rejectedCount === 1 ? "file was" : "files were"} not added because ${model.label} cannot embed that type.`);
    } else {
      setFileMessage("");
    }
  }

  function handleInputDragOver(event: DragEvent<HTMLElement>) {
    const canDrop = Array.from(event.dataTransfer.items).some(
      (item) => item.kind === "file" && (!item.type || isMimeCompatibleWithModel(item.type, model)),
    );
    event.preventDefault();
    event.dataTransfer.dropEffect = canDrop ? "copy" : "none";
  }

  function handleInputDrop(event: DragEvent<HTMLElement>) {
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

        <div className="topbarStatus" aria-live="polite">
          <div className={`topbarStatusSegment primary ${status.phase}`}>
            {isWorking ? <Loader2 size={16} className="spin" /> : status.phase === "error" ? <X size={16} /> : <Check size={16} />}
            <span>{status.message}</span>
          </div>
          <div className="topbarStatusSegment">{totalVisible.toLocaleString()} visible points</div>
        </div>

        <div className="topbarActions">
          <div className="topbarReduction segmented" aria-label="Reduction method">
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
          <button className="runButton" type="button" onClick={handleRun} disabled={!canRun} data-testid="run-projection">
            {isWorking ? <Loader2 size={17} className="spin" /> : <Play size={17} fill="currentColor" />}
            Run
          </button>
        </div>
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
          </section>

          {activeOutputMode === "tokens" ? (
            <section className="controlSection">
              <span className="fieldLabel">Token table</span>
              <p className="notice tokenModeNotice">Plots the tokenizer vocabulary with a WebGL point layer.</p>
            </section>
          ) : (
            <section className="controlSection inputWidgetSection">
              <div className="inputWidgetHeader">
                <span className="fieldLabel">Inputs</span>
                <span className="counter">{candidateInputCount.toLocaleString()} items</span>
              </div>

              <div className="inputWidget">
                <textarea
                  value={composerText}
                  onChange={(event) => setComposerText(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.altKey) {
                      event.preventDefault();
                      addComposerText();
                    }
                  }}
                  onDragOver={handleInputDragOver}
                  onDrop={handleInputDrop}
                  placeholder={
                    isImageModel
                      ? "Drag image files here. This model embeds images only."
                      : "Type text or drag files/images here. Enter adds, Alt+Enter newline."
                  }
                  readOnly={isImageModel}
                  aria-label="Input text or dropped files"
                  data-testid="input-composer"
                />
                {fileMessage ? <p className="inputMessage">{fileMessage}</p> : null}

                <TokenPlanOverview
                  inputPlan={inputPlan}
                  status={inputPlanStatus}
                  maxInputTokens={model.maxInputTokens}
                  isImageModel={isImageModel}
                  isClipTextModel={isClipTextModel}
                />

                <div className="unifiedInputList" aria-label="Added inputs">
                  {!candidateInputCount ? <p className="emptyInputList">Type text, or drag files into the box above.</p> : null}
                  {!isImageModel
                    ? snippets.map((snippet) => (
                        <div className="unifiedInputRow" key={snippet.id} data-testid="input-row">
                          <Type size={16} />
                          <div className="unifiedInputText">
                            <strong title={snippet.text}>{trimText(snippet.text, 54)}</strong>
                            <span>Typed text</span>
                            <InputItemMetadata item={inputPlanItemsById.get(snippet.id)} status={inputPlanStatus} isClipTextModel={isClipTextModel} />
                          </div>
                          <button type="button" title="Remove input" onClick={() => removeSnippet(snippet.id)} data-testid="remove-input">
                            <X size={15} />
                          </button>
                        </div>
                      ))
                    : null}
                  {files.map((file) => {
                    const itemId = filePlanItemId(file);
                    const isImageFile = classifyFile(file) === "image";
                    const FileIcon = isImageFile ? ImageIcon : FileText;
                    return (
                      <div className="unifiedInputRow" key={itemId} data-testid="input-row">
                        <FileIcon size={16} />
                        <div className="unifiedInputText">
                          <strong title={file.name}>{file.name}</strong>
                          <span>{isImageFile ? "Image" : "File"} · {fileDetailLabel(file)}</span>
                          <InputItemMetadata
                            item={inputPlanItemsById.get(itemId)}
                            status={inputPlanStatus}
                            isImageModel={isImageModel}
                            isClipTextModel={isClipTextModel}
                          />
                        </div>
                        <button type="button" title="Remove input" onClick={() => removeFile(itemId)} data-testid="remove-input">
                          <X size={15} />
                        </button>
                      </div>
                    );
                  })}
                </div>
              </div>
            </section>
          )}

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
            <h2>Inspector</h2>
            <span>{runs.length.toLocaleString()} {runs.length === 1 ? "run" : "runs"}</span>
          </div>

          <div className="runsList">
            {runs.length === 0 ? (
              <div className="emptyRuns">
                <span>Run comparison appears after projection.</span>
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

          <div className="selectedPanel" aria-label="Selected point">
            <h2>Selected point</h2>
            {selectedPoint ? (
              <>
                <span className="metaLabel">
                  {selectedPoint.kind === "token" ? "Subword token" : selectedPoint.kind === "image" ? "Image" : "Label"}
                </span>
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
                      <span>{selectedRun ? `Cosine similarity from ${selectedRun.name}` : "Cosine similarity"}</span>
                    </div>
                    <label className="neighborhoodToggle">
                      <input type="checkbox" checked={showNeighborhood} onChange={(event) => setShowNeighborhood(event.target.checked)} />
                      <span>Show neighborhood</span>
                    </label>
                  </div>
                  <div className="nearestGroups">
                    {nearestNeighbors.length === 0 ? <p className="muted">No comparable points in visible runs.</p> : null}
                    {nearestNeighbors.map((neighbor) => (
                      <button
                        className="nearestRow"
                        type="button"
                        key={neighbor.point.id}
                        onClick={() => setSelectedPointId(neighbor.point.id)}
                        data-testid="nearest-row"
                      >
                        <span className="nearestRowText">
                          <strong>{neighbor.point.label}</strong>
                          <small>{neighbor.point.source}</small>
                          <small className="nearestRunLine">
                            <span className="swatch" style={{ backgroundColor: neighbor.run.color }} />
                            {neighbor.run.name} · {neighbor.run.current ? "Current" : "Previous"}
                          </small>
                        </span>
                        <code>{neighbor.similarity.toFixed(4)}</code>
                      </button>
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

function TokenPlanOverview({
  inputPlan,
  status,
  maxInputTokens,
  isImageModel,
  isClipTextModel,
}: {
  inputPlan: EmbeddingInputPlan | null;
  status: PipelineStatus;
  maxInputTokens: number;
  isImageModel: boolean;
  isClipTextModel: boolean;
}) {
  if (isImageModel) {
    return (
      <div className="unifiedTokenPlan" data-testid="token-plan-overview">
        <div className="tokenPlanTitle">
          <span>Input plan</span>
          <small>Vision run</small>
        </div>
        <div className="tokenPlanStats">
          <span>No token chunks</span>
          <span>Embeds images directly</span>
        </div>
      </div>
    );
  }

  if (isClipTextModel) {
    return (
      <div className="unifiedTokenPlan" data-testid="token-plan-overview">
        <div className="tokenPlanTitle">
          <span>Input plan</span>
          <small>CLIP multimodal run</small>
        </div>
        <div className="tokenPlanStats">
          <span>{maxInputTokens.toLocaleString()} token model max</span>
          <span>Text/image encoders route by type</span>
        </div>
      </div>
    );
  }

  if (status.phase === "loading") {
    return (
      <div className="unifiedTokenPlan" data-testid="token-plan-overview">
        <div className="tokenPlanTitle">
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
      <div className="unifiedTokenPlan warning" data-testid="token-plan-overview">
        <div className="tokenPlanTitle">
          <span>Token plan</span>
          <small>Error</small>
        </div>
        <p>{status.message}</p>
      </div>
    );
  }

  if (!inputPlan) {
    return (
      <div className="unifiedTokenPlan" data-testid="token-plan-overview">
        <div className="tokenPlanTitle">
          <span>Token plan</span>
          <small>Pending run</small>
        </div>
        <div className="tokenPlanStats">
          <span>{maxInputTokens.toLocaleString()} token model max</span>
          <span>Chunks appear after Run</span>
        </div>
      </div>
    );
  }

  return (
    <div className="unifiedTokenPlan" data-testid="token-plan-overview">
      <div className="tokenPlanTitle">
        <span>Token plan</span>
        <small>{inputPlan.totalChunks.toLocaleString()} plot points</small>
      </div>
      <div className="inputPlanStats">
        <span>{inputPlan.totalTokens.toLocaleString()} tokens</span>
        <span>{inputPlan.totalChunks.toLocaleString()} chunks</span>
        <span>{inputPlan.chunkSize.toLocaleString()} usable / {maxInputTokens.toLocaleString()} max</span>
        <span>{inputPlan.overlapTokens.toLocaleString()} token overlap</span>
      </div>
    </div>
  );
}

function InputItemMetadata({
  item,
  status,
  isImageModel = false,
  isClipTextModel = false,
}: {
  item?: InputPlanItem;
  status: PipelineStatus;
  isImageModel?: boolean;
  isClipTextModel?: boolean;
}) {
  if (isImageModel) {
    return (
      <div className="inputItemMeta image" data-testid="input-item-meta">
        <span>Image embedding</span>
        <small>not token chunked</small>
      </div>
    );
  }

  if (isClipTextModel) {
    return (
      <div className="inputItemMeta pending" data-testid="input-item-meta">
        <span>CLIP routed</span>
        <small>by input type</small>
      </div>
    );
  }

  if (item) {
    return (
      <div className={`inputItemMeta ${item.status}`} data-testid="input-item-meta">
        <span>{item.tokenCount.toLocaleString()} tokens</span>
        <span>{item.chunkCount.toLocaleString()} {item.chunkCount === 1 ? "chunk" : "chunks"}</span>
        <small title={item.message}>{item.message}</small>
      </div>
    );
  }

  if (status.phase === "loading") {
    return (
      <div className="inputItemMeta pending" data-testid="input-item-meta">
        <span>Counting...</span>
        <small>tokenizing input</small>
      </div>
    );
  }

  if (status.phase === "error") {
    return (
      <div className="inputItemMeta skipped" data-testid="input-item-meta">
        <span>Plan failed</span>
        <small>{status.message}</small>
      </div>
    );
  }

  return (
    <div className="inputItemMeta pending" data-testid="input-item-meta">
      <span>Pending</span>
      <small>Run to count tokens</small>
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
    buildEmbeddingInputPlan: async ({ model, inputType, snippets, files, onStatus }) => {
      const fileInputs = await Promise.all(
        files.map(async (file) => ({
          id: filePlanItemId(file),
          label: file.name,
          text: (await file.text()).trim() || file.name,
          source: file.name,
        })),
      );
      const textInputs = snippets.map((snippet, index) => {
        const label = snippet.text.trim();
        return {
          id: snippet.id,
          label: label || `Snippet ${index + 1}`,
          text: label,
          source: "Typed text",
        };
      });
      const rawInputs = inputType === "files" ? fileInputs : [...textInputs, ...fileInputs];
      const items = rawInputs.map((input) => ({
        id: input.id,
        label: input.label,
        source: input.source,
        tokenCount: input.text ? 3 : 0,
        chunkCount: input.text ? 1 : 0,
        status: input.text ? ("ready" as const) : ("skipped" as const),
        message: input.text ? "Fits in one model call" : "Skipped empty input",
      }));
      const samples = rawInputs
        .filter((input) => input.text)
        .map((input, index) => ({
          id: `mock-input-${index}`,
          text: input.text,
          label: input.label,
          parentLabel: input.label,
          source: input.source,
          kind: "input" as const,
          tokenCount: 3,
          chunkIndex: 1,
          chunkCount: 1,
          tokenStart: 1,
          tokenEnd: 3,
        }));

      onStatus({ phase: "ready", message: `${samples.length} chunks`, progress: 1 });
      return {
        inputType,
        modelId: model.id,
        chunkSize: model.maxInputTokens,
        overlapTokens: 24,
        items,
        samples,
        totalTokens: samples.length * 3,
        totalChunks: samples.length,
        skippedCount: items.filter((item) => item.status === "skipped").length,
      };
    },
    createEmbeddingRun: async ({ snippets, inputPlan, reduction, onStatus }) => {
      const labels = inputPlan?.samples.map((sample) => sample.label) ?? snippets.filter((snippet) => snippet.text.trim()).map((snippet) => snippet.text.trim());
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
          source: "Typed text",
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
  run: RunRecord;
  point: EmbeddingPoint;
  distance: number;
  similarity: number;
}

function nearestNeighborsForPoint(selectedPoint: EmbeddingPoint, runs: RunRecord[], limit: number): NearestNeighbor[] {
  return runs
    .filter((run) => run.visible)
    .flatMap((run) =>
      run.points
        .filter((point) => point.id !== selectedPoint.id)
        .map((point) => {
          const similarity = cosineSimilarity(selectedPoint.vector, point.vector);
          return {
            run,
            point,
            similarity,
            distance: 1 - similarity,
          };
        }),
    )
    .filter((neighbor) => Number.isFinite(neighbor.similarity) && Number.isFinite(neighbor.distance))
    .sort((a, b) => b.similarity - a.similarity || a.distance - b.distance || a.point.label.localeCompare(b.point.label))
    .slice(0, limit);
}

function cosineSimilarity(a: ArrayLike<number>, b: ArrayLike<number>) {
  const length = Math.min(a.length, b.length);
  if (length === 0) return Number.NEGATIVE_INFINITY;

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

  if (normA === 0 || normB === 0) return Number.NEGATIVE_INFINITY;
  const similarity = dot / (Math.sqrt(normA) * Math.sqrt(normB));
  return Math.min(Math.max(similarity, -1), 1);
}

function runName(inputType: InputType) {
  if (inputType === "tokens") return "Token table";
  if (inputType === "files") return "Image inputs";
  return "Inputs";
}

function outputLabel(outputMode: OutputMode, task?: typeof MODEL_PRESETS[number]["task"]) {
  if (task === "clip-text") return "CLIP embedding";
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

function filePlanItemId(file: File) {
  return `${file.name}-${file.size}-${file.lastModified}`;
}

function fileDetailLabel(file: File) {
  const type = file.type || "unknown type";
  return `${type} · ${formatBytes(file.size)}`;
}

function trimText(value: string, maxLength: number) {
  const normalized = value.replace(/\s+/g, " ").trim();
  return normalized.length > maxLength ? `${normalized.slice(0, maxLength - 1)}…` : normalized;
}

function formatBytes(bytes: number) {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024).toLocaleString()} KB`;
  return `${bytes.toLocaleString()} B`;
}

export default App;
