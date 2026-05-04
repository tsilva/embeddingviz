import { useMemo, useState, type DragEvent } from "react";
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
import { createEmbeddingRun, runColor } from "./lib/embeddings";
import type { EmbeddingPoint, InputType, OutputMode, PipelineStatus, ReductionMethod, RunRecord, TextSnippet } from "./types";

const initialRuns: RunRecord[] = [];

function App() {
  const [modelId, setModelId] = useState(MODEL_PRESETS[0].id);
  const [outputMode, setOutputMode] = useState<OutputMode>("final");
  const [inputType, setInputType] = useState<InputType>("text");
  const [reduction, setReduction] = useState<ReductionMethod>("PCA");
  const [snippets, setSnippets] = useState<TextSnippet[]>(SAMPLE_SNIPPETS.slice(0, 3));
  const [files, setFiles] = useState<File[]>([]);
  const [fileMessage, setFileMessage] = useState("");
  const [runs, setRuns] = useState<RunRecord[]>(initialRuns);
  const [selectedPointId, setSelectedPointId] = useState<string | null>(null);
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
  const totalVisible = runs.filter((run) => run.visible).reduce((sum, run) => sum + run.count, 0);
  const effectiveInputType = activeOutputMode === "tokens" ? "tokens" : inputType;
  const isWorking = status.phase === "loading" || status.phase === "embedding" || status.phase === "projecting";

  async function handleRun() {
    try {
      setStatus({ phase: "loading", message: "Preparing model", progress: 0.02 });
      const result = await createEmbeddingRun({
        model,
        outputMode: activeOutputMode,
        inputType: effectiveInputType,
        reduction,
        snippets,
        files,
        onStatus: setStatus,
      });

      const runIndex = runs.length;
      const run: RunRecord = {
        id: `${Date.now()}`,
        name: runName(effectiveInputType),
        model: model.label,
        output: outputLabel(activeOutputMode, model.task),
        reduction,
        color: runColor(runIndex),
        count: result.points.length,
        current: true,
        visible: true,
        points: result.points,
      };

      setRuns((current) => [run, ...current.map((item) => ({ ...item, current: false }))].slice(0, 4));
      setSelectedPointId(result.points[0]?.id ?? null);
    } catch (error) {
      setStatus({
        phase: "error",
        message: error instanceof Error ? error.message : "Embedding run failed",
        progress: 0,
      });
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

        <button className="runButton" type="button" onClick={handleRun} disabled={isWorking}>
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
                      <input value={snippet.text} onChange={(event) => updateSnippet(snippet.id, event.target.value)} aria-label="Snippet text" />
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
          onQueryChange={setQuery}
          onPointSelect={(point: EmbeddingPoint) => setSelectedPointId(point.id)}
          onToggle3d={setIs3d}
        />

        <aside className="rightPanel">
          <div className="panelHeader">
            <h2>Runs</h2>
            <button type="button" onClick={handleRun}>
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
                <strong>{selectedPoint.label}</strong>
                <p>
                  {selectedPoint.kind === "token"
                    ? `Raw token ${selectedPoint.rawToken ?? selectedPoint.snippet}${selectedPoint.tokenId === undefined ? "" : ` · id ${selectedPoint.tokenId}`}`
                    : selectedPoint.snippet}
                </p>
                <span className="metaLabel">Source</span>
                <p>{selectedPoint.source} · {selectedPoint.output}</p>
                <span className="metaLabel">Dimensions</span>
                <p>{selectedPoint.vector.length}</p>
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
