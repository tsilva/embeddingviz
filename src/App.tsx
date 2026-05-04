import { useMemo, useState } from "react";
import {
  BrainCircuit,
  Check,
  ChevronDown,
  Database,
  FileText,
  Image,
  Layers,
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
  const selectedPoint = useMemo(
    () => runs.flatMap((run) => run.points).find((point) => point.id === selectedPointId) ?? runs[0]?.points[0] ?? null,
    [runs, selectedPointId],
  );
  const totalVisible = runs.filter((run) => run.visible).reduce((sum, run) => sum + run.count, 0);

  async function handleRun() {
    try {
      setStatus({ phase: "loading", message: "Preparing model", progress: 0.02 });
      const result = await createEmbeddingRun({
        model,
        outputMode,
        inputType,
        snippets,
        files,
        onStatus: setStatus,
      });

      const runIndex = runs.length;
      const run: RunRecord = {
        id: `${Date.now()}`,
        name: runName(inputType),
        output: outputLabel(outputMode, model.task),
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

  function updateSnippet(id: string, field: "text" | "label", value: string) {
    setSnippets((current) => current.map((snippet) => (snippet.id === id ? { ...snippet, [field]: value } : snippet)));
  }

  function addSnippet() {
    setSnippets((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        text: "",
        label: "",
        group: "ML / NLP",
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
    if (inputType === "images" && !next.supportsImages) {
      setInputType("text");
    }
  }

  function chooseInputType(nextInputType: InputType) {
    setInputType(nextInputType);
    if (nextInputType === "tokens") {
      setOutputMode("tokens");
    } else if (outputMode === "tokens") {
      setOutputMode(model.recommendedOutput === "tokens" ? "final" : model.recommendedOutput);
    }
  }

  function chooseOutputMode(nextOutputMode: OutputMode) {
    setOutputMode(nextOutputMode);
    if (nextOutputMode === "tokens") {
      setInputType("tokens");
    }
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
        <div className={`statusPill ${status.phase}`}>
          {status.phase === "loading" || status.phase === "embedding" || status.phase === "projecting" ? <Loader2 size={15} className="spin" /> : <Check size={15} />}
          {status.phase === "error" ? "Needs attention" : "Ready"}
        </div>

        <button className="runButton" type="button" onClick={handleRun} disabled={status.phase === "loading" || status.phase === "embedding" || status.phase === "projecting"}>
          {status.phase === "loading" || status.phase === "embedding" || status.phase === "projecting" ? <Loader2 size={17} className="spin" /> : <Play size={17} fill="currentColor" />}
          Run
        </button>
        <div className="progressTrack" aria-hidden="true">
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
              <select id="output" value={outputMode} onChange={(event) => chooseOutputMode(event.target.value as OutputMode)}>
                <option value="final">Final embedding</option>
                <option value="hidden-4">Layer 4 · hidden state</option>
                <option value="tokens">Token table · embeddings</option>
              </select>
              <ChevronDown size={16} />
            </div>
            <p className="recommend">
              <Sparkles size={13} />
              Recommended
            </p>
          </section>

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
              <button className={inputType === "images" ? "active" : ""} type="button" onClick={() => chooseInputType("images")}>
                <Image size={15} />
                Images
              </button>
              <button className={inputType === "tokens" ? "active" : ""} type="button" onClick={() => chooseInputType("tokens")}>
                <Layers size={15} />
                Tokens
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
                    <input value={snippet.text} onChange={(event) => updateSnippet(snippet.id, "text", event.target.value)} aria-label="Snippet text" />
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
              <div className="fileDrop">
                <input
                  type="file"
                  multiple
                  accept=".txt,.md,.csv,.json"
                  onChange={(event) => setFiles(Array.from(event.target.files ?? []))}
                />
                <span>{files.length ? `${files.length} files selected` : "Choose text files"}</span>
              </div>
            ) : null}

            {inputType === "images" ? <p className="notice">Image inputs require an image-feature-extraction model.</p> : null}
            {inputType === "tokens" ? <p className="notice">Plots the tokenizer vocabulary with a WebGL point layer.</p> : null}
          </section>

          <section className="controlSection">
            <span className="fieldLabel">Reduction</span>
            <div className="segmented">
              {(["PCA", "UMAP", "t-SNE"] as ReductionMethod[]).map((method) => (
                <button
                  key={method}
                  className={reduction === method ? "active" : ""}
                  type="button"
                  onClick={() => setReduction(method)}
                  title={method === "PCA" ? "Available now" : "PCA fallback is used in this MVP"}
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
                  <span>{run.output}</span>
                  <small>{run.count} points</small>
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
                <span className="metaLabel">{selectedPoint.kind === "token" ? "Subword token" : "Label / Snippet"}</span>
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

      <footer className="statusBar">
        <div className="statusSegment">
          <Check size={18} />
          <span>{status.message}</span>
        </div>
        <div className="statusSegment">{totalVisible} visible points</div>
        <div className="statusSegment">{reduction === "PCA" ? "PCA projected" : `${reduction} selected · PCA fallback`}</div>
      </footer>
    </div>
  );
}

function runName(inputType: InputType) {
  if (inputType === "tokens") return "Token table";
  if (inputType === "files") return "Text files";
  if (inputType === "images") return "Images";
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

export default App;
