import type { InputType, ModelPreset, OutputMode, PipelineStatus } from "../types";
import { extractVectorsCore, type ResolvedSample } from "./embeddings";

interface EmbeddingRequest {
  id: string;
  model: ModelPreset;
  samples: ResolvedSample[];
  inputType: InputType;
  outputMode: OutputMode;
}

type EmbeddingWorkerMessage =
  | { id: string; type: "status"; status: PipelineStatus }
  | { id: string; type: "result"; rows: number; dimensions: number; buffer: ArrayBuffer }
  | { id: string; type: "error"; message: string };

self.onmessage = async (event: MessageEvent<EmbeddingRequest>) => {
  const { id, model, samples, inputType, outputMode } = event.data;
  try {
    const vectors = await extractVectorsCore(model, samples, inputType, outputMode, (status) => {
      postWorkerMessage({ id, type: "status", status });
    });
    const { rows, dimensions, data } = packVectors(vectors);
    postWorkerMessage({ id, type: "result", rows, dimensions, buffer: data.buffer }, [data.buffer]);
  } catch (error) {
    postWorkerMessage({
      id,
      type: "error",
      message: error instanceof Error ? error.message : "Embedding worker failed",
    });
  }
};

function packVectors(vectors: ArrayLike<number>[]) {
  const rows = vectors.length;
  const dimensions = vectors[0]?.length ?? 0;
  const data = new Float32Array(rows * dimensions);

  vectors.forEach((row, rowIndex) => {
    const offset = rowIndex * dimensions;
    for (let col = 0; col < dimensions; col += 1) {
      data[offset + col] = Number(row[col] ?? 0);
    }
  });

  return { rows, dimensions, data };
}

function postWorkerMessage(message: EmbeddingWorkerMessage, transfer?: Transferable[]) {
  self.postMessage(message, { transfer });
}

export {};
