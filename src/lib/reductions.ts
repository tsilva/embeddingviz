import type { PipelineStatus, ReductionMethod } from "../types";
import { projectReductionCore } from "./reductionCore";
import type { ProjectionResult, VectorRow } from "./pca";

type StatusReporter = (status: PipelineStatus) => void;

interface ReductionWorkerResult {
  id: string;
  type: "result";
  coordinates: Float32Array;
  explained: [number, number, number];
}

interface ReductionWorkerStatus {
  id: string;
  type: "status";
  status: PipelineStatus;
}

interface ReductionWorkerError {
  id: string;
  type: "error";
  message: string;
}

type ReductionWorkerMessage = ReductionWorkerResult | ReductionWorkerStatus | ReductionWorkerError;

export async function projectReduction(vectors: VectorRow[], method: ReductionMethod, onStatus: StatusReporter): Promise<ProjectionResult> {
  if (typeof Worker === "undefined" || vectors.length === 0) {
    return projectReductionCore(vectors, method, onStatus);
  }

  try {
    return await projectReductionInWorker(vectors, method, onStatus);
  } catch (error) {
    console.warn("Projection worker failed, falling back to main thread projection.", error);
    return projectReductionCore(vectors, method, onStatus);
  }
}

async function projectReductionInWorker(vectors: VectorRow[], method: ReductionMethod, onStatus: StatusReporter) {
  const { data, rows, dimensions } = packVectors(vectors);
  const id = crypto.randomUUID();
  const worker = await createReductionWorker();

  return new Promise<ProjectionResult>((resolve, reject) => {
    worker.onmessage = (event: MessageEvent<ReductionWorkerMessage>) => {
      const message = event.data;
      if (message.id !== id) return;

      if (message.type === "status") {
        onStatus(message.status);
        return;
      }

      worker.terminate();
      if (message.type === "error") {
        reject(new Error(message.message));
        return;
      }

      resolve({
        coordinates: unpackCoordinates(message.coordinates, rows),
        explained: message.explained,
      });
    };

    worker.onerror = (event) => {
      worker.terminate();
      reject(new Error(event.message || "Projection worker failed"));
    };

    worker.postMessage({ id, method, rows, dimensions, buffer: data.buffer }, [data.buffer]);
  });
}

async function createReductionWorker() {
  const { default: ReductionWorker } = await import("./reductions.worker?worker");
  return new ReductionWorker();
}

function packVectors(vectors: VectorRow[]) {
  const rows = vectors.length;
  const dimensions = vectors[0]?.length ?? 0;
  const data = new Float32Array(rows * dimensions);

  vectors.forEach((row, rowIndex) => {
    const offset = rowIndex * dimensions;
    for (let col = 0; col < dimensions; col += 1) {
      data[offset + col] = Number(row[col] ?? 0);
    }
  });

  return { data, rows, dimensions };
}

function unpackCoordinates(data: Float32Array, rows: number): Array<[number, number, number]> {
  return Array.from({ length: rows }, (_, index) => {
    const offset = index * 3;
    return [data[offset] ?? 0, data[offset + 1] ?? 0, data[offset + 2] ?? 0] as [number, number, number];
  });
}
