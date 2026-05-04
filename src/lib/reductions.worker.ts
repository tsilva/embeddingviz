import type { PipelineStatus, ReductionMethod } from "../types";
import { projectReductionCore } from "./reductionCore";

interface ReductionRequest {
  id: string;
  method: ReductionMethod;
  rows: number;
  dimensions: number;
  buffer: ArrayBuffer;
}

type ReductionWorkerMessage =
  | { id: string; type: "status"; status: PipelineStatus }
  | { id: string; type: "result"; coordinates: Float32Array; explained: [number, number, number] }
  | { id: string; type: "error"; message: string };

self.onmessage = async (event: MessageEvent<ReductionRequest>) => {
  const { id, method, rows, dimensions, buffer } = event.data;
  try {
    const data = new Float32Array(buffer);
    const vectors = Array.from({ length: rows }, (_, index) => data.subarray(index * dimensions, (index + 1) * dimensions));
    const projection = await projectReductionCore(vectors, method, (status) => {
      postWorkerMessage({ id, type: "status", status });
    });
    const coordinates = packCoordinates(projection.coordinates);
    postWorkerMessage({ id, type: "result", coordinates, explained: projection.explained }, [coordinates.buffer]);
  } catch (error) {
    postWorkerMessage({
      id,
      type: "error",
      message: error instanceof Error ? error.message : "Projection worker failed",
    });
  }
};

function packCoordinates(coordinates: Array<[number, number, number]>) {
  const packed = new Float32Array(coordinates.length * 3);
  coordinates.forEach(([x, y, z], index) => {
    const offset = index * 3;
    packed[offset] = x;
    packed[offset + 1] = y;
    packed[offset + 2] = z;
  });
  return packed;
}

function postWorkerMessage(message: ReductionWorkerMessage, transfer?: Transferable[]) {
  self.postMessage(message, { transfer });
}

export {};
