import { getWasm } from '../wasm/init.js';
import { MetricType } from './metrics.js';

export interface SearchResult {
  id: bigint;
  score: number;
}

export interface HnswSearchOptions {
  k?: number;
  metric?: MetricType;
}

/**
 * HNSW search engine backed by the RVF WASM microkernel.
 * Wraps the low-level rvf_store_query into a typed API.
 */
export class HnswEngine {
  constructor(
    private storeHandle: number,
    private dimension: number,
  ) {}

  search(queryVector: Float32Array, options: HnswSearchOptions = {}): SearchResult[] {
    const { k = 10, metric = MetricType.Cosine } = options;

    if (queryVector.length !== this.dimension) {
      throw new Error(
        `Query dimension mismatch: expected ${this.dimension}, got ${queryVector.length}`,
      );
    }

    const wasm = getWasm();

    // Allocate input buffer for query vector
    const queryBytes = queryVector.byteLength;
    const queryPtr = wasm.rvf_alloc(queryBytes);
    const mem = new Float32Array(wasm.memory.buffer, queryPtr, this.dimension);
    mem.set(queryVector);

    // Allocate output buffer: k results, each = bigint id (8 bytes) + f32 score (4 bytes) = 12 bytes
    const resultSize = k * 12;
    const outPtr = wasm.rvf_alloc(resultSize);

    const count = wasm.rvf_store_query(this.storeHandle, queryPtr, k, metric, outPtr);

    const results: SearchResult[] = [];
    if (count > 0) {
      const view = new DataView(wasm.memory.buffer, outPtr, count * 12);
      for (let i = 0; i < count; i++) {
        const offset = i * 12;
        const id = view.getBigUint64(offset, true);
        const score = view.getFloat32(offset + 8, true);
        results.push({ id, score });
      }
    }

    wasm.rvf_free(queryPtr, queryBytes);
    wasm.rvf_free(outPtr, resultSize);

    return results;
  }

  get vectorCount(): number {
    return getWasm().rvf_store_count(this.storeHandle);
  }

  close(): void {
    getWasm().rvf_store_close(this.storeHandle);
  }
}
