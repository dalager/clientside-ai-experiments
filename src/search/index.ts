import { initWasm } from '../wasm/init.js';
import { fetchRvfCached, type RvfLoadResult } from '../loader/rvf-loader.js';
import { loadProgressive, type LoadLayer, type ProgressiveLoadResult } from '../loader/progressive-loader.js';
import { HnswEngine, type SearchResult, type HnswSearchOptions } from './hnsw-engine.js';
import { MetricType, metricFromString } from './metrics.js';

export { MetricType, metricFromString } from './metrics.js';
export type { SearchResult } from './hnsw-engine.js';
export type { RvfLoadResult } from '../loader/rvf-loader.js';
export type { ProgressiveLoadResult, LoadLayer } from '../loader/progressive-loader.js';

export interface SearchEngineOptions {
  rvfUrl: string;
  layer?: LoadLayer;
  wasmBytes?: ArrayBuffer;
  cacheName?: string;
}

export interface SearchOptions {
  k?: number;
  metric?: MetricType | string;
}

export class SearchEngine {
  private engine: HnswEngine | null = null;
  private loadResult: ProgressiveLoadResult | null = null;

  async init(options: SearchEngineOptions): Promise<ProgressiveLoadResult> {
    await initWasm(options.wasmBytes);

    const rvf = await fetchRvfCached(options.rvfUrl, options.cacheName);
    const result = await loadProgressive(rvf.buffer, options.layer ?? 'B');

    this.engine = new HnswEngine(result.storeHandle, result.dimension);
    this.loadResult = result;
    return result;
  }

  search(queryVector: Float32Array, options: SearchOptions = {}): SearchResult[] {
    if (!this.engine) throw new Error('SearchEngine not initialized. Call init() first.');

    const metric = typeof options.metric === 'string'
      ? metricFromString(options.metric)
      : options.metric;

    return this.engine.search(queryVector, { k: options.k, metric });
  }

  get stats(): { dimension: number; vectorCount: number; layers: string[] } | null {
    if (!this.loadResult) return null;
    return {
      dimension: this.loadResult.dimension,
      vectorCount: this.loadResult.vectorCount,
      layers: this.loadResult.layers.filter(l => l.loaded).map(l => l.layer),
    };
  }

  close(): void {
    this.engine?.close();
    this.engine = null;
    this.loadResult = null;
  }
}
