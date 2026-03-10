import { getWasm } from '../wasm/init.js';

export type LoadLayer = 'A' | 'B' | 'C';

export interface LayerInfo {
  layer: LoadLayer;
  description: string;
  loaded: boolean;
  loadTimeMs: number;
}

export interface ProgressiveLoadResult {
  storeHandle: number;
  dimension: number;
  vectorCount: number;
  layers: LayerInfo[];
  totalLoadTimeMs: number;
}

/**
 * Progressive RVF loader.
 *
 * Layer A (Cold): Header + metadata only — instant, validates the file.
 * Layer B (Warm): Hot weights + adjacency graphs — ~500ms target, search-ready.
 * Layer C (Hot):  Full deep load — all data, maximum precision.
 */
export async function loadProgressive(
  rvfBuffer: ArrayBuffer,
  targetLayer: LoadLayer = 'B',
): Promise<ProgressiveLoadResult> {
  const wasm = getWasm();
  const bytes = new Uint8Array(rvfBuffer);
  const layers: LayerInfo[] = [];
  const totalStart = performance.now();

  // Layer A: Header verification
  const layerAStart = performance.now();
  const headerPtr = wasm.rvf_alloc(bytes.byteLength);
  const mem = new Uint8Array(wasm.memory.buffer);
  mem.set(bytes, headerPtr);

  const headerValid = wasm.rvf_verify_header(headerPtr);
  if (headerValid !== 0) {
    wasm.rvf_free(headerPtr, bytes.byteLength);
    throw new Error(`Invalid RVF header (code: ${headerValid})`);
  }

  layers.push({
    layer: 'A',
    description: 'Header + metadata verification',
    loaded: true,
    loadTimeMs: performance.now() - layerAStart,
  });

  if (targetLayer === 'A') {
    wasm.rvf_free(headerPtr, bytes.byteLength);
    return {
      storeHandle: -1,
      dimension: 0,
      vectorCount: 0,
      layers,
      totalLoadTimeMs: performance.now() - totalStart,
    };
  }

  // Layer B: Open store (hot weights + adjacency)
  const layerBStart = performance.now();
  const storeHandle = wasm.rvf_store_open(headerPtr, bytes.byteLength);
  if (storeHandle < 0) {
    wasm.rvf_free(headerPtr, bytes.byteLength);
    throw new Error(`Failed to open RVF store (code: ${storeHandle})`);
  }

  const dimension = wasm.rvf_store_dimension(storeHandle);
  const vectorCount = wasm.rvf_store_count(storeHandle);

  layers.push({
    layer: 'B',
    description: 'Hot weights + adjacency graphs',
    loaded: true,
    loadTimeMs: performance.now() - layerBStart,
  });

  if (targetLayer === 'B') {
    return {
      storeHandle,
      dimension,
      vectorCount,
      layers,
      totalLoadTimeMs: performance.now() - totalStart,
    };
  }

  // Layer C: Full checksum verification
  const layerCStart = performance.now();
  const checksumValid = wasm.rvf_verify_checksum(headerPtr, bytes.byteLength);
  layers.push({
    layer: 'C',
    description: 'Full deep load + checksum verification',
    loaded: checksumValid === 0,
    loadTimeMs: performance.now() - layerCStart,
  });

  if (checksumValid !== 0) {
    console.warn('RVF checksum verification failed; data may be corrupted');
  }

  return {
    storeHandle,
    dimension,
    vectorCount,
    layers,
    totalLoadTimeMs: performance.now() - totalStart,
  };
}
