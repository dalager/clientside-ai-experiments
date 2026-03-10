import wasmInit, { type RvfWasmExports } from '@ruvector/rvf-wasm';

let instance: RvfWasmExports | null = null;

export async function initWasm(wasmBytes?: ArrayBuffer): Promise<RvfWasmExports> {
  if (instance) return instance;
  instance = await wasmInit(wasmBytes);
  return instance;
}

export function getWasm(): RvfWasmExports {
  if (!instance) throw new Error('WASM not initialized. Call initWasm() first.');
  return instance;
}

export function resetWasm(): void {
  instance = null;
}
