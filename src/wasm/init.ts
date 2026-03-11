import wasmInit, { type RvfWasmExports } from '@ruvector/rvf-wasm';

let instance: RvfWasmExports | null = null;
let pending: Promise<RvfWasmExports> | null = null;

export async function initWasm(wasmBytes?: ArrayBuffer): Promise<RvfWasmExports> {
  if (instance) return instance;
  if (pending) return pending;
  pending = wasmInit(wasmBytes).then((w) => {
    instance = w;
    pending = null;
    return w;
  });
  return pending;
}

export function getWasm(): RvfWasmExports {
  if (!instance) throw new Error('WASM not initialized. Call initWasm() first.');
  return instance;
}

export function resetWasm(): void {
  instance = null;
  pending = null;
}
