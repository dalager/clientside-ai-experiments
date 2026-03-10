import { describe, it, expect, beforeEach, vi } from 'vitest';

vi.mock('@ruvector/rvf-wasm', () => ({
  default: vi.fn().mockResolvedValue({ memory: {} }),
}));

const { getWasm, resetWasm } = await import('../../src/wasm/init.js');

describe('wasm/init', () => {
  beforeEach(() => {
    resetWasm();
  });

  it('throws when getWasm called before init', () => {
    expect(() => getWasm()).toThrow('WASM not initialized');
  });
});
