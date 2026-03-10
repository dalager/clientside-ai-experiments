import { describe, it, expect, vi, beforeEach } from 'vitest';
import { fetchRvf } from '../../src/loader/rvf-loader.js';

describe('fetchRvf', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('returns buffer and timing on successful fetch', async () => {
    const mockBuffer = new ArrayBuffer(1024);
    const mockResponse = {
      ok: true,
      arrayBuffer: vi.fn().mockResolvedValue(mockBuffer),
    } as unknown as Response;
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(mockResponse));

    const result = await fetchRvf('https://example.com/test.rvf');

    expect(result.buffer).toBe(mockBuffer);
    expect(result.byteLength).toBe(1024);
    expect(result.loadTimeMs).toBeGreaterThanOrEqual(0);
  });

  it('throws on non-ok response', async () => {
    const mockResponse = {
      ok: false,
      status: 404,
      statusText: 'Not Found',
    } as unknown as Response;
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(mockResponse));

    await expect(fetchRvf('https://example.com/missing.rvf')).rejects.toThrow(
      'Failed to fetch RVF: 404 Not Found',
    );
  });

  it('passes abort signal to fetch', async () => {
    const controller = new AbortController();
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      arrayBuffer: vi.fn().mockResolvedValue(new ArrayBuffer(0)),
    });
    vi.stubGlobal('fetch', fetchMock);

    await fetchRvf('https://example.com/test.rvf', controller.signal);

    expect(fetchMock).toHaveBeenCalledWith('https://example.com/test.rvf', {
      signal: controller.signal,
    });
  });
});
