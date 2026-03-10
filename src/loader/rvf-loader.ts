export interface RvfLoadResult {
  buffer: ArrayBuffer;
  byteLength: number;
  loadTimeMs: number;
}

export async function fetchRvf(url: string, signal?: AbortSignal): Promise<RvfLoadResult> {
  const start = performance.now();
  const response = await fetch(url, { signal });

  if (!response.ok) {
    throw new Error(`Failed to fetch RVF: ${response.status} ${response.statusText}`);
  }

  const buffer = await response.arrayBuffer();
  const loadTimeMs = performance.now() - start;

  return { buffer, byteLength: buffer.byteLength, loadTimeMs };
}

export async function fetchRvfCached(
  url: string,
  cacheName = 'rvf-cache',
): Promise<RvfLoadResult> {
  if (typeof caches === 'undefined') {
    return fetchRvf(url);
  }

  const cache = await caches.open(cacheName);
  const cached = await cache.match(url);

  if (cached) {
    const start = performance.now();
    const buffer = await cached.arrayBuffer();
    return { buffer, byteLength: buffer.byteLength, loadTimeMs: performance.now() - start };
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch RVF: ${response.status} ${response.statusText}`);
  }

  await cache.put(url, response.clone());

  const start = performance.now();
  const buffer = await response.arrayBuffer();
  return { buffer, byteLength: buffer.byteLength, loadTimeMs: performance.now() - start };
}
