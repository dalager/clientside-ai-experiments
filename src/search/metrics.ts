export const enum MetricType {
  Cosine = 0,
  DotProduct = 1,
  Euclidean = 2,
}

export function metricFromString(name: string): MetricType {
  switch (name.toLowerCase()) {
    case 'cosine': return MetricType.Cosine;
    case 'dot':
    case 'dotproduct':
    case 'dot_product': return MetricType.DotProduct;
    case 'euclidean':
    case 'l2': return MetricType.Euclidean;
    default: throw new Error(`Unknown metric: ${name}`);
  }
}

/** Pure-JS fallback distance functions (used when WASM is unavailable). */
export function cosineDistance(a: Float32Array, b: Float32Array): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 1 : 1 - dot / denom;
}

export function dotProductDistance(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return -dot; // negate so lower = more similar
}

export function euclideanDistance(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}
