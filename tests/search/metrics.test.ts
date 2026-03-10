import { describe, it, expect } from 'vitest';
import {
  cosineDistance,
  dotProductDistance,
  euclideanDistance,
  metricFromString,
  MetricType,
} from '../../src/search/metrics.js';

describe('metricFromString', () => {
  it('maps cosine', () => {
    expect(metricFromString('cosine')).toBe(MetricType.Cosine);
  });

  it('maps dot product variants', () => {
    expect(metricFromString('dot')).toBe(MetricType.DotProduct);
    expect(metricFromString('dotproduct')).toBe(MetricType.DotProduct);
    expect(metricFromString('dot_product')).toBe(MetricType.DotProduct);
  });

  it('maps euclidean variants', () => {
    expect(metricFromString('euclidean')).toBe(MetricType.Euclidean);
    expect(metricFromString('l2')).toBe(MetricType.Euclidean);
  });

  it('throws on unknown metric', () => {
    expect(() => metricFromString('hamming')).toThrow('Unknown metric');
  });
});

describe('cosineDistance', () => {
  it('returns 0 for identical unit vectors', () => {
    const v = new Float32Array([1, 0, 0]);
    expect(cosineDistance(v, v)).toBeCloseTo(0, 5);
  });

  it('returns ~1 for orthogonal vectors', () => {
    const a = new Float32Array([1, 0]);
    const b = new Float32Array([0, 1]);
    expect(cosineDistance(a, b)).toBeCloseTo(1, 5);
  });

  it('returns ~2 for opposite vectors', () => {
    const a = new Float32Array([1, 0]);
    const b = new Float32Array([-1, 0]);
    expect(cosineDistance(a, b)).toBeCloseTo(2, 5);
  });
});

describe('dotProductDistance', () => {
  it('returns negative dot product', () => {
    const a = new Float32Array([1, 2, 3]);
    const b = new Float32Array([4, 5, 6]);
    // dot = 4 + 10 + 18 = 32
    expect(dotProductDistance(a, b)).toBeCloseTo(-32, 5);
  });
});

describe('euclideanDistance', () => {
  it('returns 0 for identical vectors', () => {
    const v = new Float32Array([1, 2, 3]);
    expect(euclideanDistance(v, v)).toBeCloseTo(0, 5);
  });

  it('computes correct L2 distance', () => {
    const a = new Float32Array([0, 0]);
    const b = new Float32Array([3, 4]);
    expect(euclideanDistance(a, b)).toBeCloseTo(5, 5);
  });
});
