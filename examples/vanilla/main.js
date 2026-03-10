import init from '@ruvector/rvf-wasm';

const statusEl = document.getElementById('status');
const searchBox = document.getElementById('search-box');
const resultsEl = document.getElementById('results');
const vecCountEl = document.getElementById('vec-count');
const dimCountEl = document.getElementById('dim-count');
const loadTimeEl = document.getElementById('load-time');

let wasm = null;
let storeHandle = -1;
let dimension = 0;

async function initialize() {
  try {
    statusEl.textContent = 'Loading WASM runtime (5.5 KB)...';
    wasm = await init();

    statusEl.textContent = 'WASM ready. Creating demo vector store...';

    // Create an in-memory store with 128 dimensions, cosine metric (0)
    dimension = 128;
    storeHandle = wasm.rvf_store_create(dimension, 0);

    if (storeHandle < 0) {
      throw new Error(`Failed to create store (code: ${storeHandle})`);
    }

    // Ingest demo vectors
    const numVectors = 1000;
    const vecBytes = numVectors * dimension * 4; // float32
    const idBytes = numVectors * 8; // bigint64

    const vecsPtr = wasm.rvf_alloc(vecBytes);
    const idsPtr = wasm.rvf_alloc(idBytes);

    const vecView = new Float32Array(wasm.memory.buffer, vecsPtr, numVectors * dimension);
    const idView = new BigInt64Array(wasm.memory.buffer, idsPtr, numVectors);

    for (let i = 0; i < numVectors; i++) {
      idView[i] = BigInt(i);
      for (let d = 0; d < dimension; d++) {
        vecView[i * dimension + d] = Math.random() * 2 - 1;
      }
      // Normalize
      let norm = 0;
      for (let d = 0; d < dimension; d++) {
        norm += vecView[i * dimension + d] ** 2;
      }
      norm = Math.sqrt(norm);
      for (let d = 0; d < dimension; d++) {
        vecView[i * dimension + d] /= norm;
      }
    }

    const start = performance.now();
    const ingested = wasm.rvf_store_ingest(storeHandle, vecsPtr, idsPtr, numVectors);
    const loadTime = performance.now() - start;

    wasm.rvf_free(vecsPtr, vecBytes);
    wasm.rvf_free(idsPtr, idBytes);

    if (ingested < 0) {
      throw new Error(`Ingest failed (code: ${ingested})`);
    }

    const count = wasm.rvf_store_count(storeHandle);
    vecCountEl.textContent = count.toString();
    dimCountEl.textContent = dimension.toString();
    loadTimeEl.textContent = `${loadTime.toFixed(1)}ms`;

    statusEl.textContent = `Ready — ${count} vectors indexed. Type to search.`;
    searchBox.disabled = false;
    searchBox.focus();
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
    console.error(err);
  }
}

function search(queryVector) {
  const k = 10;
  const queryBytes = dimension * 4;
  const queryPtr = wasm.rvf_alloc(queryBytes);
  const qView = new Float32Array(wasm.memory.buffer, queryPtr, dimension);
  qView.set(queryVector);

  const resultSize = k * 12; // 8 bytes id + 4 bytes score
  const outPtr = wasm.rvf_alloc(resultSize);

  const start = performance.now();
  const count = wasm.rvf_store_query(storeHandle, queryPtr, k, 0, outPtr);
  const searchTime = performance.now() - start;

  const results = [];
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

  return { results, searchTime };
}

function generateQueryFromText(text) {
  // Simple deterministic hash-based vector generation from text
  // In production, this would use a text embedding model
  const vec = new Float32Array(dimension);
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    hash = ((hash << 5) - hash + text.charCodeAt(i)) | 0;
  }
  const rng = seedRng(hash);
  for (let i = 0; i < dimension; i++) {
    vec[i] = rng() * 2 - 1;
  }
  // Normalize
  let norm = 0;
  for (let i = 0; i < dimension; i++) norm += vec[i] ** 2;
  norm = Math.sqrt(norm);
  for (let i = 0; i < dimension; i++) vec[i] /= norm;
  return vec;
}

function seedRng(seed) {
  return () => {
    seed = (seed * 1664525 + 1013904223) | 0;
    return (seed >>> 0) / 4294967296;
  };
}

function renderResults(results, searchTime) {
  if (results.length === 0) {
    resultsEl.innerHTML = '<div style="color:#64748b;text-align:center;padding:1rem;">No results</div>';
    return;
  }
  resultsEl.innerHTML = results
    .map(
      (r) =>
        `<div class="result">
          <span class="result-id">ID: ${r.id.toString()}</span>
          <span class="result-score">${r.score.toFixed(6)}</span>
        </div>`,
    )
    .join('');
  statusEl.textContent = `Found ${results.length} results in ${searchTime.toFixed(3)}ms`;
}

let debounceTimer;
searchBox.addEventListener('input', (e) => {
  clearTimeout(debounceTimer);
  const query = e.target.value.trim();
  if (!query) {
    resultsEl.innerHTML = '';
    statusEl.textContent = 'Ready — type to search.';
    return;
  }
  debounceTimer = setTimeout(() => {
    const queryVec = generateQueryFromText(query);
    const { results, searchTime } = search(queryVec);
    renderResults(results, searchTime);
  }, 150);
});

initialize();
