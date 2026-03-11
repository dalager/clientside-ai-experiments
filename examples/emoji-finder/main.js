import init from '@ruvector/rvf-wasm';
import { EMOJIS } from './emoji-data.js';

const searchBox = document.getElementById('search-box');
const resultsEl = document.getElementById('results');
const statusEl = document.getElementById('status');
const countEl = document.getElementById('emoji-count');
const dimEl = document.getElementById('dim-count');
const loadTimeEl = document.getElementById('load-time');

let wasm = null;
let storeHandle = -1;
const DIMENSION = 512;

// --- Text → Vector via multi-probe feature hashing ---
// Each word activates NUM_HASHES dimensions with random signs.
// Multiple probes average out collision noise (like a Bloom filter for similarity).

const NUM_HASHES = 4;

function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, '')
    .split(/\s+/)
    .filter(w => w.length > 1);
}

// FNV-1a with configurable seed
function fnv1a(str, seed = 0x811c9dc5) {
  let h = seed;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

// Different seeds for each probe
const SEEDS = [0x811c9dc5, 0x1a2b3c4d, 0x5e6f7a8b, 0xdeadbeef];

function textToVector(text) {
  const tokens = tokenize(text);
  const vec = new Float32Array(DIMENSION);

  for (const token of tokens) {
    for (let p = 0; p < NUM_HASHES; p++) {
      const h = fnv1a(token, SEEDS[p]);
      const idx = h % DIMENSION;
      const sign = (h & 0x100) ? 1.0 : -1.0;
      vec[idx] += sign;
    }
  }

  // L2 normalize
  let norm = 0;
  for (let i = 0; i < DIMENSION; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (let i = 0; i < DIMENSION; i++) vec[i] /= norm;
  }
  return vec;
}

// --- WASM store operations ---

async function initialize() {
  statusEl.textContent = 'Loading WASM runtime...';
  wasm = await init();

  statusEl.textContent = 'Building emoji index...';
  const start = performance.now();

  storeHandle = wasm.rvf_store_create(DIMENSION, 0); // cosine metric
  if (storeHandle < 0) {
    throw new Error(`Failed to create store (code: ${storeHandle})`);
  }

  const numEmojis = EMOJIS.length;
  const vecBytes = numEmojis * DIMENSION * 4;
  const idBytes = numEmojis * 8;
  const vecsPtr = wasm.rvf_alloc(vecBytes);
  const idsPtr = wasm.rvf_alloc(idBytes);

  const vecView = new Float32Array(wasm.memory.buffer, vecsPtr, numEmojis * DIMENSION);
  const idView = new BigInt64Array(wasm.memory.buffer, idsPtr, numEmojis);

  for (let i = 0; i < numEmojis; i++) {
    const [_emoji, ...descriptions] = EMOJIS[i];
    const text = descriptions.join(' ');
    const vec = textToVector(text);
    vecView.set(vec, i * DIMENSION);
    idView[i] = BigInt(i);
  }

  const ingested = wasm.rvf_store_ingest(storeHandle, vecsPtr, idsPtr, numEmojis);
  if (ingested < 0) {
    throw new Error(`Ingest failed (code: ${ingested})`);
  }
  const loadTime = performance.now() - start;

  wasm.rvf_free(vecsPtr, vecBytes);
  wasm.rvf_free(idsPtr, idBytes);

  countEl.textContent = numEmojis.toString();
  dimEl.textContent = DIMENSION.toString();
  loadTimeEl.textContent = `${loadTime.toFixed(1)}ms`;

  statusEl.textContent = 'Type a feeling, concept, or word...';
  searchBox.disabled = false;
  searchBox.focus();
}

function search(query) {
  const queryVec = textToVector(query);
  const k = 20;
  const queryPtr = wasm.rvf_alloc(DIMENSION * 4);
  new Float32Array(wasm.memory.buffer, queryPtr, DIMENSION).set(queryVec);

  const outSize = k * 12;
  const outPtr = wasm.rvf_alloc(outSize);

  const t0 = performance.now();
  const count = wasm.rvf_store_query(storeHandle, queryPtr, k, 0, outPtr);
  const searchTime = performance.now() - t0;

  const results = [];
  if (count > 0) {
    const view = new DataView(wasm.memory.buffer, outPtr, count * 12);
    for (let i = 0; i < count; i++) {
      const id = Number(view.getBigUint64(i * 12, true));
      const distance = view.getFloat32(i * 12 + 8, true);
      // cosine distance: 0 = identical, 2 = opposite
      // convert to similarity percentage: (1 - distance/2) * 100
      const similarity = Math.max(0, (1 - distance / 2) * 100);
      results.push({ ...emojiEntry(id), distance, similarity });
    }
  }

  wasm.rvf_free(queryPtr, DIMENSION * 4);
  wasm.rvf_free(outPtr, outSize);

  return { results, searchTime, count };
}

function emojiEntry(id) {
  const [emoji, name, ...tags] = EMOJIS[id];
  return { emoji, name, tags };
}

// --- Rendering ---

function renderResults(results, searchTime) {
  if (results.length === 0) {
    resultsEl.innerHTML = '<div class="empty">No matches found</div>';
    return;
  }

  resultsEl.innerHTML = results
    .map(r => {
      const barWidth = Math.max(4, r.similarity);
      return `<div class="result">
        <span class="emoji">${r.emoji}</span>
        <div class="info">
          <div class="name">${r.name}</div>
          <div class="tags">${r.tags.slice(0, 5).join(' · ')}</div>
          <div class="bar-bg"><div class="bar" style="width:${barWidth}%"></div></div>
        </div>
        <span class="score">${r.similarity.toFixed(0)}%</span>
      </div>`;
    })
    .join('');

  statusEl.textContent = `${results.length} matches in ${searchTime.toFixed(3)}ms`;
}

// --- Suggestions ---

const SUGGESTIONS = [
  'happy', 'sad', 'love', 'angry', 'food', 'travel', 'music',
  'work', 'animal', 'weather', 'party', 'money', 'scared',
  'strong', 'smart', 'fast', 'nature', 'funny', 'tired',
];

function renderSuggestions() {
  resultsEl.innerHTML = '<div class="suggestions">' +
    SUGGESTIONS.map(s => `<button class="chip" data-query="${s}">${s}</button>`).join('') +
    '</div>';
}

resultsEl.addEventListener('click', (e) => {
  const chip = e.target.closest('.chip');
  if (chip) {
    searchBox.value = chip.dataset.query;
    searchBox.dispatchEvent(new Event('input'));
  }
});

// --- Search handler ---

let debounceTimer;
searchBox.addEventListener('input', (e) => {
  clearTimeout(debounceTimer);
  const query = e.target.value.trim();
  if (!query) {
    renderSuggestions();
    statusEl.textContent = 'Type a feeling, concept, or word...';
    return;
  }
  debounceTimer = setTimeout(() => {
    const { results, searchTime } = search(query);
    renderResults(results, searchTime);
  }, 100);
});

initialize().then(renderSuggestions).catch(err => {
  statusEl.textContent = `Error: ${err.message}`;
  console.error(err);
});
