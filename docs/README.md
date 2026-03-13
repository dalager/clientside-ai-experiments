# wasmllm

Browser-native semantic search powered by WebAssembly and the RuVector Format (RVF). Runs entirely in a browser tab with zero backend infrastructure.

## Features

- **Zero-backend architecture** — all vector operations run client-side via a 42 KB WASM microkernel
- **HNSW graph search** — nearest-neighbor lookups in sub-millisecond time
- **Progressive loading** — three-layer loading strategy (Cold/Warm/Hot) for fast time-to-interactive
- **Multiple distance metrics** — cosine, dot product, Euclidean
- **Offline-capable** — works without network after initial asset load via Cache API
- **Privacy-first** — no data leaves the browser

## Architecture

```
src/
├── index.ts                    # Public barrel export
├── wasm/
│   └── init.ts                 # WASM singleton initialization
├── loader/
│   ├── rvf-loader.ts           # Fetch RVF files (with Cache API support)
│   └── progressive-loader.ts   # Layer A/B/C progressive loading
└── search/
    ├── metrics.ts              # Distance metric enum + JS fallbacks
    ├── hnsw-engine.ts          # Low-level WASM store wrapper
    └── index.ts                # SearchEngine high-level API
```

### Progressive Loading Layers

| Layer | Name | What loads | Target time |
|-------|------|-----------|-------------|
| A | Cold | Header + metadata verification | Instant |
| B | Warm | Hot weights + adjacency graphs | ~500ms |
| C | Hot | Full data + checksum verification | Variable |

Layer B is the default — it makes search functional immediately without waiting for full verification.

### WASM API

The project uses `@ruvector/rvf-wasm` (v0.1.6), which exposes low-level C-ABI functions for:

- **Memory management** — `rvf_alloc`, `rvf_free`
- **In-memory vector store** — `rvf_store_create`, `rvf_store_ingest`, `rvf_store_query`, `rvf_store_close`
- **HNSW navigation** — `rvf_load_neighbors`, `rvf_greedy_step`
- **Quantization** — `rvf_dequant_i8`, `rvf_pq_distances`
- **Segment parsing & verification** — `rvf_verify_header`, `rvf_verify_checksum`, `rvf_crc32c`

Key convention: `rvf_store_ingest` returns the **count of ingested vectors** (not an error code). Negative values indicate errors.

## Quick Start

### Install

```bash
npm install
```

### Development

```bash
npm run dev
```

Opens the vanilla demo at `http://localhost:5173/`. The demo:
1. Loads the WASM runtime
2. Creates an in-memory store with 1000 random 128-dimensional normalized vectors
3. Provides a live search box that generates a deterministic query vector from text input
4. Returns top-10 nearest neighbors with cosine distance scores

### Build

```bash
npm run build
```

Compiles TypeScript to `dist/` with declarations and source maps.

### Test

```bash
# Unit tests (vitest)
npm test

# E2E tests (playwright)
npx playwright test --config config/playwright.config.ts
```

### Lint

```bash
npm run lint
```

## API Usage

### High-Level: SearchEngine

```typescript
import { SearchEngine } from 'wasmllm';

const engine = new SearchEngine();

// Initialize with an RVF file URL
const loadResult = await engine.init({
  rvfUrl: '/data/embeddings.rvf',
  layer: 'B',        // 'A' | 'B' | 'C' (default: 'B')
  cacheName: 'my-cache', // Cache API bucket name
});

console.log(loadResult.dimension);      // e.g. 128
console.log(loadResult.vectorCount);    // e.g. 50000
console.log(loadResult.totalLoadTimeMs); // e.g. 342.5

// Search with a pre-computed embedding vector
const results = engine.search(queryVector, {
  k: 10,              // top-k results (default: 10)
  metric: 'cosine',   // 'cosine' | 'dot' | 'euclidean'
});

for (const { id, score } of results) {
  console.log(`ID: ${id}, Distance: ${score}`);
}

engine.close();
```

### Low-Level: Direct WASM

```typescript
import { initWasm, getWasm } from 'wasmllm';

const wasm = await initWasm();

const handle = wasm.rvf_store_create(128, 0); // 128 dims, cosine metric

// Allocate and fill vectors + ids in WASM memory
const vecsPtr = wasm.rvf_alloc(numVectors * 128 * 4);
const idsPtr = wasm.rvf_alloc(numVectors * 8);
// ... fill Float32Array and BigInt64Array views ...

const ingested = wasm.rvf_store_ingest(handle, vecsPtr, idsPtr, numVectors);

// Query
const outPtr = wasm.rvf_alloc(k * 12); // 8 bytes id + 4 bytes score per result
const count = wasm.rvf_store_query(handle, queryPtr, k, 0, outPtr);

// Read results from DataView at outPtr
// Each result: getBigUint64(offset, true) for id, getFloat32(offset+8, true) for score

wasm.rvf_store_close(handle);
```

## Experiments

All demos run at `http://localhost:5173/` after `npm run dev`. Each is self-contained in `examples/`.

| Demo | Path | What it does | Key tech |
|------|------|-------------|----------|
| **Vector Search** | `/vanilla/` | Sub-ms nearest-neighbor search over 1,000 vectors | HNSW, RVF WASM (42 KB) |
| **Emoji Finder** | `/emoji-finder/` | Semantic emoji search by meaning | Feature hashing, cosine similarity |
| **Dansk NER** | `/ner-extractor/` | Danish named entity recognition in-browser | ModernBERT (144 MB), ONNX, Transformers.js |
| **Dansk Filmsogning** | `/semantic-search/` | Semantic search over Danish films | gte-small (33 MB), ONNX |
| **RuvLLM WASM** | `/ruvllm-wasm/` | LLM chat templates, HNSW routing, MicroLoRA | WASM (150 KB) |
| **Rock Paper Scissors** | `/rps-sona/` | Two AI predictors compete to read your patterns | N-gram + SONA WASM (227 KB) |

### Rock Paper Scissors — N-gram vs SONA

The RPS experiment pits two fundamentally different prediction approaches against each other:

- **N-gram frequency predictor** tracks what you play after specific move sequences (window sizes 1-5). It is fast, transparent, and very accurate on repetitive play.
- **SONA adaptive learner** encodes game states as 15-dimensional vectors, stores patterns via a WASM EphemeralAgent, and predicts using k-nearest-neighbor search plus a learned LoRA weight matrix. It has higher coverage and generalizes better to varied play.

Both predict independently on every round. Live charts and per-model metrics (accuracy, coverage, correct/tried) let you compare how they learn differently. See [`examples/rps-sona/RPS_ALGO_EXPLAINER.md`](../examples/rps-sona/RPS_ALGO_EXPLAINER.md) for a detailed comparison.

## Project Structure

```
wasmllm/
├── config/                  # Build, test, and dev server configuration
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── vitest.config.ts
│   └── playwright.config.ts
├── src/                     # TypeScript source
├── tests/
│   ├── wasm/                # WASM initialization tests
│   ├── loader/              # RVF fetch/load tests
│   ├── search/              # Distance metric tests
│   └── e2e/                 # Playwright browser tests
├── examples/
│   ├── vanilla/             # Vector search demo
│   ├── emoji-finder/        # Semantic emoji search
│   ├── ner-extractor/       # Danish NER
│   ├── semantic-search/     # Danish film search
│   ├── ruvllm-wasm/         # LLM toolkit demo
│   └── rps-sona/            # Rock Paper Scissors (N-gram vs SONA)
├── dist/                    # Compiled output
└── docs/                    # Documentation
```

## Performance

Measured in Chromium via Playwright e2e tests:

| Metric | Value |
|--------|-------|
| WASM + store init + 1000 vector ingest | < 50ms |
| Top-10 nearest neighbor search (128-dim, 1000 vectors) | < 1ms |

## Dependencies

| Package | Purpose | Size |
|---------|---------|------|
| `@ruvector/rvf-wasm` | WASM microkernel for vector operations | 42 KB (.wasm) |

Dev dependencies: TypeScript, Vite, Vitest, Playwright.
