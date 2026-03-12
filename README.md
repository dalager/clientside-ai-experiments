# Client-side AI Experiments 

Client-side AI playground — browser-native ML demos powered by WebAssembly. No servers, no API keys, no data leaves your device.

## Demos

| Demo | Description | Tech |
|------|-------------|------|
| [Vector Search](examples/vanilla/) | Sub-ms nearest-neighbor search over 1,000 vectors using HNSW graph traversal in a 42 KB WASM microkernel | @ruvector/rvf-wasm |
| [Emoji Finder](examples/emoji-finder/) | Semantic emoji search — type a feeling or concept and find matching emojis by meaning, not keywords | Feature hashing + WASM HNSW |
| [Semantic Search](examples/semantic-search/) | Semantic movie search in Danish — type a description and find matching films by meaning using in-browser embeddings | Transformers.js (gte-small) |
| [Dansk NER](examples/ner-extractor/) | Find persons, organizations and locations in Danish text using a ModernBERT model (~144 MB) running entirely in-browser | Transformers.js + ONNX Runtime |


## Quick start

```bash
npm install
npm run dev       # Vite dev server on http://localhost:5173
npm test          # Unit tests (vitest)
npm run build     # TypeScript lib → dist/
```

## Project structure

```
src/              TypeScript library (WASM init, loaders, HNSW engine, metrics)
examples/         Browser demos (vanilla JS, each self-contained)
tests/            Unit tests + Playwright e2e tests
config/           Vite, Vitest, Playwright, TypeScript configs
plans/            Session notes and plans
```

## Future ideas

Things to try next — all running client-side in the browser:
### Image processing client-side

Ruvector CNN samples:
https://github.com/ruvnet/RuVector/tree/main/docs/cnn

WASM and working. Extract other simple usecases.


### NLP & text

- **Sentiment analysis** — classify Danish/English text as positive/negative/neutral using a small ONNX model
- **Text summarization** — extractive summarization of articles using sentence embeddings + MMR selection
- **Spam / toxicity filter** — lightweight classifier for comment moderation, no server round-trip
- **Keyword extraction** — TF-IDF or RAKE running in WASM for instant keyword highlighting
- **Language detection** — identify language from a text snippet using character n-gram vectors

### Search & retrieval

- **Document search** — embed paragraphs with a small model (MiniLM), index with RVF HNSW, search by meaning
- **FAQ matcher** — semantic match user questions to a knowledge base of answers
- **Image search by caption** — CLIP-style text→image retrieval using quantized ONNX vision encoder
- **Code search** — embed code snippets and search by natural language description

### Classification & tagging

- **Image classification** — MobileNet or EfficientNet-Lite in ONNX for real-time object recognition from webcam
- **Document categorization** — auto-tag uploaded documents by topic using zero-shot classification
- **Auto-tagging for photos** — detect objects/scenes in browser using small vision models

### Generative / interactive

- **Text completion** — small GPT-2 or Phi-3-mini running in ONNX for local text generation
- **Translation** — Danish↔English using Helsinki-NLP OPUS models via Transformers.js
- **Speech-to-text** — Whisper tiny/base in ONNX for in-browser transcription from microphone
- **Text-to-speech** — client-side TTS using VITS or similar small vocoder models

### Tooling & infrastructure

- **WASM vector DB benchmark** — compare RVF HNSW against brute-force and other ANN approaches at various scales
- **Embedding cache** — persist computed embeddings in IndexedDB for instant reload
- **Model size explorer** — interactive comparison of model size vs. accuracy tradeoffs for browser deployment
- **Progressive model loading** — stream large models in chunks, become interactive before full download
- **WebGPU acceleration** — use WebGPU for matrix multiplication when available, WASM fallback otherwise
