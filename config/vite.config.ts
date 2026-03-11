import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  root: resolve(__dirname, '../examples'),
  build: {
    outDir: resolve(__dirname, '../dist'),
    emptyOutDir: true,
    rollupOptions: {
      input: {
        vanilla: resolve(__dirname, '../examples/vanilla/index.html'),
        emoji: resolve(__dirname, '../examples/emoji-finder/index.html'),
        ner: resolve(__dirname, '../examples/ner-extractor/index.html'),
        semanticSearch: resolve(__dirname, '../examples/semantic-search/index.html'),
      },
    },
  },
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'credentialless',
    },
  },
  optimizeDeps: {
    exclude: ['@ruvector/rvf-wasm'],
  },
});
