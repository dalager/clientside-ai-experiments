import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: '../tests/e2e',
  timeout: 30000,
  use: {
    baseURL: 'http://localhost:4174',
    headless: true,
  },
  projects: [
    { name: 'chromium', use: { browserName: 'chromium' } },
  ],
  webServer: {
    command: 'npx vite --config config/vite.config.ts --port 4174',
    port: 4174,
    reuseExistingServer: true,
  },
});
