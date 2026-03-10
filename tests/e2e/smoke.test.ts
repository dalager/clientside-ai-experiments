import { test, expect } from '@playwright/test';

test('vanilla demo loads and initializes WASM', async ({ page }) => {
  const errors: string[] = [];
  const logs: string[] = [];

  page.on('console', (msg) => {
    const text = msg.text();
    logs.push(`[${msg.type()}] ${text}`);
    if (msg.type() === 'error') {
      errors.push(text);
    }
  });

  page.on('pageerror', (err) => {
    errors.push(`PAGE ERROR: ${err.message}`);
  });

  await page.goto('/vanilla/', { waitUntil: 'networkidle' });

  await page.waitForFunction(
    () => {
      const status = document.getElementById('status');
      return status && !status.textContent?.includes('Initializing');
    },
    { timeout: 15000 },
  );

  const statusText = await page.textContent('#status');
  const vecCount = await page.textContent('#vec-count');
  const dimCount = await page.textContent('#dim-count');

  console.log(`Status: ${statusText}`);
  console.log(`Vectors: ${vecCount}, Dimensions: ${dimCount}`);

  expect(errors, `Console errors: ${errors.join('; ')}`).toHaveLength(0);
  expect(statusText).toContain('Ready');
  expect(vecCount).toBe('1000');
  expect(dimCount).toBe('128');
});

test('vanilla demo search returns results', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (err) => errors.push(err.message));
  page.on('console', (msg) => {
    if (msg.type() === 'error') errors.push(msg.text());
  });

  await page.goto('/vanilla/', { waitUntil: 'networkidle' });

  await page.waitForFunction(
    () => {
      const status = document.getElementById('status');
      return status && status.textContent?.includes('Ready');
    },
    { timeout: 15000 },
  );

  await page.fill('#search-box', 'hello world');
  await page.waitForSelector('.result', { timeout: 5000 });

  const resultCount = await page.locator('.result').count();
  const statusText = await page.textContent('#status');

  console.log(`Search returned ${resultCount} results in ${statusText}`);

  expect(errors).toHaveLength(0);
  expect(resultCount).toBeGreaterThan(0);
  expect(resultCount).toBeLessThanOrEqual(10);
  expect(statusText).toContain('results in');
});
