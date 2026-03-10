import { test, expect } from '@playwright/test';

test('NER extractor loads model and extracts Danish entities', async ({ page }) => {
  test.setTimeout(120_000); // Model download can take a while

  const errors: string[] = [];
  page.on('pageerror', (err) => errors.push(err.message));
  page.on('console', (msg) => {
    if (msg.type() === 'error') errors.push(msg.text());
    if (msg.type() === 'log') console.log(`[page] ${msg.text()}`);
  });

  await page.goto('/ner-extractor/', { waitUntil: 'networkidle' });

  // Wait for model to finish loading (button becomes enabled)
  await page.waitForFunction(
    () => !(document.getElementById('extract-btn') as HTMLButtonElement)?.disabled,
    { timeout: 90_000 },
  );

  const status = await page.textContent('#status');
  console.log(`Model status: ${status}`);
  expect(status).toContain('klar');

  // Click example 1: Jens Peter Hansen, København, Novo Nordisk, Mette Frederiksen, Christiansborg
  await page.click('.btn-example');

  // Wait for entities to appear
  await page.waitForSelector('.entity', { timeout: 15_000 });

  const entities = await page.locator('.entity-row').allTextContents();
  console.log('Extracted entities:', entities);

  // Should find at least some entities
  expect(entities.length).toBeGreaterThan(0);

  // Check that annotated text has highlighted spans
  const annotatedHtml = await page.innerHTML('#annotated');
  expect(annotatedHtml).toContain('entity-');

  // Check for known entities (at least some should be found)
  const allText = entities.join(' ');
  const foundPerson = allText.includes('PER');
  const foundOrg = allText.includes('ORG');
  const foundLoc = allText.includes('LOC');

  console.log(`Found PER: ${foundPerson}, ORG: ${foundOrg}, LOC: ${foundLoc}`);
  // At least person or location should be detected
  expect(foundPerson || foundLoc).toBe(true);

  expect(errors).toHaveLength(0);
});
