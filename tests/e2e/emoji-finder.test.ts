import { test, expect } from '@playwright/test';

test('emoji finder initializes and searches', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (err) => errors.push(err.message));
  page.on('console', (msg) => {
    if (msg.type() === 'error') errors.push(msg.text());
  });

  await page.goto('/emoji-finder/', { waitUntil: 'networkidle' });

  await page.waitForFunction(
    () => !(document.getElementById('search-box') as HTMLInputElement)?.disabled,
    { timeout: 15000 },
  );

  const emojiCount = await page.textContent('#emoji-count');
  expect(Number(emojiCount)).toBeGreaterThan(100);

  const chips = await page.locator('.chip').count();
  expect(chips).toBeGreaterThan(5);

  // Search for "happy"
  await page.fill('#search-box', 'happy');
  await page.waitForSelector('.result', { timeout: 3000 });

  const results = await page.locator('.result').count();
  expect(results).toBeGreaterThan(0);
  expect(results).toBeLessThanOrEqual(20);

  const firstEmoji = await page.locator('.result .emoji').first().textContent();
  console.log(`"happy" → ${firstEmoji}`);

  // Search for "love"
  await page.fill('#search-box', 'love');
  await page.waitForTimeout(200);
  const loveEmojis = await page.locator('.result .emoji').allTextContents();
  console.log(`"love" → ${loveEmojis.slice(0, 5).join(' ')}`);
  expect(loveEmojis.slice(0, 3)).toContain('❤️');

  // Search for "food"
  await page.fill('#search-box', 'food');
  await page.waitForTimeout(200);
  const foodFirst = await page.locator('.result .emoji').first().textContent();
  console.log(`"food" → ${foodFirst}`);

  // Status shows timing
  const status = await page.textContent('#status');
  expect(status).toContain('ms');

  expect(errors, `Errors: ${errors.join('; ')}`).toHaveLength(0);
});
