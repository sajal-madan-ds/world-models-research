import { chromium } from '@playwright/test';
import { createServer } from 'node:http';
import { readFile, stat, mkdir } from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';

const root = path.resolve('site');
const prefix = '/world-models-research/';
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.json': 'application/json', '.png': 'image/png', '.woff2': 'font/woff2' };
const server = createServer(async (request, response) => {
  try {
    const pathname = new URL(request.url, 'http://localhost').pathname;
    if (!pathname.startsWith(prefix)) { response.writeHead(404).end(); return; }
    let file = path.resolve(root, '.' + pathname.slice(prefix.length - 1));
    if (!file.startsWith(root + path.sep) && file !== root) throw new Error('Invalid path');
    if ((await stat(file)).isDirectory()) file = path.join(file, 'index.html');
    response.setHeader('Content-Type', mime[path.extname(file)] || 'application/octet-stream');
    response.end(await readFile(file));
  } catch { response.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const base = `http://127.0.0.1:${server.address().port}${prefix}`;
let browser;
try {
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(base);
  await page.waitForSelector('[data-course-progress]');
  const chapterLinks = await page.locator('a[href*="docs/"]').evaluateAll(links => [...new Set(links.map(a => a.href))].filter(url => /\/docs\/\d{2}_/.test(url)));
  assert.equal(chapterLinks.length, 12);
  let equations = 0, diagrams = 0;
  for (const url of chapterLinks) {
    await page.goto(url);
    await page.waitForSelector('[data-lesson] button');
    await page.waitForFunction(() => [...document.querySelectorAll('.course-diagram')].every(element => element.querySelector('svg') || element.classList.contains('diagram-fallback')));
    assert.equal(await page.locator('.diagram-fallback').count(), 0, `Diagram failed: ${url}`);
    assert.equal(await page.locator('.katex-error').count(), 0, `Math failed: ${url}`);
    equations += await page.locator('.katex').count();
    diagrams += await page.locator('.course-diagram svg').count();
  }
  assert.ok(equations > 20);
  assert.ok(diagrams > 3);
  await page.goto(chapterLinks[0]);
  await page.locator('[data-lesson] button').click();
  assert.equal(await page.locator('[data-lesson] button').getAttribute('aria-pressed'), 'true');
  await page.reload();
  await page.waitForFunction(() => document.querySelector('[data-lesson] button')?.getAttribute('aria-pressed') === 'true');
  await page.goto(base);
  await page.waitForFunction(() => document.querySelector('[data-course-progress]').value === 1);
  await page.locator('label[for="__search"]').first().click();
  await page.locator('input[data-md-component="search-query"]').fill('expectile');
  await page.waitForSelector('.md-search-result__item');
  await page.keyboard.press('Escape');
  await mkdir('test-results', { recursive: true });
  await page.screenshot({ path: 'test-results/home-desktop.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(base);
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1), 'Homepage overflows on mobile');
  await page.screenshot({ path: 'test-results/home-mobile.png', fullPage: true });
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ chapters: chapterLinks.length, equations, diagrams, progress: 'passed', search: 'passed', mobile: 'passed', runtimeErrors: errors.length }));
} finally {
  await browser?.close();
  server.close();
}
