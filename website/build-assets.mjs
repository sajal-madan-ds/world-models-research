import { build } from 'esbuild';
import { cp, mkdir } from 'node:fs/promises';
await mkdir('built', { recursive: true });
await build({
  entryPoints: ['assets/runtime.js'],
  outdir: 'built',
  bundle: true,
  splitting: true,
  format: 'esm',
  minify: true,
  target: 'es2022',
  entryNames: '[name]',
  chunkNames: 'chunks/[name]-[hash]',
});
await cp('node_modules/katex/dist', 'built/katex', { recursive: true });
