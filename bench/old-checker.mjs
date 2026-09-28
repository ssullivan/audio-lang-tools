#!/usr/bin/env node
/*
 * Scores learn-cantonese's first audio-check.mjs (commit c163428: Azure
 * pronunciation-assessment timings, a hand-written pitch tracker and
 * hand-tuned tone rules) on this benchmark, as the baseline.
 *
 *   uv run altools bench items > items.json
 *   node bench/old-checker.mjs <learn-cantonese checkout at c163428> items.json > old.json
 *   uv run altools bench score --results old.json --name old --no-fail
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const [site, itemsFile] = process.argv.slice(2);
const { checkClip } = await import(pathToFileURL(join(site, 'tools/audio-check.mjs')));
const items = JSON.parse(readFileSync(itemsFile, 'utf8'));
const out = [];
for (const it of items) {
  const r = await checkClip('unit1', { id: it.id, hanzi: it.text, jyutping: it.jyutping }, it.voice, it.path);
  // Its tone messages start "tone shape"; score() counts those as tone flags.
  out.push({ id: it.id, problems: r.problems, notes: r.notes });
  process.stderr.write(`\r${out.length}/${items.length}`);
}
process.stderr.write('\n');
console.log(JSON.stringify(out));
