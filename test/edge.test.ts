import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { builtinModules } from 'node:module';
import { dirname, join, resolve } from 'node:path';
import { test } from 'node:test';

test('the package imports no Node built-ins, so it runs on edge and browser runtimes', () => {
  const seen = new Map<string, string[]>();
  const visit = (file: string) => {
    if (seen.has(file)) return;
    const src = readFileSync(file, 'utf8');
    const specs = [...src.matchAll(/(?:^|\n)\s*(?:import|export)\s[^;]*?from\s*'([^']+)'|import\(\s*'([^']+)'\s*\)/g)].map((m) => m[1] ?? m[2]);
    seen.set(file, specs);
    for (const s of specs) if (s.startsWith('.')) visit(resolve(dirname(file), s));
  };
  visit(join(import.meta.dirname, '..', 'src', 'index.ts'));
  assert.ok(seen.size > 5, 'the walker found the source modules');
  const builtins = new Set(builtinModules);
  for (const [file, specs] of seen) for (const s of specs) assert.ok(!s.startsWith('node:') && !builtins.has(s.split('/')[0]), `${file} imports ${s}`);
});
