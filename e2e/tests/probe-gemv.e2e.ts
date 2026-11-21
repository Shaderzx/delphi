import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { expect, test } from 'e2e';

const repo = join(import.meta.dirname, '..', '..');
const build = join(repo, 'engine', 'build.sh');
const GOLDEN = 0x9e3779b9;
const INPUT_SALT = 0xa5a5a5a5;

type Gemv = {
  rows: number;
  n: number;
  k: number;
  first_row: number;
  row_hashes: string[];
  outputs?: number[][];
  bytes: number;
  seconds_median: number;
  gb_per_s_median: number;
};

type Chain = { layers: number; n: number; rows: number; mode: string; bytes_per_step: number; output_hash: string; seconds_median: number; gb_per_s_median: number };

function withEngine<T>(use: (delphi: (args: string[]) => string) => T): T {
  const out = mkdtempSync(join(tmpdir(), 'delphi-build-'));
  try {
    const built = spawnSync('bash', [build, out], { encoding: 'utf8', timeout: 300_000 });
    if (built.status !== 0) throw new Error(`engine build failed: ${built.stderr.slice(-3000)}`);
    return use((args) => {
      const run = spawnSync(join(out, 'delphi'), ['probe', ...args], { encoding: 'utf8', timeout: 300_000 });
      if (run.status !== 0) throw new Error(`delphi probe ${args.join(' ')} exited ${run.status}: ${run.stderr}`);
      return run.stdout;
    });
  } finally {
    rmSync(out, { recursive: true, force: true });
  }
}

function lowbias32(value: number): number {
  let a = value >>> 0;
  a ^= a >>> 16;
  a = Math.imul(a, 0x7feb352d) >>> 0;
  a ^= a >>> 15;
  a = Math.imul(a, 0x846ca68b) >>> 0;
  a ^= a >>> 16;
  return a >>> 0;
}

function gridValue(seed: number, index: number): number {
  return (lowbias32((Math.imul(seed, GOLDEN) + index) >>> 0) >>> 24) - 128;
}

function exactOutputs(seed: number, rows: number, firstRow: number, n: number, k: number): number[][] {
  const inputSeed = (seed ^ INPUT_SALT) >>> 0;
  return Array.from({ length: rows }, (_, r) =>
    Array.from({ length: n }, (_, i) => {
      let sum = 0;
      for (let j = 0; j < k; j++) sum += gridValue(seed, i * k + j) * gridValue(inputSeed, (firstRow + r) * k + j);
      return sum / 16384;
    }),
  );
}

test('probe gemv returns the exact dot product of every weight row with every input row at verify widths 1 to 32', () => {
  const seed = 11;
  const n = 48;
  const k = 1024;
  withEngine((delphi) => {
    for (const rows of [1, 2, 5, 8, 32]) {
      const result = JSON.parse(delphi(['gemv', '--rows', String(rows), '--n', String(n), '--k', String(k), '--seed', String(seed), '--values', 'exact', '--outputs'])) as Gemv;
      expect(result.rows).toBe(rows);
      expect(result.bytes).toBe(n * k * 2);
      expect(result.outputs).toEqual(exactOutputs(seed, rows, 0, n, k));
    }
  });
});

test('probe gemv gives each input row the same output bits at every verify width from 1 to 32, as when it runs alone', () => {
  const shape = ['--n', '2304', '--k', '2304', '--seed', '5', '--values', 'rough'];
  withEngine((delphi) => {
    const alone = new Map<number, string>();
    for (const row of [0, 1, 7, 31]) {
      const result = JSON.parse(delphi(['gemv', '--rows', '1', '--first-row', String(row), ...shape])) as Gemv;
      alone.set(row, result.row_hashes[0]);
    }
    expect(new Set(alone.values()).size).toBe(alone.size);
    for (let rows = 1; rows <= 32; rows++) {
      const result = JSON.parse(delphi(['gemv', '--rows', String(rows), ...shape])) as Gemv;
      expect(result.row_hashes.length).toBe(rows);
      for (const [row, hash] of alone) {
        if (row < rows) expect(result.row_hashes[row]).toBe(hash);
      }
    }
  });
});

test('probe chain runs 192 dependent GEMVs in one GPU submission and ends on the same bits as one submission per layer', () => {
  const layers = 192;
  const n = 2304;
  withEngine((delphi) => {
    const args = ['chain', '--layers', String(layers), '--n', String(n), '--seed', '3', '--repeats', '6'];
    const chained = JSON.parse(delphi([...args, '--mode', 'chained'])) as Chain;
    const stepwise = JSON.parse(delphi([...args, '--mode', 'stepwise'])) as Chain;
    expect(chained.mode).toBe('chained');
    expect(stepwise.mode).toBe('stepwise');
    expect(chained.bytes_per_step).toBe(layers * n * n * 2);
    expect(chained.output_hash).toBe(stepwise.output_hash);
    expect(chained.seconds_median).toBeGreaterThan(0);
    expect(chained.gb_per_s_median).toBeCloseTo(chained.bytes_per_step / chained.seconds_median / 1e9, 3);
  });
});
test('probe chain carries 8 verify rows through 192 layers on one read of each weight and ends on the same bits as one submission per layer', () => {
  const layers = 192;
  const n = 2304;
  withEngine((delphi) => {
    const args = ['chain', '--layers', String(layers), '--n', String(n), '--rows', '8', '--seed', '3', '--repeats', '6'];
    const chained = JSON.parse(delphi([...args, '--mode', 'chained'])) as Chain;
    const stepwise = JSON.parse(delphi([...args, '--mode', 'stepwise'])) as Chain;
    const single = JSON.parse(delphi(['chain', '--layers', String(layers), '--n', String(n), '--seed', '3', '--repeats', '6'])) as Chain;
    expect(chained.rows).toBe(8);
    expect(single.rows).toBe(1);
    expect(chained.bytes_per_step).toBe(layers * n * n * 2);
    expect(chained.output_hash).toBe(stepwise.output_hash);
    expect(chained.output_hash).not.toBe(single.output_hash);
  });
});
