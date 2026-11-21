import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { expect, test } from 'e2e';

const repo = join(import.meta.dirname, '..', '..');
const build = join(repo, 'engine', 'build.sh');
const PATTERN = 2654435761n;

type Probe = {
  device: string;
  bytes: number;
  repeats: number;
  seconds_best: number;
  gb_per_s_best: number;
  seconds_median: number;
  gb_per_s_median: number;
  checksum: string;
};

type Run = { status: number | null; stdout: string; stderr: string };

function expectedChecksum(bytes: number): string {
  const n = BigInt(bytes / 4);
  const sum = (PATTERN * ((n * (n - 1n)) / 2n)) % 2n ** 32n;
  return '0x' + sum.toString(16).padStart(8, '0');
}

function probe(args: string[]): Run {
  const out = mkdtempSync(join(tmpdir(), 'delphi-build-'));
  try {
    const built = spawnSync('bash', [build, out], { encoding: 'utf8', timeout: 300_000 });
    if (built.status !== 0) throw new Error(`engine build failed: ${built.stderr.slice(-3000)}`);
    return spawnSync(join(out, 'delphi'), ['probe', 'bandwidth', ...args], { encoding: 'utf8', timeout: 300_000 });
  } finally {
    rmSync(out, { recursive: true, force: true });
  }
}

function expectRefused(run: Run, flag: string): void {
  expect(run.status).toBe(2);
  expect(run.stderr).toContain(flag);
  expect(run.stdout).toBe('');
}

test('probe bandwidth streams every byte of the buffer on the GPU and reports best and median rates', () => {
  const mib = 256;
  const run = probe(['--mib', String(mib), '--repeats', '6']);
  expect(run.stderr).toBe('');
  expect(run.status).toBe(0);
  const result = JSON.parse(run.stdout) as Probe;
  expect(result.bytes).toBe(mib * 1024 * 1024);
  expect(result.repeats).toBe(6);
  expect(result.checksum).toBe(expectedChecksum(result.bytes));
  expect(result.device.length).toBeGreaterThan(0);
  expect(result.seconds_best).toBeGreaterThan(0);
  expect(result.seconds_median).toBeGreaterThanOrEqual(result.seconds_best);
  expect(result.gb_per_s_best).toBeCloseTo(result.bytes / result.seconds_best / 1e9, 3);
  expect(result.gb_per_s_median).toBeCloseTo(result.bytes / result.seconds_median / 1e9, 3);
});

test('probe bandwidth refuses a buffer size that is not a whole number of MiB above zero', () => {
  expectRefused(probe(['--mib', '0']), '--mib');
});

test('probe bandwidth refuses a buffer size whose byte count would overflow', () => {
  expectRefused(probe(['--mib', '17592186044417']), '--mib');
});

test('probe bandwidth refuses a repeat count larger than the engine can count', () => {
  expectRefused(probe(['--repeats', '4294967296']), '--repeats');
});