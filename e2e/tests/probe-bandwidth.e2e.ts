import { spawnSync } from 'node:child_process';
import { join } from 'node:path';
import { expect, test } from 'e2e';

const repo = join(import.meta.dirname, '..', '..');
const build = join(repo, 'engine', 'build.sh');
const binary = join(repo, 'engine', 'build', 'delphi');
const PATTERN = 2654435761n;

type Probe = { device: string; bytes: number; repeats: number; seconds_best: number; gb_per_s_best: number; checksum: string };

function expectedChecksum(bytes: number): string {
  const n = BigInt(bytes / 4);
  const sum = (PATTERN * ((n * (n - 1n)) / 2n)) % 2n ** 32n;
  return '0x' + sum.toString(16).padStart(8, '0');
}

function probe(args: string[]): { status: number | null; stdout: string; stderr: string } {
  const built = spawnSync('bash', [build], { encoding: 'utf8', timeout: 300_000 });
  if (built.status !== 0) throw new Error(`engine build failed: ${built.stderr.slice(-3000)}`);
  return spawnSync(binary, ['probe', 'bandwidth', ...args], { encoding: 'utf8', timeout: 300_000 });
}

test('probe bandwidth streams every byte of the buffer on the GPU and reports a positive rate', () => {
  const mib = 256;
  const run = probe(['--mib', String(mib), '--repeats', '3']);
  expect(run.stderr).toBe('');
  expect(run.status).toBe(0);
  const result = JSON.parse(run.stdout) as Probe;
  expect(result.bytes).toBe(mib * 1024 * 1024);
  expect(result.repeats).toBe(3);
  expect(result.checksum).toBe(expectedChecksum(result.bytes));
  expect(result.device.length).toBeGreaterThan(0);
  expect(result.seconds_best).toBeGreaterThan(0);
  expect(result.gb_per_s_best).toBeCloseTo(result.bytes / result.seconds_best / 1e9, 3);
});

test('probe bandwidth refuses a buffer size that is not a whole number of MiB above zero', () => {
  const run = probe(['--mib', '0']);
  expect(run.status).toBe(2);
  expect(run.stderr).toContain('--mib');
  expect(run.stdout).toBe('');
});