import { spawnSync, type SpawnSyncReturns } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { expect, test } from 'e2e';

const repo = join(import.meta.dirname, '..', '..');
const model = process.env.DELPHI_MODEL ?? join(homedir(), 'models', 'Inkling-2-Mini');
const CLEANUP_RESERVE_MS = 30_000;

type Case = { text: string; input_ids: number[]; output_ids: number[] };
type Fanout = { branches: { output_ids: number[] }[]; branch_count: number; generated_tokens: number; decode_tokens_per_s: number; mode: string; lossy: boolean; bitwise: boolean };
type Generated = { output_ids: number[] };

const truth = JSON.parse(readFileSync(join(import.meta.dirname, '..', 'fixtures', 'g4-sglang-greedy.json'), 'utf8')) as { cases: Case[] };
const story = truth.cases.find((c) => c.text.startsWith('Once upon a time'))!;
const fibonacci = truth.cases.find((c) => c.text.startsWith('def fibonacci'))!;
const prefix = [...story.input_ids, ...fibonacci.input_ids];
const fixtureTokens = [...new Set(truth.cases.flatMap((c) => c.input_ids))].slice(0, 32);
const branchHeads = [...fixtureTokens.map((id) => [id]), ...fixtureTokens.map((id, i) => [id, fixtureTokens[(i + 1) % fixtureTokens.length]])];

function remainingMs(deadline: number): number {
  const left = deadline - Date.now() - CLEANUP_RESERVE_MS;
  if (left <= 0) throw new Error('test deadline exhausted before subprocess launch');
  return left;
}

function checkRun(label: string, run: SpawnSyncReturns<string>): void {
  if (run.error) throw new Error(`${label} failed to launch: ${run.error.message}`);
  if (run.status !== 0) throw new Error(`${label} exited ${run.status}: ${(run.stderr ?? '').slice(-3000)}`);
}

function withBuiltDelphi<T>(deadline: number, fn: (bin: string) => T): T {
  const out = mkdtempSync(join(tmpdir(), 'delphi-build-'));
  try {
    checkRun('engine build', spawnSync('bash', [join(repo, 'engine', 'build.sh'), out], { encoding: 'utf8', timeout: remainingMs(deadline) }));
    return fn(join(out, 'delphi'));
  } finally {
    rmSync(out, { recursive: true, force: true });
  }
}

function fanout(bin: string, branches: number[][], steps: number, deadline: number, extra: string[] = []): Fanout {
  const run = spawnSync(bin, ['fanout', '--model', model, '--prefix-ids', JSON.stringify(prefix), '--branch-ids', JSON.stringify(branches), '--steps', String(steps), ...extra], { encoding: 'utf8', timeout: remainingMs(deadline) });
  checkRun('delphi fanout', run);
  return JSON.parse(run.stdout) as Fanout;
}

test('delphi fanout refuses missing, empty or malformed branch lists', { timeout: 300_000 }, () => {
  const deadline = Date.now() + 300_000;
  withBuiltDelphi(deadline, (bin) => {
    const tooMany = JSON.stringify(Array.from({ length: 65 }, () => [1]));
    for (const bad of ['[]', '[[]]', '[1, 2]', '[[1], [-3]]', 'not-json', tooMany]) {
      const run = spawnSync(bin, ['fanout', '--model', '/nonexistent', '--prefix-ids', '[1]', '--branch-ids', bad], { encoding: 'utf8', timeout: remainingMs(deadline) });
      if (run.error) throw run.error;
      expect(run.status).toBe(2);
      expect(run.stderr).toContain('--branch-ids');
    }
    const missing = spawnSync(bin, ['fanout', '--model', '/nonexistent', '--prefix-ids', '[1]'], { encoding: 'utf8', timeout: remainingMs(deadline) });
    if (missing.error) throw missing.error;
    expect(missing.status).toBe(2);
    const okShape = spawnSync(bin, ['fanout', '--model', '/nonexistent', '--prefix-ids', '[1]', '--branch-ids', '[[2], [3, 4]]'], { encoding: 'utf8', timeout: remainingMs(deadline) });
    if (okShape.error) throw okShape.error;
    expect(okShape.status).toBe(3);
  });
});

test('delphi fanout --bitwise decodes each of 64 branches token-for-token like decoding that branch alone', { tags: ['model'], timeout: 1_800_000 }, () => {
  const deadline = Date.now() + 1_800_000;
  const steps = 24;
  withBuiltDelphi(deadline, (bin) => {
    const out = fanout(bin, branchHeads, steps, deadline, ['--bitwise']);
    expect(out.bitwise).toBe(true);
    expect(out.branch_count).toBe(branchHeads.length);
    expect(out.mode).toBe('exact');
    expect(out.lossy).toBe(false);
    branchHeads.forEach((head, b) => {
      const alone = spawnSync(bin, ['generate', '--model', model, '--input-ids', JSON.stringify([...prefix, ...head]), '--steps', String(steps), '--spec', 'off'], { encoding: 'utf8', timeout: remainingMs(deadline) });
      checkRun(`delphi generate branch ${b}`, alone);
      expect(out.branches[b].output_ids).toEqual((JSON.parse(alone.stdout) as Generated).output_ids);
    });
  });
});

test('delphi fanout generates at least 200 tokens per second across 64 branches with Metal 4 expert kernels', { tags: ['model'], timeout: 900_000 }, () => {
  const deadline = Date.now() + 900_000;
  const steps = 128;
  withBuiltDelphi(deadline, (bin) => {
    const out = fanout(bin, branchHeads, steps, deadline);
    expect(out.branch_count).toBe(branchHeads.length);
    expect(out.generated_tokens).toBe(branchHeads.length * steps);
    expect(out.branches.every((b) => b.output_ids.length === steps)).toBe(true);
    expect(out.lossy).toBe(false);
    expect(out.bitwise).toBe(false);
    expect(out.decode_tokens_per_s).toBeGreaterThanOrEqual(200);
  });
});
test('delphi fanout without --bitwise keeps 16 branches over a 160-token prefix token-for-token like decoding each alone', { tags: ['model'], timeout: 1_200_000 }, () => {
  const deadline = Date.now() + 1_200_000;
  const steps = 24;
  const longPrefix = Array.from({ length: 8 }, () => fibonacci.input_ids).flat();
  const heads = branchHeads.slice(0, 16);
  withBuiltDelphi(deadline, (bin) => {
    const run = spawnSync(bin, ['fanout', '--model', model, '--prefix-ids', JSON.stringify(longPrefix), '--branch-ids', JSON.stringify(heads), '--steps', String(steps)], { encoding: 'utf8', timeout: remainingMs(deadline) });
    checkRun('delphi fanout', run);
    const out = JSON.parse(run.stdout) as Fanout;
    expect(out.bitwise).toBe(true);
    heads.forEach((head, b) => {
      const alone = spawnSync(bin, ['generate', '--model', model, '--input-ids', JSON.stringify([...longPrefix, ...head]), '--steps', String(steps), '--spec', 'off'], { encoding: 'utf8', timeout: remainingMs(deadline) });
      checkRun(`delphi generate branch ${b}`, alone);
      expect(out.branches[b].output_ids).toEqual((JSON.parse(alone.stdout) as Generated).output_ids);
    });
  });
});
