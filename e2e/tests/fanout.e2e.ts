import { spawnSync, type SpawnSyncReturns } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { expect, test } from 'e2e';

const repo = join(import.meta.dirname, '..', '..');
const model = process.env.DELPHI_MODEL ?? join(homedir(), 'models', 'Inkling-2-Mini');
const CLEANUP_RESERVE_MS = 30_000;

type Case = { text: string; input_ids: number[]; output_ids: number[] };
type Fanout = {
  branches: { output_ids: number[] }[];
  branch_count: number;
  generated_tokens: number;
  lockstep_steps: number;
  decode_tokens_per_s: number;
  per_branch_tokens_per_s: number;
  ms_per_step: number;
  mode: string;
  lossy: boolean;
  bitwise: boolean;
};
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

function fanout(bin: string, prefixIds: number[], branches: number[][], steps: number, deadline: number, extra: string[] = []): Fanout {
  const run = spawnSync(bin, ['fanout', '--model', model, '--prefix-ids', JSON.stringify(prefixIds), '--branch-ids', JSON.stringify(branches), '--steps', String(steps), ...extra], { encoding: 'utf8', timeout: remainingMs(deadline) });
  checkRun('delphi fanout', run);
  return JSON.parse(run.stdout) as Fanout;
}

function expectEachBranchMatchesSolo(bin: string, prefixIds: number[], heads: number[][], out: Fanout, steps: number, deadline: number): void {
  heads.forEach((head, b) => {
    const alone = spawnSync(bin, ['generate', '--model', model, '--input-ids', JSON.stringify([...prefixIds, ...head]), '--steps', String(steps), '--spec', 'off'], { encoding: 'utf8', timeout: remainingMs(deadline) });
    checkRun(`delphi generate branch ${b}`, alone);
    expect(out.branches[b].output_ids).toEqual((JSON.parse(alone.stdout) as Generated).output_ids);
  });
}

test('delphi fanout refuses missing, empty, malformed or more than 64 branch lists', { timeout: 300_000 }, () => {
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
    const okShape = spawnSync(bin, ['fanout', '--model', '/nonexistent', '--prefix-ids', '[1]', '--branch-ids', JSON.stringify(Array.from({ length: 64 }, () => [2]))], { encoding: 'utf8', timeout: remainingMs(deadline) });
    if (okShape.error) throw okShape.error;
    expect(okShape.status).toBe(3);
  });
});

test('delphi fanout decodes each of 64 branches token-for-token like decoding that branch alone', { tags: ['model'], timeout: 1_800_000 }, () => {
  const deadline = Date.now() + 1_800_000;
  const steps = 24;
  withBuiltDelphi(deadline, (bin) => {
    const out = fanout(bin, prefix, branchHeads, steps, deadline);
    expect(out.branch_count).toBe(64);
    expect(out.mode).toBe('exact');
    expect(out.lossy).toBe(false);
    expect(out.bitwise).toBe(true);
    expectEachBranchMatchesSolo(bin, prefix, branchHeads, out, steps, deadline);
  });
});

test('delphi fanout matches solo decoding past the 1024-token local attention window with branch heads of 1 to 5 tokens', { tags: ['model'], timeout: 1_800_000 }, () => {
  const deadline = Date.now() + 1_800_000;
  const steps = 24;
  const longPrefix = Array.from({ length: 56 }, () => fibonacci.input_ids).flat();
  const heads = [1, 2, 3, 4, 5, 1, 3, 5].map((len, b) => Array.from({ length: len }, (_, j) => fixtureTokens[(b * 5 + j) % fixtureTokens.length]));
  withBuiltDelphi(deadline, (bin) => {
    expect(longPrefix.length).toBeGreaterThan(1024);
    const out = fanout(bin, longPrefix, heads, steps, deadline);
    expect(out.bitwise).toBe(true);
    expect(out.lockstep_steps).toBe(5 - 1 + steps);
    expectEachBranchMatchesSolo(bin, longPrefix, heads, out, steps, deadline);
  });
});

test('delphi fanout --tensor-experts generates at least 200 tokens per second across 64 branches timed from outside and reports itself lossy', { tags: ['model'], timeout: 900_000 }, () => {
  const deadline = Date.now() + 900_000;
  const steps = 256;
  withBuiltDelphi(deadline, (bin) => {
    const started = process.hrtime.bigint();
    const out = fanout(bin, prefix, branchHeads, steps, deadline, ['--tensor-experts']);
    const elapsedS = Number(process.hrtime.bigint() - started) / 1e9;
    expect(out.branch_count).toBe(64);
    expect(out.generated_tokens).toBe(64 * steps);
    expect(out.branches.every((b) => b.output_ids.length === steps)).toBe(true);
    expect(out.lossy).toBe(true);
    expect(out.bitwise).toBe(false);
    expect(out.per_branch_tokens_per_s).toBeCloseTo(out.decode_tokens_per_s / 64, 1);
    expect(out.generated_tokens / elapsedS).toBeGreaterThanOrEqual(200);
  });
});