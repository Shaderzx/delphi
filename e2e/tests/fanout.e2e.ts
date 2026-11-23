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
  end_to_end_tokens_per_s: number;
  ms_per_step: number;
  mode: string;
  lossy: boolean;
  solo_identical: boolean;
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

function expectEachBranchMatchesSolo(bin: string, prefixIds: number[], heads: number[][], out: Fanout, steps: number, deadline: number, extra: string[] = []): void {
  heads.forEach((head, b) => {
    const alone = spawnSync(bin, ['generate', '--model', model, '--input-ids', JSON.stringify([...prefixIds, ...head]), '--steps', String(steps), '--spec', 'off', ...extra], { encoding: 'utf8', timeout: remainingMs(deadline) });
    checkRun(`delphi generate branch ${b}`, alone);
    expect(out.branches[b].output_ids).toEqual((JSON.parse(alone.stdout) as Generated).output_ids);
  });
}

test('delphi fanout refuses missing, empty, malformed or more than 64 branch lists, unknown switches such as --tensor-dense, and runs that would exceed the 32768-token context', { timeout: 300_000 }, () => {
  const deadline = Date.now() + 300_000;
  withBuiltDelphi(deadline, (bin) => {
    const dense = spawnSync(bin, ['fanout', '--model', '/nonexistent', '--prefix-ids', '[1]', '--branch-ids', '[[1]]', '--tensor-dense'], { encoding: 'utf8', timeout: remainingMs(deadline) });
    if (dense.error) throw dense.error;
    expect(dense.status).toBe(2);
    expect(dense.stderr).toContain('--tensor-dense');
    const tooMany = JSON.stringify(Array.from({ length: 65 }, () => [1]));
    for (const bad of ['[]', '[[]]', '[1, 2]', '[[1], [-3]]', 'not-json', tooMany]) {
      const run = spawnSync(bin, ['fanout', '--model', '/nonexistent', '--prefix-ids', '[1]', '--branch-ids', bad], { encoding: 'utf8', timeout: remainingMs(deadline) });
      if (run.error) throw run.error;
      expect(run.status).toBe(2);
      expect(run.stderr).toContain('--branch-ids');
    }
    const overflow = spawnSync(bin, ['fanout', '--model', '/nonexistent', '--prefix-ids', '[1]', '--branch-ids', JSON.stringify(Array.from({ length: 64 }, () => [2])), '--steps', '512'], { encoding: 'utf8', timeout: remainingMs(deadline) });
    if (overflow.error) throw overflow.error;
    expect(overflow.status).toBe(2);
    expect(overflow.stderr).toContain('32768');
    const missing = spawnSync(bin, ['fanout', '--model', '/nonexistent', '--prefix-ids', '[1]'], { encoding: 'utf8', timeout: remainingMs(deadline) });
    if (missing.error) throw missing.error;
    expect(missing.status).toBe(2);
    const okShape = spawnSync(bin, ['fanout', '--model', '/nonexistent', '--prefix-ids', '[1]', '--branch-ids', JSON.stringify(Array.from({ length: 64 }, () => [2]))], { encoding: 'utf8', timeout: remainingMs(deadline) });
    if (okShape.error) throw okShape.error;
    expect(okShape.status).toBe(3);
  });
});

test('delphi fanout decodes each of 64 branches token-for-token like decoding that branch alone for 128 steps at 140 tok/s or more, and reports a slower end-to-end rate that includes prefill', { tags: ['model'], timeout: 2_400_000 }, () => {
  const deadline = Date.now() + 2_400_000;
  const steps = 128;
  withBuiltDelphi(deadline, (bin) => {
    const out = fanout(bin, prefix, branchHeads, steps, deadline);
    expect(out.branch_count).toBe(64);
    expect(out.mode).toBe('exact');
    expect(out.lossy).toBe(false);
    expect(out.solo_identical).toBe(true);
    expect(out.decode_tokens_per_s).toBeGreaterThanOrEqual(140);
    expect(out.end_to_end_tokens_per_s).toBeGreaterThan(0);
    expect(out.end_to_end_tokens_per_s).toBeLessThan(out.decode_tokens_per_s);
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
    expect(out.solo_identical).toBe(true);
    expect(out.lockstep_steps).toBe(5 - 1 + steps);
    expectEachBranchMatchesSolo(bin, longPrefix, heads, out, steps, deadline);
  });
});

test('delphi fanout keeps two branches token-for-token like solo decoding for 128 steps', { tags: ['model'], timeout: 900_000 }, () => {
  const deadline = Date.now() + 900_000;
  const steps = 128;
  const heads = branchHeads.slice(0, 2);
  withBuiltDelphi(deadline, (bin) => {
    const out = fanout(bin, prefix, heads, steps, deadline);
    expect(out.solo_identical).toBe(true);
    expectEachBranchMatchesSolo(bin, prefix, heads, out, steps, deadline);
  });
});

test('delphi fanout matches solo decoding token-for-token at 1, 7 and 25 branches', { tags: ['model'], timeout: 1_800_000 }, () => {
  const deadline = Date.now() + 1_800_000;
  const steps = 16;
  withBuiltDelphi(deadline, (bin) => {
    for (const count of [1, 7, 25]) {
      const heads = branchHeads.slice(0, count);
      const out = fanout(bin, prefix, heads, steps, deadline);
      expect(out.branch_count).toBe(count);
      expect(out.solo_identical).toBe(true);
      expectEachBranchMatchesSolo(bin, prefix, heads, out, steps, deadline);
    }
  });
});

test('delphi fanout --tensor-experts stays token-for-token like solo decoding at 23 branches, one below the tensor threshold', { tags: ['model'], timeout: 1_200_000 }, () => {
  const deadline = Date.now() + 1_200_000;
  const steps = 24;
  const heads = branchHeads.slice(0, 23);
  withBuiltDelphi(deadline, (bin) => {
    const out = fanout(bin, prefix, heads, steps, deadline, ['--tensor-experts']);
    expect(out.lossy).toBe(false);
    expect(out.solo_identical).toBe(true);
    expectEachBranchMatchesSolo(bin, prefix, heads, out, steps, deadline);
  });
});

test('delphi fanout --tensor-experts at 64 branches keeps sampled branches on the solo tokens for their first 8 steps and at least 80% of 24', { tags: ['model'], timeout: 1_200_000 }, () => {
  const deadline = Date.now() + 1_200_000;
  const steps = 24;
  withBuiltDelphi(deadline, (bin) => {
    const out = fanout(bin, prefix, branchHeads, steps, deadline, ['--tensor-experts']);
    expect(out.solo_identical).toBe(false);
    let same = 0;
    for (let b = 0; b < branchHeads.length; b += 8) {
      const alone = spawnSync(bin, ['generate', '--model', model, '--input-ids', JSON.stringify([...prefix, ...branchHeads[b]]), '--steps', String(steps), '--spec', 'off'], { encoding: 'utf8', timeout: remainingMs(deadline) });
      checkRun(`delphi generate branch ${b}`, alone);
      const solo = (JSON.parse(alone.stdout) as Generated).output_ids;
      expect(out.branches[b].output_ids.slice(0, 8)).toEqual(solo.slice(0, 8));
      same += solo.filter((id, i) => out.branches[b].output_ids[i] === id).length;
    }
    expect(same / (8 * steps)).toBeGreaterThanOrEqual(0.8);
  });
});
test('delphi fanout --mode lean matches lean solo decoding token-for-token across 8 branches', { tags: ['model'], timeout: 1_200_000 }, () => {
  const deadline = Date.now() + 1_200_000;
  const steps = 24;
  const heads = branchHeads.slice(0, 8);
  withBuiltDelphi(deadline, (bin) => {
    const out = fanout(bin, prefix, heads, steps, deadline, ['--mode', 'lean']);
    expect(out.mode).toBe('lean');
    expect(out.lossy).toBe(true);
    expect(out.solo_identical).toBe(true);
    heads.forEach((head, b) => {
      const alone = spawnSync(bin, ['generate', '--model', model, '--input-ids', JSON.stringify([...prefix, ...head]), '--steps', String(steps), '--spec', 'off', '--mode', 'lean'], { encoding: 'utf8', timeout: remainingMs(deadline) });
      checkRun(`delphi generate --mode lean branch ${b}`, alone);
      expect(out.branches[b].output_ids).toEqual((JSON.parse(alone.stdout) as Generated).output_ids);
    });
  });
});

test('delphi fanout --tensor-experts decodes over 200 tokens per second across 64 branches and over 180 timed from outside including load, and reports itself lossy', { tags: ['model'], timeout: 900_000 }, () => {
  const deadline = Date.now() + 900_000;
  const steps = 500;
  withBuiltDelphi(deadline, (bin) => {
    const started = process.hrtime.bigint();
    const out = fanout(bin, prefix, branchHeads, steps, deadline, ['--tensor-experts']);
    const elapsedS = Number(process.hrtime.bigint() - started) / 1e9;
    expect(out.branch_count).toBe(64);
    expect(out.generated_tokens).toBe(64 * steps);
    expect(out.branches.every((b) => b.output_ids.length === steps)).toBe(true);
    expect(out.lossy).toBe(true);
    expect(out.solo_identical).toBe(false);
    expect(out.per_branch_tokens_per_s).toBeCloseTo(out.decode_tokens_per_s / 64, 1);
    expect(out.decode_tokens_per_s).toBeGreaterThanOrEqual(200);
    expect(out.generated_tokens / elapsedS).toBeGreaterThanOrEqual(180);
  });
});
test('delphi fanout stays token-for-token like solo decoding when 64 branches fill the 32768-slot KV buffer', { tags: ['model'], timeout: 1_800_000 }, () => {
  const deadline = Date.now() + 1_800_000;
  const steps = Math.floor((32768 - prefix.length) / 64) - 1;
  withBuiltDelphi(deadline, (bin) => {
    const out = fanout(bin, prefix, branchHeads, steps, deadline);
    expect(out.lockstep_steps).toBe(steps + 1);
    expect(prefix.length + out.lockstep_steps * 64).toBeLessThanOrEqual(32768);
    expect(out.solo_identical).toBe(true);
    const sampled = [0, 63];
    sampled.forEach((b) => {
      const alone = spawnSync(bin, ['generate', '--model', model, '--input-ids', JSON.stringify([...prefix, ...branchHeads[b]]), '--steps', String(steps), '--spec', 'off'], { encoding: 'utf8', timeout: remainingMs(deadline) });
      checkRun(`delphi generate branch ${b}`, alone);
      expect(out.branches[b].output_ids).toEqual((JSON.parse(alone.stdout) as Generated).output_ids);
    });
  });
});

test('delphi fanout --head full matches solo --head full decoding token-for-token across 4 branches', { tags: ['model'], timeout: 900_000 }, () => {
  const deadline = Date.now() + 900_000;
  const steps = 24;
  const heads = branchHeads.slice(0, 4);
  withBuiltDelphi(deadline, (bin) => {
    const out = fanout(bin, prefix, heads, steps, deadline, ['--head', 'full']);
    expect(out.solo_identical).toBe(true);
    heads.forEach((head, b) => {
      const alone = spawnSync(bin, ['generate', '--model', model, '--input-ids', JSON.stringify([...prefix, ...head]), '--steps', String(steps), '--spec', 'off', '--head', 'full'], { encoding: 'utf8', timeout: remainingMs(deadline) });
      checkRun(`delphi generate --head full branch ${b}`, alone);
      expect(out.branches[b].output_ids).toEqual((JSON.parse(alone.stdout) as Generated).output_ids);
    });
  });
});

test('delphi fanout --mode lean and --mode turbo match solo decoding in the same mode token-for-token at 25 branches', { tags: ['model'], timeout: 1_800_000 }, () => {
  const deadline = Date.now() + 1_800_000;
  const steps = 16;
  const heads = branchHeads.slice(0, 25);
  withBuiltDelphi(deadline, (bin) => {
    for (const mode of ['lean', 'turbo']) {
      const out = fanout(bin, prefix, heads, steps, deadline, ['--mode', mode]);
      expect(out.mode).toBe(mode);
      expect(out.lossy).toBe(true);
      expect(out.solo_identical).toBe(true);
      expectEachBranchMatchesSolo(bin, prefix, heads, out, steps, deadline, ['--mode', mode]);
    }
  });
});

test('delphi fanout matches solo decoding token-for-token on 4 branches over a 4096-token prefix', { tags: ['model'], timeout: 1_800_000 }, () => {
  const deadline = Date.now() + 1_800_000;
  const steps = 16;
  const repeated = Array.from({ length: Math.ceil(4096 / fibonacci.input_ids.length) }, () => fibonacci.input_ids).flat();
  const longPrefix = repeated.slice(0, 4096);
  const heads = branchHeads.slice(0, 4);
  withBuiltDelphi(deadline, (bin) => {
    const out = fanout(bin, longPrefix, heads, steps, deadline);
    expect(out.solo_identical).toBe(true);
    expectEachBranchMatchesSolo(bin, longPrefix, heads, out, steps, deadline);
  });
});

test('delphi fanout matches solo decoding token-for-token on all 64 branches past the 1024-token local attention window', { tags: ['model'], timeout: 2_400_000 }, () => {
  const deadline = Date.now() + 2_400_000;
  const steps = 16;
  const longPrefix = Array.from({ length: 56 }, () => fibonacci.input_ids).flat();
  withBuiltDelphi(deadline, (bin) => {
    expect(longPrefix.length).toBeGreaterThan(1100);
    const out = fanout(bin, longPrefix, branchHeads, steps, deadline);
    expect(out.branch_count).toBe(64);
    expect(out.solo_identical).toBe(true);
    expectEachBranchMatchesSolo(bin, longPrefix, branchHeads, out, steps, deadline);
  });
});

test('delphi fanout --tensor-experts reports itself lossy from exactly 24 branches', { tags: ['model'], timeout: 600_000 }, () => {
  const deadline = Date.now() + 600_000;
  withBuiltDelphi(deadline, (bin) => {
    const out = fanout(bin, prefix, branchHeads.slice(0, 24), 4, deadline, ['--tensor-experts']);
    expect(out.branch_count).toBe(24);
    expect(out.lossy).toBe(true);
    expect(out.solo_identical).toBe(false);
  });
});
