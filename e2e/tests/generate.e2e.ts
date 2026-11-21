import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { expect, test } from 'e2e';

const repo = join(import.meta.dirname, '..', '..');
const model = process.env.DELPHI_MODEL ?? join(homedir(), 'models', 'Inkling-2-Mini');
const oraclePython = process.env.DELPHI_ORACLE_PYTHON ?? 'python3';

type Case = { text: string; input_ids: number[]; output_ids: number[] };
type Generated = { output_ids: number[]; prompt_tokens: number; decode_tokens_per_s: number };

const truth = JSON.parse(readFileSync(join(import.meta.dirname, '..', 'fixtures', 'g4-sglang-greedy.json'), 'utf8')) as { cases: Case[] };
const fibonacci = truth.cases.find((c) => c.text.startsWith('def fibonacci'))!;
const paris = truth.cases.find((c) => c.text.startsWith('The capital of France'))!;

function generate(inputIds: number[], steps: number): Generated {
  const out = mkdtempSync(join(tmpdir(), 'delphi-build-'));
  try {
    const built = spawnSync('bash', [join(repo, 'engine', 'build.sh'), out], { encoding: 'utf8', timeout: 300_000 });
    if (built.status !== 0) throw new Error(`engine build failed: ${built.stderr.slice(-3000)}`);
    const run = spawnSync(join(out, 'delphi'), ['generate', '--model', model, '--input-ids', JSON.stringify(inputIds), '--steps', String(steps)], { encoding: 'utf8', timeout: 900_000 });
    if (run.status !== 0) throw new Error(`delphi generate exited ${run.status}: ${run.stderr}`);
    return JSON.parse(run.stdout) as Generated;
  } finally {
    rmSync(out, { recursive: true, force: true });
  }
}

test('delphi generate continues the fibonacci prompt with the same 32 greedy tokens as the SGLang server on a G4', { tags: ['model'], timeout: 900_000 }, () => {
  const result = generate(fibonacci.input_ids, fibonacci.output_ids.length);
  expect(result.prompt_tokens).toBe(fibonacci.input_ids.length);
  expect(result.output_ids).toEqual(fibonacci.output_ids);
  expect(result.decode_tokens_per_s).toBeGreaterThan(0);
});

test('delphi generate picks the same 12 greedy tokens as the numpy oracle on the Paris prompt', { tags: ['model'], timeout: 1_800_000 }, () => {
  const steps = 12;
  const oracle = spawnSync(oraclePython, ['-I', join(repo, 'oracle', 'inkling.py'), '--model', model, '--input-ids', JSON.stringify(paris.input_ids), '--steps', String(steps)], { encoding: 'utf8', timeout: 1_700_000 });
  if (oracle.status !== 0) throw new Error(`oracle exited ${oracle.status}: ${oracle.stderr.slice(-3000)}`);
  const expected = (JSON.parse(oracle.stdout) as { output_ids: number[] }).output_ids;
  expect(expected.slice(0, 5)).toEqual(paris.output_ids.slice(0, 5));
  expect(generate(paris.input_ids, steps).output_ids).toEqual(expected);
});