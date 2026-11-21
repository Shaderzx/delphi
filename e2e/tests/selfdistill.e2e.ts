import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { expect, test } from 'e2e';
import { completionFor, idsFor, startTarget } from '../fixtures/sglang-target.fixture.mjs';

const repo = join(import.meta.dirname, '..', '..');
const cli = join(repo, 'drafter', 'selfdistill.py');
const template = join(repo, 'drafter', 'templates', 'inkling-2-mini.jinja');
const python = process.env.DELPHI_PYTHON ?? 'python3';
const inklingPath = '/content/inkling';

type Prompt = { id: string; messages: { role: string; content: string }[]; reasoning_effort?: string };
type Record = { id: string; prompt: string; completion: string; output_ids: number[]; finish_reason: string; completion_tokens: number; model_path: string };

const prompts: Prompt[] = [
  { id: 'merge', messages: [{ role: 'user', content: 'Merge two sorted lists in Python.' }] },
  { id: 'tcp', messages: [{ role: 'user', content: 'Explain TCP slow start.' }], reasoning_effort: 'low' },
  { id: 'weather', messages: [{ role: 'system', content: 'You call tools.' }, { role: 'user', content: 'Weather in Paris?' }] },
  { id: 'haiku', messages: [{ role: 'user', content: 'A haiku about Metal shaders.' }] },
];

const expectedPrompts: { [id: string]: string } = {
  merge: '<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Merge two sorted lists in Python.<|end_message|>',
  tcp: '<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|><|message_user|><|content_text|>Explain TCP slow start.<|end_message|>',
  weather: '<|message_system|><|content_text|>You call tools.<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Weather in Paris?<|end_message|>',
  haiku: '<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>A haiku about Metal shaders.<|end_message|>',
};

function workspace(): string {
  return mkdtempSync(join(tmpdir(), 'delphi-selfdistill-'));
}

function writePrompts(dir: string, items: Prompt[]): string {
  const file = join(dir, 'prompts.jsonl');
  writeFileSync(file, items.map((p) => JSON.stringify(p)).join('\n') + '\n');
  return file;
}

function readRecords(file: string): Record[] {
  return readFileSync(file, 'utf8').trim().split('\n').map((line) => JSON.parse(line) as Record);
}

function selfdistill(args: string[]): Promise<{ code: number | null; stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn(python, ['-I', cli, ...args], { stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (c) => (stdout += c));
    child.stderr.on('data', (c) => (stderr += c));
    child.on('error', reject);
    child.on('close', (code) => resolve({ code, stdout, stderr }));
  });
}

function baseArgs(server: string, promptsFile: string, out: string): string[] {
  return ['--server', server, '--prompts', promptsFile, '--out', out, '--template', template, '--max-new-tokens', '64', '--temperature', '0', '--expect-model', inklingPath];
}

test('selfdistill writes one record per prompt holding the target completion for the rendered Inkling chat prompt', async () => {
  const dir = workspace();
  const target = await startTarget({ modelPath: inklingPath });
  try {
    const out = join(dir, 'out.jsonl');
    const run = await selfdistill(baseArgs(target.url, writePrompts(dir, prompts.slice(0, 3)), out));
    expect(run.stderr).toBe('');
    expect(run.code).toBe(0);

    const records = readRecords(out);
    expect(records.map((r) => r.id).sort()).toEqual(['merge', 'tcp', 'weather']);
    for (const record of records) {
      expect(record.prompt).toBe(expectedPrompts[record.id]);
      expect(record.completion).toBe(completionFor(expectedPrompts[record.id]));
      expect(record.output_ids).toEqual(idsFor(completionFor(expectedPrompts[record.id])));
      expect(record.finish_reason).toBe('stop');
      expect(record.model_path).toBe(inklingPath);
    }

    expect(target.generations).toHaveLength(3);
    for (const generation of target.generations) {
      expect(generation.body.sampling_params).toEqual({ temperature: 0, max_new_tokens: 64, skip_special_tokens: false });
    }
  } finally {
    await target.close();
    rmSync(dir, { recursive: true, force: true });
  }
});

test('selfdistill resumes by generating only the prompts missing from the output file', async () => {
  const dir = workspace();
  const target = await startTarget({ modelPath: inklingPath });
  try {
    const out = join(dir, 'out.jsonl');
    const first = await selfdistill(baseArgs(target.url, writePrompts(dir, prompts.slice(0, 2)), out));
    expect(first.code).toBe(0);
    const firstRecords = readRecords(out);

    const generatedBeforeResume = target.generations.length;

    const second = await selfdistill([...baseArgs(target.url, writePrompts(dir, prompts), out), '--resume']);
    expect(second.code).toBe(0);

    const records = readRecords(out);
    expect(records.map((r) => r.id).sort()).toEqual(['haiku', 'merge', 'tcp', 'weather']);
    expect(records.slice(0, 2)).toEqual(firstRecords);
    expect(generatedBeforeResume).toBe(2);
    const resumedTexts = target.generations.slice(generatedBeforeResume).map((g) => g.body.text).sort();
    expect(resumedTexts).toEqual([expectedPrompts.haiku, expectedPrompts.weather].sort());
  } finally {
    await target.close();
    rmSync(dir, { recursive: true, force: true });
  }
});

test('selfdistill refuses to generate when the server serves a different model', async () => {
  const dir = workspace();
  const target = await startTarget({ modelPath: '/content/some-other-model' });
  try {
    const out = join(dir, 'out.jsonl');
    const run = await selfdistill(baseArgs(target.url, writePrompts(dir, prompts.slice(0, 1)), out));
    expect(run.code).toBe(2);
    expect(run.stderr).toContain('/content/some-other-model');
    expect(run.stderr).toContain(inklingPath);
    expect(existsSync(out)).toBe(false);
    expect(target.generations).toHaveLength(0);
  } finally {
    await target.close();
    rmSync(dir, { recursive: true, force: true });
  }
});