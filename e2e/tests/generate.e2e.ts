import { spawnSync, type SpawnSyncReturns } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { expect, test } from 'e2e';

const repo = join(import.meta.dirname, '..', '..');
const model = process.env.DELPHI_MODEL ?? join(homedir(), 'models', 'Inkling-2-Mini');
const oraclePython = process.env.DELPHI_ORACLE_PYTHON ?? 'python3';
const CLEANUP_RESERVE_MS = 30_000;

type Case = { text: string; input_ids: number[]; output_ids: number[] };
type Generated = { output_ids: number[]; prompt_tokens: number; decode_tokens_per_s: number; accepted_per_step?: number; mode?: string; lossy?: boolean };

const truth = JSON.parse(readFileSync(join(import.meta.dirname, '..', 'fixtures', 'g4-sglang-greedy.json'), 'utf8')) as { cases: Case[] };
const fibonacci = truth.cases.find((c) => c.text.startsWith('def fibonacci'))!;
const paris = truth.cases.find((c) => c.text.startsWith('The capital of France'))!;

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

function generate(inputIds: number[], steps: number, deadline: number): Generated {
  return withBuiltDelphi(deadline, (bin) => {
    const run = spawnSync(bin, ['generate', '--model', model, '--input-ids', JSON.stringify(inputIds), '--steps', String(steps)], { encoding: 'utf8', timeout: remainingMs(deadline) });
    checkRun('delphi generate', run);
    return JSON.parse(run.stdout) as Generated;
  });
}

test('delphi generate rejects non-integer or negative input ids and truncated checkpoints', { timeout: 300_000 }, () => {
  const deadline = Date.now() + 300_000;
  withBuiltDelphi(deadline, (bin) => {
    const dir = mkdtempSync(join(tmpdir(), 'delphi-bad-model-'));
    try {
      const cfg = {
        text_config: {
          hidden_size: 8, num_hidden_layers: 1, num_attention_heads: 1, num_key_value_heads: 1, head_dim: 8,
          d_rel: 2, rel_extent: 4, sliding_window_size: 4, n_layers_per_attn_res_block: 1, n_routed_experts: 8,
          num_experts_per_tok: 8, intermediate_size: 8, unpadded_vocab_size: 8, vocab_size: 8,
          loop_config: { core_start: 0, core_end: 0, num_passes: 1 }, log_scaling_n_floor: 8,
          qk_l2_clip_threshold: 1, rel_l2_clip_threshold: 1, log_scaling_alpha: 0.1,
          mlp_gate_softcap: 4, mlp_up_softcap: 25, route_scale: 8, final_logit_softcapping: 30, local_layer_ids: [0],
        },
      };
      writeFileSync(join(dir, 'config.json'), JSON.stringify(cfg));
      const header = Buffer.from(JSON.stringify({ 'model.llm.embed.weight': { dtype: 'BF16', shape: [8, 8], data_offsets: [0, 128] } }));
      const lenBuf = Buffer.alloc(8);
      lenBuf.writeBigUInt64LE(BigInt(header.length));
      writeFileSync(join(dir, 'model.safetensors'), Buffer.concat([lenBuf, header, Buffer.alloc(16)]));
      for (const badIds of ['[-1]', '[1.5]', 'not-json']) {
        const bad = spawnSync(bin, ['generate', '--model', dir, '--input-ids', badIds, '--steps', '1'], { encoding: 'utf8', timeout: remainingMs(deadline) });
        if (bad.error) throw bad.error;
        expect(bad.status).toBe(2);
      }
      const trunc = spawnSync(bin, ['generate', '--model', dir, '--input-ids', '[0]', '--steps', '1'], { encoding: 'utf8', timeout: remainingMs(deadline) });
      if (trunc.error) throw trunc.error;
      expect(trunc.status).toBe(3);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

test('delphi bench refuses missing model or invalid repeat counts', { timeout: 300_000 }, () => {
  const deadline = Date.now() + 300_000;
  withBuiltDelphi(deadline, (bin) => {
    const missing = spawnSync(bin, ['bench', '--steps', '8'], { encoding: 'utf8', timeout: remainingMs(deadline) });
    if (missing.error) throw missing.error;
    expect(missing.status).toBe(2);
    expect(missing.stderr).toContain('--model');
    const badRepeats = spawnSync(bin, ['bench', '--model', '/nonexistent', '--input-ids', '[1]', '--repeats', '0'], { encoding: 'utf8', timeout: remainingMs(deadline) });
    if (badRepeats.error) throw badRepeats.error;
    expect(badRepeats.status).toBe(2);
    expect(badRepeats.stderr).toContain('--repeats');
    const badHead = spawnSync(bin, ['generate', '--model', '/nonexistent', '--input-ids', '[1]', '--head', 'bogus'], { encoding: 'utf8', timeout: remainingMs(deadline) });
    if (badHead.error) throw badHead.error;
    expect(badHead.status).toBe(2);
    expect(badHead.stderr).toContain('--head');
    const okHead = spawnSync(bin, ['generate', '--model', '/nonexistent', '--input-ids', '[1]', '--head', 'exact'], { encoding: 'utf8', timeout: remainingMs(deadline) });
    if (okHead.error) throw okHead.error;
    expect(okHead.status).toBe(3);
    const badPack = spawnSync(bin, ['generate', '--model', '/nonexistent', '--input-ids', '[1]', '--pack', 'bogus'], { encoding: 'utf8', timeout: remainingMs(deadline) });
    if (badPack.error) throw badPack.error;
    expect(badPack.status).toBe(2);
    expect(badPack.stderr).toContain('--pack');
    const okPack = spawnSync(bin, ['generate', '--model', '/nonexistent', '--input-ids', '[1]', '--pack', 'bf12'], { encoding: 'utf8', timeout: remainingMs(deadline) });
    if (okPack.error) throw okPack.error;
    expect(okPack.status).toBe(3);
    const badSpec = spawnSync(bin, ['generate', '--model', '/nonexistent', '--input-ids', '[1]', '--spec', 'bogus'], { encoding: 'utf8', timeout: remainingMs(deadline) });
    if (badSpec.error) throw badSpec.error;
    expect(badSpec.status).toBe(2);
    expect(badSpec.stderr).toContain('--spec');
    const okSpec = spawnSync(bin, ['generate', '--model', '/nonexistent', '--input-ids', '[1]', '--spec', 'auto'], { encoding: 'utf8', timeout: remainingMs(deadline) });
    if (okSpec.error) throw okSpec.error;
    expect(okSpec.status).toBe(3);
    const badMode = spawnSync(bin, ['generate', '--model', '/nonexistent', '--input-ids', '[1]', '--mode', 'bogus'], { encoding: 'utf8', timeout: remainingMs(deadline) });
    if (badMode.error) throw badMode.error;
    expect(badMode.status).toBe(2);
    expect(badMode.stderr).toContain('--mode');
    for (const m of ['exact', 'lean', 'turbo']) {
      const okMode = spawnSync(bin, ['generate', '--model', '/nonexistent', '--input-ids', '[1]', '--mode', m], { encoding: 'utf8', timeout: remainingMs(deadline) });
      if (okMode.error) throw okMode.error;
      expect(okMode.status).toBe(3);
    }
  });
});

test('delphi generate opt-in --mode lean and --mode turbo report mode, stay spec-exact, and preserve Paris answer', { tags: ['model'], timeout: 900_000 }, () => {
  const deadline = Date.now() + 900_000;
  withBuiltDelphi(deadline, (bin) => {
    for (const m of ['lean', 'turbo']) {
      const offRun = spawnSync(bin, ['generate', '--model', model, '--input-ids', JSON.stringify(paris.input_ids), '--steps', '16', '--mode', m, '--spec', 'off'], { encoding: 'utf8', timeout: remainingMs(deadline) });
      checkRun(`delphi generate --mode ${m} --spec off`, offRun);
      const run = spawnSync(bin, ['generate', '--model', model, '--input-ids', JSON.stringify(paris.input_ids), '--steps', '16', '--mode', m], { encoding: 'utf8', timeout: remainingMs(deadline) });
      checkRun(`delphi generate --mode ${m}`, run);
      const out = JSON.parse(run.stdout) as Generated;
      expect(out.mode).toBe(m);
      expect(out.lossy).toBe(true);
      expect(out.output_ids[0]).toBe(paris.output_ids[0]);
      expect(out.output_ids).toEqual((JSON.parse(offRun.stdout) as Generated).output_ids);
    }
  });
});

test('delphi generate --mode lean retrieves a needle past 4096 tokens', { tags: ['model'], timeout: 900_000 }, () => {
  const deadline = Date.now() + 900_000;
  withBuiltDelphi(deadline, (bin) => {
    const code = [19, 22, 17, 24, 16];
    const needle = [371, 153536, 150406, 149884, 220, ...code, 13, 198];
    const filler = [...truth.cases[2].input_ids, 198];
    const prompt: number[] = [...needle];
    while (prompt.length < 4600) prompt.push(...filler);
    prompt.push(371, 153536, 150406, 149884, 220);
    const leanNeedle = spawnSync(bin, ['generate', '--model', model, '--input-ids', JSON.stringify(prompt), '--steps', '6', '--mode', 'lean', '--spec', 'auto'], { encoding: 'utf8', timeout: remainingMs(deadline) });
    checkRun('delphi generate lean 4.6k needle', leanNeedle);
    expect((JSON.parse(leanNeedle.stdout) as Generated).output_ids.slice(0, code.length)).toEqual(code);
  });
});

test('delphi generate batched prefill and speculative verify match --spec off token-for-token', { tags: ['model'], timeout: 900_000 }, () => {
  const deadline = Date.now() + 900_000;
  withBuiltDelphi(deadline, (bin) => {
    const widePrompt = Array.from({ length: 8 }, () => fibonacci.input_ids).flat();
    for (const ids of [fibonacci.input_ids, truth.cases[2].input_ids, widePrompt]) {
      const offRun = spawnSync(bin, ['generate', '--model', model, '--input-ids', JSON.stringify(ids), '--steps', '64', '--spec', 'off'], { encoding: 'utf8', timeout: remainingMs(deadline) });
      checkRun('delphi generate --spec off', offRun);
      const specRun = spawnSync(bin, ['generate', '--model', model, '--input-ids', JSON.stringify(ids), '--steps', '64', '--spec', 'auto'], { encoding: 'utf8', timeout: remainingMs(deadline) });
      checkRun('delphi generate --spec auto', specRun);
      expect((JSON.parse(specRun.stdout) as Generated).output_ids).toEqual((JSON.parse(offRun.stdout) as Generated).output_ids);
    }
  });
});

test('delphi generate retrieves a needle past 4096 tokens using 32K ring-KV', { tags: ['model'], timeout: 900_000 }, () => {
  const deadline = Date.now() + 900_000;
  withBuiltDelphi(deadline, (bin) => {
    const code = [19, 22, 17, 24, 16];
    const needle = [371, 153536, 150406, 149884, 220, ...code, 13, 198];
    const filler = [...truth.cases[2].input_ids, 198];
    const prompt: number[] = [...needle];
    while (prompt.length < 4600) prompt.push(...filler);
    prompt.push(371, 153536, 150406, 149884, 220);
    const run = spawnSync(bin, ['generate', '--model', model, '--input-ids', JSON.stringify(prompt), '--steps', '6', '--spec', 'auto'], { encoding: 'utf8', timeout: remainingMs(deadline) });
    checkRun('delphi generate 4.6k needle', run);
    const out = JSON.parse(run.stdout) as Generated;
    expect(out.output_ids.slice(0, code.length)).toEqual(code);
  });
});

test('delphi generate --spec mtp matches --spec off token-for-token with multi-token tree acceptance', { tags: ['model'], timeout: 900_000 }, () => {
  const deadline = Date.now() + 900_000;
  withBuiltDelphi(deadline, (bin) => {
    for (const c of [fibonacci, truth.cases[2]]) {
      const offRun = spawnSync(bin, ['generate', '--model', model, '--input-ids', JSON.stringify(c.input_ids), '--steps', '64', '--spec', 'off'], { encoding: 'utf8', timeout: remainingMs(deadline) });
      checkRun('delphi generate --spec off', offRun);
      const mtpRun = spawnSync(bin, ['generate', '--model', model, '--input-ids', JSON.stringify(c.input_ids), '--steps', '64', '--spec', 'mtp'], { encoding: 'utf8', timeout: remainingMs(deadline) });
      checkRun('delphi generate --spec mtp', mtpRun);
      const mtpOut = JSON.parse(mtpRun.stdout) as Generated;
      expect(mtpOut.output_ids).toEqual((JSON.parse(offRun.stdout) as Generated).output_ids);
      expect(mtpOut.accepted_per_step ?? 0).toBeGreaterThan(1.55);
    }
  });
});

test('delphi generate 12-bit exact BF16 packing matches unpacked BF16 token-for-token', { tags: ['model'], timeout: 900_000 }, () => {
  const deadline = Date.now() + 900_000;
  withBuiltDelphi(deadline, (bin) => {
    for (const c of [fibonacci, truth.cases[2]]) {
      const offRun = spawnSync(bin, ['generate', '--model', model, '--input-ids', JSON.stringify(c.input_ids), '--steps', '32', '--pack', 'off'], { encoding: 'utf8', timeout: remainingMs(deadline) });
      checkRun('delphi generate --pack off', offRun);
      const packRun = spawnSync(bin, ['generate', '--model', model, '--input-ids', JSON.stringify(c.input_ids), '--steps', '32', '--pack', 'bf12'], { encoding: 'utf8', timeout: remainingMs(deadline) });
      checkRun('delphi generate --pack bf12', packRun);
      expect((JSON.parse(packRun.stdout) as Generated).output_ids).toEqual((JSON.parse(offRun.stdout) as Generated).output_ids);
    }
  });
});

test('delphi generate certified exact head matches full BF16 head token-for-token', { tags: ['model'], timeout: 900_000 }, () => {
  const deadline = Date.now() + 900_000;
  withBuiltDelphi(deadline, (bin) => {
    for (const c of [fibonacci, truth.cases[2]]) {
      const fullRun = spawnSync(bin, ['generate', '--model', model, '--input-ids', JSON.stringify(c.input_ids), '--steps', '32', '--head', 'full'], { encoding: 'utf8', timeout: remainingMs(deadline) });
      checkRun('delphi generate --head full', fullRun);
      const exactRun = spawnSync(bin, ['generate', '--model', model, '--input-ids', JSON.stringify(c.input_ids), '--steps', '32', '--head', 'exact'], { encoding: 'utf8', timeout: remainingMs(deadline) });
      checkRun('delphi generate --head exact', exactRun);
      const fullOut = (JSON.parse(fullRun.stdout) as Generated).output_ids;
      const exactOut = (JSON.parse(exactRun.stdout) as Generated).output_ids;
      expect(exactOut).toEqual(fullOut);
    }
  });
});

test('delphi generate continues the fibonacci prompt with the same 32 greedy tokens as the SGLang server on a G4', { tags: ['model'], timeout: 900_000 }, () => {
  const deadline = Date.now() + 900_000;
  const result = generate(fibonacci.input_ids, fibonacci.output_ids.length, deadline);
  expect(result.prompt_tokens).toBe(fibonacci.input_ids.length);
  expect(result.output_ids).toEqual(fibonacci.output_ids);
});

test('delphi generate picks the same 12 greedy tokens as the numpy oracle on the Paris prompt', { tags: ['model'], timeout: 1_800_000 }, () => {
  const deadline = Date.now() + 1_800_000;
  const steps = 12;
  const oracle = spawnSync(oraclePython, ['-I', join(repo, 'oracle', 'inkling.py'), '--model', model, '--input-ids', JSON.stringify(paris.input_ids), '--steps', String(steps)], { encoding: 'utf8', timeout: remainingMs(deadline) });
  checkRun('oracle', oracle);
  const expected = (JSON.parse(oracle.stdout) as { output_ids: number[] }).output_ids;
  expect(expected.slice(0, 5)).toEqual(paris.output_ids.slice(0, 5));
  expect(generate(paris.input_ids, steps, deadline).output_ids).toEqual(expected);
});