#!/usr/bin/env node
import { execFileSync, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { realpathSync, closeSync, copyFileSync, existsSync, fstatSync, linkSync, lstatSync, mkdirSync, mkdtempSync, openSync, readFileSync, renameSync, rmSync, statSync, writeFileSync, writeSync } from 'node:fs';
import { createRequire } from 'node:module';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const PROFILE = ['package.json', 'package-lock.json', 'oxlintrc.json'];
const SOURCE = /\.[cm]?[jt]sx?$/;
const SKIPPED = /(?:^|\/)(?:node_modules|vendor|third_party|\.git)\/|\.(?:min|umd)\.[cm]?js$/;
const MIN_NODE = [22, 18];
const FAKE_CLOCK = 'useFakeTimers';
const UNMASKED_CLOCK = 'useFakeTimerX';
const SLEEP_RULE = 'landmines-testing/no-fixed-sleep';

function verdict(kind, detail) {
  console.log('landmines: ' + kind + ' (' + detail + ')');
  process.exit({ PASS: 0, FAIL: 1, BLOCKED: 3 }[kind]);
}

function parseArgs(argv) {
  const args = { base: null };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--base' && argv[i + 1]) args.base = argv[++i];
    else if (argv[i] === '--help' || argv[i] === '-h') {
      console.log('usage: landmines.mjs [--base <ref>]\n  no --base: every tracked JS/TS file must be clean\n  --base: only findings the branch adds against its merge base fail');
      process.exit(0);
    } else verdict('BLOCKED', 'unknown argument ' + argv[i] + '; usage: landmines.mjs [--base <ref>]');
  }
  return args;
}

function nodeIsRecentEnough() {
  const [major, minor] = process.versions.node.split('.').map(Number);
  return major > MIN_NODE[0] || (major === MIN_NODE[0] && minor >= MIN_NODE[1]);
}

function git(cwd, args, env = process.env) {
  return execFileSync('git', args, { cwd, env, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], maxBuffer: 256 * 1024 * 1024 });
}

function tryGit(cwd, args) {
  try {
    return git(cwd, args);
  } catch {
    return null;
  }
}

function profileHash() {
  const hash = createHash('sha256');
  hash.update([process.platform, process.arch, process.versions.node.split('.')[0]].join('\0'));
  for (const name of [...PROFILE, 'landmines.mjs']) hash.update('\0' + name + '\0' + readFileSync(join(here, name)));
  return hash.digest('hex').slice(0, 16);
}

function cacheRoot() {
  return process.env.LANDMINES_CACHE || join(process.env.XDG_CACHE_HOME || join(homedir(), '.cache'), 'fleet-landmines');
}

function run(cmd, args, cwd) {
  const r = spawnSync(cmd, args, { cwd, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], maxBuffer: 256 * 1024 * 1024 });
  return { code: r.status ?? 1, out: (r.stdout || '') + (r.stderr || '') || String(r.error?.message ?? ''), stdout: r.stdout || '', missing: r.error?.code === 'ENOENT' };
}

const lastLines = (text) => text.trim().split('\n').slice(-3).join(' | ');

function pinPluginPaths(dir) {
  const configPath = join(dir, 'oxlintrc.json');
  const config = JSON.parse(readFileSync(configPath, 'utf8'));
  const resolve = createRequire(join(dir, 'package.json')).resolve;
  config.jsPlugins = config.jsPlugins.map((plugin) => ({ ...plugin, specifier: './' + relative(realpathSync(dir), resolve(plugin.specifier)).split(sep).join('/') }));
  writeFileSync(configPath, JSON.stringify(config, null, 2) + '\n');
}

function readLock(lock) {
  try {
    const fd = openSync(lock, 'r');
    try {
      return { ino: fstatSync(fd).ino, pid: Number(readFileSync(fd, 'utf8').trim()) };
    } finally {
      closeSync(fd);
    }
  } catch (error) {
    if (error.code === 'ENOENT') return null;
    throw error;
  }
}

function alive(pid) {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return error.code === 'EPERM';
  }
}

function clearDeadLock(lock, dead) {
  const moved = lock + '.' + process.pid + '-' + Date.now();
  try {
    renameSync(lock, moved);
  } catch (error) {
    if (error.code === 'ENOENT') return;
    throw error;
  }
  if (statSync(moved).ino === dead.ino) {
    rmSync(moved, { force: true });
    return;
  }
  try {
    linkSync(moved, lock);
  } finally {
    rmSync(moved, { force: true });
  }
}

function takeLock(lock) {
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const fd = openSync(lock, 'wx');
      try {
        writeSync(fd, String(process.pid) + '\n');
        return fstatSync(fd).ino;
      } finally {
        closeSync(fd);
      }
    } catch (error) {
      if (error.code !== 'EEXIST') throw error;
    }
    const holder = readLock(lock);
    if (holder && alive(holder.pid)) return null;
    if (holder) {
      try {
        clearDeadLock(lock, holder);
      } catch {
        return null;
      }
    }
  }
  return null;
}

function moveInto(staging, ready) {
  try {
    renameSync(staging, ready);
    return true;
  } catch {
    return false;
  }
}

function replaceStale(root, staging, ready) {
  const lock = ready + '.lock';
  const owned = takeLock(lock);
  if (owned === null) return false;
  try {
    if (existsSync(join(ready, '.ready'))) return true;
    const stale = join(root, 'stale-' + process.pid + '-' + Date.now());
    try {
      renameSync(ready, stale);
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
    if (existsSync(join(stale, '.ready')) && moveInto(stale, ready)) return true;
    const placed = moveInto(staging, ready);
    rmSync(stale, { recursive: true, force: true });
    return placed || existsSync(join(ready, '.ready'));
  } finally {
    if (readLock(lock)?.ino === owned) rmSync(lock, { force: true });
  }
}

function publish(root, staging, ready) {
  if (moveInto(staging, ready) || existsSync(join(ready, '.ready')) || replaceStale(root, staging, ready)) {
    rmSync(staging, { recursive: true, force: true });
    return ready;
  }
  process.on('exit', () => rmSync(staging, { recursive: true, force: true }));
  return staging;
}

function install() {
  const root = cacheRoot();
  const ready = join(root, profileHash());
  if (existsSync(join(ready, '.ready'))) return ready;
  mkdirSync(root, { recursive: true });
  const staging = mkdtempSync(join(root, 'staging-'));
  const abandon = (kind, detail) => {
    rmSync(staging, { recursive: true, force: true });
    verdict(kind, detail);
  };
  for (const name of PROFILE) copyFileSync(join(here, name), join(staging, name));
  const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';
  const ci = run(npm, ['ci', '--ignore-scripts', '--no-audit', '--no-fund', '--silent'], staging);
  if (ci.missing) abandon('BLOCKED', 'npm is not on PATH: install Node.js with npm, or put it on PATH');
  if (ci.code !== 0) abandon('BLOCKED', 'npm ci of the pinned profile failed: ' + lastLines(ci.out));
  const signatures = run(npm, ['audit', 'signatures'], staging);
  if (signatures.code !== 0) abandon('BLOCKED', 'npm audit signatures could not verify the pinned profile (the registry may be unreachable, or a signature is wrong): ' + lastLines(signatures.out));
  pinPluginPaths(staging);
  writeFileSync(join(staging, '.ready'), new Date().toISOString() + '\n');
  return publish(root, staging, ready);
}

function linted(dir, f) {
  return !!f && SOURCE.test(f) && !SKIPPED.test(f) && lstatSync(join(dir, f), { throwIfNoEntry: false })?.isFile() === true;
}

function trackedSources(top) {
  return git(top, ['ls-files', '-z', '--cached', '--others', '--exclude-standard'])
    .split('\0')
    .filter((f) => linted(top, f));
}

function ruleName(code) {
  if (!code) return 'parse-error';
  const m = /^([\w-]+)\(([\w-]+)\)$/.exec(code);
  return m ? m[1] + '/' + m[2] : String(code);
}

function spanText(content, span) {
  if (!span) return '';
  return content.subarray(span.offset, span.offset + span.length).toString('utf8').replace(/\s+/g, ' ').trim();
}

function oxlint(installed, dir, files) {
  const bin = join(installed, 'node_modules', 'oxlint', 'bin', 'oxlint');
  const config = join(installed, 'oxlintrc.json');
  const diagnostics = [];
  for (let i = 0; i < files.length; i += 400) {
    const chunk = files.slice(i, i + 400).map((f) => './' + f);
    const r = run(process.execPath, [bin, '-c', config, '--disable-nested-config', '-f', 'json', '--no-ignore', ...chunk], dir);
    let parsed;
    try {
      parsed = JSON.parse(r.stdout);
    } catch {
      verdict('BLOCKED', 'oxlint did not run: ' + lastLines(r.out));
    }
    if (r.code !== 0 && r.code !== 1) verdict('BLOCKED', 'oxlint exited ' + r.code + ': ' + lastLines(r.out));
    diagnostics.push(...(parsed.diagnostics || []));
  }
  return diagnostics.map((d) => ({ ...d, file: relative(dir, join(dir, d.filename)).split(sep).join('/') }));
}

function finding(dir, d) {
  const span = d.labels?.[0]?.span;
  const rule = ruleName(d.code);
  const severity = rule === 'parse-error' || d.severity === 'error' ? 'error' : 'warning';
  const content = readFileSync(join(dir, d.file));
  const text = rule === 'parse-error' ? 'whole file ' + createHash('sha256').update(content).digest('hex') : spanText(content, span);
  return { file: d.file, line: span?.line ?? 1, rule, severity, message: d.message, text };
}

const TEST_CALLEES = new Set(['test', 'it']);
const SUITE_CALLEES = new Set(['describe', 'suite', 'context']);

function calleeRoot(callee) {
  let node = callee;
  while (node && (node.type === 'MemberExpression' || node.type === 'CallExpression')) node = node.type === 'MemberExpression' ? node.object : node.callee;
  return node?.type === 'Identifier' ? node.name : null;
}

function clockMap(installed, text) {
  let ast;
  try {
    ast = createRequire(join(installed, 'package.json'))('@babel/parser').parse(text, { sourceType: 'unambiguous', errorRecovery: true, plugins: ['typescript', 'jsx'] });
  } catch {
    return null;
  }
  const tests = [];
  const suites = [];
  const clocks = [];
  const visit = (node) => {
    if (!node || typeof node.type !== 'string') return;
    if (node.type === 'CallExpression') {
      const root = calleeRoot(node.callee);
      if (TEST_CALLEES.has(root)) tests.push([node.start, node.end]);
      if (SUITE_CALLEES.has(root)) suites.push([node.start, node.end]);
      if (node.callee.type === 'MemberExpression' && node.callee.property?.name === FAKE_CLOCK) clocks.push(node.start);
    }
    for (const key of Object.keys(node)) {
      if (key === 'loc' || key === 'leadingComments' || key === 'trailingComments') continue;
      const value = node[key];
      if (Array.isArray(value)) value.forEach(visit);
      else if (value && typeof value === 'object') visit(value);
    }
  };
  visit(ast.program);
  const within = (at) => ([s, e]) => s <= at && at < e;
  const smallest = (spans) => spans.sort((a, b) => a[1] - a[0] - (b[1] - b[0]))[0] ?? null;
  const innermost = (at) => smallest(tests.filter(within(at)));
  const hookScopes = clocks.filter((at) => !innermost(at)).map((at) => smallest(suites.filter(within(at))));
  const inTestClock = (at) => {
    const t = innermost(at);
    return !!t && clocks.some((c) => within(c)(t));
  };
  return {
    wholeFile: hookScopes.includes(null),
    fakeAt: (at) => inTestClock(at) || hookScopes.some((scope) => !!scope && within(at)(scope)),
  };
}

function sleepsBehindFakeClocks(installed, dir, files) {
  const clocked = files.map((f) => [f, readFileSync(join(dir, f), 'utf8')]).filter(([, text]) => text.includes(FAKE_CLOCK));
  if (!clocked.length) return [];
  const maps = new Map(clocked.map(([f, text]) => [f, { text, map: clockMap(installed, text) }]));
  const mirrored = clocked.filter(([f]) => !maps.get(f).map?.wholeFile).map(([f]) => f);
  if (!mirrored.length) return [];
  const mirror = mkdtempSync(join(tmpdir(), 'fleet-landmines-clock-'));
  try {
    for (const f of mirrored) {
      mkdirSync(dirname(join(mirror, f)), { recursive: true });
      writeFileSync(join(mirror, f), maps.get(f).text.replaceAll(FAKE_CLOCK, UNMASKED_CLOCK));
    }
    return oxlint(installed, mirror, mirrored).filter((d) => {
      if (ruleName(d.code) !== SLEEP_RULE) return false;
      const { text, map } = maps.get(d.file);
      const at = Buffer.from(text).subarray(0, d.labels?.[0]?.span?.offset ?? 0).toString('utf8').length;
      return !map || !map.fakeAt(at);
    });
  } finally {
    rmSync(mirror, { recursive: true, force: true });
  }
}

function lint(installed, dir, files) {
  const direct = oxlint(installed, dir, files);
  const seen = new Set(direct.map((d) => d.file + '\0' + d.code + '\0' + d.labels?.[0]?.span?.offset));
  const hidden = sleepsBehindFakeClocks(installed, dir, files).filter((d) => !seen.has(d.file + '\0' + d.code + '\0' + d.labels?.[0]?.span?.offset));
  return [...direct, ...hidden].map((d) => finding(dir, d)).sort((a, b) => a.file.localeCompare(b.file) || a.line - b.line);
}

function renames(top, mergeBase) {
  const parts = git(top, ['diff', '-M', '--name-status', '-z', mergeBase]).split('\0').filter(Boolean);
  const moved = new Map();
  for (let i = 0; i < parts.length; ) {
    const status = parts[i++];
    if (status.startsWith('R') || status.startsWith('C')) {
      const from = parts[i++];
      moved.set(parts[i++], from);
    } else i++;
  }
  return moved;
}

function checkoutBase(top, mergeBase) {
  const dir = mkdtempSync(join(tmpdir(), 'fleet-landmines-base-'));
  const index = join(dir, '.landmines-index');
  const env = { ...process.env, GIT_INDEX_FILE: index };
  try {
    git(top, ['read-tree', mergeBase], env);
    git(top, ['checkout-index', '--all', '--force', '--prefix=' + join(dir, 'tree') + sep], env);
  } catch (error) {
    rmSync(dir, { recursive: true, force: true });
    verdict('BLOCKED', 'the base ' + mergeBase.slice(0, 12) + ' could not be checked out: ' + lastLines(String(error.stderr || error.message)));
  }
  return { dir, tree: join(dir, 'tree') };
}

function report(findings) {
  for (const f of findings) console.log((f.severity === 'warning' ? 'warning ' : '') + f.file + ':' + f.line + ' ' + f.rule + ': ' + f.message);
}

const plural = (n, word) => n + ' ' + word + (n === 1 ? '' : 's');

function full(installed, top) {
  const files = trackedSources(top);
  const findings = lint(installed, top, files);
  const errors = findings.filter((f) => f.severity === 'error');
  report(findings);
  if (errors.length) verdict('FAIL', plural(errors.length, 'finding'));
  verdict('PASS', plural(files.length, 'file') + ' clean');
}

function baseFindings(installed, top, mergeBase, headFiles, moved) {
  const { dir, tree } = checkoutBase(top, mergeBase);
  try {
    const files = [...new Set(headFiles.map((f) => moved.get(f) ?? f))].filter((f) => linted(tree, f));
    const kept = new Map();
    for (const f of lint(installed, tree, files)) {
      const key = f.file + '\0' + f.rule + '\0' + f.text;
      kept.set(key, (kept.get(key) ?? 0) + 1);
    }
    return kept;
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

function comparisonPoint(top, base) {
  const mergeBase = tryGit(top, ['merge-base', base, 'HEAD'])?.trim();
  if (mergeBase) return mergeBase;
  const tip = tryGit(top, ['rev-parse', '--verify', '-q', base + '^{commit}'])?.trim();
  const shallow = tryGit(top, ['rev-parse', '--is-shallow-repository'])?.trim() === 'true';
  if (!tip || !shallow) verdict('BLOCKED', 'the base ' + base + ' cannot be read: git fetch it, or pass another --base');
  console.log('shallow checkout: comparing with the tip of ' + base + ', since the merge base is not in this history');
  return tip;
}

function ratchet(installed, top, base) {
  const mergeBase = comparisonPoint(top, base);
  const moved = renames(top, mergeBase);
  const headFiles = trackedSources(top);
  if (!headFiles.length) verdict('PASS', '0 files, 0 old findings kept');
  const head = lint(installed, top, headFiles);
  const kept = baseFindings(installed, top, mergeBase, headFiles, moved);
  const fresh = [];
  let keptCount = 0;
  for (const f of head) {
    const key = (moved.get(f.file) ?? f.file) + '\0' + f.rule + '\0' + f.text;
    const left = kept.get(key) ?? 0;
    if (left > 0) {
      kept.set(key, left - 1);
      if (f.severity === 'error') keptCount++;
    } else fresh.push(f);
  }
  report(fresh);
  const errors = fresh.filter((f) => f.severity === 'error');
  if (errors.length) verdict('FAIL', plural(errors.length, 'new finding'));
  verdict('PASS', plural(headFiles.length, 'file') + ', ' + plural(keptCount, 'old finding') + ' kept');
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!nodeIsRecentEnough()) verdict('BLOCKED', 'Node ' + process.versions.node + ' is too old: oxlint JS plugins need Node ' + MIN_NODE.join('.') + ' or newer');
  const top = tryGit(process.cwd(), ['rev-parse', '--show-toplevel'])?.trim();
  if (!top) verdict('BLOCKED', 'run it inside a git checkout');
  const installed = install();
  if (args.base) ratchet(installed, top, args.base);
  else full(installed, top);
}

try {
  main();
} catch (error) {
  verdict('BLOCKED', 'landmines stopped on an unexpected error: ' + lastLines(String(error?.stack ?? error)));
}