import { createServer } from 'node:http';
import { createHash } from 'node:crypto';

export function completionFor(text) {
  const digest = createHash('sha256').update(text).digest('hex').slice(0, 16);
  return `<|message_model|><|content_text|>reply ${digest}<|end_message|>`;
}

export function idsFor(completion) {
  return [...Buffer.from(completion, 'utf8')];
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

function send(res, status, value) {
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(JSON.stringify(value));
}

export async function startTarget({ modelPath }) {
  const generations = [];
  const server = createServer(async (req, res) => {
    if (req.method === 'GET' && req.url === '/health') return send(res, 200, {});
    if (req.method === 'GET' && req.url === '/model_info') return send(res, 200, { model_path: modelPath, is_generation: true });
    if (req.method === 'POST' && req.url === '/generate') {
      const body = JSON.parse(await readBody(req));
      const text = completionFor(body.text);
      const outputIds = idsFor(text);
      generations.push({ body, text, outputIds });
      return send(res, 200, {
        text,
        output_ids: outputIds,
        meta_info: { prompt_tokens: body.text.length, completion_tokens: outputIds.length, finish_reason: { type: 'stop', matched: 200006 } },
      });
    }
    send(res, 404, { error: 'not found' });
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address();
  return {
    url: `http://127.0.0.1:${port}`,
    generations,
    close: () => new Promise((resolve) => server.close(resolve)),
  };
}