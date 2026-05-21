import { afterEach, test } from 'node:test';
import assert from 'node:assert/strict';

import {
  analyzeRewrite,
  getDefaultApiBaseUrl,
  getHealth,
  predict,
  rewrite,
  toneSuggestions,
} from '../src/api.js';
import { clampUrl, errorMessage, formatConfidence, summarize } from '../src/utils/appUtils.js';

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

function makeJsonResponse(body, { ok = true, status = 200, statusText = 'OK', contentType = 'application/json' } = {}) {
  return {
    ok,
    status,
    statusText,
    headers: {
      get(name) {
        return name.toLowerCase() === 'content-type' ? contentType : null;
      },
    },
    async json() {
      return body;
    },
    async text() {
      return typeof body === 'string' ? body : JSON.stringify(body);
    },
  };
}

test('utility helpers normalize and format values', () => {
  assert.equal(clampUrl('http://localhost:8000///'), 'http://localhost:8000');
  assert.equal(formatConfidence(0.5238), '52.4%');
  assert.equal(formatConfidence('n/a'), '-');
  assert.equal(errorMessage(new Error('Boom')), 'Boom');
  assert.equal(
    summarize('analyze', { prediction: { emotion: 'joy' }, rewrite: { applied_tone: 'professional' } }),
    'joy -> professional (-)',
  );
  assert.equal(getDefaultApiBaseUrl(), '/api');
});

test('request helpers call the correct endpoints with expected payloads', async () => {
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    return makeJsonResponse({ ok: true, url });
  };

  const health = await getHealth('http://example.com///', 'secret');
  const prediction = await predict('http://example.com', { text: 'Hello', top_k: 3 }, 'secret');
  const rewritten = await rewrite('http://example.com', { text: 'Hello', target_tone: 'professional' }, 'secret');
  const analyzed = await analyzeRewrite('http://example.com', { text: 'Hello', target_tone: 'professional' }, 'secret');
  const tones = await toneSuggestions('http://example.com', 'joy', 'secret');

  assert.deepEqual(health, { ok: true, url: 'http://example.com/health' });
  assert.deepEqual(prediction, { ok: true, url: 'http://example.com/predict' });
  assert.deepEqual(rewritten, { ok: true, url: 'http://example.com/rewrite' });
  assert.deepEqual(analyzed, { ok: true, url: 'http://example.com/analyze-rewrite' });
  assert.deepEqual(tones, { ok: true, url: 'http://example.com/tones/suggestions?emotion=joy' });

  assert.equal(calls.length, 5);
  assert.equal(calls[0].options.headers['X-API-Key'], 'secret');
  assert.equal(calls[1].options.method, 'POST');
  assert.equal(calls[1].options.headers['Content-Type'], 'application/json');
  assert.equal(calls[1].options.body, JSON.stringify({ text: 'Hello', top_k: 3 }));
  assert.equal(calls[4].options.headers['X-API-Key'], 'secret');
});

test('request helpers surface backend errors clearly', async () => {
  globalThis.fetch = async () =>
    makeJsonResponse({ detail: 'Unauthorized' }, { ok: false, status: 401, statusText: 'Unauthorized' });

  await assert.rejects(() => getHealth('http://example.com', 'bad-key'), /Unauthorized/);
});

test('toneSuggestions encodes emotion safely', async () => {
  globalThis.fetch = async (url) => makeJsonResponse({ url });

  const response = await toneSuggestions('http://example.com', 'very happy', undefined);
  assert.deepEqual(response, { url: 'http://example.com/tones/suggestions?emotion=very%20happy' });
});

