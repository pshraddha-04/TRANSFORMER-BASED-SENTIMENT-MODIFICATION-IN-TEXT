const DEFAULT_API_BASE_URL = import.meta?.env?.VITE_API_BASE_URL?.trim() || '/api';

function buildHeaders(options = {}) {
  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {}),
  };
  if (options.apiKey) {
    headers['X-API-Key'] = options.apiKey;
  }
  return headers;
}

function normalizeBaseUrl(baseUrl) {
  return baseUrl.trim().replace(/\/+$/, '');
}

function buildUrl(baseUrl, path) {
  const normalizedBase = normalizeBaseUrl(baseUrl);
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${normalizedBase}${normalizedPath}`;
}

async function parseResponse(response) {
  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    return response.json();
  }
  return response.text();
}

async function requestJson(baseUrl, path, options = {}) {
  const { apiKey, ...fetchOptions } = options;
  const response = await fetch(buildUrl(baseUrl, path), {
    ...fetchOptions,
    headers: buildHeaders({ ...fetchOptions, apiKey }),
  });

  const payload = await parseResponse(response);
  if (!response.ok) {
    const message =
      (payload && typeof payload === 'object' && (payload.detail || payload.message)) ||
      (typeof payload === 'string' && payload) ||
      response.statusText ||
      'Request failed';
    throw new Error(message);
  }

  return payload;
}

export function getDefaultApiBaseUrl() {
  return DEFAULT_API_BASE_URL;
}

export function getHealth(baseUrl = DEFAULT_API_BASE_URL, apiKey) {
  return requestJson(baseUrl, '/health', { apiKey });
}

export function predict(baseUrl, payload, apiKey) {
  return requestJson(baseUrl, '/predict', {
    method: 'POST',
    body: JSON.stringify(payload),
    apiKey,
  });
}

export function rewrite(baseUrl, payload, apiKey) {
  return requestJson(baseUrl, '/rewrite', {
    method: 'POST',
    body: JSON.stringify(payload),
    apiKey,
  });
}

export function analyzeRewrite(baseUrl, payload, apiKey) {
  return requestJson(baseUrl, '/analyze-rewrite', {
    method: 'POST',
    body: JSON.stringify(payload),
    apiKey,
  });
}

export function toneSuggestions(baseUrl, emotion, apiKey) {
  return requestJson(baseUrl, `/tones/suggestions?emotion=${encodeURIComponent(emotion)}`, { apiKey });
}

