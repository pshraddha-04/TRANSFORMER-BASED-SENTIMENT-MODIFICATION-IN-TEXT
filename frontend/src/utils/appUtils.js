export const MODES = [
  { id: 'predict', label: 'Emotion Detect', hint: 'Classifier output with confidence' },
  { id: 'rewrite', label: 'Tone Rewrite', hint: 'Rewrite text with target tone' },
  { id: 'analyze', label: 'Analyze + Rewrite', hint: 'Predict and rewrite in one request' },
  { id: 'tones', label: 'Tone Suggestions', hint: 'Recommend tones from emotion' },
];

export const TONES = ['professional', 'empathetic', 'friendly', 'concise', 'formal'];
export const EMOTIONS = ['anger', 'sadness', 'fear', 'joy', 'surprise', 'neutral'];

export function clampUrl(value) {
  return value.trim().replace(/\/+$/, '');
}

export function formatConfidence(value) {
  if (typeof value !== 'number') return '-';
  return `${(value * 100).toFixed(1)}%`;
}

export function errorMessage(error) {
  return error?.message || 'Request failed.';
}

export function summarize(mode, result) {
  if (mode === 'predict') {
    const top = result?.sentiment_percentages ? Object.entries(result.sentiment_percentages).sort((a, b) => b[1] - a[1])[0] : null;
    return `${result.emotion} (${formatConfidence(result.confidence)})${top ? ` • ${top[0]} ${top[1].toFixed(1)}%` : ''}`;
  }
  if (mode === 'rewrite') return `${result.detected_emotion} -> ${result.applied_tone} (${formatConfidence(result.rewrite_confidence)})`;
  if (mode === 'analyze') return `${result?.prediction?.emotion || '-'} -> ${result?.rewrite?.applied_tone || '-'} (${formatConfidence(result?.rewrite?.rewrite_confidence)})`;
  return `${(result.suggestions || []).length} suggestions for ${result.emotion}`;
}

