import { formatConfidence } from '../utils/appUtils';

const CRITICAL_REWRITE_FLAGS = new Set(['empty_output', 'number_mismatch', 'negation_mismatch', 'meaning_drift', 'subject_shift', 'repeated_phrase', 'transformer_unavailable']);

const EMOTION_VALENCE = {
  anger: 0,
  sadness: 0,
  fear: 0,
  surprise: 1,
  neutral: 2,
  joy: 3,
};

function Stat({ title, value }) {
  return (
    <article className="stat">
      <span>{title}</span>
      <strong>{value}</strong>
    </article>
  );
}

function ScoreBars({ title, scores, emptyLabel }) {
  const safeScores = Array.isArray(scores) ? scores : [];
  return (
    <section className="result-card full-width">
      <h3>{title}</h3>
      {safeScores.length === 0 ? <p className="muted">{emptyLabel}</p> : null}
      {safeScores.map((item) => (
        <div className="line" key={item.label}>
          <span>{item.label}</span>
          <div className="confidence-cell">
            <strong>{formatConfidence(item.score)}</strong>
            <div className="confidence-track" aria-hidden="true">
              <span className="confidence-fill" style={{ width: `${Math.round((item.score || 0) * 100)}%` }} />
            </div>
          </div>
        </div>
      ))}
    </section>
  );
}

function PercentageBars({ title, percentages, emptyLabel }) {
  const entries = Object.entries(percentages || {}).filter(([, value]) => typeof value === 'number');
  return (
    <section className="result-card full-width">
      <h3>{title}</h3>
      {entries.length === 0 ? <p className="muted">{emptyLabel}</p> : null}
      {entries.map(([label, value]) => (
        <div className="line" key={label}>
          <span>{label}</span>
          <div className="confidence-cell">
            <strong>{`${value.toFixed(1)}%`}</strong>
            <div className="confidence-track" aria-hidden="true">
              <span className="confidence-fill" style={{ width: `${Math.max(0, Math.round(value))}%` }} />
            </div>
          </div>
        </div>
      ))}
    </section>
  );
}

function getShiftStatus(beforeEmotion, afterEmotion) {
  const before = String(beforeEmotion || '').toLowerCase();
  const after = String(afterEmotion || '').toLowerCase();
  if (!Object.prototype.hasOwnProperty.call(EMOTION_VALENCE, before) || !Object.prototype.hasOwnProperty.call(EMOTION_VALENCE, after)) {
    return 'unknown';
  }
  if (EMOTION_VALENCE[after] > EMOTION_VALENCE[before]) return 'improved';
  if (EMOTION_VALENCE[after] < EMOTION_VALENCE[before]) return 'worsened';
  return 'unchanged';
}

function ShiftBadge({ status, ui }) {
  const labelMap = {
    improved: ui.shiftImproved,
    unchanged: ui.shiftUnchanged,
    worsened: ui.shiftWorsened,
    unknown: ui.shiftUnknown,
  };
  return <span className={`shift-badge ${status}`}>{labelMap[status] || ui.shiftUnknown}</span>;
}

function ComparisonChart({ title, beforeScores, afterScores, ui }) {
  const before = Array.isArray(beforeScores) ? beforeScores : [];
  const after = Array.isArray(afterScores) ? afterScores : [];
  const labelSet = new Set([...before.map((item) => item.label), ...after.map((item) => item.label)]);
  const beforeMap = new Map(before.map((item) => [item.label, Number(item.score || 0)]));
  const afterMap = new Map(after.map((item) => [item.label, Number(item.score || 0)]));
  const labels = Array.from(labelSet).sort((a, b) => String(a).localeCompare(String(b)));

  return (
    <section className="result-card full-width">
      <div className="result-card-heading">
        <h3>{title}</h3>
        <div className="chart-legend" aria-hidden="true">
          <span><i className="legend-dot before" />{ui.chartBeforeLegend}</span>
          <span><i className="legend-dot after" />{ui.chartAfterLegend}</span>
        </div>
      </div>
      {labels.length === 0 ? <p className="muted">{ui.resultNoScores}</p> : null}
      <div className="comparison-chart">
        {labels.map((label) => {
          const beforeValue = beforeMap.get(label) || 0;
          const afterValue = afterMap.get(label) || 0;
          return (
            <div className="comparison-row" key={label}>
              <span className="comparison-label">{label}</span>
              <div className="comparison-bars">
                <span className="comparison-bar before" style={{ width: `${Math.round(beforeValue * 100)}%` }} />
                <span className="comparison-bar after" style={{ width: `${Math.round(afterValue * 100)}%` }} />
              </div>
              <span className="comparison-values">{formatConfidence(beforeValue)} / {formatConfidence(afterValue)}</span>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function ResultView({ mode, result, ui }) {
  if (mode === 'predict') {
    const lowConfidence = result.low_confidence || false;
    const hasWarning = lowConfidence || result.model_source === 'lightweight';
    const explanationTokens = Array.isArray(result.explanation_tokens) ? result.explanation_tokens : [];
    const sentenceDistributions = Array.isArray(result.sentence_distributions) ? result.sentence_distributions : [];
    return (
      <div className="result-grid">
        {hasWarning ? (
          <section className="result-card full-width warning-card">
            <h3>⚠️ {ui.confidenceWarning || 'Confidence Notice'}</h3>
            {lowConfidence && <p>{ui.lowConfidenceMessage || 'The model confidence is below the threshold. Results may be less reliable.'}</p>}
            {result.model_source === 'lightweight' && <p>{ui.fallbackModelMessage || 'Using fallback heuristics instead of trained model. Results may be approximate.'}</p>}
          </section>
        ) : null}
        <Stat title={ui.resultEmotion} value={result.emotion} />
        <Stat title={ui.resultConfidence} value={formatConfidence(result.confidence)} />
        <Stat title={ui.resultModelSource} value={result.model_source || '-'} />
        <PercentageBars title={ui.resultSentimentPercentages} percentages={result.sentiment_percentages} emptyLabel={ui.resultNoScores} />
        {sentenceDistributions.length > 0 ? (
          <section className="result-card full-width">
            <h3>{ui.resultSentenceDistributions}</h3>
            <div className="chips">
              {sentenceDistributions.map((item, index) => (
                <span key={index} className="chip">
                  {ui.resultSentenceLabel} {index + 1}: {Object.entries(item || {})
                    .map(([label, value]) => `${label} ${value.toFixed(1)}%`)
                    .join(', ')}
                </span>
              ))}
            </div>
          </section>
        ) : null}
        <ScoreBars title={ui.resultTopScores} scores={result.scores} emptyLabel={ui.resultNoScores} />
        <section className="result-card full-width">
          <h3>{ui.resultExplanationTokens}</h3>
          {explanationTokens.length === 0 ? <p className="muted">{ui.resultNoExplanationTokens}</p> : null}
          <div className="chips">
            {explanationTokens.map((item, index) => (
              <span key={`${item.token}-${index}`} className="chip">
                {item.token} ({formatConfidence(item.importance)})
              </span>
            ))}
          </div>
        </section>
      </div>
    );
  }

  if (mode === 'rewrite') {
    const qualityFlags = Array.isArray(result.quality_flags) ? result.quality_flags : [];
    const lowConfidence = result.rewrite_confidence < 0.65;
    const hasCriticalQualityIssues = qualityFlags.some((flag) => CRITICAL_REWRITE_FLAGS.has(flag));
    const hasWarning = lowConfidence || hasCriticalQualityIssues;
    const hasInfoNote = !hasWarning && Boolean(result.fallback_reason);
    const hasSoftQualityNotes = qualityFlags.length > 0 && !hasCriticalQualityIssues;
    return (
      <div className="result-grid">
        {hasWarning ? (
          <section className="result-card full-width warning-card">
            <h3>⚠️ {ui.rewriteWarning || 'Rewrite quality alert'}</h3>
            {lowConfidence && <p>{ui.lowConfidenceRewriteMessage || 'Low rewrite confidence. The output may not fully preserve meaning.'}</p>}
            {hasCriticalQualityIssues && <p>{ui.qualityConcernsMessage || 'Detected critical quality concerns. Review the output carefully.'}</p>}
            {result.fallback_reason && <p>{ui.fallbackReasonMessage || `Used fallback: ${result.fallback_reason}`}</p>}
          </section>
        ) : null}
        {hasInfoNote ? (
          <section className="result-card full-width">
            <h3>{ui.rewriteInfoTitle || 'Rewrite note'}</h3>
            <p>{ui.rewriteInfoMessage || 'Rewrite completed with fallback heuristics.'}</p>
          </section>
        ) : null}
        <Stat title={ui.resultDetected} value={result.detected_emotion} />
        <Stat title={ui.resultAppliedTone} value={result.applied_tone} />
        <Stat title={ui.resultRewriteSource} value={result.rewrite_source || '-'} />
        <Stat title={ui.resultRewriteConfidence} value={formatConfidence(result.rewrite_confidence)} />
        <section className="result-card full-width">
          <h3>{ui.resultRewrittenText}</h3>
          <p>{result.rewritten_text}</p>
        </section>
        <section className="result-card full-width">
          <h3>{ui.resultRewriteQuality}</h3>
          {qualityFlags.length === 0 ? (
            <p className="muted">{ui.resultNoQualityFlags}</p>
          ) : (
            <>
              {hasSoftQualityNotes && !hasCriticalQualityIssues ? <p className="muted">{ui.qualityNotesMessage || 'Informational quality notes only; no critical issues were detected.'}</p> : null}
            <div className="chips">
              {qualityFlags.map((item) => (
                <span key={item} className="chip">{item}</span>
              ))}
            </div>
            </>
          )}
          {result.fallback_reason ? <p className="muted">{ui.resultFallbackReason}: {result.fallback_reason}</p> : null}
        </section>
      </div>
    );
  }

  if (mode === 'analyze') {
    const beforeEmotion = result?.prediction?.emotion || '-';
    const afterEmotion = result?.rewritten_prediction?.emotion || '-';
    const shiftStatus = getShiftStatus(beforeEmotion, afterEmotion);
    const explanationTokens = Array.isArray(result?.prediction?.explanation_tokens) ? result.prediction.explanation_tokens : [];
    const qualityFlags = Array.isArray(result?.rewrite?.quality_flags) ? result.rewrite.quality_flags : [];

    return (
      <div className="result-grid">
        <Stat title={ui.resultPrediction} value={beforeEmotion} />
        <Stat title={ui.resultFinalEmotion} value={afterEmotion} />
        <Stat title={ui.resultRewriteConfidence} value={formatConfidence(result?.rewrite?.rewrite_confidence)} />
        <article className="stat">
          <span>{ui.resultSentimentShift}</span>
          <div className="shift-wrap">
            <strong>{`${beforeEmotion} -> ${afterEmotion}`}</strong>
            <ShiftBadge status={shiftStatus} ui={ui} />
          </div>
        </article>
        <Stat title={ui.resultRewriteTone} value={result?.rewrite?.applied_tone || '-'} />
        <ComparisonChart
          title={ui.resultSentimentChartTitle}
          beforeScores={result?.prediction?.scores}
          afterScores={result?.rewritten_prediction?.scores}
          ui={ui}
        />
        <ScoreBars title={ui.resultBeforeScores} scores={result?.prediction?.scores} emptyLabel={ui.resultNoScores} />
        <ScoreBars title={ui.resultAfterScores} scores={result?.rewritten_prediction?.scores} emptyLabel={ui.resultNoScores} />
        <section className="result-card full-width">
          <h3>{ui.resultFinalRewrite}</h3>
          <p>{result?.rewrite?.rewritten_text}</p>
        </section>
        <section className="result-card full-width">
          <h3>{ui.resultExplanationTokens}</h3>
          {explanationTokens.length === 0 ? <p className="muted">{ui.resultNoExplanationTokens}</p> : null}
          <div className="chips">
            {explanationTokens.map((item, index) => (
              <span key={`${item.token}-${index}`} className="chip">
                {item.token} ({formatConfidence(item.importance)})
              </span>
            ))}
          </div>
        </section>
        <section className="result-card full-width">
          <h3>{ui.resultRewriteQuality}</h3>
          {qualityFlags.length === 0 ? (
            <p className="muted">{ui.resultNoQualityFlags}</p>
          ) : (
            <>
              {qualityFlags.some((flag) => CRITICAL_REWRITE_FLAGS.has(flag)) ? null : <p className="muted">{ui.qualityNotesMessage || 'Informational quality notes only; no critical issues were detected.'}</p>}
            <div className="chips">
              {qualityFlags.map((item) => (
                <span key={item} className="chip">{item}</span>
              ))}
            </div>
            </>
          )}
          {result?.rewrite?.fallback_reason ? <p className="muted">{ui.resultFallbackReason}: {result.rewrite.fallback_reason}</p> : null}
        </section>
      </div>
    );
  }

  return (
    <div className="result-grid">
      <Stat title={ui.resultEmotion} value={result.emotion} />
      <Stat title={ui.resultSuggestions} value={String((result.suggestions || []).length)} />
      <section className="result-card full-width">
        <h3>{ui.resultToneStrategy}</h3>
        <div className="chips">
          {(result.suggestions || []).map((item) => (
            <span key={item} className="chip">
              {item}
            </span>
          ))}
        </div>
      </section>
    </div>
  );
}

export default function ResultPanel({ mode, result, onCopy, loading, error, ui }) {
  if (loading) {
    return (
      <section className="card result-panel result-surface">
        <div className="head-row">
          <h2>{ui.resultTitle}</h2>
        </div>
        <div className="stack" aria-hidden="true">
          <div className="skeleton skeleton-title" />
          <div className="skeleton skeleton-line" />
          <div className="skeleton skeleton-line" />
          <div className="skeleton skeleton-line short" />
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="card result-panel result-surface">
        <div className="head-row">
          <h2>{ui.resultTitle}</h2>
        </div>
        <div className="error" role="alert">{error}</div>
      </section>
    );
  }

  if (!result) {
    return (
      <section className="card result-panel result-surface">
        <div className="head-row">
          <h2>{ui.resultTitle}</h2>
          <button type="button" className="ghost" onClick={onCopy}>
            {ui.resultCopy}
          </button>
        </div>
        <p className="muted">{ui.resultRunPrompt}</p>
      </section>
    );
  }

  return (
    <section className="card result-panel result-surface">
      <div className="head-row">
        <h2>{ui.resultTitle}</h2>
        <button type="button" className="ghost" onClick={onCopy}>
          {ui.resultCopy}
        </button>
      </div>
      <ResultView mode={mode} result={result} ui={ui} />
    </section>
  );
}

