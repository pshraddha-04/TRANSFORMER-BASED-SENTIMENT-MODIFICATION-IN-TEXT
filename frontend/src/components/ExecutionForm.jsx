import { EMOTIONS, TONES } from '../utils/appUtils';

export default function ExecutionForm({
  mode,
  activeMode,
  text,
  topK,
  targetTone,
  instruction,
  emotion,
  onText,
  onTopK,
  onTargetTone,
  onInstruction,
  onEmotion,
  onSubmit,
  onApplyTemplate,
  loading,
  error,
  ui,
}) {
  const textLength = text.trim().length;
  const needsText = mode !== 'tones';
  const disableSubmit = loading || (needsText && textLength === 0);
  const templates = [
    {
      id: 'customer-support',
      label: ui.templates.customerSupport,
      targetTone: 'professional',
      instruction: 'Use calm language, acknowledge issue, and propose next step in 2 sentences.',
    },
    {
      id: 'team-update',
      label: ui.templates.teamUpdate,
      targetTone: 'concise',
      instruction: 'Keep this to one short paragraph with direct action items.',
    },
    {
      id: 'empathetic',
      label: ui.templates.empatheticReply,
      targetTone: 'empathetic',
      instruction: 'Acknowledge emotion first, then respond with reassurance and clarity.',
    },
  ];

  return (
    <section className="card form-card">
      <div className="head-row">
        <div>
          <h2>{activeMode.label}</h2>
          <p className="muted">{activeMode.hint}</p>
        </div>
        <span className="badge">{mode}</span>
      </div>

      <form className="stack" onSubmit={onSubmit}>
        {mode !== 'tones' ? (
          <label>
            {ui.formMessage}
            <textarea
              rows="7"
              value={text}
              onChange={(event) => onText(event.target.value)}
              placeholder={ui.formMessagePlaceholder}
            />
            <small className="field-hint">
              {ui.formMessageHintPrefix}: {textLength}. {ui.formMessageHintSuffix}
            </small>
          </label>
        ) : null}

        {mode === 'predict' ? (
          <label>
            {ui.formTopK}
            <select value={topK} onChange={(event) => onTopK(Number(event.target.value))}>
              {[1, 3, 5, 7].map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>
        ) : null}

        {mode === 'rewrite' || mode === 'analyze' ? (
          <>
            <div className="template-row" role="group" aria-label={ui.formTemplatesLabel}>
              {templates.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className="chip template-chip"
                  onClick={() => onApplyTemplate(item)}
                >
                  {item.label}
                </button>
              ))}
            </div>
            <div className="two-col">
              <label>
                {ui.formTargetTone}
                <select value={targetTone} onChange={(event) => onTargetTone(event.target.value)}>
                  {TONES.map((tone) => (
                    <option key={tone} value={tone}>
                      {tone}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                {ui.formUserInstruction}
                <input
                  value={instruction}
                  onChange={(event) => onInstruction(event.target.value)}
                  placeholder={ui.formUserInstructionPlaceholder}
                />
                <small className="field-hint">{ui.formUserInstructionHint}</small>
              </label>
            </div>
          </>
        ) : null}

        {mode === 'tones' ? (
          <label>
            {ui.formEmotion}
            <select value={emotion} onChange={(event) => onEmotion(event.target.value)}>
              {EMOTIONS.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>
        ) : null}

        <button type="submit" disabled={disableSubmit}>
          {loading ? ui.formRunning : `${ui.formRun} ${activeMode.label}`}
        </button>
        {needsText && textLength === 0 ? <small className="field-hint error-text">{ui.formMessageRequired}</small> : null}
      </form>

      {error ? <div className="error" role="alert" aria-live="assertive">{error}</div> : null}
    </section>
  );
}

