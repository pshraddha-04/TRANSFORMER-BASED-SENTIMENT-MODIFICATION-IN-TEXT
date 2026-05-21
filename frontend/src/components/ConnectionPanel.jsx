export default function ConnectionPanel({
  healthState,
  healthMessage,
  notice,
  draftBaseUrl,
  draftApiKey,
  onDraftChange,
  onDraftApiKeyChange,
  onSave,
  onVerify,
  ui,
}) {
  return (
    <section className="card connection-card">
      <div className="head-row compact">
        <h2>{ui.connectionTitle}</h2>
        <span className="badge">{ui.connectionBackend}</span>
      </div>
      <div className={`status-pill ${healthState}`}>
        <span className="status-dot" />
        {ui.connectionStatus[healthState] || ui.connectionStatus.idle}
      </div>

      <p className="muted spaced">{healthMessage}</p>
      {notice.message ? <div className={`notice ${notice.kind}`}>{notice.message}</div> : null}

      <form className="stack" onSubmit={onSave}>
        <label>
          {ui.connectionApiLabel}
          <input
            value={draftBaseUrl}
            onChange={(event) => onDraftChange(event.target.value)}
            placeholder="/api or http://127.0.0.1:8000"
          />
          <small className="field-hint">{ui.connectionApiHint}</small>
        </label>
        <label>
          {ui.connectionApiKeyLabel}
          <input
            type="password"
            value={draftApiKey}
            onChange={(event) => onDraftApiKeyChange(event.target.value)}
            placeholder="optional"
            autoComplete="off"
          />
          <small className="field-hint">{ui.connectionApiKeyHint}</small>
        </label>
        <div className="inline-buttons">
          <button type="submit" className="ghost">
            {ui.connectionSave}
          </button>
          <button type="button" className="ghost" onClick={onVerify}>
            {ui.connectionVerify}
          </button>
        </div>
      </form>
    </section>
  );
}

