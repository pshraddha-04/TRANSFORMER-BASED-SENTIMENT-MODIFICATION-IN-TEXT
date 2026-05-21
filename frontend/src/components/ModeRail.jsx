export default function ModeRail({ modes, activeMode, onChange, ui }) {
  return (
    <section className="card rail-card">
      <div className="head-row compact">
        <h2>{ui.workspaceTitle}</h2>
        <span className="badge">
          {modes.length} {ui.workspaceModeCount}
        </span>
      </div>
      <p className="muted">{ui.workspaceHint}</p>
      <div className="rail-list" role="tablist" aria-label="Operation mode selector">
        {modes.map((item, index) => (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={activeMode === item.id}
            className={item.id === activeMode ? 'rail-item mode-tab active is-active' : 'rail-item mode-tab'}
            onClick={() => onChange(item.id)}
          >
            <strong>
              {index + 1}. {item.label}
            </strong>
            <span>{item.hint}</span>
          </button>
        ))}
      </div>
    </section>
  );
}

