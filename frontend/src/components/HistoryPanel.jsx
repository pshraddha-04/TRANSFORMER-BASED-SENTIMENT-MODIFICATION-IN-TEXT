import { useMemo, useState } from 'react';

export default function HistoryPanel({ history, expanded, onToggle, onClear, onUse, locale, ui }) {
  const [filter, setFilter] = useState('all');
  const filtered = useMemo(() => {
    if (filter === 'success') return history.filter((item) => !item.isError);
    if (filter === 'error') return history.filter((item) => item.isError);
    return history;
  }, [filter, history]);
  const modeCounts = useMemo(() => {
    const counts = history.reduce((acc, item) => {
      acc[item.mode] = (acc[item.mode] || 0) + 1;
      return acc;
    }, {});
    const max = Math.max(1, ...Object.values(counts));
    return Object.entries(counts).map(([name, count]) => ({ name, count, width: Math.round((count / max) * 100) }));
  }, [history]);
  const formatter = useMemo(
    () =>
      new Intl.DateTimeFormat(locale || 'en', {
        dateStyle: 'short',
        timeStyle: 'short',
      }),
    [locale],
  );

  function formatWhen(item) {
    if (item.createdAt) {
      return formatter.format(new Date(item.createdAt));
    }
    return item.when || '-';
  }

  return (
    <section className="card">
      <div className="head-row">
        <h2>{ui.historyTitle}</h2>
        <div className="inline-buttons">
          <button type="button" className="ghost" onClick={onToggle}>
            {expanded ? ui.historyHide : ui.historyShow}
          </button>
          <button type="button" className="ghost" onClick={onClear} disabled={history.length === 0}>
            {ui.historyClear}
          </button>
        </div>
      </div>

      {expanded ? (
        <div className="history-list">
          {modeCounts.length > 0 ? (
            <section className="result-card">
              <h3>{ui.historyTrendTitle}</h3>
              <div className="trend-list">
                {modeCounts.map((item) => (
                  <div key={item.name} className="trend-row">
                    <span>{ui.modeLabels?.[item.name] || item.name}</span>
                    <div className="trend-track" aria-hidden="true">
                      <span className="trend-fill" style={{ width: `${item.width}%` }} />
                    </div>
                    <strong>{item.count}</strong>
                  </div>
                ))}
              </div>
            </section>
          ) : null}
          <label>
            {ui.historyFilter}
            <select value={filter} onChange={(event) => setFilter(event.target.value)}>
              <option value="all">{ui.historyFilterAll}</option>
              <option value="success">{ui.historyFilterSuccess}</option>
              <option value="error">{ui.historyFilterError}</option>
            </select>
          </label>
          {filtered.length === 0 ? (
            <p className="muted">{ui.historyEmpty}</p>
          ) : (
            filtered.map((item) => (
              <article key={item.id} className={item.isError ? 'history-item error' : 'history-item'}>
                <div className="head-row compact">
                  <strong>{ui.modeLabels?.[item.mode] || item.mode}</strong>
                  <span>{formatWhen(item)}</span>
                </div>
                <p>{item.summary}</p>
                <div className="inline-buttons history-actions">
                  <button type="button" className="ghost" onClick={() => onUse(item)} disabled={!item.request}>
                    {ui.historyReuse}
                  </button>
                </div>
              </article>
            ))
          )}
        </div>
      ) : (
        <p className="muted">{ui.historyHidden}</p>
      )}
    </section>
  );
}

