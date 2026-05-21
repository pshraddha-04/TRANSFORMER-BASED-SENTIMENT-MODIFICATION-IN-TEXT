import { useEffect, useMemo, useState } from 'react';

import {
  analyzeRewrite,
  getDefaultApiBaseUrl,
  getHealth,
  predict,
  rewrite,
  toneSuggestions,
} from './api';
import ConnectionPanel from './components/ConnectionPanel';
import ExecutionForm from './components/ExecutionForm';
import HistoryPanel from './components/HistoryPanel';
import ModeRail from './components/ModeRail';
import ResultPanel from './components/ResultPanel';
import { getDictionary, LOCALES } from './utils/i18n';
import { MODES, clampUrl, errorMessage, summarize } from './utils/appUtils';

function App() {
  const [mode, setMode] = useState('predict');
  const [apiBaseUrl, setApiBaseUrl] = useState(() => {
    if (typeof window === 'undefined') return getDefaultApiBaseUrl();
    return localStorage.getItem('sentiment-api-base-url') || getDefaultApiBaseUrl();
  });
  const [draftBaseUrl, setDraftBaseUrl] = useState(apiBaseUrl);
  const [apiKey, setApiKey] = useState(() => {
    if (typeof window === 'undefined') return '';
    return localStorage.getItem('sentiment-api-key') || '';
  });
  const [draftApiKey, setDraftApiKey] = useState(apiKey);

  const [healthState, setHealthState] = useState('idle');
  const [healthMessage, setHealthMessage] = useState('');
  const [notice, setNotice] = useState({ kind: 'info', message: '' });

  const [text, setText] = useState('I am upset because no one has responded to my ticket for three days.');
  const [topK, setTopK] = useState(3);
  const [targetTone, setTargetTone] = useState('professional');
  const [instruction, setInstruction] = useState('Keep it concise and respectful.');
  const [emotion, setEmotion] = useState('anger');

  const [runState, setRunState] = useState({ loading: false, error: '', result: null });
  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [theme, setTheme] = useState(() => (typeof window === 'undefined' ? 'dark' : localStorage.getItem('sentiment-ui-theme') || 'dark'));
  const [locale, setLocale] = useState(() => (typeof window === 'undefined' ? 'en' : localStorage.getItem('sentiment-ui-locale') || 'en'));

  const ui = useMemo(() => getDictionary(locale), [locale]);
  const localizedModes = useMemo(
    () =>
      MODES.map((item) => ({
        ...item,
        label: ui.modeLabels?.[item.id] || item.label,
        hint: ui.modeHints?.[item.id] || item.hint,
      })),
    [ui],
  );
  const activeMode = useMemo(() => localizedModes.find((item) => item.id === mode) || localizedModes[0], [mode, localizedModes]);

  useEffect(() => {
    verifyBackend(apiBaseUrl);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    document.body.setAttribute('data-theme', theme);
    localStorage.setItem('sentiment-ui-theme', theme);
  }, [theme]);

  useEffect(() => {
    localStorage.setItem('sentiment-ui-locale', locale);
  }, [locale]);

  useEffect(() => {
    setDraftApiKey(apiKey);
  }, [apiKey]);

  useEffect(() => {
    if (healthState === 'idle') {
      setHealthMessage(ui.healthMessageIdle);
    }
  }, [healthState, ui]);

  function formatText(template, values = {}) {
    return Object.entries(values).reduce(
      (text, [key, value]) => text.replace(`{${key}}`, String(value)),
      template,
    );
  }

  function flash(kind, message) {
    setNotice({ kind, message });
  }

  function addHistory(item) {
    setHistory((prev) => [{ id: Date.now(), ...item }, ...prev].slice(0, 10));
  }

  function clearHistory() {
    setHistory([]);
    flash('info', ui.flashes.historyCleared);
  }

  function useHistoryItem(item) {
    if (!item?.request) return;
    setMode(item.mode);
    if (item.request.text !== undefined) setText(item.request.text);
    if (item.request.top_k !== undefined) setTopK(item.request.top_k);
    if (item.request.target_tone !== undefined) setTargetTone(item.request.target_tone);
    if (item.request.user_instruction !== undefined) setInstruction(item.request.user_instruction || '');
    if (item.request.emotion !== undefined) setEmotion(item.request.emotion);
    flash('info', ui.flashes.loadedFromHistory);
  }

  function applyTemplate(template) {
    if (!template) return;
    if (template.text) setText(template.text);
    if (template.instruction) setInstruction(template.instruction);
    if (template.targetTone) setTargetTone(template.targetTone);
    flash('info', ui.flashes.templateApplied);
  }

  async function copyResult() {
    const textToCopy = runState.result ? JSON.stringify(runState.result, null, 2) : '';
    if (!textToCopy) {
      flash('error', ui.flashes.noResultToCopy);
      return;
    }

    try {
      await navigator.clipboard.writeText(textToCopy);
      flash('success', ui.flashes.resultCopied);
    } catch {
      flash('error', ui.flashes.copyFailed);
    }
  }

  async function verifyBackend(baseUrl = apiBaseUrl, providedApiKey = apiKey) {
    const normalized = clampUrl(baseUrl);
    const displayBaseUrl = normalized || 'backend';
    setHealthState('checking');
    setHealthMessage(formatText(ui.healthMessageChecking, { baseUrl: displayBaseUrl }));

    try {
      await getHealth(normalized, providedApiKey || undefined);
      setHealthState('ok');
      setHealthMessage(formatText(ui.healthMessageConnected, { baseUrl: displayBaseUrl }));
      flash('success', ui.flashes.backendVerified);
      return true;
    } catch (firstError) {
      if (normalized === '/api') {
        const fallback = 'http://127.0.0.1:8000';
        try {
          await getHealth(fallback, providedApiKey || undefined);
          setApiBaseUrl(fallback);
          setDraftBaseUrl(fallback);
          localStorage.setItem('sentiment-api-base-url', fallback);
          setHealthState('ok');
          setHealthMessage(formatText(ui.healthMessageFallbackConnected, { baseUrl: fallback }));
          flash('info', ui.flashes.autoSwitchedBaseUrl);
          return true;
        } catch {
          // fall through to error message
        }
      }

      const msg = errorMessage(firstError);
      setHealthState('error');
      setHealthMessage(msg);
      flash('error', msg);
      return false;
    }
  }

  async function saveBaseUrl(event) {
    event.preventDefault();
    const nextUrl = clampUrl(draftBaseUrl) || getDefaultApiBaseUrl();
    const nextApiKey = draftApiKey.trim();
    setApiBaseUrl(nextUrl);
    setApiKey(nextApiKey);
    localStorage.setItem('sentiment-api-base-url', nextUrl);
    localStorage.setItem('sentiment-api-key', nextApiKey);
    await verifyBackend(nextUrl, nextApiKey);
  }

  async function runMode(event) {
    event.preventDefault();
    setRunState({ loading: true, error: '', result: null });

    try {
      let result;
      let requestPayload;
      if (mode === 'predict') {
        requestPayload = { text, top_k: Number(topK) };
        result = await predict(apiBaseUrl, requestPayload, apiKey || undefined);
      } else if (mode === 'rewrite') {
        requestPayload = {
          text,
          target_tone: targetTone,
          user_instruction: instruction || null,
        };
        result = await rewrite(apiBaseUrl, requestPayload, apiKey || undefined);
      } else if (mode === 'analyze') {
        requestPayload = {
          text,
          target_tone: targetTone,
          user_instruction: instruction || null,
        };
        result = await analyzeRewrite(apiBaseUrl, requestPayload, apiKey || undefined);
      } else {
        requestPayload = { emotion };
        result = await toneSuggestions(apiBaseUrl, emotion, apiKey || undefined);
      }

      setRunState({ loading: false, error: '', result });
      flash('success', formatText(ui.flashes.modeCompleted, { mode: activeMode.label }));
      addHistory({
        mode,
        createdAt: Date.now(),
        summary: summarize(mode, result),
        result,
        request: requestPayload,
      });
    } catch (err) {
      const msg = errorMessage(err);
      setRunState({ loading: false, error: msg, result: null });
      flash('error', msg);
      addHistory({ mode, createdAt: Date.now(), summary: msg, result: null, isError: true });
    }
  }

  const successRuns = history.filter((item) => !item.isError).length;
  const failedRuns = history.filter((item) => item.isError).length;

  return (
    <div className="studio-shell">
      <header className="hero">
        <div className="hero-top-row">
          <div>
            <p className="eyebrow">Sentiment Transformation</p>
            <h1>{ui.appTitle}</h1>
            <p className="subtext">{ui.appSubtitle}</p>
          </div>
          <div className="hero-controls">
            <div className="hero-control">
              <label htmlFor="theme-select" className="hero-control-title">
                {ui.appearanceLabel}
              </label>
              <select
                id="theme-select"
                className="hero-control-select"
                value={theme}
                onChange={(event) => setTheme(event.target.value)}
              >
                <option value="dark">{ui.themeDark}</option>
                <option value="light">{ui.themeLight}</option>
              </select>
            </div>
            <div className="hero-control">
              <label htmlFor="locale-select" className="hero-control-title">
                {ui.languageLabel}
              </label>
              <select
                id="locale-select"
                className="hero-control-select"
                value={locale}
                onChange={(event) => setLocale(event.target.value)}
              >
                {LOCALES.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
        <div className="hero-metrics" role="status" aria-live="polite">
          <span className="metric-pill">{ui.modeLabel}: {activeMode.label}</span>
          <span className="metric-pill">{ui.runsLabel}: {history.length}</span>
          <span className="metric-pill">{ui.successLabel}: {successRuns}</span>
          <span className="metric-pill">{ui.errorsLabel}: {failedRuns}</span>
        </div>
      </header>

      <div className="studio-grid">
        <aside className="studio-sidebar">
          <ModeRail modes={localizedModes} activeMode={mode} onChange={setMode} ui={ui} />
        </aside>

        <main className="studio-main">
          <ExecutionForm
            mode={mode}
            activeMode={activeMode}
            text={text}
            topK={topK}
            targetTone={targetTone}
            instruction={instruction}
            emotion={emotion}
            onText={setText}
            onTopK={setTopK}
            onTargetTone={setTargetTone}
            onInstruction={setInstruction}
            onEmotion={setEmotion}
            onSubmit={runMode}
            loading={runState.loading}
            error={runState.error}
            onApplyTemplate={applyTemplate}
            ui={ui}
          />

          <ResultPanel mode={mode} result={runState.result} onCopy={copyResult} loading={runState.loading} error={runState.error} ui={ui} />
        </main>

        <aside className="studio-right">
          <ConnectionPanel
            healthState={healthState}
            healthMessage={healthMessage}
            notice={notice}
            draftBaseUrl={draftBaseUrl}
            draftApiKey={draftApiKey}
            onDraftChange={setDraftBaseUrl}
            onDraftApiKeyChange={setDraftApiKey}
            onSave={saveBaseUrl}
            onVerify={() => verifyBackend(draftBaseUrl, draftApiKey.trim())}
            ui={ui}
          />

          <HistoryPanel
            history={history}
            expanded={showHistory}
            onToggle={() => setShowHistory((prev) => !prev)}
            onClear={clearHistory}
            onUse={useHistoryItem}
            locale={locale}
            ui={ui}
          />
        </aside>
      </div>
    </div>
  );
}

export default App;

