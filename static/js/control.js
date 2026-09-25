/**
 * AI Day Trader - guided setup, API keys, and Paper / Live bot control.
 * Talks to /api/settings/* and /api/control/* (see config/api/settings.py, control.py).
 */
const Control = (() => {
  const $ = (id) => document.getElementById(id);
  const esc = (v) => {
    const d = document.createElement('div');
    d.textContent = v == null ? '' : String(v);
    return d.innerHTML;
  };
  const toast = (msg, type = 'info') => (typeof App !== 'undefined' && App.toast ? App.toast(msg, type) : alert(msg));
  const money = (v) => (v == null || isNaN(v) ? '—'
    : new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(v));
  const pct = (v) => (v == null || isNaN(v) ? '—' : (v >= 0 ? '+' : '') + Number(v).toFixed(2) + '%');
  const signCls = (v) => (v > 0 ? 'pos' : v < 0 ? 'neg' : '');
  const api = (method, path, body) => API.request(method, path, body);

  const ARM_PHRASE = 'I UNDERSTAND THIS USES REAL MONEY';
  const TIMEFRAMES = [
    ['5m', '5 minutes (recommended for day trading)'],
    ['15m', '15 minutes'],
    ['1m', '1 minute (fastest, noisiest)'],
    ['1h', '1 hour (swing)'],
    ['1d', '1 day (swing)'],
  ];

  const state = { view: 'start', overview: null, timers: [], user: null, built: {} };

  /* ── Login screen: first-run account creation ── */
  async function onLogin() {
    stopTimers();
    try {
      const s = await api('GET', '/settings/setup/status');
      const card = $('setup-card');
      if (s.needs_admin) {
        card.hidden = false;
        $('login-form').style.display = 'none';
        $('login-toggle').style.display = 'none';
        if (!s.local_request) {
          $('setup-error').textContent = 'Open this page on the computer running the bot (http://127.0.0.1:8000) to create the first account.';
          $('setup-error').style.display = 'block';
        }
      } else {
        card.hidden = true;
      }
    } catch { /* API down: the normal login form still shows */ }
  }

  async function handleSetup(e) {
    e.preventDefault();
    const err = $('setup-error');
    err.style.display = 'none';
    const username = $('setup-username').value.trim();
    const password = $('setup-password').value;
    try {
      await api('POST', '/settings/setup/admin', { username, email: $('setup-email').value.trim(), password });
      await API.login(username, password);
      window.location.reload();
    } catch (ex) {
      err.textContent = ex.message || 'Could not create the account';
      err.style.display = 'block';
    }
  }

  /* ── Navigation ── */
  function show(view) {
    state.view = view;
    document.querySelectorAll('#sidebar .nav-item').forEach((b) => b.classList.toggle('active', b.dataset.view === view));
    document.querySelectorAll('#views .view').forEach((v) => { v.hidden = v.id !== `view-${view}`; });
    try { localStorage.setItem('adt_view', view); } catch {}
    refreshActive();
  }

  function onDashboard(user) {
    state.user = user;
    buildModeView('paper');
    buildModeView('live');
    let saved = null;
    try { saved = localStorage.getItem('adt_view'); } catch {}
    show(saved && $(`view-${saved}`) ? saved : 'start');
    loadOverview();
    loadKeys();
    loadAutostart();
    stopTimers();
    state.timers.push(setInterval(loadOverview, 15000));
    state.timers.push(setInterval(refreshActive, 5000));
  }

  function stopTimers() {
    state.timers.forEach(clearInterval);
    state.timers = [];
  }

  function refreshActive() {
    if (state.view === 'paper' || state.view === 'live') loadMode(state.view);
    if (state.view === 'start') loadValidation();
  }

  /* ── Overview: checklist + sidebar badges ── */
  async function loadOverview() {
    try {
      const o = state.overview = await api('GET', '/control/overview');
      setStep('step-keys', o.keys.paper, o.keys.paper ? 'Paper keys connected' : null);
      setStep('step-validate', o.edge.passed, o.edge.exists && !o.edge.passed ? 'fail' : null);
      setStep('step-paper', o.bots.paper.running, null);
      setStep('step-live', o.bots.live.running, null);
      badge('badge-keys', o.keys.paper ? '✓' : '!', o.keys.paper ? 'ok' : 'warn');
      badge('badge-start', o.edge.passed ? '✓' : '', 'ok');
      badge('badge-paper', o.bots.paper.running ? 'ON' : '', 'ok');
      badge('badge-live', o.bots.live.running ? 'ON' : o.live_armed ? 'ARMED' : '', o.bots.live.running ? 'live' : 'warn');
      renderEdge(o.edge);
    } catch (ex) {
      if (ex.status === 403) toast('Only the owner (admin) account can control the bot.', 'error');
    }
  }

  function setStep(id, done, extra) {
    const el = $(id);
    if (!el) return;
    el.classList.toggle('done', !!done);
    el.classList.toggle('failed', extra === 'fail');
  }

  function badge(id, text, cls) {
    const el = $(id);
    if (!el) return;
    el.textContent = text;
    el.className = `nav-badge ${text ? cls : ''}`;
  }

  let reportFor = null;

  async function loadReport(edge) {
    const target = $('val-details');
    if (!target || !edge || !edge.exists) return;
    if (reportFor === edge.created_at && target.dataset.filled) return;
    target.innerHTML = '<div class="empty-state">Loading…</div>';
    reportFor = edge.created_at;
    try {
      const r = await api('GET', '/control/validate/report');
      const table = (rows, cols) => (rows.length
        ? `<table class="table"><thead><tr>${cols.map(([, h]) => `<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>
            ${rows.map((row) => `<tr>${cols.map(([k, , f]) => `<td>${esc(f ? f(row[k]) : row[k])}</td>`).join('')}</tr>`).join('')}</tbody></table>`
        : '<div class="empty-state">No data.</div>');
      const day = (v) => (v ? new Date(v).toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' }) : '');
      const hour = (v) => (v === 'n/a' ? v : `${v}:00 ET`);
      target.innerHTML = `
        <h4>Each test period (the model never saw these days before trading them)</h4>
        ${table(r.folds, [['fold', 'Period'], ['start', 'From', day], ['end', 'To', day], ['trades', 'Trades'],
          ['win_rate_pct', 'Win %'], ['avg_r', 'Avg R'], ['return_pct', 'Return %'], ['buy_hold_pct', 'Buy & hold %'], ['max_dd_pct', 'Max drawdown %']])}
        <h4>By time of day</h4>
        ${table(r.by_entry_hour, [['entry_hour_et', 'Entry hour', hour], ['trades', 'Trades'], ['win_rate_pct', 'Win %'],
          ['avg_r', 'Avg R'], ['total_pnl', 'P&L $'], ['share_of_pnl_pct', 'Share of P&L %']])}
        <h4>By market regime</h4>
        ${table(r.by_regime, [['regime', 'Regime'], ['trades', 'Trades'], ['win_rate_pct', 'Win %'], ['avg_r', 'Avg R'],
          ['total_pnl', 'P&L $']])}
        <p class="muted small">Look for an edge that shows up in most periods and hours, not one lucky stretch.</p>`;
      target.dataset.filled = '1';
    } catch { reportFor = null; }
  }

  function renderEdge(edge) {
    const box = $('val-result');
    if (!box) return;
    const key = edge && edge.exists ? `${edge.created_at}|${edge.passed}` : '';
    if (box.dataset.key === key) return;  // unchanged: keep the open details panel as it is
    box.dataset.key = key;
    if (!edge || !edge.exists) { box.innerHTML = ''; return; }
    const m = edge.metrics || {};
    const when = edge.created_at ? new Date(edge.created_at).toLocaleString() : '';
    const stats = `${esc(m.trades)} trades · profit factor ${esc(m.profit_factor)} · avg ${esc(m.avg_r)}R per trade ·
      ${esc(m.positive_folds)}/${esc(m.folds)} test periods profitable · worst drawdown ${esc(m.max_drawdown_pct)}%`;
    box.innerHTML = edge.passed
      ? `<div class="notice ok"><b>✓ Strategy validated</b> on ${esc(edge.timeframe)} bars (${esc(when)}).<br>${stats}</div>`
      : `<div class="notice bad"><b>✗ No reliable edge found</b> (${esc(when)}). The bot will not open trades with this setup.
          <ul>${(edge.failures || []).map((f) => `<li>${esc(f)}</li>`).join('')}</ul>
          <span class="muted">Try other liquid stocks or more history, but don't keep tweaking until something passes: that finds luck, not an edge.</span></div>`;
    if (!box.querySelector('#val-details-box')) {
      box.insertAdjacentHTML('beforeend', '<details class="journal" id="val-details-box"><summary>See the test details</summary><div id="val-details"></div></details>');
    }
    loadReport(edge);
  }

  /* ── Validation job ── */
  async function startValidation() {
    const symbols = $('val-symbols').value.split(/[\s,]+/).filter(Boolean);
    try {
      await api('POST', '/control/validate', {
        symbols, timeframe: $('val-timeframe').value, days: Number($('val-days').value), market: $('val-market').checked,
      });
      $('val-log-box').open = true;
      toast('Validation started. This usually takes a few minutes.', 'info');
      loadValidation();
    } catch (ex) { toast(ex.message, 'error'); }
  }

  async function loadValidation() {
    try {
      const v = await api('GET', '/control/validate');
      $('val-start').disabled = v.running;
      $('val-start').textContent = v.running ? '⏳ Validating…' : '▶ Validate strategy';
      $('val-stop').hidden = !v.running;
      const log = $('val-log');
      const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 10;
      log.textContent = (v.log || []).join('\n') || 'No validation run yet.';
      if (atBottom) log.scrollTop = log.scrollHeight;
      renderEdge(v.edge);
    } catch { /* ignore */ }
  }

  /* ── API keys ── */
  async function loadKeys() {
    try { renderKeys(await api('GET', '/settings/keys')); } catch { /* not admin */ }
  }

  function renderKeys(k) {
    for (const mode of ['paper', 'live']) {
      const pill = $(`${mode}-key-status`);
      pill.textContent = k[mode].configured ? `saved · ${k[mode].key_id_hint}` : 'not set';
      pill.className = `pill ${k[mode].configured ? 'ok' : ''}`;
    }
    $('alerts-status').textContent = k.alerts.configured ? 'on' : 'off';
    $('alerts-status').className = `pill ${k.alerts.configured ? 'ok' : ''}`;
    $('env-file-note').textContent = `Stored in ${k.env_file}`;
  }

  async function saveKeys(mode) {
    const id = $(`${mode}-key-id`).value.trim();
    const secret = $(`${mode}-secret`).value.trim();
    const out = $(`${mode}-key-result`);
    if (!id || !secret) { out.innerHTML = '<div class="notice bad">Paste both the Key ID and the Secret Key.</div>'; return; }
    try {
      const res = await api('PUT', '/settings/keys', { [`${mode}_key_id`]: id, [`${mode}_secret_key`]: secret });
      $(`${mode}-key-id`).value = '';
      $(`${mode}-secret`).value = '';
      renderKeys(res);
      out.innerHTML = `<div class="notice ok">Saved. Testing the connection…</div>`
        + (res.warnings || []).map((w) => `<div class="notice warn">${esc(w)}</div>`).join('');
      await testKeys(mode, true);
      loadOverview();
    } catch (ex) { out.innerHTML = `<div class="notice bad">${esc(ex.message)}</div>`; }
  }

  async function testKeys(mode, append = false) {
    const out = $(`${mode}-key-result`);
    try {
      const r = await api('POST', `/settings/test/${mode}`);
      const html = r.ok
        ? `<div class="notice ok">✓ ${esc(r.message)} Account ${esc(r.account_number_hint)} · equity ${money(r.equity)} · buying power ${money(r.buying_power)}${r.trading_blocked ? ' · <b>trading blocked by Alpaca</b>' : ''}</div>`
        : `<div class="notice bad">✗ ${esc(r.message)}</div>`;
      out.innerHTML = append ? out.innerHTML.replace(/<div class="notice ok">Saved\. Testing the connection…<\/div>/, '') + html : html;
    } catch (ex) { out.innerHTML = `<div class="notice bad">${esc(ex.message)}</div>`; }
  }

  async function removeKeys(group) {
    const label = group === 'alerts' ? 'the alert webhook' : `your ${group} keys`;
    if (!confirm(`Remove ${label} from this computer?`)) return;
    try { renderKeys(await api('DELETE', `/settings/keys/${group}`)); toast(`Removed ${label}.`, 'success'); loadOverview(); }
    catch (ex) { toast(ex.message, 'error'); }
  }

  async function loadAutostart() {
    try { renderAutostart(await api('GET', '/settings/autostart')); } catch {}
  }

  function renderAutostart(a) {
    $('autostart-toggle').checked = !!a.enabled;
    $('autostart-status').textContent = a.enabled ? 'starts at login' : 'manual start';
    $('autostart-status').className = `pill ${a.enabled ? 'ok' : ''}`;
    $('autostart-path').textContent = a.enabled ? `Login item: ${a.path}` : '';
  }

  async function toggleAutostart(e) {
    try {
      renderAutostart(await api('PUT', '/settings/autostart', { enabled: e.target.checked }));
      toast(e.target.checked ? 'The dashboard will start when you log in.' : 'Login start turned off.', 'success');
    } catch (ex) { e.target.checked = !e.target.checked; toast(ex.message, 'error'); }
  }

  async function saveAlerts() {
    const url = $('alert-url').value.trim();
    if (!url) return;
    try { renderKeys(await api('PUT', '/settings/keys', { alert_webhook_url: url })); $('alert-url').value = ''; toast('Alerts on.', 'success'); }
    catch (ex) { toast(ex.message, 'error'); }
  }

  /* ── Paper / Live views ── */
  function buildModeView(mode) {
    if (state.built[mode]) return;
    state.built[mode] = true;
    const live = mode === 'live';
    const tfOptions = TIMEFRAMES.map(([v, l]) => `<option value="${v}">${esc(l)}</option>`).join('');
    $(`view-${mode}`).innerHTML = `
      <div class="mode-banner ${mode}">
        ${live ? '💵 <b>LIVE TRADING: REAL MONEY.</b> Orders here buy and sell real stocks with your money.'
               : '🧪 <b>PAPER TRADING: fake money.</b> A safe practice account: nothing here costs real money.'}
      </div>
      <div class="view-inner">
        <div class="mode-head">
          <h1 class="view-title">${live ? 'Live trading' : 'Paper trading'}</h1>
          <span class="pill big" id="${mode}-run-pill">…</span>
        </div>

        ${live ? `<div class="card" id="live-arm-card"></div>` : ''}

        <div class="grid-2">
          <div class="card">
            <h3>Auto-trader</h3>
            <div id="${mode}-gate-note"></div>
            <label class="field"><span>Stocks to trade</span>
              <input class="form-input mono" id="${mode}-symbols" value="AAPL,MSFT,NVDA,AMD,META,AMZN,GOOGL,TSLA"></label>
            <label class="field"><span>Bar size</span><select class="form-input" id="${mode}-timeframe">${tfOptions}</select></label>
            <div class="radio-row">
              <label class="check"><input type="radio" name="${mode}-exec" value="1" checked> Place ${live ? 'real' : 'paper'} orders</label>
              <label class="check"><input type="radio" name="${mode}-exec" value="0"> Watch only (signals, no orders)</label>
            </div>
            <div class="btn-row">
              <button class="btn ${live ? 'btn-danger' : 'btn-success'}" id="${mode}-start">▶ Start bot</button>
              <button class="btn btn-outline" id="${mode}-stop">■ Stop bot</button>
            </div>
            <p class="muted small">Stopping finishes the current check and exits. Open positions keep their broker-side stop-loss
              and take-profit. In day-trading mode everything is closed by 3:50 pm ET.</p>
            <div id="${mode}-heartbeat" class="heartbeat"></div>
          </div>

          <div class="card">
            <h3>Account</h3>
            <div id="${mode}-account"><div class="empty-state">Loading…</div></div>
          </div>
        </div>

        <div class="card">
          <h3>Open positions</h3>
          <div id="${mode}-positions"><div class="empty-state">No positions</div></div>
        </div>

        <div class="card perf-card">
          <div class="card-head">
            <h3>Performance</h3>
            <div class="seg" role="group" aria-label="Time range" id="${mode}-periods">
              ${['1D', '1W', '1M', '3M', '1Y'].map((p) => `<button type="button" data-period="${p}" class="${p === '1M' ? 'active' : ''}">${p}</button>`).join('')}
            </div>
          </div>
          <div class="chart-wrap" id="${mode}-equity"><div class="empty-state">Loading…</div></div>
          <div id="${mode}-verdict"></div>
          <div class="tiles" id="${mode}-tiles"></div>
          <details class="journal">
            <summary>Trade journal <span class="muted small" id="${mode}-journal-count"></span></summary>
            <div id="${mode}-journal"></div>
            <button type="button" class="btn btn-outline btn-sm" id="${mode}-csv">Download all trades (CSV)</button>
          </details>
        </div>

        ${live ? '' : `
        <div class="card">
          <h3>Manual paper order</h3>
          <p class="muted small">Buys go through the same risk checks and get an automatic stop-loss and take-profit.</p>
          <div class="form-row">
            <label class="field"><span>Symbol</span><input class="form-input mono" id="paper-order-symbol" maxlength="10" placeholder="AAPL"></label>
            <label class="field"><span>Side</span><select class="form-input" id="paper-order-side"><option>BUY</option><option>SELL</option></select></label>
            <label class="field"><span>Shares</span><input class="form-input" type="number" min="1" step="1" id="paper-order-qty" value="1"></label>
            <div class="field end"><button class="btn btn-primary" id="paper-order-submit">Submit paper order</button></div>
          </div>
        </div>`}

        <div class="card">
          <div class="card-head"><h3>Activity</h3></div>
          <ul class="events" id="${mode}-events"><li class="empty-state">Nothing yet</li></ul>
        </div>

        <details class="card log-box">
          <summary>Bot log (technical)</summary>
          <pre class="log" id="${mode}-log"></pre>
        </details>

        <div class="card danger-zone">
          <h3>Emergency</h3>
          <p class="muted small">Stops the bot, cancels every open order and sells every position at market in this
            ${live ? '<b>live</b>' : 'paper'} account.</p>
          <button class="btn btn-danger" id="${mode}-flatten">Close everything now</button>
        </div>
      </div>`;

    $(`${mode}-start`).addEventListener('click', () => startBot(mode));
    $(`${mode}-stop`).addEventListener('click', () => stopBot(mode));
    $(`${mode}-flatten`).addEventListener('click', () => flatten(mode));
    $(`${mode}-periods`).addEventListener('click', (e) => {
      const b = e.target.closest('[data-period]');
      if (!b) return;
      perf[mode].period = b.dataset.period;
      $(`${mode}-periods`).querySelectorAll('button').forEach((x) => x.classList.toggle('active', x === b));
      loadPerformance(mode, true);
    });
    $(`${mode}-csv`).addEventListener('click', () => downloadCsv(mode));
    if (!live) $('paper-order-submit').addEventListener('click', submitPaperOrder);
    try {
      const saved = JSON.parse(localStorage.getItem(`adt_${mode}_settings`) || 'null');
      if (saved) { $(`${mode}-symbols`).value = saved.symbols; $(`${mode}-timeframe`).value = saved.timeframe; }
    } catch {}
  }

  const lastAccountLoad = { paper: 0, live: 0 };

  async function loadMode(mode) {
    try {
      const s = await api('GET', `/control/${mode}/status`);
      renderStatus(mode, s);
    } catch (ex) { if (ex.status === 403) return; }
    loadPerformance(mode);
    if (Date.now() - lastAccountLoad[mode] > 15000) {
      lastAccountLoad[mode] = Date.now();
      try { renderAccount(mode, await api('GET', `/control/${mode}/account`)); } catch {}
    }
  }

  function renderStatus(mode, s) {
    const live = mode === 'live';
    const pill = $(`${mode}-run-pill`);
    const running = s.running;
    const execute = s.settings && s.settings.execute;
    pill.textContent = running ? (execute ? '● RUNNING, placing orders' : '● RUNNING, watch only') : '○ Stopped';
    pill.className = `pill big ${running ? (live && execute ? 'live' : 'ok') : ''}`;
    $(`${mode}-start`).disabled = running || !s.keys_configured || (live && !s.live_armed);
    $(`${mode}-stop`).disabled = !running;

    const o = state.overview;
    const notes = [];
    if (!s.keys_configured) notes.push(`<div class="notice warn">Add your ${mode} API keys on the <a href="#" data-goto="keys">API Keys</a> tab first.</div>`);
    if (o && !o.edge.passed) notes.push(`<div class="notice warn">The strategy isn't validated yet, so the bot ${live ? "can't trade real money" : 'will only watch and manage exits'}.
      <a href="#" data-goto="start">Validate it on Get Started</a>.</div>`);
    if (live && !s.live_armed) notes.push('<div class="notice warn">Live trading is not armed (see above).</div>');
    $(`${mode}-gate-note`).innerHTML = notes.join('');

    const hb = s.heartbeat;
    $(`${mode}-heartbeat`).innerHTML = running && hb
      ? `Last check ${esc(hb.age_seconds)}s ago · market phase <b>${esc(hb.phase)}</b>${hb.entry_block ? ` · new entries paused: ${esc(hb.entry_block)}` : ''}
         ${hb.age_seconds > 900 ? '<div class="notice bad">No heartbeat for 15+ minutes: check the log.</div>' : ''}`
      : (s.exit_code != null && s.exit_code !== 0 ? `<div class="notice bad">The bot exited with an error (code ${esc(s.exit_code)}). Check the log below.</div>` : '');

    if (running && s.settings && s.settings.symbols) {
      $(`${mode}-symbols`).value = s.settings.symbols.join(',');
      $(`${mode}-timeframe`).value = s.settings.timeframe;
    }
    const log = $(`${mode}-log`);
    log.textContent = (s.log || []).join('\n') || 'No log yet.';
    renderEvents(mode, s.events || []);
    if (live) renderArm(s);
  }

  function renderArm(s) {
    const card = $('live-arm-card');
    if (!card) return;
    if (s.live_armed) {
      if (card.dataset.state === 'armed') return;
      card.dataset.state = 'armed';
      card.className = 'card armed';
      card.innerHTML = `<div class="card-head"><h3>🔓 Live trading is ARMED</h3>
          <button class="btn btn-outline" id="live-disarm">Disarm (turn real-money trading off)</button></div>
        <p class="muted small">Disarming stops the live bot and blocks new live orders. Open positions keep their stop-losses.</p>`;
      $('live-disarm').addEventListener('click', disarm);
      return;
    }
    if (card.dataset.state === 'disarmed') return;
    card.dataset.state = 'disarmed';
    card.className = 'card arm';
    card.innerHTML = `
      <h3>🔒 Arm live trading</h3>
      <ul class="warn-list">
        <li>You can lose money, including more than you expect on gaps or fast markets.</li>
        <li>Past results (even validated ones) don't guarantee future results.</li>
        <li>Paper trade first. Start with a small amount you can afford to lose.</li>
        <li>With under $25,000 in the account, US pattern-day-trader rules limit you to 3 day trades per 5 days (the bot respects this).</li>
      </ul>
      <label class="field"><span>Type <b class="mono">${esc(ARM_PHRASE)}</b></span>
        <input class="form-input mono" id="arm-phrase" autocomplete="off" spellcheck="false"></label>
      <label class="field"><span>Your dashboard password</span>
        <input class="form-input" type="password" id="arm-password" autocomplete="current-password"></label>
      <button class="btn btn-danger" id="arm-btn">Arm live trading</button>`;
    $('arm-btn').addEventListener('click', arm);
  }

  function renderAccount(mode, a) {
    const box = $(`${mode}-account`);
    if (!a.connected) { box.innerHTML = `<div class="empty-state">${esc(a.message)}</div>`; renderPositions(mode, []); return; }
    box.innerHTML = `
      <div class="stats">
        <div><span>Equity</span><b>${money(a.equity)}</b></div>
        <div><span>Today</span><b class="${signCls(a.day_pnl)}">${money(a.day_pnl)} <small>${pct(a.day_pnl_pct)}</small></b></div>
        <div><span>Buying power</span><b>${money(a.buying_power)}</b></div>
        <div><span>Market</span><b>${a.market_open ? '<span class="pos">Open</span>' : 'Closed'}</b></div>
      </div>
      <p class="muted small">${a.open_orders} open order(s) · day trades (5 days): ${esc(a.daytrade_count ?? '—')}
        ${a.pattern_day_trader ? ' · <b>flagged pattern day trader</b>' : ''}${a.trading_blocked ? ' · <b class="neg">trading blocked by Alpaca</b>' : ''}</p>`;
    renderPositions(mode, a.positions || []);
  }

  function renderPositions(mode, positions) {
    const box = $(`${mode}-positions`);
    if (!positions.length) { box.innerHTML = '<div class="empty-state">No open positions</div>'; return; }
    box.innerHTML = `<table class="table positions-table"><thead><tr><th>Symbol</th><th>Shares</th><th>Avg cost</th><th>Price</th><th>Value</th><th>P&amp;L</th></tr></thead><tbody>
      ${positions.map((p) => `<tr><td class="mono">${esc(p.symbol)}</td><td>${esc(p.qty)}</td><td>${money(p.avg_entry_price)}</td>
        <td>${money(p.current_price)}</td><td>${money(p.market_value)}</td>
        <td class="${signCls(p.unrealized_pl)}">${money(p.unrealized_pl)} <small>${pct(p.unrealized_plpc)}</small></td></tr>`).join('')}
      </tbody></table>`;
  }

  function describeEvent(e) {
    const sym = e.symbol ? ` ${e.symbol}` : '';
    switch (e.event) {
      case 'order_submitted': return ['🟢', `${(e.side || '').toUpperCase()} ${e.qty}${sym} submitted${e.stop_loss ? ` (stop ${e.stop_loss}, target ${e.take_profit})` : ''}`];
      case 'order_filled': return ['✅', `${(e.side || '').toUpperCase()} ${e.qty}${sym} filled at ${e.fill_price}${e.slippage_bps != null ? ` (${e.slippage_bps > 0 ? '+' : ''}${e.slippage_bps} bps slippage)` : ''}`];
      case 'order_rejected': return ['🔴', `${sym} order rejected: ${e.message || e.category}`];
      case 'order_recovered': return ['🟡', `${sym} order retried after ${e.category}`];
      case 'entry_skipped_spread': return ['↔️', `${sym} skipped: spread ${e.spread_bps} bps too wide`];
      case 'entry_expired': return ['⌛', `${sym} unfilled entry cancelled`];
      case 'flatten': return ['🌙', `Closed all positions (${e.reason})${(e.failures || []).length ? ' with FAILURES' : ''}`];
      case 'breaker_tripped': return ['🛑', e.reason];
      case 'edge_gate': return ['🚧', `New trades blocked: ${e.reason}`];
      case 'edge_decay': return ['📉', `Live results fell short of the test; new trades paused: ${e.reason}`];
      case 'session_report': return ['📊', `Day summary: P&L ${money(e.pnl)} (${pct(e.pnl_pct)}), ${e.fills} fills`];
      case 'reconciliation': return (e.discrepancies || []).length ? ['⚠️', 'Broker and local records differed (auto-synced)'] : null;
      case 'trailing_stop': return ['⤴️', `${sym} stop raised to ${e.new_stop}`];
      case 'trailing_stop_failed': return ['⚠️', `${sym} could not raise stop`];
      case 'bot_started': return ['▶️', 'Bot started'];
      case 'bot_stopped': return ['⏹️', `Bot stopped (${e.reason || 'requested'})`];
      case 'bot_error': return ['❗', `Error: ${e.message}`];
      case 'bot_restarted': return ['🔁', 'Bot restarted automatically'];
      case 'bot_restart_blocked': return ['⚠️', `Bot is down and was not restarted: ${e.reason}`];
      default: return null;
    }
  }

  function renderEvents(mode, events) {
    const items = events.map((e) => [e, describeEvent(e)]).filter(([, d]) => d).slice(0, 30);
    $(`${mode}-events`).innerHTML = items.length
      ? items.map(([e, [icon, text]]) => `<li><span class="ev-icon">${icon}</span><span class="ev-text">${esc(text)}</span>
          <span class="ev-time">${e.ts ? esc(new Date(e.ts).toLocaleTimeString()) : ''}</span></li>`).join('')
      : '<li class="empty-state">Nothing yet. Activity appears here once the bot runs during market hours.</li>';
  }

  /* ── Performance ── */
  const perf = { paper: { period: '1M', at: 0 }, live: { period: '1M', at: 0 } };

  async function loadPerformance(mode, force = false) {
    if (!force && Date.now() - perf[mode].at < 60000) return;
    perf[mode].at = Date.now();
    try {
      const p = await api('GET', `/control/${mode}/performance?period=${perf[mode].period}`);
      perf[mode].data = p;
      renderPerformance(mode, p);
    } catch { perf[mode].at = 0; }
  }

  function renderPerformance(mode, p) {
    drawEquity($(`${mode}-equity`), p.equity || [], p.equity_message);
    const c = p.comparison || {};
    const cls = { on_track: 'ok', behind: 'bad', watch: 'warn' }[c.status] || '';
    $(`${mode}-verdict`).innerHTML = c.message
      ? `<div class="notice ${cls}"><b>${esc({ on_track: 'On track', behind: 'Behind the test', watch: 'Keep watching',
          too_early: 'Too early to tell', no_backtest: 'No test to compare' }[c.status] || 'Live vs test')}:</b> ${esc(c.message)}</div>` : '';
    const l = p.live || {};
    const b = p.backtest || {};
    const tile = (label, value, sub) => `<div class="tile"><span>${esc(label)}</span><b>${value}</b>${sub ? `<small>${sub}</small>` : ''}</div>`;
    const num = (v, d = 2) => (v == null ? '—' : Number(v).toFixed(d));
    $(`${mode}-tiles`).innerHTML = [
      tile('Bot trades closed', esc(l.trades ?? 0)),
      tile('Bot P&L', `<span class="${signCls(l.total_pnl)}">${money(l.total_pnl)}</span>`),
      tile('Win rate', l.win_rate_pct == null ? '—' : `${esc(l.win_rate_pct)}%`, b.win_rate_pct != null ? `test ${esc(b.win_rate_pct)}%` : ''),
      tile('Avg per trade', l.avg_r == null ? '—' : `${l.avg_r >= 0 ? '+' : ''}${num(l.avg_r)}R`, b.avg_r != null ? `test ${b.avg_r >= 0 ? '+' : ''}${num(b.avg_r)}R` : ''),
      tile('Profit factor', l.profit_factor == null ? (l.trades && !l.losing_trades ? 'no losses yet' : '—') : num(l.profit_factor),
        b.profit_factor != null ? `test ${num(b.profit_factor)}` : ''),
    ].join('');
    const trades = p.recent_trades || [];
    $(`${mode}-journal-count`).textContent = trades.length ? `(latest ${trades.length})` : '(none yet)';
    $(`${mode}-journal`).innerHTML = trades.length
      ? `<table class="table journal-table"><thead><tr><th>Symbol</th><th>Shares</th><th>Entry</th><th>Exit</th><th>P&amp;L</th><th>R</th><th>Closed</th></tr></thead><tbody>
        ${trades.map((t) => `<tr><td class="mono">${esc(t.symbol)}</td><td>${esc(t.qty)}</td><td>${money(t.entry_price)}</td>
          <td>${money(t.exit_price)}</td><td class="${signCls(t.pnl)}">${money(t.pnl)}</td>
          <td>${t.r_multiple == null ? '—' : (t.r_multiple >= 0 ? '+' : '') + Number(t.r_multiple).toFixed(2)}</td>
          <td>${t.exit_at ? esc(new Date(t.exit_at).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })) : ''}</td></tr>`).join('')}
        </tbody></table>`
      : '<div class="empty-state">Closed trades appear here. R = profit measured in units of the risk taken (1R = the stop distance).</div>';
  }

  function niceTicks(min, max, count = 4) {
    if (min === max) { min -= 1; max += 1; }
    const raw = (max - min) / count;
    const mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
    const lo = Math.floor(min / step) * step;
    const ticks = [];
    for (let v = lo; v <= max + step * 0.5; v += step) ticks.push(Number(v.toFixed(10)));
    return ticks;
  }

  function drawEquity(box, points, message) {
    if (!points.length) {
      box.innerHTML = `<div class="empty-state">${esc(message || 'No equity history yet.')}</div>`;
      return;
    }
    const W = Math.max(280, box.clientWidth || 600);
    const H = 220;
    const m = { l: 64, r: 16, t: 14, b: 26 };
    const vals = points.map((p) => p.equity);
    const ticks = niceTicks(Math.min(...vals), Math.max(...vals));
    const yMin = ticks[0];
    const yMax = ticks[ticks.length - 1];
    const x = (i) => m.l + (points.length === 1 ? (W - m.l - m.r) / 2 : (i / (points.length - 1)) * (W - m.l - m.r));
    const y = (v) => m.t + (1 - (v - yMin) / (yMax - yMin || 1)) * (H - m.t - m.b);
    const line = points.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p.equity).toFixed(1)}`).join('');
    const area = `${line}L${x(points.length - 1).toFixed(1)},${y(yMin).toFixed(1)}L${x(0).toFixed(1)},${y(yMin).toFixed(1)}Z`;
    const fmtT = (t) => new Date(t).toLocaleDateString([], { month: 'short', day: 'numeric' });
    const intraday = points.length > 1 && (new Date(points[points.length - 1].t) - new Date(points[0].t)) < 2 * 86400000;
    const fmtX = (t) => (intraday ? new Date(t).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }) : fmtT(t));
    const xl = [0, Math.floor((points.length - 1) / 2), points.length - 1].filter((v, i, a) => a.indexOf(v) === i);
    const last = points[points.length - 1];
    const first = points[0];
    const change = last.equity - first.equity;
    // One format for the whole axis (never "$102K" next to "$99,000").
    const big = Math.max(...ticks.map(Math.abs)) >= 100000;
    const axisFmt = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD',
      notation: big ? 'compact' : 'standard', maximumFractionDigits: big ? 1 : 0 });
    const compact = (v) => axisFmt.format(v);
    box.innerHTML = `
      <div class="chart-head"><span class="muted small">Account equity</span>
        <b>${money(last.equity)}</b> <span class="small ${signCls(change)}">${change >= 0 ? '+' : ''}${money(change)} this period</span></div>
      <svg class="equity" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img"
           aria-label="Account equity from ${esc(fmtT(first.t))} (${esc(money(first.equity))}) to ${esc(fmtT(last.t))} (${esc(money(last.equity))})">
        ${ticks.map((v) => `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/>
          <text class="axis" x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end">${esc(compact(v))}</text>`).join('')}
        ${xl.map((i) => `<text class="axis" x="${x(i)}" y="${H - 6}" text-anchor="${i === 0 ? 'start' : i === points.length - 1 ? 'end' : 'middle'}">${esc(fmtX(points[i].t))}</text>`).join('')}
        <path class="area" d="${area}"/>
        <path class="line" d="${line}"/>
        <circle class="end" cx="${x(points.length - 1)}" cy="${y(last.equity)}" r="4"/>
        <g class="hover" visibility="hidden"><line class="cross" y1="${m.t}" y2="${H - m.b}"/><circle class="dot" r="4"/></g>
        <rect class="hit" x="${m.l}" y="0" width="${W - m.l - m.r}" height="${H}" fill="transparent"/>
      </svg>
      <div class="chart-tip" hidden></div>`;
    const svg = box.querySelector('svg');
    const hover = svg.querySelector('.hover');
    const tip = box.querySelector('.chart-tip');
    const move = (ev) => {
      const rect = svg.getBoundingClientRect();
      const px = (ev.clientX - rect.left) * (W / rect.width);
      const i = Math.max(0, Math.min(points.length - 1, Math.round(((px - m.l) / (W - m.l - m.r)) * (points.length - 1))));
      const p = points[i];
      hover.setAttribute('visibility', 'visible');
      hover.querySelector('.cross').setAttribute('x1', x(i));
      hover.querySelector('.cross').setAttribute('x2', x(i));
      hover.querySelector('.dot').setAttribute('cx', x(i));
      hover.querySelector('.dot').setAttribute('cy', y(p.equity));
      tip.hidden = false;
      tip.innerHTML = `<div class="muted small">${esc(new Date(p.t).toLocaleString([], { dateStyle: 'medium', ...(intraday ? { timeStyle: 'short' } : {}) }))}</div>
        <div><span class="key"></span>Equity <b>${money(p.equity)}</b></div>
        ${p.pnl != null ? `<div class="small">P&amp;L <span class="${signCls(p.pnl)}">${money(p.pnl)}${p.pnl_pct != null ? ` (${pct(p.pnl_pct)})` : ''}</span></div>` : ''}`;
      const left = (x(i) / W) * rect.width;
      tip.style.left = `${Math.min(Math.max(left + 12, 0), rect.width - tip.offsetWidth - 4)}px`;
      tip.style.top = `${(y(p.equity) / H) * rect.height - 10}px`;
    };
    svg.querySelector('.hit').addEventListener('mousemove', move);
    svg.querySelector('.hit').addEventListener('mouseleave', () => { hover.setAttribute('visibility', 'hidden'); tip.hidden = true; });
  }

  async function downloadCsv(mode) {
    try {
      await API.getMe();  // refreshes an expired login token first (plain fetch doesn't)
      const token = localStorage.getItem('adt_access_token');
      const resp = await fetch(`/api/control/${mode}/trades.csv`, { headers: { Authorization: `Bearer ${token}` } });
      if (!resp.ok) throw new Error(`Download failed (${resp.status})`);
      const url = URL.createObjectURL(await resp.blob());
      const a = Object.assign(document.createElement('a'), { href: url, download: `${mode}_trades.csv` });
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (ex) { toast(ex.message, 'error'); }
  }

  /* ── Actions ── */
  function modal(html, onConfirm, { confirmText = 'Confirm', danger = false, check = null } = {}) {
    const root = $('modal-root');
    root.innerHTML = `<div class="modal-backdrop"><div class="modal" role="dialog" aria-modal="true">
      ${html}
      ${check ? `<label class="check"><input type="checkbox" id="modal-check"> ${esc(check)}</label>` : ''}
      <div class="btn-row end"><button class="btn btn-outline" id="modal-cancel">Cancel</button>
      <button class="btn ${danger ? 'btn-danger' : 'btn-primary'}" id="modal-ok" ${check ? 'disabled' : ''}>${esc(confirmText)}</button></div>
    </div></div>`;
    const close = () => { root.innerHTML = ''; };
    $('modal-cancel').addEventListener('click', close);
    if (check) $('modal-check').addEventListener('change', (e) => { $('modal-ok').disabled = !e.target.checked; });
    $('modal-ok').addEventListener('click', async () => { $('modal-ok').disabled = true; await onConfirm(); close(); });
  }

  function startBot(mode) {
    const live = mode === 'live';
    const symbols = $(`${mode}-symbols`).value.split(/[\s,]+/).filter(Boolean);
    const timeframe = $(`${mode}-timeframe`).value;
    const execute = document.querySelector(`input[name="${mode}-exec"]:checked`).value === '1';
    try { localStorage.setItem(`adt_${mode}_settings`, JSON.stringify({ symbols: symbols.join(','), timeframe })); } catch {}
    const go = async () => {
      try {
        await api('POST', `/control/${mode}/start`, { symbols, timeframe, execute, confirm_live: live && execute });
        toast(`${live ? 'LIVE' : 'Paper'} bot started${execute ? '' : ' (watch only)'}.`, 'success');
        loadMode(mode); loadOverview();
      } catch (ex) { toast(ex.message, 'error'); }
    };
    if (live && execute) {
      modal(`<h3>Start trading with real money?</h3>
        <p>The bot will place <b>real orders</b> in your live Alpaca account on: <b class="mono">${esc(symbols.join(', '))}</b>.</p>
        <p class="muted small">Risk limits stay on (per-trade risk, 2% daily drawdown stop, total-risk cap, closing everything before the close).</p>`,
      go, { confirmText: 'Start live trading', danger: true, check: 'I understand this trades real money and I can lose it.' });
    } else {
      go();
    }
  }

  async function stopBot(mode) {
    try { const r = await api('POST', `/control/${mode}/stop`); toast(r.message, 'info'); setTimeout(() => loadMode(mode), 1500); loadOverview(); }
    catch (ex) { toast(ex.message, 'error'); }
  }

  function flatten(mode) {
    const live = mode === 'live';
    modal(`<h3>Close everything in the ${live ? 'LIVE' : 'paper'} account?</h3>
      <p>This stops the bot, cancels all open orders and sells all positions at market price now.</p>`,
    async () => {
      try {
        const r = await api('POST', `/control/${mode}/flatten`, { confirm: true });
        const closed = (r.closed || []).map((c) => c.symbol).join(', ') || 'nothing to close';
        toast(`Cancelled ${r.cancelled_orders} orders; closed ${closed}${(r.failures || []).length ? ' — SOME FAILED, check Alpaca' : ''}.`,
          (r.failures || []).length ? 'error' : 'success');
        lastAccountLoad[mode] = 0; loadMode(mode);
      } catch (ex) { toast(ex.message, 'error'); }
    }, { confirmText: 'Close everything', danger: true, check: live ? 'Yes, sell all my live positions now.' : null });
  }

  async function arm() {
    try {
      await api('POST', '/control/live/arm', { phrase: $('arm-phrase').value, password: $('arm-password').value });
      toast('Live trading armed.', 'success');
      $('live-arm-card').dataset.state = '';
      loadMode('live'); loadOverview();
    } catch (ex) { toast(ex.message, 'error'); }
  }

  async function disarm() {
    try {
      await api('POST', '/control/live/disarm');
      toast('Live trading disarmed and the live bot stopped.', 'success');
      $('live-arm-card').dataset.state = '';
      loadMode('live'); loadOverview();
    } catch (ex) { toast(ex.message, 'error'); }
  }

  async function submitPaperOrder() {
    const symbol = $('paper-order-symbol').value.trim().toUpperCase();
    const side = $('paper-order-side').value;
    const qty = Number($('paper-order-qty').value);
    if (!symbol || !(qty > 0)) { toast('Enter a symbol and a share count.', 'error'); return; }
    try {
      const r = await API.submitPaperOrder(symbol, side, qty);
      toast(r.submitted ? `Paper ${side} ${qty} ${symbol} submitted.` : `Not submitted: ${r.skipped_reason}`, r.submitted ? 'success' : 'error');
      lastAccountLoad.paper = 0; loadMode('paper');
    } catch (ex) { toast(ex.message, 'error'); }
  }

  /* ── Wiring ── */
  function init() {
    $('setup-form').addEventListener('submit', handleSetup);
    document.querySelectorAll('#sidebar .nav-item').forEach((b) => b.addEventListener('click', () => show(b.dataset.view)));
    document.addEventListener('click', (e) => {
      const go = e.target.closest('[data-goto]');
      if (go) { e.preventDefault(); show(go.dataset.goto); }
      const reveal = e.target.closest('.reveal');
      if (reveal) {
        const input = $(reveal.dataset.target);
        input.type = input.type === 'password' ? 'text' : 'password';
        reveal.textContent = input.type === 'password' ? 'Show' : 'Hide';
      }
      const save = e.target.closest('[data-save]');
      if (save) saveKeys(save.dataset.save);
      const test = e.target.closest('[data-test]');
      if (test) testKeys(test.dataset.test);
      const remove = e.target.closest('[data-remove]');
      if (remove) removeKeys(remove.dataset.remove);
    });
    $('alerts-save').addEventListener('click', saveAlerts);
    $('autostart-toggle').addEventListener('change', toggleAutostart);
    let resizeTimer = null;
    window.addEventListener('resize', () => {  // redraw at the new width so text stays readable
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        ['paper', 'live'].forEach((mode) => {
          const data = perf[mode].data;
          if (data && $(`${mode}-equity`)) drawEquity($(`${mode}-equity`), data.equity || [], data.equity_message);
        });
      }, 200);
    });
    $('val-start').addEventListener('click', startValidation);
    $('val-stop').addEventListener('click', async () => { await api('POST', '/control/validate/stop'); loadValidation(); });
  }

  document.addEventListener('DOMContentLoaded', init);
  return { onLogin, onDashboard, show };
})();
