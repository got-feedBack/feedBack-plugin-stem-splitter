/* Stem Splitter — client screen + v3 card-action integration.
 *
 * Classic self-executing script (feedpak plugin spec §6.1). Idempotent: guarded
 * so a re-hydration never double-registers card actions, double-binds
 * screen:changed, or opens a second WebSocket.
 */
(function () {
  'use strict';
  if (window.__stemSplitterLoaded) return;
  window.__stemSplitterLoaded = true;

  var API = '/api/plugins/stem_splitter';
  var SCREEN_ID = 'plugin-stem_splitter';
  var fb = window.feedBack;

  var state = {
    missingStems: new Set(),
    missingLyrics: new Set(),
    splitEngine: null,
    lyricsEngine: null,
    ws: null,
    inited: false,
    pendingAfterSetup: [],   // jobs waiting on a one-time model download
    preparingModels: false,  // guard: only ever one prepare_models in flight
  };

  // The model download takes many minutes, and we promise the user their job "starts
  // automatically when it finishes". Keeping the queue in memory broke that promise on
  // any reload/navigation, so persist it.
  var PENDING_KEY = 'stem_splitter.pendingAfterSetup';
  function loadPending() {
    try {
      var raw = window.localStorage.getItem(PENDING_KEY);
      var arr = raw ? JSON.parse(raw) : [];
      state.pendingAfterSetup = Array.isArray(arr) ? arr : [];
    } catch (e) { state.pendingAfterSetup = []; }
  }
  var warnedPersist = false;
  function savePending() {
    try {
      window.localStorage.setItem(PENDING_KEY, JSON.stringify(state.pendingAfterSetup));
    } catch (e) {
      // The queue IS the "starts automatically after the download, even across a
      // reload" promise. If it can't be persisted (quota, storage disabled), the jobs
      // still run in this page life — but say so rather than quietly breaking the
      // promise. QuotaExceeded on a big batch is the realistic case.
      console.warn('[stem_splitter] could not persist pending queue', e);
      if (warnedPersist) return;   // savePending runs on every enqueue - warn once
      warnedPersist = true;
      toast('Queue not saved',
            'These jobs will still start when the download finishes, but they '
            + 'will be lost if you reload the page first.', 'warn');
    }
  }

  function $(id) { return document.getElementById(id); }

  function toast(title, message, accent) {
    try {
      if (window.fbNotify && window.fbNotify.show) {
        window.fbNotify.show({ title: title, message: message || '', accent: accent || 'info' });
        return;
      }
    } catch (e) {}
    console.log('[stem_splitter]', title, message || '');
  }

  function api(path, opts) {
    return fetch(API + path, opts).then(function (r) { return r.json(); });
  }

  function post(kind, body) {
    return api('/' + kind, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  }

  // Enqueue, but never let a job silently stall on a lazy multi-GB model fetch:
  // the backend answers `needs_setup` when the split would go to the managed local
  // server and its weights aren't downloaded. Ask, then set it up, then run.
  function enqueue(kind, filenames) {
    var body = Array.isArray(filenames) ? { filenames: filenames } : { filename: filenames };
    return post(kind, body).then(function (res) {
      if (!res || !res.needs_setup) return res;
      if (!window.confirm(res.message + '\n\nThe download runs in the background; your '
                          + (kind === 'split' ? 'split' : 'transcription')
                          + ' starts automatically when it finishes.')) {
        toast('Cancelled', 'Models are needed before this can run.');
        return res;
      }
      // Keep a handle on THIS entry: if the prepare_models request is refused or
      // fails we have to remove exactly the job we just queued. pop() would remove
      // whatever is last, which is the wrong one if another enqueue landed while our
      // request was in flight.
      var entry = { kind: kind, body: body };
      state.pendingAfterSetup.push(entry);
      savePending();
      connectWS();

      function dropEntry() {
        var i = state.pendingAfterSetup.indexOf(entry);
        if (i !== -1) state.pendingAfterSetup.splice(i, 1);
        savePending();
      }
      // Several songs can hit needs_setup at once (a batch, or rapid clicks).
      // Only ever kick off ONE model download — the rest just queue behind it.
      if (state.preparingModels) {
        toast('Already downloading models', 'This job will start when it finishes.');
        return res;
      }
      state.preparingModels = true;
      return api('/server/prepare_models', { method: 'POST' }).then(function (r) {
        // The backend serialises server ops, so this can be REFUSED if one is already
        // running. Swallowing that would leave preparingModels stuck true and strand
        // the pending queue forever — reconcile only settles a prepare_models op, and
        // that op never started. So undo the optimism and say why.
        if (r && r.ok === false) {
          state.preparingModels = false;
          dropEntry();   // remove exactly the job we queued, not whatever is last
          toast('Server busy', (r.message || 'Another server operation is running.')
                               + ' Try again when it finishes.', 'warn');
          return res;
        }
        toast('Downloading models', 'This is a one-time ~2 GB download.');
        return res;
      }).catch(function (e) {
        state.preparingModels = false;
        dropEntry();
        toast('Could not start model download', String(e), 'warn');
        return res;
      });
    });
  }

  // Settle a prepare_models that completed while we weren't listening.
  function reconcilePrepareFromSnapshot(srv) {
    if (!srv || srv.op !== 'prepare_models') return;
    if (srv.active) {
      // Authoritative: a prepare IS running (maybe started by another page/session).
      state.preparingModels = true;
      return;
    }
    // Keyed off the persisted queue, not the in-memory flag: after a reload the flag is
    // gone but the user's approved jobs are still owed to them.
    if (!state.preparingModels && !state.pendingAfterSetup.length) return;
    state.preparingModels = false;
    if (srv.error) {
      toast('Model download failed', srv.error, 'warn');
      state.pendingAfterSetup.length = 0;
      savePending();
    } else {
      toast('Models ready', 'The local server is warmed up.', 'ok');
      flushPendingAfterSetup();
    }
    refreshConfig();
  }

  function flushPendingAfterSetup() {
    var pending = state.pendingAfterSetup.splice(0);
    savePending();
    pending.forEach(function (p) {
      var body = Object.assign({}, p.body, { skip_setup_check: true });
      post(p.kind, body).then(function (r) {
        if (r && r.enqueued) toast('Models ready', 'Queued ' + p.kind + ' now.');
        else toast('Could not queue ' + p.kind, (r && r.message) || 'The request was rejected.', 'warn');
      }).catch(function (e) {
        // Don't let a failed re-queue vanish as an unhandled rejection: the user
        // agreed to a 2 GB download for this job, so say that it didn't run.
        toast('Could not queue ' + p.kind, String(e), 'warn');
      });
    });
  }

  // ── missing-set + engine availability ─────────────────────────────────────
  function refreshConfig() {
    return api('/config').then(function (cfg) {
      state.splitEngine = cfg.split && cfg.split.engine;
      state.lyricsEngine = cfg.lyrics && cfg.lyrics.engine;
      var se = $('ss-split-engine'), le = $('ss-lyrics-engine');
      if (se) se.textContent = (cfg.split && cfg.split.reason) || 'unavailable';
      if (le) le.textContent = (cfg.lyrics && cfg.lyrics.reason) || 'unavailable';
    }).catch(function () {});
  }

  function refreshMissing() {
    return Promise.all([
      api('/missing_stems').catch(function () { return { songs: [] }; }),
      api('/missing_lyrics').catch(function () { return { songs: [] }; }),
    ]).then(function (res) {
      state.missingStems = new Set((res[0].songs || []).map(function (s) { return s.filename; }));
      state.missingLyrics = new Set((res[1].songs || []).map(function (s) { return s.filename; }));
      var a = $('ss-missing-stems-n'), b = $('ss-missing-lyrics-n');
      if (a) a.textContent = state.missingStems.size;
      if (b) b.textContent = state.missingLyrics.size;
    });
  }

  // ── v3 song-card actions (the official API that replaced DOM injection) ────
  function registerCardActions() {
    if (!fb || !fb.libraryCardActions) return;
    var reg = fb.libraryCardActions;
    reg.register({
      id: 'stem_splitter.split',
      pluginId: 'stem_splitter',
      label: 'Split stems',
      placement: 'menu',
      order: 30,
      applies: function (song) { return !!(song && song.filename); },
      enabled: function (song) { return state.missingStems.has(song.filename); },
      run: function (song) {
        if (!state.splitEngine) { toast('No split engine', 'Open Stem Splitter settings to configure a server or download a local engine.', 'warn'); return; }
        enqueue('split', song.filename).then(function (r) {
          if (r && r.enqueued) toast('Split queued', song.filename);
        });
      },
    });
    reg.register({
      id: 'stem_splitter.transcribe',
      pluginId: 'stem_splitter',
      label: 'Transcribe lyrics',
      placement: 'menu',
      order: 31,
      applies: function (song) { return !!(song && song.filename); },
      enabled: function (song) { return state.missingLyrics.has(song.filename); },
      run: function (song) {
        if (!state.lyricsEngine) { toast('No lyrics engine', 'Open Stem Splitter settings to configure a server or download whisperx.', 'warn'); return; }
        enqueue('transcribe', song.filename).then(function (r) {
          if (r && r.enqueued) toast('Transcription queued', song.filename);
        });
      },
    });
  }

  // ── job queue rendering ────────────────────────────────────────────────────
  function renderJobs(snap) {
    var host = $('ss-jobs');
    if (!host) return;
    var jobs = (snap && snap.jobs) || [];
    if (!jobs.length) { host.innerHTML = '<p class="ss-muted">No jobs yet.</p>'; return; }
    host.innerHTML = '';
    jobs.forEach(function (j) {
      var row = document.createElement('div');
      row.className = 'ss-job';
      var pct = Math.round((j.progress || 0) * 100);
      var label = j.title
        ? j.title + (j.artist ? ' — ' + j.artist : '')
        : baseName(j.filename);
      row.innerHTML =
        '<span class="ss-pill ' + (j.kind === 'split' ? 'split' : 'transcribe') + '">' + j.kind + '</span>' +
        '<span class="name" title="' + esc(j.filename) + '">' + esc(label) +
        '<br><span class="msg">' + esc(j.message || '') + '</span></span>' +
        '<div class="ss-bar"><i style="width:' + pct + '%"></i></div>' +
        '<span class="ss-status ' + esc(j.status) + '">' + esc(j.status) + '</span>' +
        '<button class="ss-btn secondary" data-del="' + esc(j.id) + '">✕</button>';
      host.appendChild(row);
    });
    host.querySelectorAll('[data-del]').forEach(function (b) {
      b.addEventListener('click', function () {
        api('/jobs/' + b.getAttribute('data-del'), { method: 'DELETE' });
      });
    });
  }

  function baseName(p) { return String(p || '').split(/[\\/]/).pop(); }
  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }

  // ── websocket ──────────────────────────────────────────────────────────────
  function connectWS() {
    if (state.ws) return;
    var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    var ws = new WebSocket(proto + '//' + location.host + API + '/events');
    state.ws = ws;
    ws.onmessage = function (ev) {
      var msg;
      try { msg = JSON.parse(ev.data); } catch (e) { return; }
      if (msg.type === 'jobs') {
        renderJobs(msg);
        // Snapshot recovery: if the WS dropped while prepare_models was running we'd
        // have missed the live server_done/server_error, leaving preparingModels stuck
        // true and the jobs the user approved a multi-GB download for never queued.
        // The snapshot carries the server op's terminal state, so settle it from there.
        reconcilePrepareFromSnapshot(msg.server);
        // A finished job may have changed missing-sets → refresh them (debounced).
        scheduleMissingRefresh();
      } else if (msg.type === 'install_done') {
        toast('Local engine ready', msg.which, 'ok');
        refreshConfig();
      } else if (msg.type === 'install_error') {
        toast('Install failed', msg.error, 'warn');
      } else if (msg.type === 'server_done') {
        if (msg.op === 'prepare_models') {
          state.preparingModels = false;
          toast('Models ready', 'The local server is warmed up.', 'ok');
          flushPendingAfterSetup();
        }
        refreshConfig();
      } else if (msg.type === 'server_error') {
        toast('Server error', msg.error, 'warn');
        // ONLY a failed prepare_models invalidates the pending queue. A failed
        // start/install/stop says nothing about the jobs waiting on the models, and
        // binning them here would break the "starts automatically" promise for a
        // completely unrelated failure.
        if (msg.op === 'prepare_models') {
          state.preparingModels = false;
          state.pendingAfterSetup.length = 0;
          savePending();
        }
      }
    };
    ws.onclose = function () { state.ws = null; setTimeout(connectWS, 2000); };
    ws.onerror = function () { try { ws.close(); } catch (e) {} };
  }

  var _missingTimer = null;
  function scheduleMissingRefresh() {
    if (_missingTimer) return;
    _missingTimer = setTimeout(function () { _missingTimer = null; refreshMissing(); }, 1500);
  }

  // ── screen wiring ──────────────────────────────────────────────────────────
  function wireScreen() {
    if (state.inited) return;
    state.inited = true;

    bind('ss-split-all', function () {
      var list = Array.from(state.missingStems);
      if (!list.length) { toast('Nothing to split', 'No songs are missing stems.'); return; }
      enqueue('split', list).then(function (r) {
        if (r && r.enqueued) toast('Queued', r.enqueued + ' split job(s)');
      });
    });
    bind('ss-transcribe-all', function () {
      var list = Array.from(state.missingLyrics);
      if (!list.length) { toast('Nothing to transcribe', 'No songs are missing lyrics.'); return; }
      enqueue('transcribe', list).then(function (r) {
        if (r && r.enqueued) toast('Queued', r.enqueued + ' transcription job(s)');
      });
    });
    bind('ss-refresh-missing', function () { refreshMissing(); refreshConfig(); });
    bind('ss-pause', function () { api('/pause', { method: 'POST' }); });
    bind('ss-resume', function () { api('/resume', { method: 'POST' }); });
    bind('ss-retry', function () { api('/retry_failed', { method: 'POST' }); });
    bind('ss-cancel', function () { api('/cancel_queued', { method: 'POST' }); });
    bind('ss-clear', function () { api('/clear_finished', { method: 'POST' }); });
    bind('ss-open-settings', function () {
      if (fb && fb.navigate) fb.navigate('settings');
      else if (window.showScreen) window.showScreen('settings');
    });

    connectWS();
    api('/jobs').then(renderJobs).catch(function () {});
  }

  function bind(id, fn) { var el = $(id); if (el) el.addEventListener('click', fn); }

  function onScreenChanged(ev) {
    var id = ev && ev.detail && ev.detail.id;
    if (id === SCREEN_ID) { wireScreen(); refreshConfig(); refreshMissing(); }
  }

  // ── boot ───────────────────────────────────────────────────────────────────
  function boot() {
    loadPending();
    // Jobs are owed from a previous page life. Stay connected so the completion
    // snapshot can flush them — that's what makes "starts automatically" true.
    //
    // Deliberately do NOT assume a download is still running: preparingModels is
    // in-memory, and forcing it true here would leave it stuck true forever if no
    // prepare_models op is actually active, blocking every future setup attempt. The
    // snapshot tells us the truth; and if the models finished while we were away,
    // flush the queue right now.
    if (state.pendingAfterSetup.length) {
      connectWS();
      api('/server_status').then(function (st) {
        if (st && st.models_downloaded && state.pendingAfterSetup.length) {
          flushPendingAfterSetup();
        }
      }).catch(function () {});
    }
    registerCardActions();
    refreshConfig();
    refreshMissing();
    if (fb && fb.on) fb.on('screen:changed', onScreenChanged);
    // If we're already on our screen at load, wire immediately.
    var el = document.getElementById(SCREEN_ID);
    if (el && el.classList.contains('active')) wireScreen();
  }

  if (fb && fb.on) {
    boot();
  } else {
    // Bus not ready yet (load-order safety) — poll briefly.
    var tries = 0;
    var t = setInterval(function () {
      fb = window.feedBack;
      if ((fb && fb.on) || tries++ > 50) { clearInterval(t); boot(); }
    }, 100);
  }
})();
