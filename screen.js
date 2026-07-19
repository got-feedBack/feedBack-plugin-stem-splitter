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
    missingVocals: new Set(),
    // False until the three missing-sets have ALL loaded. Actions gated on a song HAVING
    // something (re-align) must not fire while the sets are still empty-because-unknown.
    missingReady: false,
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

  // Job kinds, in the two places the user reads them. The old code asked "is it a split?" and
  // called everything else a transcription — which, with re-align added, means telling someone
  // their LYRICS ARE ABOUT TO BE REPLACED when they clicked the button that promises not to.
  var KIND = {
    split:      { noun: 'split',      pill: 'split' },
    transcribe: { noun: 'transcription', pill: 'transcribe' },
    realign:    { noun: 're-align',   pill: 'realign' },
  };
  function kindNoun(kind) { return (KIND[kind] || KIND.transcribe).noun; }
  function kindPill(kind) { return (KIND[kind] || KIND.transcribe).pill; }

  // Take the user to the queue. The plugin's own screen IS the queue, so this is just
  // navigation — but it has to work from the library, from settings, and from a toast, which is
  // three different places the user can be standing when they wonder where their job went.
  function openQueue() {
    if (fb && fb.navigate) fb.navigate(SCREEN_ID);
    else if (window.showScreen) window.showScreen(SCREEN_ID);
  }

  function toast(title, message, accent, onClick) {
    try {
      if (window.fbNotify && window.fbNotify.show) {
        // show() hands back the card, and its only built-in handler is dismiss-on-click. So a
        // click can carry us somewhere useful AND still dismiss — the two listeners coexist.
        var card = window.fbNotify.show({
          title: title,
          message: message || '',
          accent: accent || 'info',
        });
        if (card && onClick) {
          card.style.cursor = 'pointer';
          card.title = 'Open the Stem Splitter queue';
          card.addEventListener('click', onClick);
        }
        return card;
      }
    } catch (e) {}
    console.log('[stem_splitter]', title, message || '');
    return null;
  }

  // "Queued" toasts are the moment the user most wants the queue: they just started something and
  // want to watch it. Every other toast stays inert — a toast that navigates when you only meant
  // to dismiss it is worse than one that does nothing.
  function queuedToast(title, message) {
    return toast(title, (message || '') + ' — click to open the queue', 'info', openQueue);
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
  function enqueue(kind, filenames, extra) {
    var body = Array.isArray(filenames) ? { filenames: filenames } : { filename: filenames };
    if (extra) for (var k in extra) { if (Object.prototype.hasOwnProperty.call(extra, k)) body[k] = extra[k]; }
    return post(kind, body).then(function (res) {
      if (!res || !res.needs_setup) return res;
      if (!window.confirm(res.message + '\n\nThe download runs in the background; your '
                          + kindNoun(kind)
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
    // A failed fetch is UNKNOWN, not "nothing is missing".
    //
    // Split and Transcribe are gated on PRESENCE in a missing-set, so an empty set means the
    // action is disabled — a failed fetch is safe by accident. Re-align is gated on ABSENCE
    // (it needs the lyrics and the stem to be there), so an empty set means ENABLED, and a
    // failed fetch would offer a re-align on a song with neither. Same data, opposite default.
    //
    // So the sets are only trusted once they have all actually loaded. Until then re-align is
    // greyed out — the right answer to "I don't know" is not "go ahead".
    return Promise.all([
      api('/missing_stems'),
      api('/missing_lyrics'),
      // Vocals SPECIFICALLY — not the same question as /missing_stems, which asks for songs
      // lacking any of the six instrument stems. A song with vocals but no piano is in that set,
      // and re-align works fine on it: all it needs is something to align against.
      api('/missing_vocals'),
    ]).then(function (res) {
      state.missingStems = new Set((res[0].songs || []).map(function (s) { return s.filename; }));
      state.missingLyrics = new Set((res[1].songs || []).map(function (s) { return s.filename; }));
      state.missingVocals = new Set((res[2].songs || []).map(function (s) { return s.filename; }));
      state.missingReady = true;
      var a = $('ss-missing-stems-n'), b = $('ss-missing-lyrics-n');
      if (a) a.textContent = state.missingStems.size;
      if (b) b.textContent = state.missingLyrics.size;
    }).catch(function (e) {
      // Keep whatever we last knew, but stop trusting it: an action that depends on a song
      // HAVING something must not fire on a guess.
      state.missingReady = false;
      console.warn('[stem_splitter] could not refresh the missing-sets', e);
    });
  }

  // ── v3 song-card actions (the official API that replaced DOM injection) ────
  // Stems a split engine can actually produce, and therefore the only ids a
  // re-split could overwrite. Anything else in the pak (a custom "click"
  // stem, a user-added backing track) is never touched by the backend merge,
  // so it is shown as protected rather than offered as a checkbox.
  var REPLACEABLE_IDS = ['guitar', 'bass', 'drums', 'vocals', 'piano', 'other'];

  // Checkbox picker for a re-split (issue #11): people replace stems on
  // purpose (a re-recorded guitar) and add stems the engines know nothing
  // about — a re-split must only overwrite what the user says it may.
  // Defaults to everything replaceable checked.
  function openResplitPicker(filename) {
    api('/pak_stems?filename=' + encodeURIComponent(filename)).then(function (res) {
      if (!res || res.error) { toast('Could not read pak', (res && res.error) || 'unknown error', 'warn'); return; }
      var existing = (res.stems || []).map(function (s) { return s.id; });
      var replaceable = existing.filter(function (id) { return REPLACEABLE_IDS.indexOf(id) !== -1; });
      var protectedIds = existing.filter(function (id) {
        return REPLACEABLE_IDS.indexOf(id) === -1 && id !== 'full';
      });

      var old = document.getElementById('ss-resplit-overlay');
      if (old) old.remove();
      var ov = document.createElement('div');
      ov.id = 'ss-resplit-overlay';
      ov.style.cssText = 'position:fixed;inset:0;z-index:60;background:rgba(0,0,0,.55);display:flex;align-items:center;justify-content:center;';
      var boxes = replaceable.map(function (id) {
        return '<label class="flex items-center gap-2 py-1 text-sm">'
          + '<input type="checkbox" class="ss-rs-box" value="' + esc(id) + '" checked>'
          + '<span>' + esc(id) + '</span></label>';
      }).join('');
      // New ids the engine produces that the pak does not have yet are always
      // written (there is nothing to protect) — the backend guarantees that.
      var protectedNote = protectedIds.length
        ? '<div class="text-xs text-slate-400 mt-2">Never touched: ' + protectedIds.map(esc).join(', ') + ' (and the full mix).</div>'
        : '<div class="text-xs text-slate-400 mt-2">The full mix is never touched.</div>';
      ov.innerHTML =
        '<div class="bg-slate-900 border border-slate-700 rounded-lg p-4 w-80 shadow-xl">'
        + '<div class="font-semibold mb-1">Re-split stems</div>'
        + '<div class="text-xs text-slate-400 mb-2">Checked stems are replaced by the new split. Uncheck any you have replaced yourself and want to keep.</div>'
        + boxes + protectedNote
        + '<div class="flex justify-end gap-2 mt-3">'
        + '<button id="ss-rs-cancel" class="px-3 py-1 text-sm rounded bg-slate-700 hover:bg-slate-600">Cancel</button>'
        + '<button id="ss-rs-go" class="px-3 py-1 text-sm rounded bg-teal-700 hover:bg-teal-600">Re-split</button>'
        + '</div></div>';
      document.body.appendChild(ov);
      ov.addEventListener('click', function (e) { if (e.target === ov) ov.remove(); });
      ov.querySelector('#ss-rs-cancel').addEventListener('click', function () { ov.remove(); });
      ov.querySelector('#ss-rs-go').addEventListener('click', function () {
        var picked = Array.prototype.slice.call(ov.querySelectorAll('.ss-rs-box'))
          .filter(function (b) { return b.checked; })
          .map(function (b) { return b.value; });
        ov.remove();
        if (!picked.length) { toast('Nothing selected', 'No stems were chosen to replace.', 'warn'); return; }
        enqueue('split', filename, { replace_stems: picked }).then(function (r) {
          if (r && r.enqueued) queuedToast('Re-split queued', filename + ' (' + picked.join(', ') + ')');
        });
      });
    });
  }

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
          if (r && r.enqueued) queuedToast('Split queued', song.filename);
        });
      },
    });
    reg.register({
      id: 'stem_splitter.resplit',
      pluginId: 'stem_splitter',
      label: 'Re-split stems…',
      placement: 'menu',
      order: 30.5,
      applies: function (song) { return !!(song && song.filename); },
      // The mirror image of Split: only for songs that already HAVE stems.
      // Unknown is not "yes" (same rule as re-align below).
      enabled: function (song) {
        return state.missingReady && !state.missingStems.has(song.filename);
      },
      run: function (song) {
        if (!state.splitEngine) { toast('No split engine', 'Open Stem Splitter settings to configure a server or download a local engine.', 'warn'); return; }
        openResplitPicker(song.filename);
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
          if (r && r.enqueued) queuedToast('Transcription queued', song.filename);
        });
      },
    });
    reg.register({
      id: 'stem_splitter.realign',
      pluginId: 'stem_splitter',
      label: 'Re-align lyrics to vocals',
      placement: 'menu',
      order: 32,
      applies: function (song) { return !!(song && song.filename); },
      // Needs BOTH: lyrics to re-time, and a vocal stem to time them against. The mirror image
      // of Transcribe on the first count — that one needs lyrics to be MISSING, this one needs
      // them present (words already right, timings wrong: the case where transcribing again
      // would "fix" the timing by throwing the correct words away). The backend refuses either
      // way, but a menu item that is clickable and then fails is a worse answer than one that
      // is greyed out.
      enabled: function (song) {
        // Unknown is not "yes". Before the sets load — or after a failed refresh — an
        // absence-gated action would be enabled on every song, including ones with no lyrics
        // and no stem, and the user's click would travel all the way to the backend to be told
        // no. Split and Transcribe are gated on PRESENCE, so they default to disabled and never
        // had this problem; this one is the mirror image and needs the guard.
        return state.missingReady &&
               !state.missingLyrics.has(song.filename) &&
               !state.missingVocals.has(song.filename);
      },
      run: function (song) {
        // Server-only: /align is "here are the words, when are they sung", and the local engine
        // has no equivalent entry point. Falling back to transcription would replace the user's
        // lyrics with Whisper's guesses — precisely what they clicked re-align to avoid.
        if (state.lyricsEngine !== 'remote') {
          toast('Re-align needs a server',
                'Re-aligning keeps your words and only fixes their timing, which needs a demucs/WhisperX server. The local engine can transcribe, but not re-align. Configure a server in Stem Splitter settings.',
                'warn');
          return;
        }
        enqueue('realign', song.filename).then(function (r) {
          if (r && r.enqueued) queuedToast('Re-align queued', song.filename);
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
      // A failed job's error is the whole reason the row is worth reading, and it used to be
      // clipped to one ellipsised line inside .name (nowrap + overflow:hidden) — so the part
      // that says WHY was exactly the part that got cut. Errors now get their own full-width
      // block below the row: wrapped, selectable, and copyable, because where these end up is
      // a bug report.
      var failed = j.status === 'failed';
      var detail = failed ? (j.error || j.message || '') : '';

      row.innerHTML =
        '<div class="ss-job-main">' +
          '<span class="ss-pill ' + kindPill(j.kind) + '">' + esc(j.kind) + '</span>' +
          '<span class="name" title="' + esc(j.filename) + '">' + esc(label) +
          (failed ? '' : '<br><span class="msg">' + esc(j.message || '') + '</span>') + '</span>' +
          '<div class="ss-bar"><i style="width:' + pct + '%"></i></div>' +
          '<span class="ss-status ' + esc(j.status) + '">' + esc(j.status) + '</span>' +
          '<button class="ss-btn secondary" data-del="' + esc(j.id) + '">✕</button>' +
        '</div>' +
        (detail
          ? '<div class="ss-job-error">' +
              '<pre class="ss-err-text fb-selectable">' + esc(detail) + '</pre>' +
              '<button class="ss-btn secondary ss-err-copy" data-copy="' + esc(j.id) + '">Copy</button>' +
            '</div>'
          : '');
      if (detail) row._errText = detail;
      host.appendChild(row);
    });

    host.querySelectorAll('[data-copy]').forEach(function (b) {
      b.addEventListener('click', function () {
        var row = b.closest('.ss-job');
        var text = (row && row._errText) || '';
        var flash = function (label) {
          b.textContent = label;
          setTimeout(function () { b.textContent = 'Copy'; }, 1500);
        };
        var done = function () { flash('Copied'); };
        // If we couldn't copy, say so — and select the text so Ctrl+C still works. Claiming
        // success we didn't achieve just moves the failure to the moment they paste.
        var fail = function () {
          flash('Press Ctrl+C');
          // This is the fallback for a copy that ALREADY failed, so it must not be the thing
          // that throws: getSelection() returns null in some contexts (and the text stays
          // selectable by hand regardless — .ss-err-text is fb-selectable). Best-effort only.
          var pre = row && row.querySelector('.ss-err-text');
          var sel = window.getSelection ? window.getSelection() : null;
          if (!pre || !sel) return;
          try {
            var range = document.createRange();
            range.selectNodeContents(pre);
            sel.removeAllRanges();
            sel.addRange(range);
          } catch (e) { /* the user can still select it themselves */ }
        };
        // navigator.clipboard is undefined on a non-secure origin (plain http to a NAS, which
        // is how plenty of people run this). Fall back rather than throwing into the console.
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(done, function () { fallbackCopy(text, done, fail); });
        } else {
          fallbackCopy(text, done, fail);
        }
      });
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

  // navigator.clipboard only exists on a secure origin. A NAS or LAN install served over plain
  // http has no clipboard API at all, and that is exactly the deployment whose users most need
  // to paste an error into an issue.
  function fallbackCopy(text, done, fail) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    var ok = false;
    // execCommand returns FALSE on failure rather than throwing, so a bare try/catch would
    // report success while the clipboard still held whatever was there before. Telling someone
    // their error is copied when it isn't is worse than telling them it failed: they paste the
    // wrong thing into the issue and don't find out until someone asks them to try again.
    try { ok = document.execCommand('copy') === true; } catch (e) { ok = false; }
    document.body.removeChild(ta);
    if (ok) done(); else fail();
  }

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
