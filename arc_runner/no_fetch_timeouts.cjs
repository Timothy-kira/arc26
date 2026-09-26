// Preloaded into MiniMax Code (NODE_OPTIONS=--require). Node's fetch aborts a response after
// 300 s without bytes ("terminated"); a busy local vLLM can queue a long prompt for longer.
// Replace the global undici dispatcher with one of the same class without those timeouts.
'use strict';
const key = Symbol.for('undici.globalDispatcher.1');
// Node installs its default Agent on the first fetch call; make one that is aborted at once.
const probe = new AbortController();
probe.abort();
globalThis.fetch('http://127.0.0.1:9/', { signal: probe.signal }).catch(() => {});
const current = globalThis[key];
if (current && typeof current.constructor === 'function') {
  const ms = Number(process.env.ARC_FETCH_TIMEOUT_MS || 0);
  globalThis[key] = new current.constructor({ headersTimeout: ms, bodyTimeout: ms, connectTimeout: 60000 });
}
