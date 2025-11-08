/* eslint-disable no-restricted-globals */
(() => {
  'use strict';

  const scope = typeof self !== 'undefined' ? self : globalThis;

  const state = {
    running: false,
    baseElapsed: 0,
    startPerf: 0,
    timerId: null,
    tickInterval: 40,
  };

  const now = () => {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
      return performance.now();
    }
    return Date.now();
  };

  const clearTimer = () => {
    if (state.timerId) {
      clearInterval(state.timerId);
      state.timerId = null;
    }
  };

  const emitTick = () => {
    const elapsed = state.running
      ? state.baseElapsed + Math.max(0, now() - state.startPerf)
      : state.baseElapsed;
    scope.postMessage({ type: 'tick', elapsedMs: Math.max(0, Math.round(elapsed)) });
  };

  const start = (elapsedMs) => {
    state.baseElapsed = Math.max(0, Number(elapsedMs) || 0);
    state.startPerf = now();
    state.running = true;
    emitTick();
    clearTimer();
    state.timerId = setInterval(emitTick, state.tickInterval);
  };

  const stop = () => {
    if (!state.running) {
      emitTick();
      return;
    }
    state.baseElapsed = Math.max(0, state.baseElapsed + Math.max(0, now() - state.startPerf));
    state.running = false;
    clearTimer();
    emitTick();
  };

  const reset = () => {
    state.running = false;
    clearTimer();
    state.baseElapsed = 0;
    state.startPerf = now();
    emitTick();
  };

  scope.onmessage = (event) => {
    const data = event.data || {};
    switch (data.type) {
      case 'start':
        start(data.elapsedMs);
        break;
      case 'stop':
        stop();
        break;
      case 'reset':
        reset();
        break;
      default:
        break;
    }
  };

  scope.postMessage({ type: 'ready' });
})();
