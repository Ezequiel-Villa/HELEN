/* eslint-disable no-restricted-globals */
(function () {
  'use strict';

  const scope = typeof self !== 'undefined' ? self : globalThis;
  const connections = new Set();
  let nodeParentPort = null;
  let nodePortAdapter = null;

  if (typeof process !== 'undefined' && process.versions && process.versions.node && typeof require === 'function') {
    try {
      const { parentPort } = require('worker_threads');
      if (parentPort) {
        nodeParentPort = parentPort;
        nodePortAdapter = {
          postMessage: (message) => parentPort.postMessage(message),
          start: null,
        };
      }
    } catch (error) {
      nodeParentPort = null;
      nodePortAdapter = null;
    }
  }
  const state = {
    items: new Map(),
    tickTimer: null,
    baseEpoch: Date.now(),
    basePerf: (typeof performance !== 'undefined' && performance.now) ? performance.now() : 0,
  };

  const now = () => {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
      return state.baseEpoch + (performance.now() - state.basePerf);
    }
    return Date.now();
  };

  const postToPort = (port, message) => {
    try {
      port.postMessage(message);
    } catch (error) {
      // Ignore stale ports
    }
  };

  const broadcast = (message) => {
    connections.forEach((port) => postToPort(port, message));
  };

  const clearTickTimer = () => {
    if (state.tickTimer) {
      clearTimeout(state.tickTimer);
      state.tickTimer = null;
    }
  };

  const scheduleTick = () => {
    clearTickTimer();
    let nextDelay = null;
    const current = now();
    state.items.forEach((item) => {
      if (!item || item.state !== 'running' || typeof item.targetEpochMs !== 'number') {
        return;
      }
      const remaining = item.targetEpochMs - current;
      if (remaining <= 0) {
        nextDelay = 0;
      } else if (nextDelay === null || remaining < nextDelay) {
        nextDelay = remaining;
      }
    });

    if (nextDelay === null) {
      return;
    }
    const delay = Math.max(30, Math.min(nextDelay, 1000));
    state.tickTimer = setTimeout(runTick, delay);
  };

  const runTick = () => {
    state.tickTimer = null;
    const current = now();
    const ticks = [];
    const fired = [];

    state.items.forEach((item) => {
      if (!item || item.state !== 'running' || typeof item.targetEpochMs !== 'number') {
        return;
      }
      const remaining = item.targetEpochMs - current;
      const safeRemaining = remaining > 0 ? remaining : 0;
      ticks.push({ id: item.id, remainingMs: Math.max(0, Math.round(safeRemaining)) });
      if (remaining <= 0) {
        fired.push({ id: item.id, type: item.type, firedAt: current });
        item.state = 'completed';
        item.targetEpochMs = null;
        item.remainingMs = 0;
      }
    });

    if (ticks.length) {
      broadcast({ type: 'tick', payload: ticks });
    }
    if (fired.length) {
      broadcast({ type: 'fired', payload: fired });
    }

    scheduleTick();
  };

  const setItems = (items) => {
    state.items.clear();
    if (!Array.isArray(items)) {
      scheduleTick();
      return;
    }
    items.forEach((raw) => {
      if (!raw || !raw.id) return;
      const entry = {
        id: raw.id,
        type: raw.type,
        state: raw.state,
        targetEpochMs: (typeof raw.targetEpochMs === 'number') ? raw.targetEpochMs : null,
        remainingMs: (typeof raw.remainingMs === 'number') ? raw.remainingMs : null,
      };
      state.items.set(entry.id, entry);
    });
    scheduleTick();
  };

  const handleMessage = (data, port) => {
    if (!data || typeof data !== 'object') {
      return;
    }
    switch (data.type) {
      case 'set-items':
        setItems(data.items);
        break;
      case 'ping':
        postToPort(port, { type: 'pong', timestamp: now() });
        break;
      case 'debug-fast-forward':
        if (typeof data.milliseconds === 'number' && Number.isFinite(data.milliseconds)) {
          state.baseEpoch += data.milliseconds;
          runTick();
        }
        break;
      default:
        break;
    }
  };

  const attachPort = (port) => {
    if (!port) return;
    connections.add(port);
    if (typeof port.start === 'function') {
      port.start();
    }
    port.onmessage = (event) => handleMessage(event.data, port);
    postToPort(port, { type: 'ready' });
    scheduleTick();
  };

  if (nodeParentPort && nodePortAdapter) {
    const adapter = nodePortAdapter;
    connections.add(adapter);
    if (typeof nodeParentPort.on === 'function') {
      nodeParentPort.on('message', (data) => handleMessage(data, adapter));
      nodeParentPort.on('close', () => connections.delete(adapter));
    } else if (typeof nodeParentPort.addEventListener === 'function') {
      nodeParentPort.addEventListener('message', (event) => handleMessage(event.data, adapter));
    }
    postToPort(adapter, { type: 'ready' });
    scheduleTick();
  } else if (typeof scope !== 'undefined' && typeof scope.onconnect === 'function') {
    scope.onconnect = (event) => {
      const [port] = event.ports;
      attachPort(port);
    };
  } else {
    const dedicatedPort = {
      postMessage: (message) => scope.postMessage(message),
      start: null,
    };
    connections.add(dedicatedPort);
    scope.onmessage = (event) => handleMessage(event.data, dedicatedPort);
    postToPort(dedicatedPort, { type: 'ready' });
  }
})();
