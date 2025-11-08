/* =========================================================
   HELEN – Motor persistente de alarmas y temporizadores
   Ejecuta el cálculo en un Web Worker / SharedWorker para
   evitar throttling y mantiene el estado en localStorage.
   ========================================================= */

(function () {
  'use strict';

  if (window.HelenScheduler && window.HelenScheduler.version === 2) {
    return;
  }

  const STORAGE_KEY = 'helen:timekeeper:v1';
  const VERSION = 2;
  const ONE_SECOND = 1000;
  const ONE_MINUTE = 60 * ONE_SECOND;
  const ONE_HOUR = 60 * ONE_MINUTE;
  const ONE_DAY = 24 * ONE_HOUR;
  const TIMER_TOAST_DURATION = 6500;
  const AUTO_REMOVE_DELAY = TIMER_TOAST_DURATION + 2200;
  const SOUND_CONFIG = {
    url: 'https://actions.google.com/sounds/v1/alarms/digital_watch_alarm_long.ogg',
    fallbackInterval: 2600,
    autoStopAfterMs: null,
    gain: 0.9,
    chimeGain: 0.6,
    elementVolume: 1,
  };
  const PENDING_STORAGE_KEY = 'helen:timekeeper:pendingQueue:v1';
  const GLOBAL_EVENT_NAME = 'helen:timekeeper:fired';
  const GLOBAL_BUS_VERSION = 1;

  const dispatchWindowEvent = (name, detail) => {
    if (typeof window === 'undefined' || typeof window.dispatchEvent !== 'function') {
      return;
    }
    let event = null;
    const CustomEventCtor = (typeof window.CustomEvent === 'function' && window.CustomEvent)
      || (typeof CustomEvent === 'function' && CustomEvent);
    if (CustomEventCtor) {
      try {
        event = new CustomEventCtor(name, { detail });
      } catch (error) {
        event = null;
      }
    }
    if (!event && window.document && typeof window.document.createEvent === 'function') {
      try {
        event = window.document.createEvent('CustomEvent');
        event.initCustomEvent(name, false, false, detail);
      } catch (error) {
        event = null;
      }
    }
    if (!event) {
      event = { type: name, detail };
    }
    try {
      window.dispatchEvent(event);
    } catch (error) {
      // Ignorar errores silenciosos en ambientes de prueba.
    }
  };

  const createGlobalBus = () => {
    const listenersMap = new Map();
    const bus = {
      version: GLOBAL_BUS_VERSION,
      on(eventName, handler) {
        if (typeof handler !== 'function') {
          return () => {};
        }
        if (!listenersMap.has(eventName)) {
          listenersMap.set(eventName, new Set());
        }
        const set = listenersMap.get(eventName);
        set.add(handler);
        return () => bus.off(eventName, handler);
      },
      off(eventName, handler) {
        const set = listenersMap.get(eventName);
        if (!set) {
          return;
        }
        if (handler) {
          set.delete(handler);
          if (set.size === 0) {
            listenersMap.delete(eventName);
          }
        } else {
          set.clear();
          listenersMap.delete(eventName);
        }
      },
      emit(eventName, detail) {
        dispatchWindowEvent(eventName, detail);
        const set = listenersMap.get(eventName);
        if (!set) {
          return;
        }
        set.forEach((callback) => {
          try {
            callback(detail);
          } catch (error) {
            console.error('[HelenScheduler] Error en listener global', error);
          }
        });
      },
    };
    return bus;
  };

  const ensureGlobalBus = () => {
    if (typeof window === 'undefined') {
      return null;
    }
    const existing = window.HelenTimekeeperBus;
    if (existing && existing.version === GLOBAL_BUS_VERSION) {
      return existing;
    }
    const bus = createGlobalBus();
    window.HelenTimekeeperBus = bus;
    return bus;
  };

  const globalBus = ensureGlobalBus();

  const listeners = {
    update: new Set(),
    fired: new Set(),
  };

  const store = new Map();
  const pendingMessages = [];
  const readyResolvers = [];
  let workerPort = null;
  let workerReady = false;
  let audioContext = null;
  let audioUnlocked = false;
  let audioQueueTime = 0;
  let fallbackAudio = null;
  let toastContainer = null;
  let reconciled = false;
  let modalController = null;
  let pendingNotifications = [];
  let currentNotification = null;
  let modalSoundNeedsUnlock = false;
  let soundAutoStopTimer = null;
  let pendingSoundPromise = null;
  let fallbackLoopInterval = null;
  let htmlAudioElement = null;
  let activeLoopSource = null;
  let loopGainNode = null;
  let audioBuffer = null;
  let audioBufferPromise = null;

  const readyPromise = new Promise((resolve) => {
    readyResolvers.push(resolve);
  });

  const now = () => Date.now();

  const uuid = () => {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
      return crypto.randomUUID();
    }
    const random = Math.random().toString(16).slice(2, 10);
    const time = now().toString(16);
    return `helen-${time}-${random}`;
  };

  const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

  const toNumber = (value, fallback = 0) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  };

  const clone = (value) => {
    try {
      return JSON.parse(JSON.stringify(value));
    } catch (error) {
      return value;
    }
  };

  const domReadyPromise = new Promise((resolve) => {
    if (typeof document === 'undefined') {
      resolve();
      return;
    }
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => resolve(), { once: true });
    } else {
      resolve();
    }
  });

  const onDomReady = (callback) => {
    if (typeof callback !== 'function') return;
    domReadyPromise.then(() => {
      try {
        callback();
      } catch (error) {
        console.error('[HelenScheduler] Error en callback DOM listo', error);
      }
    });
  };

  const emit = (event, payload) => {
    const callbacks = listeners[event];
    if (!callbacks) return;
    callbacks.forEach((handler) => {
      try {
        handler(payload);
      } catch (error) {
        console.error('[HelenScheduler] Error en listener', error);
      }
    });
  };

  const on = (event, handler) => {
    if (listeners[event]) {
      listeners[event].add(handler);
    }
  };

  const off = (event, handler) => {
    if (listeners[event]) {
      listeners[event].delete(handler);
    }
  };

  const safeLocalStorage = {
    get(key) {
      try {
        return window.localStorage.getItem(key);
      } catch (error) {
        console.warn('[HelenScheduler] No se pudo leer localStorage:', error);
        return null;
      }
    },
    set(key, value) {
      try {
        window.localStorage.setItem(key, value);
      } catch (error) {
        console.warn('[HelenScheduler] No se pudo guardar localStorage:', error);
      }
    },
  };

  const loadFromStorage = () => {
    const raw = safeLocalStorage.get(STORAGE_KEY);
    if (!raw) {
      return [];
    }
    try {
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch (error) {
      console.warn('[HelenScheduler] No se pudo parsear el estado guardado:', error);
      return [];
    }
  };

  const updatePendingNotificationsFromStorage = (rawValue) => {
    let value = rawValue;
    if (typeof value !== 'string') {
      value = safeLocalStorage.get(PENDING_STORAGE_KEY);
    }
    if (!value) {
      pendingNotifications = [];
      return;
    }
    try {
      const parsed = JSON.parse(value);
      if (!Array.isArray(parsed)) {
        pendingNotifications = [];
        return;
      }
      pendingNotifications = parsed
        .filter((entry) => entry && typeof entry === 'object' && typeof entry.eventId === 'string')
        .map((entry) => ({ ...entry }))
        .sort((a, b) => toNumber(a?.firedAt, 0) - toNumber(b?.firedAt, 0));
    } catch (error) {
      pendingNotifications = [];
    }
  };

  const persistPendingNotifications = () => {
    const snapshot = pendingNotifications.map((entry) => ({ ...entry }));
    safeLocalStorage.set(PENDING_STORAGE_KEY, JSON.stringify(snapshot));
  };

  updatePendingNotificationsFromStorage();

  const persistState = () => {
    const snapshot = Array.from(store.values()).map((item) => clone(item));
    safeLocalStorage.set(STORAGE_KEY, JSON.stringify(snapshot));
  };

  const sendToWorker = (message) => {
    if (!workerPort) return;
    if (!workerReady) {
      pendingMessages.push(message);
      return;
    }
    try {
      workerPort.postMessage(message);
    } catch (error) {
      console.warn('[HelenScheduler] No se pudo enviar mensaje al worker:', error);
    }
  };

  const syncWithWorker = () => {
    const payload = Array.from(store.values()).map((item) => ({
      id: item.id,
      type: item.type,
      state: item.state,
      targetEpochMs: typeof item.targetEpochMs === 'number' ? item.targetEpochMs : null,
      remainingMs: typeof item.remainingMs === 'number' ? item.remainingMs : null,
    }));
    sendToWorker({ type: 'set-items', items: payload });
  };

  const notifyUpdate = () => {
    emit('update', list());
    persistState();
    syncWithWorker();
  };

  const ensureToastContainer = () => {
    if (toastContainer || !document || !document.body) return toastContainer;
    const container = document.createElement('div');
    container.className = 'helen-toast-stack';
    container.setAttribute('data-toast-stack', '');
    document.body.appendChild(container);
    toastContainer = container;
    return container;
  };

  const pushToast = ({ title, body, tone = 'info', duration = TIMER_TOAST_DURATION }) => {
    const container = ensureToastContainer();
    if (!container) return;
    const card = document.createElement('div');
    card.className = `helen-toast-card helen-toast-card--${tone}`;

    const icon = document.createElement('div');
    icon.className = 'helen-toast-card__icon';
    icon.innerHTML = tone === 'alarm'
      ? '<i class="bi bi-bell" aria-hidden="true"></i>'
      : '<i class="bi bi-stopwatch" aria-hidden="true"></i>';

    const content = document.createElement('div');
    content.className = 'helen-toast-card__body';
    const heading = document.createElement('div');
    heading.className = 'helen-toast-card__title';
    heading.textContent = title || '';
    const message = document.createElement('div');
    message.className = 'helen-toast-card__message';
    message.textContent = body || '';
    content.append(heading, message);

    card.append(icon, content);
    container.append(card);

    requestAnimationFrame(() => {
      card.classList.add('is-visible');
    });

    const remove = () => {
      card.classList.remove('is-visible');
      setTimeout(() => {
        card.remove();
        if (container.childElementCount === 0) {
          container.remove();
          toastContainer = null;
        }
      }, 320);
    };

    if (duration > 0) {
      setTimeout(remove, duration);
    }

    card.addEventListener('click', remove, { once: true });
  };

  const scheduleAutoRemoval = (item, delay = AUTO_REMOVE_DELAY) => {
    if (!item || !item.id || !item.metadata || !item.metadata.autoRemove) {
      return;
    }
    window.setTimeout(() => {
      const current = store.get(item.id);
      if (!current || current.type !== 'timer') {
        return;
      }
      if (current.state === 'completed' && current.metadata && current.metadata.autoRemove) {
        store.delete(current.id);
        notifyUpdate();
      }
    }, Math.max(1200, delay));
  };

  const formatDurationForNotification = (durationMs) => {
    const safeDuration = Math.max(0, toNumber(durationMs, 0));
    if (!safeDuration) {
      return null;
    }
    const totalSeconds = Math.round(safeDuration / ONE_SECOND);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    const digital = hours > 0
      ? `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`
      : `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    let natural = '';
    if (hours > 0) {
      natural = `${hours}h ${minutes.toString().padStart(2, '0')}m`;
    } else if (minutes > 0) {
      natural = seconds ? `${minutes} min ${seconds} s` : `${minutes} min`;
    } else {
      natural = `${seconds} s`;
    }
    return { digital, natural };
  };

  const formatTimeOfDay = (epochMs) => {
    if (!Number.isFinite(epochMs)) return '';
    try {
      const date = new Date(epochMs);
      return date.toLocaleTimeString('es-MX', { hour: '2-digit', minute: '2-digit' });
    } catch (error) {
      return '';
    }
  };

  const formatAlarmTargetDisplay = (item) => {
    const meta = item && item.metadata ? item.metadata : {};
    const hours24 = toNumber(meta.hours24, NaN);
    const minutes = clamp(toNumber(meta.minutes, 0), 0, 59);
    if (!Number.isFinite(hours24)) {
      return null;
    }
    try {
      const reference = new Date();
      reference.setHours(hours24);
      reference.setMinutes(minutes);
      reference.setSeconds(0, 0);
      return {
        digital: reference.toLocaleTimeString('es-MX', { hour: '2-digit', minute: '2-digit' }),
        spoken: reference.toLocaleTimeString('es-MX', { hour: 'numeric', minute: '2-digit' }),
      };
    } catch (error) {
      return null;
    }
  };

  const buildNotificationSnapshot = (item, tone, firedAt) => {
    if (!item || !item.id) return null;
    const baseLabel = item.label || (item.type === 'timer' ? 'Temporizador' : 'Alarma');
    const eventId = `${item.id}:${toNumber(firedAt, now())}`;
    const firedTime = formatTimeOfDay(firedAt);
    let detail = '';
    let meta = '';
    let secondaryLabel = 'Descartar';

    if (item.type === 'timer') {
      const formattedDuration = formatDurationForNotification(item.metadata && item.metadata.durationMs);
      if (formattedDuration) {
        detail = `Duración: ${formattedDuration.digital}${formattedDuration.natural ? ` (${formattedDuration.natural})` : ''}`;
      } else {
        detail = 'El temporizador ha finalizado.';
      }
      if (firedTime) {
        meta = `Se activó a las ${firedTime}`;
      }
    } else {
      const target = formatAlarmTargetDisplay(item);
      if (target) {
        detail = `Programada para las ${target.digital}`;
        meta = `Se activó a las ${target.digital}`;
      } else {
        detail = 'La alarma se activó.';
      }
      const repeatDays = Array.isArray(item.metadata && item.metadata.repeatDays)
        ? item.metadata.repeatDays.filter((day) => Number.isFinite(day))
        : [];
      if (repeatDays.length) {
        secondaryLabel = 'Repetir';
      }
    }

    return {
      id: item.id,
      eventId,
      tone: tone || (item.type === 'timer' ? 'timer' : 'alarm'),
      type: item.type,
      firedAt,
      title: item.type === 'timer' ? 'Temporizador finalizado' : 'Alarma activada',
      label: baseLabel,
      detail,
      meta,
      secondaryLabel,
    };
  };

  const ensureModalController = () => {
    if (typeof window === 'undefined') {
      return null;
    }
    if (modalController && typeof modalController.show === 'function') {
      return modalController;
    }
    const api = window.HelenNotifications;
    if (!api) {
      return null;
    }
    try {
      modalController = typeof api.ensure === 'function' ? api.ensure() : api;
    } catch (error) {
      modalController = null;
    }
    return modalController;
  };

  const setModalStatus = (message, visible = true) => {
    const controller = ensureModalController();
    if (!controller || typeof controller.setStatus !== 'function') {
      return;
    }
    if (!message) {
      controller.setStatus('', false);
      return;
    }
    controller.setStatus(message, visible);
  };

  const setPrimaryDisabled = (disabled) => {
    const controller = ensureModalController();
    if (!controller || typeof controller.setPrimaryDisabled !== 'function') {
      return;
    }
    controller.setPrimaryDisabled(Boolean(disabled));
  };

  const updateModalContent = (entry) => {
    const controller = ensureModalController();
    if (!controller) {
      return;
    }
    if (typeof controller.update === 'function') {
      controller.update(entry);
    }
  };

  const updatePendingIndicator = () => {
    const controller = ensureModalController();
    if (!controller || typeof controller.setPending !== 'function') {
      return;
    }
    const count = Math.max(0, pendingNotifications.length - (currentNotification ? 1 : 0));
    controller.setPending(count);
  };

  const showModalEntry = (entry) => {
    const controller = ensureModalController();
    currentNotification = entry;
    pendingSoundPromise = null;
    modalSoundNeedsUnlock = false;
    updateModalContent(entry);
    if (controller && typeof controller.show === 'function') {
      controller.show(entry, {
        pendingCount: Math.max(0, pendingNotifications.length - 1),
        onPrimary: (source) => acknowledgeCurrentNotification(source || 'primary'),
      });
      setModalStatus('', false);
      setPrimaryDisabled(false);
    }
    updatePendingIndicator();
    tryEnsureSoundPlayback();
  };

  const hideModal = (reason) => {
    const controller = ensureModalController();
    if (!controller || typeof controller.hide !== 'function') {
      return;
    }
    controller.hide(reason);
  };

  const clearSoundAutoStop = () => {
    if (soundAutoStopTimer && typeof window !== 'undefined') {
      window.clearTimeout(soundAutoStopTimer);
    }
    soundAutoStopTimer = null;
  };

  const stopActiveSound = () => {
    clearSoundAutoStop();
    audioEngine.stop();
  };

  const scheduleSoundAutoStop = () => {
    clearSoundAutoStop();
    if (typeof window === 'undefined') {
      return;
    }
    const limit = Number(SOUND_CONFIG.autoStopAfterMs);
    if (!Number.isFinite(limit) || limit <= 0) {
      return;
    }
    soundAutoStopTimer = window.setTimeout(() => {
      audioEngine.stop();
      soundAutoStopTimer = null;
      setModalStatus('Sonido detenido automáticamente.', true);
      setPrimaryDisabled(false);
    }, limit);
  };

  const tryEnsureSoundPlayback = () => {
    if (!currentNotification) return;
    const controller = ensureModalController();
    if (!controller) return;
    if (pendingSoundPromise) {
      return;
    }
    pendingSoundPromise = audioEngine.playPersistent()
      .then((result) => {
        pendingSoundPromise = null;
        if (!currentNotification) {
          stopActiveSound();
          return;
        }
        if (result && result.ok) {
          modalSoundNeedsUnlock = false;
          setModalStatus('', false);
          setPrimaryDisabled(false);
          scheduleSoundAutoStop();
        } else {
          modalSoundNeedsUnlock = true;
          stopActiveSound();
          setPrimaryDisabled(false);
          setModalStatus('Activa el sonido tocando la pantalla.', true);
        }
      })
      .catch(() => {
        pendingSoundPromise = null;
        modalSoundNeedsUnlock = true;
        stopActiveSound();
        setModalStatus('No se pudo reproducir el sonido.', true);
        setPrimaryDisabled(false);
      });
  };

  const removeNotificationByEventId = (eventId) => {
    if (!eventId) return;
    const index = pendingNotifications.findIndex((entry) => entry && entry.eventId === eventId);
    if (index !== -1) {
      pendingNotifications.splice(index, 1);
      persistPendingNotifications();
    }
    updatePendingIndicator();
  };

  const acknowledgeCurrentNotification = (reason = 'acknowledge') => {
    if (!currentNotification) {
      hideModal(reason);
      return;
    }
    const snapshot = currentNotification;
    stopActiveSound();
    modalSoundNeedsUnlock = false;
    currentNotification = null;
    removeNotificationByEventId(snapshot.eventId);
    hideModal(reason);
    if (snapshot.type === 'timer' && window.HelenScheduler && typeof window.HelenScheduler.cancelTimer === 'function') {
      try {
        window.HelenScheduler.cancelTimer(snapshot.id);
      } catch (error) {
        console.warn('[HelenScheduler] No se pudo cancelar temporizador finalizado', error);
      }
    }
    processNotificationQueue();
  };

  const processNotificationQueue = () => {
    if (currentNotification) return;
    if (!pendingNotifications.length) {
      stopActiveSound();
      hideModal('queue-empty');
      updatePendingIndicator();
      return;
    }
    const next = pendingNotifications[0];
    showModalEntry(next);
  };

  const queueNotificationSnapshot = (snapshot) => {
    if (!snapshot || !snapshot.eventId) return;
    if (pendingNotifications.some((entry) => entry.eventId === snapshot.eventId)) {
      return;
    }
    pendingNotifications.push(snapshot);
    pendingNotifications.sort((a, b) => toNumber(a?.firedAt, 0) - toNumber(b?.firedAt, 0));
    persistPendingNotifications();
    updatePendingIndicator();
    const payload = { ...snapshot };
    const channel = snapshot.type === 'timer' ? 'timer:finished' : 'alarm:triggered';
    if (globalBus) {
      try {
        globalBus.emit(GLOBAL_EVENT_NAME, payload);
        globalBus.emit(channel, payload);
      } catch (error) {
        console.error('[HelenScheduler] Error emitiendo evento global', error);
      }
    } else {
      dispatchWindowEvent(GLOBAL_EVENT_NAME, payload);
      dispatchWindowEvent(channel, payload);
    }
    onDomReady(() => {
      if (!currentNotification) {
        processNotificationQueue();
      }
    });
  };

  const queueDueNotification = (item, tone, firedAt) => {
    const snapshot = buildNotificationSnapshot(item, tone, firedAt);
    if (!snapshot) return;
    queueNotificationSnapshot(snapshot);
  };

  onDomReady(() => {
    ensureModalController();
    if (pendingNotifications.length && !currentNotification) {
      processNotificationQueue();
    }
  });

  const ensureAudioContext = () => {
    if (audioContext) {
      return audioContext;
    }
    if (!audioUnlocked) {
      audioUnlocked = true;
      try {
        const Ctor = window.AudioContext || window.webkitAudioContext;
        if (Ctor) {
          audioContext = new Ctor();
        }
      } catch (error) {
        audioContext = null;
      }
    }
    if (!fallbackAudio && typeof Audio === 'function') {
      try {
        fallbackAudio = new Audio('data:audio/wav;base64,UklGRhQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=');
        const volume = clamp(Number(SOUND_CONFIG.elementVolume) || 1, 0, 1);
        fallbackAudio.volume = volume;
      } catch (error) {
        fallbackAudio = null;
      }
    }
    return audioContext;
  };

  const ensureHtmlAudioElement = () => {
    if (htmlAudioElement) {
      return htmlAudioElement;
    }
    if (typeof Audio !== 'function') {
      return null;
    }
    try {
      const element = new Audio(SOUND_CONFIG.url);
      element.loop = true;
      element.preload = 'auto';
      const volume = clamp(Number(SOUND_CONFIG.elementVolume) || 1, 0, 1);
      element.volume = volume;
      htmlAudioElement = element;
      return element;
    } catch (error) {
      htmlAudioElement = null;
      return null;
    }
  };

  const stopHtmlAudioElement = () => {
    if (!htmlAudioElement) return;
    try {
      htmlAudioElement.pause();
      htmlAudioElement.currentTime = 0;
    } catch (error) {
      // ignore
    }
  };

  const stopFallbackLoop = () => {
    if (fallbackLoopInterval && typeof window !== 'undefined') {
      window.clearInterval(fallbackLoopInterval);
    }
    fallbackLoopInterval = null;
  };

  const startFallbackLoop = () => {
    stopFallbackLoop();
    if (typeof window === 'undefined') {
      playChime();
      return;
    }
    fallbackLoopInterval = window.setInterval(() => {
      playChime();
    }, SOUND_CONFIG.fallbackInterval);
    playChime();
  };

  const decodeAudioBuffer = (context, arrayBuffer) => new Promise((resolve, reject) => {
    if (!context || typeof context.decodeAudioData !== 'function') {
      reject(new Error('decodeAudioData no disponible'));
      return;
    }
    try {
      const result = context.decodeAudioData(
        arrayBuffer,
        (buffer) => resolve(buffer),
        (error) => reject(error)
      );
      if (result && typeof result.then === 'function') {
        result.then(resolve).catch(reject);
      }
    } catch (error) {
      reject(error);
    }
  });

  const loadAudioBuffer = async () => {
    if (audioBuffer) return audioBuffer;
    if (audioBufferPromise) return audioBufferPromise;
    const context = ensureAudioContext();
    if (!context || typeof fetch !== 'function') {
      return null;
    }
    audioBufferPromise = fetch(SOUND_CONFIG.url, { cache: 'force-cache' })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`status ${response.status}`);
        }
        return response.arrayBuffer();
      })
      .then((arrayBuffer) => decodeAudioBuffer(context, arrayBuffer))
      .then((buffer) => {
        audioBuffer = buffer;
        return buffer;
      })
      .catch((error) => {
        console.warn('[HelenScheduler] No se pudo precargar audio de alarma:', error);
        audioBufferPromise = null;
        audioBuffer = null;
        return null;
      });
    return audioBufferPromise;
  };

  const audioEngine = {
    async preload() {
      const context = ensureAudioContext();
      if (!context) return null;
      try {
        return await loadAudioBuffer();
      } catch (error) {
        return null;
      }
    },
    async playPersistent() {
      stopFallbackLoop();
      stopHtmlAudioElement();
      if (activeLoopSource) {
        try { activeLoopSource.stop(); } catch (error) {}
        activeLoopSource = null;
      }
      if (loopGainNode) {
        try { loopGainNode.disconnect(); } catch (error) {}
        loopGainNode = null;
      }

      const context = ensureAudioContext();
      if (context) {
        if (context.state === 'suspended') {
          try { await context.resume(); } catch (error) {}
        }
        if (context.state !== 'running') {
          return { ok: false, reason: 'suspended' };
        }
        const buffer = await loadAudioBuffer();
        if (buffer) {
          try {
            const source = context.createBufferSource();
            source.buffer = buffer;
            source.loop = true;
            const gain = context.createGain ? context.createGain() : null;
            if (gain) {
              const targetGain = clamp(Number(SOUND_CONFIG.gain) || 0.9, 0, 1);
              gain.gain.setValueAtTime(targetGain, context.currentTime);
              source.connect(gain);
              gain.connect(context.destination);
              loopGainNode = gain;
            } else {
              source.connect(context.destination);
              loopGainNode = null;
            }
            activeLoopSource = source;
            source.onended = () => {
              if (activeLoopSource === source) {
                activeLoopSource = null;
              }
            };
            source.start();
            return { ok: true, mode: 'buffer' };
          } catch (error) {
            console.warn('[HelenScheduler] No se pudo iniciar loop de audio:', error);
          }
        }
        startFallbackLoop();
        return { ok: true, mode: 'beep' };
      }

      const element = ensureHtmlAudioElement();
      if (element) {
        try {
          await element.play();
          return { ok: true, mode: 'element' };
        } catch (error) {
          return { ok: false, reason: 'blocked' };
        }
      }

      return { ok: false, reason: 'unsupported' };
    },
    stop() {
      stopFallbackLoop();
      if (activeLoopSource) {
        try { activeLoopSource.stop(); } catch (error) {}
        activeLoopSource = null;
      }
      if (loopGainNode) {
        try { loopGainNode.disconnect(); } catch (error) {}
        loopGainNode = null;
      }
      stopHtmlAudioElement();
    },
    isPlaying() {
      const elementPlaying = htmlAudioElement && !htmlAudioElement.paused;
      return Boolean(activeLoopSource || fallbackLoopInterval || elementPlaying);
    },
  };

  const unlockAudioOnGesture = () => {
    const unlock = () => {
      const context = ensureAudioContext();
      if (context && context.state === 'suspended') {
        context.resume().catch(() => {});
      }
      audioEngine.preload().catch(() => {});
      if (modalSoundNeedsUnlock) {
        tryEnsureSoundPlayback();
      }
      window.removeEventListener('pointerdown', unlock, true);
      window.removeEventListener('keydown', unlock, true);
      window.removeEventListener('touchstart', unlock, true);
      window.removeEventListener('click', unlock, true);
    };
    window.addEventListener('pointerdown', unlock, true);
    window.addEventListener('keydown', unlock, true);
    window.addEventListener('touchstart', unlock, true);
    window.addEventListener('click', unlock, true);
  };

  const playChime = () => {
    const context = ensureAudioContext();
    if (context) {
      if (context.state === 'suspended') {
        context.resume().catch(() => {});
      }
      const startAt = Math.max(context.currentTime, audioQueueTime);
      const gain = context.createGain();
      gain.gain.setValueAtTime(0, startAt);
      gain.connect(context.destination);

      const oscillator = context.createOscillator();
      oscillator.type = 'sine';
      oscillator.frequency.setValueAtTime(880, startAt);
      const chimeGain = clamp(Number(SOUND_CONFIG.chimeGain) || 0.6, 0, 1);
      gain.gain.linearRampToValueAtTime(chimeGain, startAt + 0.02);
      gain.gain.linearRampToValueAtTime(0, startAt + 1.1);
      oscillator.connect(gain);
      oscillator.start(startAt);
      oscillator.stop(startAt + 1.2);

      audioQueueTime = startAt + 0.4;
      return;
    }

    if (fallbackAudio) {
      try {
        const cloneAudio = fallbackAudio.cloneNode();
        const volume = clamp(Number(SOUND_CONFIG.elementVolume) || 1, 0, 1);
        try { cloneAudio.volume = volume; } catch (error) {}
        cloneAudio.play().catch(() => {});
      } catch (error) {
        console.warn('[HelenScheduler] No se pudo reproducir sonido de respaldo:', error);
      }
    }
  };

  const computeAlarmTarget = (item, reference = now()) => {
    const meta = item.metadata || {};
    const hours24 = toNumber(meta.hours24, 0);
    const minutes = clamp(toNumber(meta.minutes, 0), 0, 59);
    if (!Number.isFinite(hours24)) {
      return null;
    }

    const base = new Date(reference);
    base.setMilliseconds(0);
    base.setSeconds(0);
    base.setMinutes(minutes);
    base.setHours(hours24);

    let candidate = base.getTime();
    const days = Array.isArray(meta.repeatDays) ? meta.repeatDays.map((day) => Number(day)).filter((day) => Number.isFinite(day)) : [];
    const currentDay = new Date(reference).getDay();

    if (days.length) {
      let best = null;
      days.sort((a, b) => a - b);
      for (const day of days) {
        let delta = day - currentDay;
        if (delta < 0) delta += 7;
        let scheduled = candidate + delta * ONE_DAY;
        if (delta === 0 && scheduled <= reference + ONE_SECOND) {
          scheduled += 7 * ONE_DAY;
        }
        if (best === null || scheduled < best) {
          best = scheduled;
        }
      }
      return best;
    }

    if (candidate <= reference + ONE_SECOND) {
      candidate += ONE_DAY;
    }
    return candidate;
  };

  const reconcilePastDueItems = () => {
    if (reconciled) return;
    reconciled = true;
    const current = now();
    const triggered = [];
    store.forEach((item) => {
      if (item.type === 'timer' && item.targetEpochMs && item.targetEpochMs <= current) {
        handleTimerTriggered(item, current, { silent: true });
        triggered.push({ item, tone: 'timer' });
      } else if (item.type === 'alarm' && item.metadata && item.metadata.active && item.targetEpochMs && item.targetEpochMs <= current) {
        handleAlarmTriggered(item, current, { silent: true });
        triggered.push({ item, tone: 'alarm' });
      }
    });
    if (triggered.length) {
      notifyUpdate();
      triggered.forEach(({ item, tone }) => {
        const body = tone === 'alarm'
          ? `${item.label ? `“${item.label}”` : 'La alarma'} se activó mientras estabas fuera.`
          : `${item.label ? `“${item.label}”` : 'El temporizador'} finalizó mientras estabas fuera.`;
        pushToast({
          title: tone === 'alarm' ? 'Alarma activada' : 'Temporizador finalizado',
          body,
          tone,
        });
      });
    } else {
      syncWithWorker();
    }
  };

  const handleTimerTriggered = (item, firedAt, { silent } = {}) => {
    item.state = 'completed';
    item.targetEpochMs = null;
    item.remainingMs = 0;
    item.metadata = item.metadata || {};
    item.metadata.lastTriggeredAt = firedAt;
    queueDueNotification(item, 'timer', firedAt);
    if (!silent) {
      const bodyLabel = item.label ? `“${item.label}”` : 'El temporizador';
      pushToast({
        title: 'Temporizador finalizado',
        body: `${bodyLabel} ha finalizado.`,
        tone: 'timer',
      });
    }
    scheduleAutoRemoval(item);
  };

  const handleAlarmTriggered = (item, firedAt, { silent } = {}) => {
    item.metadata = item.metadata || {};
    item.metadata.lastTriggeredAt = firedAt;
    const nextTarget = computeAlarmTarget(item, firedAt + ONE_SECOND);
    if (nextTarget && item.metadata.active) {
      item.targetEpochMs = nextTarget;
      item.remainingMs = nextTarget - now();
      item.state = 'running';
    } else {
      item.metadata.active = false;
      item.targetEpochMs = null;
      item.remainingMs = null;
      item.state = 'completed';
    }
    queueDueNotification(item, 'alarm', firedAt);
    if (!silent) {
      const bodyLabel = item.label ? `“${item.label}”` : 'La alarma';
      pushToast({
        title: 'Alarma activada',
        body: `${bodyLabel} se activó.`,
        tone: 'alarm',
      });
    }
  };

  const handleWorkerFired = (payload) => {
    if (!Array.isArray(payload)) return;
    const firedAt = now();
    payload.forEach((entry) => {
      if (!entry || !entry.id) return;
      const item = store.get(entry.id);
      if (!item) return;
      if (item.type === 'timer') {
        handleTimerTriggered(item, firedAt);
      } else if (item.type === 'alarm') {
        handleAlarmTriggered(item, firedAt);
      }
      emit('fired', clone(item));
    });
    notifyUpdate();
  };

  const handleWorkerTick = (payload) => {
    if (!Array.isArray(payload)) return;
    let changed = false;
    payload.forEach((entry) => {
      if (!entry || !entry.id) return;
      const item = store.get(entry.id);
      if (!item) return;
      const remaining = toNumber(entry.remainingMs, 0);
      if (item.remainingMs !== remaining) {
        item.remainingMs = remaining;
        changed = true;
      }
    });
    if (changed) {
      emit('update', list());
    }
  };

  const handleWorkerMessage = (event) => {
    const data = event.data;
    if (!data || typeof data !== 'object') {
      return;
    }
    switch (data.type) {
      case 'ready': {
        workerReady = true;
        while (pendingMessages.length) {
          const message = pendingMessages.shift();
          sendToWorker(message);
        }
        syncWithWorker();
        readyResolvers.forEach((resolve) => resolve());
        readyResolvers.length = 0;
        break;
      }
      case 'tick':
        handleWorkerTick(data.payload);
        break;
      case 'fired':
        handleWorkerFired(data.payload);
        break;
      default:
        break;
    }
  };

  const resolveWorkerScriptUrl = () => {
    const script = document.currentScript;
    if (script && script.src) {
      return new URL('./time-worker.js', script.src).toString();
    }
    return new URL('./time-worker.js', window.location.href).toString();
  };

  const setupWorker = () => {
    const url = resolveWorkerScriptUrl();
    try {
      if (typeof SharedWorker !== 'undefined') {
        const shared = new SharedWorker(url);
        workerPort = shared.port;
        workerPort.start();
        workerPort.onmessage = handleWorkerMessage;
        return;
      }
    } catch (error) {
      console.warn('[HelenScheduler] No se pudo crear SharedWorker, usando Worker estándar.', error);
    }
    const dedicated = new Worker(url);
    workerPort = dedicated;
    dedicated.onmessage = handleWorkerMessage;
  };

  const list = (type) => {
    const items = Array.from(store.values());
    if (type) {
      return items.filter((item) => item.type === type).map((item) => clone(item));
    }
    return items.map((item) => clone(item));
  };

  const saveItem = (item) => {
    item.updatedAt = now();
    store.set(item.id, item);
    notifyUpdate();
  };

  const removeItem = (id) => {
    if (store.delete(id)) {
      notifyUpdate();
    }
  };

  const formatTimerLabel = (durationMs) => {
    if (!Number.isFinite(durationMs)) return 'Temporizador';
    const totalSeconds = Math.max(0, Math.round(durationMs / ONE_SECOND));
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    if (hours > 0) {
      return `${hours}h ${minutes.toString().padStart(2, '0')}m`;
    }
    if (minutes > 0) {
      return `${minutes}m ${seconds.toString().padStart(2, '0')}s`;
    }
    return `${seconds}s`;
  };

  const createAlarm = (options = {}) => {
    const hoursRaw = options.hours != null ? String(options.hours).padStart(2, '0') : '07';
    const minutesRaw = options.minutes != null ? String(options.minutes).padStart(2, '0') : '00';
    const ampm = String(options.ampm || 'AM').toUpperCase();

    let hours24 = toNumber(hoursRaw, 0);
    const minutes = clamp(toNumber(minutesRaw, 0), 0, 59);
    if (ampm === 'PM' && hours24 < 12) {
      hours24 += 12;
    }
    if (ampm === 'AM' && hours24 === 12) {
      hours24 = 0;
    }

    const metadata = {
      hours: hoursRaw.padStart(2, '0'),
      minutes: minutes.toString().padStart(2, '0'),
      ampm,
      hours24,
      repeatDays: Array.isArray(options.repeatDays) ? options.repeatDays.map((day) => Number(day)).filter((day) => Number.isFinite(day)) : [],
      active: options.active !== false,
      label: options.label ? String(options.label) : '',
    };

    const item = {
      id: options.id || uuid(),
      type: 'alarm',
      label: options.label ? String(options.label) : 'Alarma',
      createdAt: now(),
      updatedAt: now(),
      targetEpochMs: null,
      remainingMs: null,
      state: 'paused',
      metadata,
    };

    if (metadata.active) {
      const target = computeAlarmTarget(item);
      if (target) {
        item.targetEpochMs = target;
        item.remainingMs = target - now();
        item.state = 'running';
      }
    }

    store.set(item.id, item);
    notifyUpdate();
    return item.id;
  };

  const updateAlarm = (id, updates = {}) => {
    const item = store.get(id);
    if (!item || item.type !== 'alarm') return;
    if (!item.metadata) item.metadata = {};

    if (updates.label != null) {
      item.label = String(updates.label);
    }
    if (updates.hours != null) {
      item.metadata.hours = String(updates.hours).padStart(2, '0');
    }
    if (updates.minutes != null) {
      item.metadata.minutes = String(updates.minutes).padStart(2, '0');
    }
    if (updates.ampm) {
      item.metadata.ampm = String(updates.ampm).toUpperCase();
    }
    if (updates.repeatDays) {
      item.metadata.repeatDays = Array.isArray(updates.repeatDays)
        ? updates.repeatDays.map((day) => Number(day)).filter((day) => Number.isFinite(day))
        : [];
    }

    const ampm = item.metadata.ampm || 'AM';
    let hours24 = toNumber(item.metadata.hours, 0);
    const minutes = clamp(toNumber(item.metadata.minutes, 0), 0, 59);
    if (ampm === 'PM' && hours24 < 12) hours24 += 12;
    if (ampm === 'AM' && hours24 === 12) hours24 = 0;
    item.metadata.hours24 = hours24;
    item.metadata.minutes = minutes.toString().padStart(2, '0');

    if (updates.active != null) {
      item.metadata.active = Boolean(updates.active);
    }

    if (item.metadata.active) {
      const target = computeAlarmTarget(item);
      if (target) {
        item.targetEpochMs = target;
        item.remainingMs = target - now();
        item.state = 'running';
      }
    } else {
      item.targetEpochMs = null;
      item.remainingMs = null;
      item.state = 'paused';
    }

    saveItem(item);
  };

  const toggleAlarm = (id, active) => {
    const item = store.get(id);
    if (!item || item.type !== 'alarm') return;
    item.metadata = item.metadata || {};
    item.metadata.active = Boolean(active);
    if (item.metadata.active) {
      const target = computeAlarmTarget(item);
      if (target) {
        item.targetEpochMs = target;
        item.remainingMs = target - now();
        item.state = 'running';
      }
    } else {
      item.targetEpochMs = null;
      item.remainingMs = null;
      item.state = 'paused';
    }
    saveItem(item);
  };

  const deleteAlarm = (id) => {
    removeItem(id);
  };

  const createTimer = (options = {}) => {
    const durationMs = Math.max(ONE_SECOND, toNumber(options.durationMs, 0));
    const label = options.label || formatTimerLabel(durationMs);
    const autoRemove = options.autoRemove !== false;
    const target = now() + durationMs;
    const item = {
      id: options.id || uuid(),
      type: 'timer',
      label,
      createdAt: now(),
      updatedAt: now(),
      targetEpochMs: target,
      remainingMs: durationMs,
      state: 'running',
      metadata: {
        durationMs,
        autoRemove,
      },
    };
    store.set(item.id, item);
    notifyUpdate();
    return item.id;
  };

  const pauseTimer = (id) => {
    const item = store.get(id);
    if (!item || item.type !== 'timer' || item.state !== 'running') return;
    const remaining = Math.max(0, toNumber(item.targetEpochMs, now()) - now());
    item.state = 'paused';
    item.targetEpochMs = null;
    item.remainingMs = remaining;
    saveItem(item);
  };

  const resumeTimer = (id) => {
    const item = store.get(id);
    if (!item || item.type !== 'timer') return;
    const remaining = Math.max(ONE_SECOND, toNumber(item.remainingMs, 0));
    item.targetEpochMs = now() + remaining;
    item.remainingMs = remaining;
    item.state = 'running';
    saveItem(item);
  };

  const resetTimer = (id) => {
    const item = store.get(id);
    if (!item || item.type !== 'timer') return;
    const duration = toNumber(item.metadata && item.metadata.durationMs, ONE_MINUTE);
    item.targetEpochMs = now() + duration;
    item.remainingMs = duration;
    item.state = 'running';
    saveItem(item);
  };

  const cancelTimer = (id) => {
    removeItem(id);
  };

  const hydrate = () => {
    const data = loadFromStorage();
    if (Array.isArray(data)) {
      data.forEach((item) => {
        if (!item || !item.id) return;
        store.set(item.id, item);
      });
    }
    reconcilePastDueItems();
    notifyUpdate();
  };

  setupWorker();
  hydrate();
  unlockAudioOnGesture();

  window.addEventListener('storage', (event) => {
    if (event.key === STORAGE_KEY && event.newValue) {
      store.clear();
      hydrate();
    }
    if (event.key === PENDING_STORAGE_KEY) {
      updatePendingNotificationsFromStorage(event.newValue);
      updatePendingIndicator();
      onDomReady(() => {
        if (!pendingNotifications.length) {
          if (!currentNotification) {
            hideModal('storage-sync');
          }
          return;
        }
        if (!currentNotification) {
          processNotificationQueue();
        }
      });
    }
  });

  window.HelenScheduler = {
    version: VERSION,
    ready: () => readyPromise,
    on,
    off,
    list,
    getItem: (id) => clone(store.get(id)),
    createAlarm,
    updateAlarm,
    toggleAlarm,
    deleteAlarm,
    createTimer,
    pauseTimer,
    resumeTimer,
    resetTimer,
    cancelTimer,
  };
})();
