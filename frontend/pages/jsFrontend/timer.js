// =====================================================
// HELEN – Temporizadores persistentes (UI + Scheduler)
// =====================================================

(function () {
  'use strict';

  const state = {
    timers: [],
    selectedId: null,
    statusOverride: null,
    statusTimeout: null,
  };

  let scheduler = null;
  const SCHEDULER_POLL_INTERVAL_MS = 60;
  const SCHEDULER_MAX_WAIT_MS = 5000;
  let displayEl;
  let labelEl;
  let statusEl;
  let startPauseBtn;
  let resetBtn;
  let presetsGrid;
  let openCustomBtn;
  let timersList;
  let noTimersMessage;
  let customModal;
  let modalCloseBtn;
  let customCancelBtn;
  let customSetBtn;
  let customHours;
  let customMinutes;
  let customSeconds;
  let visualTicker = null;

  const pad2 = (value) => String(value).padStart(2, '0');

  const formatForDisplay = (ms) => {
    const safe = Math.max(0, Math.round(Number(ms || 0) / 1000));
    const hours = Math.floor(safe / 3600);
    const minutes = Math.floor((safe % 3600) / 60);
    const seconds = safe % 60;
    if (hours > 0) {
      return `${pad2(hours)}:${pad2(minutes)}:${pad2(seconds)}`;
    }
    return `${pad2(minutes)}:${pad2(seconds)}`;
  };

  const computeRemainingMs = (timer) => {
    if (!timer) {
      return 0;
    }
    if (timer.state === 'completed') {
      return 0;
    }
    if (typeof timer.remainingMs === 'number') {
      return Math.max(0, timer.remainingMs);
    }
    if (typeof timer.targetEpochMs === 'number') {
      return Math.max(0, timer.targetEpochMs - Date.now());
    }
    return 0;
  };

  const describeStatus = (timer) => {
    if (!timer) {
      return { text: 'Selecciona un temporizador.', icon: 'bi-clock-history' };
    }
    switch (timer.state) {
      case 'running':
        return { text: 'En curso', icon: 'bi-play-fill' };
      case 'paused':
        return { text: 'Pausado', icon: 'bi-pause-fill' };
      case 'completed':
        return { text: 'Finalizado', icon: 'bi-flag' };
      default:
        return { text: 'En espera', icon: 'bi-clock-history' };
    }
  };

  const sortTimers = (timers) => {
    const order = { running: 0, paused: 1, completed: 2 };
    return timers.slice().sort((a, b) => {
      const aOrder = order[a.state] ?? 3;
      const bOrder = order[b.state] ?? 3;
      if (aOrder !== bOrder) {
        return aOrder - bOrder;
      }
      const aKey = typeof a.targetEpochMs === 'number' ? a.targetEpochMs : (a.updatedAt || a.createdAt || 0);
      const bKey = typeof b.targetEpochMs === 'number' ? b.targetEpochMs : (b.updatedAt || b.createdAt || 0);
      if (aKey !== bKey) {
        return aKey - bKey;
      }
      return (b.createdAt || 0) - (a.createdAt || 0);
    });
  };

  const getSelectedTimer = () => {
    if (!state.selectedId) {
      return null;
    }
    return state.timers.find((timer) => timer.id === state.selectedId) || null;
  };

  const clearStatusOverride = () => {
    if (state.statusTimeout) {
      window.clearTimeout(state.statusTimeout);
      state.statusTimeout = null;
    }
    state.statusOverride = null;
  };

  const setStatusOverride = (message, tone = 'info', duration = 0) => {
    if (!statusEl) {
      return;
    }
    if (!message) {
      clearStatusOverride();
      renderMain();
      return;
    }
    if (state.statusTimeout) {
      window.clearTimeout(state.statusTimeout);
      state.statusTimeout = null;
    }
    state.statusOverride = { message, tone };
    renderMain();
    if (duration > 0) {
      state.statusTimeout = window.setTimeout(() => {
        if (state.statusOverride && state.statusOverride.message === message) {
          clearStatusOverride();
          renderMain();
        }
      }, duration);
    }
  };

  const updateControlButtons = (timer) => {
    if (!startPauseBtn || !resetBtn) {
      return;
    }

    const setPrimaryButton = (disabled, icon, label) => {
      startPauseBtn.disabled = disabled;
      startPauseBtn.innerHTML = `<i class="bi ${icon}" aria-hidden="true"></i>`;
      startPauseBtn.setAttribute('aria-label', label);
      startPauseBtn.title = label;
    };

    const setResetButton = (disabled, label) => {
      resetBtn.disabled = disabled;
      resetBtn.innerHTML = '<i class="bi bi-arrow-counterclockwise" aria-hidden="true"></i>';
      resetBtn.setAttribute('aria-label', label);
      resetBtn.title = label;
    };

    if (!timer) {
      setPrimaryButton(true, 'bi-play-fill', 'Iniciar temporizador');
      setResetButton(true, 'Restablecer temporizador');
      return;
    }

    const hasDuration = timer.metadata && Number(timer.metadata.durationMs) > 0;
    setResetButton(!hasDuration, 'Restablecer temporizador');

    if (timer.state === 'running') {
      setPrimaryButton(false, 'bi-pause-fill', 'Pausar temporizador');
    } else if (timer.state === 'paused') {
      setPrimaryButton(false, 'bi-play-fill', 'Reanudar temporizador');
    } else {
      setPrimaryButton(false, 'bi-arrow-clockwise', 'Reiniciar temporizador');
    }
  };

  const renderMain = (sorted) => {
    if (!displayEl || !labelEl || !statusEl) {
      return;
    }

    const timers = Array.isArray(sorted) ? sorted : sortTimers(state.timers);

    if (!timers.length) {
      state.selectedId = null;
    } else if (!state.selectedId || !timers.some((timer) => timer.id === state.selectedId)) {
      state.selectedId = timers[0].id;
    }

    const timer = getSelectedTimer();

    if (!timer) {
      displayEl.textContent = '00:00';
      labelEl.textContent = 'Sin temporizador seleccionado';
      const message = state.statusOverride ? state.statusOverride.message : 'Usa los presets para crear uno nuevo.';
      statusEl.textContent = message;
      statusEl.classList.toggle('is-warning', Boolean(state.statusOverride && state.statusOverride.tone === 'warning'));
      updateControlButtons(null);
      return;
    }

    const remaining = computeRemainingMs(timer);
    displayEl.textContent = formatForDisplay(remaining);
    labelEl.textContent = timer.label || 'Temporizador';

    if (state.statusOverride) {
      statusEl.textContent = state.statusOverride.message;
      statusEl.classList.toggle('is-warning', state.statusOverride.tone === 'warning');
    } else {
      const descriptor = describeStatus(timer);
      statusEl.textContent = descriptor.text;
      statusEl.classList.remove('is-warning');
    }

    updateControlButtons(timer);
  };

  const updateTimersListTimes = () => {
    if (!timersList) {
      return;
    }
    const map = new Map(state.timers.map((timer) => [timer.id, timer]));
    timersList.querySelectorAll('.timer-item').forEach((item) => {
      const id = item.dataset.timerId;
      if (!id || !map.has(id)) {
        return;
      }
      const timer = map.get(id);
      const timeEl = item.querySelector('.timer-item__time');
      if (timeEl) {
        timeEl.textContent = formatForDisplay(computeRemainingMs(timer));
      }
      const metaEl = item.querySelector('.timer-item__meta');
      if (metaEl) {
        const descriptor = describeStatus(timer);
        const iconEl = metaEl.querySelector('i');
        const textEl = metaEl.querySelector('span');
        if (iconEl) {
          iconEl.className = `bi ${descriptor.icon}`;
        }
        if (textEl) {
          textEl.textContent = descriptor.text;
        }
      }
    });
  };

  const updateVisualCountdowns = () => {
    if (!state.timers.length) {
      renderMain();
      return;
    }
    renderMain();
    updateTimersListTimes();
  };

  const startVisualTicker = () => {
    if (visualTicker) {
      return;
    }
    visualTicker = window.setInterval(() => {
      if (!state.timers.length) {
        renderMain();
        return;
      }
      updateVisualCountdowns();
    }, 500);
  };

  const buildTimerCard = (timer) => {
    const article = document.createElement('article');
    article.className = 'timer-item';
    article.dataset.timerId = timer.id;
    article.setAttribute('role', 'listitem');
    if (timer.state) {
      article.dataset.state = timer.state;
    }
    if (timer.id === state.selectedId) {
      article.classList.add('is-selected');
    }

    const body = document.createElement('div');
    body.className = 'timer-item__body';
    body.dataset.action = 'select';
    body.setAttribute('role', 'button');
    body.setAttribute('tabindex', '0');
    body.setAttribute('aria-pressed', timer.id === state.selectedId ? 'true' : 'false');
    body.setAttribute('aria-label', `Seleccionar ${timer.label || 'temporizador'}`);

    const titleRow = document.createElement('div');
    titleRow.className = 'timer-item__title';
    const title = document.createElement('span');
    title.textContent = timer.label || 'Temporizador';
    const time = document.createElement('span');
    time.className = 'timer-item__time';
    time.textContent = formatForDisplay(computeRemainingMs(timer));
    titleRow.append(title, time);

    const meta = document.createElement('div');
    meta.className = 'timer-item__meta';
    const descriptor = describeStatus(timer);
    meta.innerHTML = `<i class="bi ${descriptor.icon}" aria-hidden="true"></i><span>${descriptor.text}</span>`;

    body.append(titleRow, meta);

    const actions = document.createElement('div');
    actions.className = 'timer-item__actions';

    const toggleBtn = document.createElement('button');
    toggleBtn.type = 'button';
    toggleBtn.className = 'timer-item__btn';
    if (timer.state === 'running') {
      toggleBtn.dataset.action = 'pause';
      toggleBtn.setAttribute('aria-label', 'Pausar temporizador');
      toggleBtn.innerHTML = '<i class="bi bi-pause-fill" aria-hidden="true"></i>';
    } else if (timer.state === 'paused') {
      toggleBtn.dataset.action = 'resume';
      toggleBtn.setAttribute('aria-label', 'Reanudar temporizador');
      toggleBtn.innerHTML = '<i class="bi bi-play-fill" aria-hidden="true"></i>';
    } else {
      toggleBtn.dataset.action = 'reset';
      toggleBtn.setAttribute('aria-label', 'Reiniciar temporizador');
      toggleBtn.innerHTML = '<i class="bi bi-arrow-clockwise" aria-hidden="true"></i>';
    }

    const resetButton = document.createElement('button');
    resetButton.type = 'button';
    resetButton.className = 'timer-item__btn';
    resetButton.dataset.action = 'reset';
    resetButton.setAttribute('aria-label', 'Restablecer temporizador');
    resetButton.innerHTML = '<i class="bi bi-arrow-counterclockwise" aria-hidden="true"></i>';

    const deleteButton = document.createElement('button');
    deleteButton.type = 'button';
    deleteButton.className = 'timer-item__btn';
    deleteButton.dataset.action = 'delete';
    deleteButton.setAttribute('aria-label', 'Eliminar temporizador');
    deleteButton.innerHTML = '<i class="bi bi-x-lg" aria-hidden="true"></i>';

    actions.append(toggleBtn, resetButton, deleteButton);
    article.append(body, actions);
    return article;
  };

  const renderTimersList = (sorted) => {
    if (!timersList || !noTimersMessage) {
      return;
    }

    const timers = Array.isArray(sorted) ? sorted : sortTimers(state.timers);
    timersList.innerHTML = '';

    if (!timers.length) {
      timersList.setAttribute('aria-hidden', 'true');
      noTimersMessage.hidden = false;
      return;
    }

    noTimersMessage.hidden = true;
    timersList.removeAttribute('aria-hidden');

    timers.forEach((timer) => {
      const card = buildTimerCard(timer);
      timersList.appendChild(card);
    });
  };

  const render = (sorted) => {
    const timers = Array.isArray(sorted) ? sorted : sortTimers(state.timers);
    if (!timers.length) {
      state.selectedId = null;
    } else if (!state.selectedId || !timers.some((timer) => timer.id === state.selectedId)) {
      state.selectedId = timers[0].id;
    }
    renderMain(timers);
    renderTimersList(timers);
  };

  const selectTimer = (timerId) => {
    if (!timerId || state.selectedId === timerId) {
      return;
    }
    state.selectedId = timerId;
    clearStatusOverride();
    render();
  };

  const handleSchedulerUpdate = (items = []) => {
    state.timers = Array.isArray(items)
      ? items.filter((item) => item && item.type === 'timer')
      : [];
    console.debug('[Timer] Evento de actualización recibido.', { total: state.timers.length });
    render();
    updateVisualCountdowns();
  };

  const createTimer = (durationMs, labelText) => {
    if (!scheduler) {
      return null;
    }
    const duration = Number(durationMs);
    if (!Number.isFinite(duration) || duration <= 0) {
      setStatusOverride('El tiempo debe ser mayor a 0.', 'warning', 3200);
      return null;
    }
    const trimmedLabel = labelText ? String(labelText).trim() : '';
    const id = scheduler.createTimer({ durationMs: duration, label: trimmedLabel || undefined, autoRemove: true });
    state.selectedId = id;
    clearStatusOverride();
    console.debug('[Timer] Temporizador creado.', { id, duracionMs: duration, etiqueta: trimmedLabel || undefined });
    return id;
  };

  const findTimer = (id) => {
    if (id) {
      return state.timers.find((timer) => timer.id === id) || null;
    }
    return getSelectedTimer();
  };

  const handlePresetClick = (event) => {
    const button = event.target.closest('.preset-btn');
    if (!button) {
      return;
    }
    const minutes = Number(button.dataset.min || 0);
    if (!Number.isFinite(minutes) || minutes <= 0) {
      setStatusOverride('Selecciona un preset válido.', 'warning', 2800);
      return;
    }
    const durationMs = minutes * 60 * 1000;
    const label = button.textContent ? `Temporizador ${button.textContent.trim()}` : undefined;
    createTimer(durationMs, label);
  };

  const handlePrimaryAction = () => {
    const timer = getSelectedTimer();
    if (!timer || !scheduler) {
      return;
    }
    console.debug('[Timer] Acción principal ejecutada.', { id: timer.id, estado: timer.state });
    if (timer.state === 'running') {
      scheduler.pauseTimer(timer.id);
    } else if (timer.state === 'paused') {
      scheduler.resumeTimer(timer.id);
    } else {
      scheduler.resetTimer(timer.id);
    }
    window.setTimeout(() => {
      updateVisualCountdowns();
    }, 80);
  };

  const handleResetAction = () => {
    const timer = getSelectedTimer();
    if (!timer || !scheduler) {
      return;
    }
    console.debug('[Timer] Reinicio solicitado.', { id: timer.id, estado: timer.state });
    scheduler.resetTimer(timer.id);
    window.setTimeout(() => {
      updateVisualCountdowns();
    }, 80);
  };

  const handleTimersListClick = (event) => {
    const actionEl = event.target.closest('[data-action]');
    if (!actionEl) {
      return;
    }
    const timerItem = actionEl.closest('.timer-item');
    if (!timerItem) {
      return;
    }
    const timerId = timerItem.dataset.timerId;
    if (!timerId) {
      return;
    }
    const action = actionEl.dataset.action;
    switch (action) {
      case 'select':
        selectTimer(timerId);
        break;
      case 'pause':
        scheduler && scheduler.pauseTimer(timerId);
        break;
      case 'resume':
        scheduler && scheduler.resumeTimer(timerId);
        break;
      case 'reset':
        scheduler && scheduler.resetTimer(timerId);
        break;
      case 'delete':
        scheduler && scheduler.cancelTimer(timerId);
        if (state.selectedId === timerId) {
          state.selectedId = null;
        }
        break;
      default:
        break;
    }
  };

  const handleTimersListKeydown = (event) => {
    if (event.key !== 'Enter' && event.key !== ' ') {
      return;
    }
    const body = event.target.closest('.timer-item__body[data-action="select"]');
    if (!body) {
      return;
    }
    const timerItem = body.closest('.timer-item');
    if (!timerItem || !timerItem.dataset.timerId) {
      return;
    }
    event.preventDefault();
    selectTimer(timerItem.dataset.timerId);
  };

  const openCustomModal = () => {
    if (!customModal) {
      return;
    }
    if (customHours) customHours.value = 0;
    if (customMinutes) customMinutes.value = 0;
    if (customSeconds) customSeconds.value = 0;
    customModal.classList.add('open');
  };

  const closeCustomModal = () => {
    if (customModal) {
      customModal.classList.remove('open');
    }
  };

  const clampInputs = () => {
    if (customHours) {
      const value = Number(customHours.value || 0);
      customHours.value = Math.min(23, Math.max(0, value));
    }
    if (customMinutes) {
      const value = Number(customMinutes.value || 0);
      customMinutes.value = Math.min(59, Math.max(0, value));
    }
    if (customSeconds) {
      const value = Number(customSeconds.value || 0);
      customSeconds.value = Math.min(59, Math.max(0, value));
    }
  };

  const setupStepperButtons = () => {
    document.querySelectorAll('.ct-plus').forEach((button) => {
      button.addEventListener('click', () => {
        const target = button.dataset.target;
        if (!target) return;
        const element = target === 'hours' ? customHours : target === 'minutes' ? customMinutes : customSeconds;
        if (!element) return;
        const max = target === 'hours' ? 23 : 59;
        element.value = Math.min(max, Number(element.value || 0) + 1);
      });
    });

    document.querySelectorAll('.ct-minus').forEach((button) => {
      button.addEventListener('click', () => {
        const target = button.dataset.target;
        if (!target) return;
        const element = target === 'hours' ? customHours : target === 'minutes' ? customMinutes : customSeconds;
        if (!element) return;
        element.value = Math.max(0, Number(element.value || 0) - 1);
      });
    });
  };

  const handleCustomCreate = () => {
    clampInputs();
    const hours = Number(customHours && customHours.value ? customHours.value : 0);
    const minutes = Number(customMinutes && customMinutes.value ? customMinutes.value : 0);
    const seconds = Number(customSeconds && customSeconds.value ? customSeconds.value : 0);
    const totalSeconds = (hours * 3600) + (minutes * 60) + seconds;
    if (totalSeconds <= 0) {
      setStatusOverride('El tiempo debe ser mayor a 0.', 'warning', 3200);
      return;
    }
    const labelParts = [];
    if (hours) labelParts.push(`${hours}h`);
    if (minutes) labelParts.push(`${minutes}m`);
    if (seconds) labelParts.push(`${seconds}s`);
    const label = labelParts.length ? `Temporizador ${labelParts.join(' ')}` : undefined;
    createTimer(totalSeconds * 1000, label);
    closeCustomModal();
  };

  const exposeTimerAPI = () => {
    window.TimerAPI = {
      set: (h = 0, m = 0, s = 0, autostart = true) => {
        const durationMs = ((Number(h) || 0) * 3600 + (Number(m) || 0) * 60 + (Number(s) || 0)) * 1000;
        const id = createTimer(durationMs);
        if (!autostart && id) {
          scheduler.pauseTimer(id);
        }
        return id;
      },
      start: (id) => {
        const timer = findTimer(id);
        if (timer && timer.state !== 'running') {
          scheduler.resumeTimer(timer.id);
        }
      },
      pause: (id) => {
        const timer = findTimer(id);
        if (timer && timer.state === 'running') {
          scheduler.pauseTimer(timer.id);
        }
      },
      reset: (id) => {
        const timer = findTimer(id);
        if (timer) {
          scheduler.resetTimer(timer.id);
        }
      },
      cancel: (id) => {
        const timer = findTimer(id);
        if (timer) {
          scheduler.cancelTimer(timer.id);
        }
      },
      running: (id) => {
        const timer = findTimer(id);
        return Boolean(timer && timer.state === 'running');
      },
      remainingMs: (id) => {
        const timer = findTimer(id);
        return computeRemainingMs(timer);
      },
      list: () => scheduler.list('timer'),
    };
  };

  const waitForScheduler = () => new Promise((resolve, reject) => {
    if (window.HelenScheduler) {
      resolve(window.HelenScheduler);
      return;
    }
    const deadline = Date.now() + SCHEDULER_MAX_WAIT_MS;
    const poll = () => {
      if (window.HelenScheduler) {
        resolve(window.HelenScheduler);
        return;
      }
      if (Date.now() >= deadline) {
        reject(new Error('HelenScheduler no disponible'));
        return;
      }
      window.setTimeout(poll, SCHEDULER_POLL_INTERVAL_MS);
    };
    poll();
  });

  const init = () => {
    displayEl = document.getElementById('timerDisplay');
    labelEl = document.getElementById('timerLabel');
    statusEl = document.getElementById('timerStatus');
    startPauseBtn = document.getElementById('startPauseBtn');
    resetBtn = document.getElementById('resetBtn');
    presetsGrid = document.getElementById('presetsGrid');
    openCustomBtn = document.getElementById('openCustomBtn');
    timersList = document.getElementById('timersList');
    noTimersMessage = document.getElementById('noTimersMessage');
    customModal = document.getElementById('customTimeModal');
    modalCloseBtn = customModal ? customModal.querySelector('.close-modal') : null;
    customCancelBtn = document.getElementById('customCancelBtn');
    customSetBtn = document.getElementById('customSetBtn');
    customHours = document.getElementById('customHours');
    customMinutes = document.getElementById('customMinutes');
    customSeconds = document.getElementById('customSeconds');

    if (presetsGrid) {
      presetsGrid.addEventListener('click', handlePresetClick);
    }
    if (startPauseBtn) {
      startPauseBtn.addEventListener('click', handlePrimaryAction);
    }
    if (resetBtn) {
      resetBtn.addEventListener('click', handleResetAction);
    }
    if (openCustomBtn) {
      openCustomBtn.addEventListener('click', openCustomModal);
    }
    if (modalCloseBtn) {
      modalCloseBtn.addEventListener('click', closeCustomModal);
    }
    if (customCancelBtn) {
      customCancelBtn.addEventListener('click', closeCustomModal);
    }
    if (customSetBtn) {
      customSetBtn.addEventListener('click', handleCustomCreate);
    }
    if (customModal) {
      customModal.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
          closeCustomModal();
        }
      });
    }
    if (timersList) {
      timersList.addEventListener('click', handleTimersListClick);
      timersList.addEventListener('keydown', handleTimersListKeydown);
    }

    setupStepperButtons();
    exposeTimerAPI();

    waitForScheduler()
      .then((instance) => {
        console.debug('[Timer] HelenScheduler detectado.');
        scheduler = instance;

        if (typeof scheduler.on === 'function') {
          scheduler.on('update', handleSchedulerUpdate);
        }

        const listTimers = () => {
          if (typeof scheduler.list === 'function') {
            try {
              return scheduler.list('timer');
            } catch (error) {
              console.warn('[Timer] No se pudo obtener la lista de temporizadores:', error);
            }
          }
          return [];
        };

        const bootstrapTimers = () => {
          const snapshot = listTimers();
          handleSchedulerUpdate(Array.isArray(snapshot) ? snapshot : []);
        };

        if (typeof scheduler.ready === 'function') {
          scheduler.ready()
            .then(() => {
              const snapshot = listTimers();
              console.debug('[Timer] HelenScheduler listo.', { temporizadores: snapshot.length });
              handleSchedulerUpdate(snapshot);
            })
            .catch((error) => {
              console.error('[Timer] Error al preparar HelenScheduler:', error);
              bootstrapTimers();
            })
            .finally(() => {
              startVisualTicker();
            });
        } else {
          bootstrapTimers();
          startVisualTicker();
        }
      })
      .catch((error) => {
        console.error('[Timer] No se pudo inicializar HelenScheduler:', error);
        setStatusOverride('No se pudo inicializar el temporizador.', 'warning', 4800);
      });
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
