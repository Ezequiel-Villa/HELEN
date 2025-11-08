(function () {
  'use strict';

  const FALLBACK_STORAGE_KEY = 'helen:alarms:fallback:v1';
  const SCHEDULER_CHECK_INTERVAL = 50;
  const SCHEDULER_MAX_ATTEMPTS = 40;

  const safeLocalStorage = {
    get(key) {
      try {
        return window.localStorage.getItem(key);
      } catch (error) {
        console.warn('[Alarmas] No se pudo leer localStorage:', error);
        return null;
      }
    },
    set(key, value) {
      try {
        window.localStorage.setItem(key, value);
      } catch (error) {
        console.warn('[Alarmas] No se pudo escribir localStorage:', error);
      }
    },
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

  const computeAlarmTarget = (metadata, reference = Date.now()) => {
    if (!metadata) return null;
    const hours24 = toNumber(metadata.hours24, NaN);
    const minutes = clamp(toNumber(metadata.minutes, 0), 0, 59);
    if (!Number.isFinite(hours24)) {
      return null;
    }

    const base = new Date(reference);
    base.setMilliseconds(0);
    base.setSeconds(0);
    base.setMinutes(minutes);
    base.setHours(hours24);

    let candidate = base.getTime();
    const repeatDays = Array.isArray(metadata.repeatDays)
      ? metadata.repeatDays.map((day) => Number(day)).filter((day) => Number.isFinite(day))
      : [];
    const currentDay = new Date(reference).getDay();

    if (repeatDays.length) {
      let best = null;
      repeatDays.sort((a, b) => a - b);
      for (const day of repeatDays) {
        let delta = day - currentDay;
        if (delta < 0) delta += 7;
        let scheduled = candidate + delta * 24 * 60 * 60 * 1000;
        if (delta === 0 && scheduled <= reference + 1000) {
          scheduled += 7 * 24 * 60 * 60 * 1000;
        }
        if (best === null || scheduled < best) {
          best = scheduled;
        }
      }
      return best;
    }

    if (candidate <= reference + 1000) {
      candidate += 24 * 60 * 60 * 1000;
    }
    return candidate;
  };

  const createFallbackScheduler = () => {
    let items = [];
    const raw = safeLocalStorage.get(FALLBACK_STORAGE_KEY);
    if (raw) {
      try {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) {
          items = parsed.map((item) => ({
            ...item,
            metadata: item.metadata || {},
          }));
        }
      } catch (error) {
        console.warn('[Alarmas] No se pudo parsear fallback de alarmas:', error);
      }
    }

    const listeners = new Set();

    const persist = () => {
      safeLocalStorage.set(FALLBACK_STORAGE_KEY, JSON.stringify(items));
    };

    const emitUpdate = () => {
      const snapshot = items.map((item) => clone(item));
      listeners.forEach((handler) => {
        try {
          handler(snapshot);
        } catch (error) {
          console.error('[Alarmas] Error en listener fallback', error);
        }
      });
      persist();
    };

    const ensureMetadata = (item) => {
      const meta = item.metadata || {};
      item.metadata = meta;
      meta.hours = String(meta.hours || '07').padStart(2, '0');
      meta.minutes = String(clamp(toNumber(meta.minutes, 0), 0, 59)).padStart(2, '0');
      meta.ampm = String(meta.ampm || 'AM').toUpperCase();
      let hours24 = toNumber(meta.hours, 0);
      if (meta.ampm === 'PM' && hours24 < 12) {
        hours24 += 12;
      }
      if (meta.ampm === 'AM' && hours24 === 12) {
        hours24 = 0;
      }
      meta.hours24 = hours24;
      meta.repeatDays = Array.isArray(meta.repeatDays)
        ? meta.repeatDays.map((day) => Number(day)).filter((day) => Number.isFinite(day))
        : [];
      meta.active = meta.active !== false;
      if (typeof meta.label === 'string') {
        item.label = meta.label || 'Alarma';
      }
      if (!item.label) {
        item.label = 'Alarma';
      }
    };

    items.forEach((item) => ensureMetadata(item));

    const recalc = (item) => {
      ensureMetadata(item);
      item.updatedAt = Date.now();
      if (item.metadata.active) {
        const target = computeAlarmTarget(item.metadata);
        if (target) {
          item.targetEpochMs = target;
          item.remainingMs = Math.max(0, target - Date.now());
          item.state = 'running';
          return;
        }
      }
      item.targetEpochMs = null;
      item.remainingMs = null;
      item.state = 'paused';
    };

    items.forEach((item) => recalc(item));

    const list = (type) => {
      const snapshot = items.map((item) => clone(item));
      if (type) {
        return snapshot.filter((item) => item.type === type);
      }
      return snapshot;
    };

    const findIndex = (id) => items.findIndex((item) => item.id === id);

    return {
      ready: () => Promise.resolve(),
      on(event, handler) {
        if (event === 'update' && typeof handler === 'function') {
          listeners.add(handler);
        }
      },
      off(event, handler) {
        if (event === 'update' && handler) {
          listeners.delete(handler);
        }
      },
      list,
      createAlarm(options = {}) {
        const hoursRaw = options.hours != null ? String(options.hours).padStart(2, '0') : '07';
        const minutesRaw = options.minutes != null ? String(options.minutes).padStart(2, '0') : '00';
        const ampm = String(options.ampm || 'AM').toUpperCase();
        const repeatDays = Array.isArray(options.repeatDays)
          ? options.repeatDays.map((day) => Number(day)).filter((day) => Number.isFinite(day))
          : [];
        const id = options.id || `fallback-${Date.now()}-${Math.random().toString(16).slice(2)}`;
        const item = {
          id,
          type: 'alarm',
          label: options.label ? String(options.label) : 'Alarma',
          createdAt: Date.now(),
          updatedAt: Date.now(),
          metadata: {
            hours: hoursRaw,
            minutes: minutesRaw,
            ampm,
            repeatDays,
            active: options.active !== false,
            label: options.label ? String(options.label) : 'Alarma',
          },
          targetEpochMs: null,
          remainingMs: null,
          state: 'paused',
        };
        ensureMetadata(item);
        recalc(item);
        items.push(item);
        emitUpdate();
        return item.id;
      },
      updateAlarm(id, updates = {}) {
        const index = findIndex(id);
        if (index === -1) return;
        const item = items[index];
        if (updates.label != null) {
          item.label = String(updates.label);
          item.metadata.label = item.label;
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
        recalc(item);
        emitUpdate();
      },
      toggleAlarm(id, active) {
        const index = findIndex(id);
        if (index === -1) return;
        const item = items[index];
        const desired = typeof active === 'boolean' ? active : !item.metadata.active;
        item.metadata.active = desired;
        recalc(item);
        emitUpdate();
      },
      deleteAlarm(id) {
        const index = findIndex(id);
        if (index === -1) return;
        items.splice(index, 1);
        emitUpdate();
      },
    };
  };

  let scheduler = null;
  let schedulerPromise = null;

  const waitForScheduler = () => {
    if (scheduler) return Promise.resolve(scheduler);
    if (schedulerPromise) return schedulerPromise;

    schedulerPromise = new Promise((resolve) => {
      const finalize = (instance) => {
        scheduler = instance;
        resolve(instance);
      };

      const attemptAssign = () => {
        if (window.HelenScheduler) {
          finalize(window.HelenScheduler);
          return true;
        }
        return false;
      };

      if (attemptAssign()) {
        return;
      }

      let attempts = 0;
      const interval = window.setInterval(() => {
        attempts += 1;
        if (attemptAssign()) {
          window.clearInterval(interval);
        } else if (attempts >= SCHEDULER_MAX_ATTEMPTS) {
          window.clearInterval(interval);
          console.warn('[Alarmas] Usando modo fallback de alarmas.');
          finalize(createFallbackScheduler());
        }
      }, SCHEDULER_CHECK_INTERVAL);
    });

    return schedulerPromise;
  };

  const alarmList = document.querySelector('.alarm-list, .alarms-list');
  const loadingElement = document.querySelector('.alarm-loading');
  const noAlarmsMessage = document.querySelector('.no-alarms-message');
  const newAlarmBtn = document.getElementById('new-alarm-btn');
  const modal = document.getElementById('newAlarmModal');
  const modalClose = modal ? modal.querySelector('.close-modal') : null;
  const modalCancel = modal ? modal.querySelector('.cancel-btn') : null;
  const modalAction = modal ? modal.querySelector('.create-btn') : null;
  const modalTitle = modal ? modal.querySelector('.modal-header h2') : null;
  const timeInput = document.getElementById('alarm-time-input');
  const labelInput = document.getElementById('alarm-label-input');
  const daysContainer = document.getElementById('modal-days');
  const deleteModal = document.getElementById('deleteConfirmModal');
  const deleteCancel = deleteModal ? deleteModal.querySelector('[data-cancel-delete]') : null;
  const deleteClose = deleteModal ? deleteModal.querySelector('[data-close-delete]') : null;
  const deleteConfirm = document.getElementById('confirmDeleteBtn');
  const toast = document.getElementById('helenToast');

  let editingAlarmId = null;
  let pendingDeleteId = null;
  let alarms = [];

  const pad2 = (value) => String(value).padStart(2, '0');

  const showToast = (type, message) => {
    if (!toast) return;
    toast.className = 'helen-toast';
    toast.textContent = message || '';
    if (type === 'error') {
      toast.classList.add('error');
    } else {
      toast.classList.add('success');
    }
    toast.style.display = 'block';
    setTimeout(() => {
      toast.style.display = 'none';
    }, 3000);
  };

  const hideLoading = () => {
    if (loadingElement) loadingElement.style.display = 'none';
  };

  const showNoAlarms = (visible) => {
    if (noAlarmsMessage) {
      noAlarmsMessage.style.display = visible ? 'flex' : 'none';
    }
  };

  const getSelectedDays = () => {
    if (!daysContainer) return [];
    return Array.from(daysContainer.querySelectorAll('.day.active'))
      .map((day) => Number(day.dataset.day))
      .filter((day) => Number.isFinite(day));
  };

  const setSelectedDays = (days) => {
    if (!daysContainer) return;
    const set = new Set(days || []);
    daysContainer.querySelectorAll('.day').forEach((day) => {
      const value = Number(day.dataset.day);
      day.classList.toggle('active', set.has(value));
    });
  };

  const formatDisplayTime = (alarm) => {
    const meta = alarm.metadata || {};
    const hours24 = Number(meta.hours24);
    const minutes = Number(meta.minutes);
    const date = new Date();
    if (Number.isFinite(hours24)) {
      date.setHours(hours24);
    }
    if (Number.isFinite(minutes)) {
      date.setMinutes(minutes);
    }
    const hours = Number.isFinite(hours24) ? hours24 : date.getHours();
    const mins = Number.isFinite(minutes) ? minutes : date.getMinutes();
    const ampm = hours >= 12 ? 'PM' : 'AM';
    const hours12 = ((hours % 12) || 12);
    return `${pad2(hours12)}:${pad2(mins)} ${ampm}`;
  };

  const formatRepeatDays = (alarm) => {
    const meta = alarm.metadata || {};
    const days = Array.isArray(meta.repeatDays) ? meta.repeatDays : [];
    return days;
  };

  const renderDaysChips = (container, daysArray) => {
    if (!container) return;
    container.innerHTML = '';
    if (!Array.isArray(daysArray) || daysArray.length === 0) return;
    const order = [1, 2, 3, 4, 5, 6, 0];
    const letter = { 0: 'D', 1: 'L', 2: 'M', 3: 'X', 4: 'J', 5: 'V', 6: 'S' };
    order.forEach((day) => {
      if (daysArray.includes(day)) {
        const span = document.createElement('span');
        span.textContent = letter[day];
        container.appendChild(span);
      }
    });
  };

  const buildAlarmElement = (alarm) => {
    const template = document.getElementById('alarm-template');
    if (!template) return null;
    const fragment = template.content.cloneNode(true);
    const alarmItem = fragment.querySelector('.alarm-item');
    const timeEl = fragment.querySelector('.alarm-time');
    const nameEl = fragment.querySelector('.alarm-name');
    const daysEl = fragment.querySelector('.alarm-days');
    const toggleInput = fragment.querySelector('.switch input');
    const editBtn = fragment.querySelector('.edit-btn');
    const deleteBtn = fragment.querySelector('.delete-alarm');

    alarmItem.dataset.id = alarm.id;
    timeEl.textContent = formatDisplayTime(alarm);
    nameEl.textContent = alarm.label || 'Alarma';
    renderDaysChips(daysEl, formatRepeatDays(alarm));

    const isActive = Boolean(alarm.metadata && alarm.metadata.active);
    toggleInput.checked = isActive;
    alarmItem.classList.toggle('inactive', !isActive);

    toggleInput.addEventListener('change', (event) => {
      if (!scheduler) return;
      scheduler.toggleAlarm(alarm.id, event.target.checked);
      showToast(event.target.checked ? 'success' : 'success', event.target.checked ? 'Alarma activada' : 'Alarma desactivada');
    });

    if (editBtn) {
      editBtn.addEventListener('click', () => openAlarmModal('edit', alarm));
    }
    if (deleteBtn) {
      deleteBtn.addEventListener('click', () => openDeleteModal(alarm.id));
    }

    return fragment;
  };

  const renderAlarms = () => {
    if (!alarmList) return;
    alarmList.querySelectorAll('.alarm-item').forEach((item) => item.remove());
    hideLoading();
    if (!alarms.length) {
      showNoAlarms(true);
      return;
    }
    showNoAlarms(false);
    const sorted = alarms.slice().sort((a, b) => {
      const aTime = a.targetEpochMs || a.createdAt || 0;
      const bTime = b.targetEpochMs || b.createdAt || 0;
      return aTime - bTime;
    });
    sorted.forEach((alarm) => {
      const fragment = buildAlarmElement(alarm);
      if (fragment) {
        alarmList.appendChild(fragment);
      }
    });
  };

  const openAlarmModal = (mode, alarm) => {
    if (!modal) return;
    editingAlarmId = null;
    if (mode === 'edit' && alarm) {
      editingAlarmId = alarm.id;
      if (modalTitle) modalTitle.textContent = 'Editar alarma';
      if (timeInput) {
        const hours24 = Number(alarm.metadata && alarm.metadata.hours24);
        const minutes = Number(alarm.metadata && alarm.metadata.minutes);
        const hoursValue = Number.isFinite(hours24) ? pad2(hours24) : '07';
        const minutesValue = Number.isFinite(minutes) ? pad2(minutes) : '00';
        timeInput.value = `${hoursValue}:${minutesValue}`;
      }
      if (labelInput) {
        labelInput.value = alarm.label || '';
      }
      setSelectedDays(formatRepeatDays(alarm));
      if (modalAction) modalAction.textContent = 'Guardar';
    } else {
      if (modalTitle) modalTitle.textContent = 'Nueva alarma';
      if (timeInput) timeInput.value = '07:00';
      if (labelInput) labelInput.value = '';
      setSelectedDays([]);
      if (modalAction) modalAction.textContent = 'Crear';
    }
    modal.classList.add('open');
  };

  const closeAlarmModal = () => {
    if (!modal) return;
    modal.classList.remove('open');
    editingAlarmId = null;
  };

  const openDeleteModal = (id) => {
    if (!deleteModal) return;
    pendingDeleteId = id;
    deleteModal.classList.add('open');
  };

  const closeDeleteModal = () => {
    if (!deleteModal) return;
    pendingDeleteId = null;
    deleteModal.classList.remove('open');
  };

  const getAlarmPayloadFromModal = () => {
    if (!timeInput) return null;
    const timeValue = timeInput.value || '07:00';
    const [hoursStr, minutesStr] = timeValue.split(':');
    const hours24 = Number(hoursStr);
    const minutes = Number(minutesStr);
    const ampm = hours24 >= 12 ? 'PM' : 'AM';
    let hours12 = hours24 % 12;
    if (hours12 === 0) hours12 = 12;
    return {
      hours: pad2(hours12),
      minutes: pad2(minutes),
      ampm,
      repeatDays: getSelectedDays(),
      label: labelInput ? labelInput.value.trim() : '',
    };
  };

  const createAlarmFromModal = () => {
    const payload = getAlarmPayloadFromModal();
    if (!payload) return;
    waitForScheduler().then(() => {
      if (!scheduler) {
        showToast('error', 'No se pudo crear la alarma.');
        return;
      }
      scheduler.createAlarm({
        hours: payload.hours,
        minutes: payload.minutes,
        ampm: payload.ampm,
        repeatDays: payload.repeatDays,
        label: payload.label,
        active: true,
      });
      showToast('success', 'Alarma creada');
      closeAlarmModal();
    });
  };

  const saveEditAlarm = () => {
    if (!editingAlarmId) return;
    const payload = getAlarmPayloadFromModal();
    if (!payload) return;
    waitForScheduler().then(() => {
      if (!scheduler) {
        showToast('error', 'No se pudo actualizar la alarma.');
        return;
      }
      scheduler.updateAlarm(editingAlarmId, {
        hours: payload.hours,
        minutes: payload.minutes,
        ampm: payload.ampm,
        repeatDays: payload.repeatDays,
        label: payload.label,
      });
      showToast('success', 'Alarma actualizada');
      closeAlarmModal();
    });
  };

  const deleteAlarm = (id) => {
    if (!scheduler) return;
    scheduler.deleteAlarm(id);
    showToast('success', 'Alarma eliminada');
  };

  const updateClock = () => {
    const nowDate = new Date();
    const timeElement = document.querySelector('.time');
    const dateElement = document.querySelector('.date-section');

    if (timeElement) {
      let hours = nowDate.getHours();
      const minutes = nowDate.getMinutes().toString().padStart(2, '0');
      hours = hours % 12;
      hours = hours ? hours : 12;
      timeElement.textContent = `${hours.toString().padStart(2, '0')}:${minutes}`;
    }

    if (dateElement) {
      const options = { weekday: 'short', day: 'numeric', month: 'short' };
      dateElement.textContent = nowDate.toLocaleDateString('es-ES', options);
    }
  };

  const refreshAlarms = () => {
    if (!scheduler) return;
    alarms = scheduler.list('alarm') || [];
    renderAlarms();
  };

  const attachScheduler = () => {
    if (!scheduler) return;
    scheduler.on && scheduler.on('update', refreshAlarms);
    if (typeof scheduler.ready === 'function') {
      scheduler.ready().then(() => {
        refreshAlarms();
      });
    } else {
      refreshAlarms();
    }
  };

  const init = () => {
    waitForScheduler().then(() => {
      attachScheduler();
    });

    updateClock();
    setInterval(updateClock, 1000);

    if (newAlarmBtn) newAlarmBtn.addEventListener('click', () => openAlarmModal('create'));
    if (modalClose) modalClose.addEventListener('click', closeAlarmModal);
    if (modalCancel) modalCancel.addEventListener('click', closeAlarmModal);
    if (modal) {
      modal.addEventListener('click', (event) => {
        if (event.target === modal) closeAlarmModal();
      });
    }
    if (modalAction) {
      modalAction.addEventListener('click', () => {
        if (editingAlarmId) {
          saveEditAlarm();
        } else {
          createAlarmFromModal();
        }
      });
    }

    if (daysContainer) {
      daysContainer.querySelectorAll('.day').forEach((day) => {
        day.addEventListener('click', () => {
          day.classList.toggle('active');
        });
      });
    }

    if (deleteCancel) deleteCancel.addEventListener('click', closeDeleteModal);
    if (deleteClose) deleteClose.addEventListener('click', closeDeleteModal);
    if (deleteModal) {
      deleteModal.addEventListener('click', (event) => {
        if (event.target === deleteModal) {
          closeDeleteModal();
        }
      });
    }
    if (deleteConfirm) {
      deleteConfirm.addEventListener('click', () => {
        if (pendingDeleteId) {
          deleteAlarm(pendingDeleteId);
        }
        closeDeleteModal();
      });
    }

    waitForScheduler().then(() => {
      refreshAlarms();
      hideLoading();
    });
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
