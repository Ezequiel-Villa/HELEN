const ZONAS_HORARIAS = {
  mexicali: 'America/Tijuana',
  ensenada: 'America/Tijuana',
  tecate: 'America/Tijuana',
  rosarito: 'America/Tijuana',
  tijuana: 'America/Tijuana',
  chiapas: 'America/Mexico_City',
};

const CONFIG = {
  STORAGE_KEY: 'selectedCity',
  FORMAT_KEY: 'clockFormatMode',
  DEFAULT_CITY: 'tijuana',
};

(function () {
  'use strict';

  const elements = {
    clock: null,
    date: null,
    toggleButton: null,
    selector: null,
  };

  const state = {
    timezoneKey: CONFIG.DEFAULT_CITY,
    is24h: true,
    tickId: null,
    localTimeZone: Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC',
    timeFormatter: null,
    dateFormatter: null,
  };

  const MONTHS = [
    'enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
    'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre',
  ];

  const pad = (value) => String(value).padStart(2, '0');

  const sanitizeDayPeriod = (value) => {
    if (!value) {
      return '';
    }
    return value.replace(/\./g, '').replace(/\s+/g, '').toUpperCase();
  };

  const setElementText = (element, value) => {
    if (!element) {
      return;
    }
    if (element.textContent !== value) {
      element.textContent = value;
    }
  };

  const resolveTimezoneKey = () => {
    if (!state.localTimeZone) {
      return CONFIG.DEFAULT_CITY;
    }
    const match = Object.entries(ZONAS_HORARIAS).find(([, zone]) => zone === state.localTimeZone);
    return match ? match[0] : CONFIG.DEFAULT_CITY;
  };

  const activeTimeZone = () => ZONAS_HORARIAS[state.timezoneKey] || state.localTimeZone || 'UTC';

  const buildFormatters = () => {
    const tz = activeTimeZone();
    try {
      state.timeFormatter = new Intl.DateTimeFormat('es-MX', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: !state.is24h,
        timeZone: tz,
      });
      state.dateFormatter = new Intl.DateTimeFormat('es-MX', {
        day: 'numeric',
        month: 'long',
        year: 'numeric',
        timeZone: tz,
      });
    } catch (error) {
      console.error('[Helen] No se pudo construir formateadores de hora:', error);
      state.timeFormatter = null;
      state.dateFormatter = null;
    }
  };

  const formatTime = (date) => {
    if (!state.timeFormatter) {
      buildFormatters();
    }
    if (!state.timeFormatter) {
      const fallback = `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
      return state.is24h ? fallback : `${pad((date.getHours() % 12) || 12)}:${pad(date.getMinutes())}:${pad(date.getSeconds())} ${date.getHours() >= 12 ? 'PM' : 'AM'}`;
    }

    const parts = state.timeFormatter.formatToParts(date);
    const map = parts.reduce((acc, part) => {
      acc[part.type] = part.value;
      return acc;
    }, {});

    const hour = pad(map.hour || date.getHours());
    const minute = pad(map.minute || date.getMinutes());
    const second = pad(map.second || date.getSeconds());
    if (state.is24h) {
      return `${hour}:${minute}:${second}`;
    }
    const suffix = sanitizeDayPeriod(map.dayPeriod || '');
    return suffix ? `${hour}:${minute}:${second} ${suffix}` : `${hour}:${minute}:${second}`;
  };

  const formatDate = (date) => {
    if (!state.dateFormatter) {
      buildFormatters();
    }
    if (!state.dateFormatter) {
      const month = MONTHS[date.getMonth()] || '';
      const monthCap = month ? month.charAt(0).toUpperCase() + month.slice(1) : '';
      return `${date.getDate()} de ${monthCap}, ${date.getFullYear()}`;
    }
    const parts = state.dateFormatter.formatToParts(date);
    const day = parts.find((part) => part.type === 'day')?.value || String(date.getDate());
    const month = parts.find((part) => part.type === 'month')?.value || (MONTHS[date.getMonth()] || '');
    const year = parts.find((part) => part.type === 'year')?.value || String(date.getFullYear());
    const monthCap = month ? month.charAt(0).toUpperCase() + month.slice(1) : '';
    return `${day} de ${monthCap || month}, ${year}`;
  };

  const updateClock = () => {
    const now = new Date();
    setElementText(elements.clock, formatTime(now));
    setElementText(elements.date, formatDate(now));
  };

  const stopTick = () => {
    if (state.tickId) {
      window.clearInterval(state.tickId);
      state.tickId = null;
    }
  };

  const startTick = () => {
    stopTick();
    buildFormatters();
    updateClock();
    state.tickId = window.setInterval(updateClock, 1000);
  };

  const persistTimezone = (key) => {
    try {
      window.localStorage.setItem(CONFIG.STORAGE_KEY, key);
    } catch (error) {
      console.warn('[Helen] No se pudo guardar la zona horaria seleccionada:', error);
    }
  };

  const loadSavedTimezone = () => {
    try {
      const saved = window.localStorage.getItem(CONFIG.STORAGE_KEY);
      if (saved && ZONAS_HORARIAS[saved]) {
        state.timezoneKey = saved;
        return;
      }
    } catch (error) {
      console.warn('[Helen] No se pudo leer la zona horaria guardada:', error);
    }
    state.timezoneKey = resolveTimezoneKey();
  };

  const persistFormat = () => {
    try {
      window.localStorage.setItem(CONFIG.FORMAT_KEY, state.is24h ? '24' : '12');
    } catch (error) {
      console.warn('[Helen] No se pudo guardar el formato de hora:', error);
    }
  };

  const loadSavedFormat = () => {
    try {
      const saved = window.localStorage.getItem(CONFIG.FORMAT_KEY);
      if (saved === '12') {
        state.is24h = false;
      }
    } catch (error) {
      console.warn('[Helen] No se pudo leer el formato de hora guardado:', error);
    }
  };

  const updateToggleLabel = () => {
    if (elements.toggleButton) {
      elements.toggleButton.textContent = state.is24h ? 'Cambiar a 12 horas' : 'Cambiar a 24 horas';
    }
  };

  const highlightSelector = () => {
    if (!elements.selector) {
      return;
    }
    const target = ZONAS_HORARIAS[state.timezoneKey] ? state.timezoneKey : '';
    if (target) {
      elements.selector.value = target;
    } else {
      elements.selector.value = '';
    }
  };

  const handleSelectorChange = (event) => {
    const selected = event.target.value;
    if (!selected || !ZONAS_HORARIAS[selected]) {
      return;
    }
    if (state.timezoneKey === selected) {
      return;
    }
    state.timezoneKey = selected;
    persistTimezone(selected);
    buildFormatters();
    updateClock();
    console.debug('[Helen] Zona horaria actualizada a', selected, activeTimeZone());
  };

  const handleVisibilityChange = () => {
    if (document.hidden) {
      stopTick();
    } else {
      startTick();
    }
  };

  const init = () => {
    elements.clock = document.querySelector('.clock-item');
    elements.date = document.querySelector('.date-item');
    elements.toggleButton = document.querySelector('.toggle-format-btn');
    elements.selector = document.getElementById('citySelector');

    loadSavedFormat();
    loadSavedTimezone();
    highlightSelector();
    updateToggleLabel();

    if (elements.selector) {
      elements.selector.addEventListener('change', handleSelectorChange);
    }

    startTick();
  };

  const cleanup = () => {
    stopTick();
    if (elements.selector) {
      elements.selector.removeEventListener('change', handleSelectorChange);
    }
  };

  document.addEventListener('visibilitychange', handleVisibilityChange);
  window.addEventListener('beforeunload', cleanup);
  window.addEventListener('pagehide', cleanup);

  document.addEventListener('DOMContentLoaded', init);

  window.alternarFormatoHora = function alternarFormatoHora() {
    state.is24h = !state.is24h;
    updateToggleLabel();
    persistFormat();
    buildFormatters();
    updateClock();
  };
})();
