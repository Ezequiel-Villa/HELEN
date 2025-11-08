(function (global) {
  'use strict';

  const VERSION = 1;
  const BODY_CLASS = 'helen-global-modal-open';

  if (!global || (global.HelenNotifications && global.HelenNotifications.version >= VERSION)) {
    return;
  }

  const document = global.document;
  if (!document || !document.body) {
    return;
  }

  const state = {
    root: null,
    card: null,
    iconGlyph: null,
    badge: null,
    title: null,
    label: null,
    detail: null,
    meta: null,
    status: null,
    primary: null,
    onPrimary: null,
    lastActive: null,
  };

  const ensureElements = () => {
    if (state.root && document.body.contains(state.root)) {
      return state;
    }

    const root = document.createElement('div');
    root.className = 'helen-global-modal';
    root.dataset.helenModal = 'timekeeper';
    root.setAttribute('aria-hidden', 'true');
    root.setAttribute('aria-live', 'assertive');

    const card = document.createElement('div');
    card.className = 'helen-global-modal__card';
    card.setAttribute('role', 'alertdialog');
    card.setAttribute('aria-modal', 'true');
    card.setAttribute('aria-labelledby', 'helenGlobalModalTitle');
    card.setAttribute('aria-describedby', 'helenGlobalModalDescription');

    const badge = document.createElement('span');
    badge.className = 'helen-global-modal__badge';
    badge.hidden = true;

    const icon = document.createElement('div');
    icon.className = 'helen-global-modal__icon';
    const iconGlyph = document.createElement('i');
    iconGlyph.className = 'bi bi-bell-fill';
    iconGlyph.setAttribute('aria-hidden', 'true');
    icon.appendChild(iconGlyph);

    const title = document.createElement('h2');
    title.className = 'helen-global-modal__title';
    title.id = 'helenGlobalModalTitle';

    const label = document.createElement('p');
    label.className = 'helen-global-modal__label';

    const detail = document.createElement('p');
    detail.className = 'helen-global-modal__detail';
    detail.id = 'helenGlobalModalDescription';

    const meta = document.createElement('p');
    meta.className = 'helen-global-modal__meta';
    meta.hidden = true;

    const status = document.createElement('p');
    status.className = 'helen-global-modal__status';
    status.hidden = true;

    const primary = document.createElement('button');
    primary.type = 'button';
    primary.className = 'helen-global-modal__primary';
    primary.textContent = 'Detener';

    card.append(badge, icon, title, label, detail, meta, status, primary);
    root.appendChild(card);
    document.body.appendChild(root);

    const handlePrimary = (source) => {
      if (typeof state.onPrimary === 'function') {
        state.onPrimary(source || 'primary');
      }
    };

    primary.addEventListener('click', () => handlePrimary('primary'));

    card.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        handlePrimary('escape');
        return;
      }
      if (event.key === 'Tab') {
        event.preventDefault();
        try {
          primary.focus();
        } catch (error) {}
      }
    });

    state.root = root;
    state.card = card;
    state.iconGlyph = iconGlyph;
    state.badge = badge;
    state.title = title;
    state.label = label;
    state.detail = detail;
    state.meta = meta;
    state.status = status;
    state.primary = primary;

    return state;
  };

  const setTone = (tone) => {
    const elements = ensureElements();
    const iconClass = tone === 'timer' ? 'bi-stopwatch-fill' : 'bi-bell-fill';
    elements.iconGlyph.className = `bi ${iconClass}`;
    elements.root.dataset.tone = tone === 'timer' ? 'timer' : 'alarm';
  };

  const updateContent = (entry) => {
    const elements = ensureElements();
    const tone = entry && entry.tone === 'timer' ? 'timer' : entry && entry.type === 'timer' ? 'timer' : 'alarm';
    setTone(tone);
    elements.title.textContent = entry && entry.title
      ? entry.title
      : tone === 'timer' ? 'Temporizador finalizado' : 'Alarma activada';
    elements.label.textContent = entry && entry.label
      ? entry.label
      : tone === 'timer' ? 'Temporizador' : 'Alarma';
    elements.detail.textContent = entry && entry.detail
      ? entry.detail
      : 'Evento completado.';
    if (entry && entry.meta) {
      elements.meta.textContent = entry.meta;
      elements.meta.hidden = false;
    } else {
      elements.meta.textContent = '';
      elements.meta.hidden = true;
    }
  };

  const setStatus = (message, visible = true) => {
    const elements = ensureElements();
    if (!message) {
      elements.status.textContent = '';
      elements.status.classList.remove('is-visible');
      elements.status.hidden = true;
      return;
    }
    elements.status.textContent = message;
    if (visible) {
      elements.status.classList.add('is-visible');
      elements.status.hidden = false;
    } else {
      elements.status.classList.remove('is-visible');
      elements.status.hidden = true;
    }
  };

  const setPending = (count) => {
    const elements = ensureElements();
    const safe = Math.max(0, Number(count) || 0);
    if (safe > 0) {
      elements.badge.textContent = `+${safe}`;
      elements.badge.classList.add('is-visible');
      elements.badge.hidden = false;
    } else {
      elements.badge.textContent = '';
      elements.badge.classList.remove('is-visible');
      elements.badge.hidden = true;
    }
  };

  const setPrimaryDisabled = (disabled) => {
    const elements = ensureElements();
    elements.primary.disabled = Boolean(disabled);
  };

  const focusPrimary = () => {
    const elements = ensureElements();
    try {
      elements.primary.focus();
    } catch (error) {}
  };

  const show = (entry, options = {}) => {
    const elements = ensureElements();
    updateContent(entry);
    setPending(options.pendingCount || 0);
    if (options.statusMessage) {
      setStatus(options.statusMessage, true);
    } else {
      setStatus('', false);
    }
    setPrimaryDisabled(Boolean(options.disablePrimary));

    elements.onPrimary = null;
    state.onPrimary = typeof options.onPrimary === 'function' ? options.onPrimary : null;

    if (document && document.body && !document.body.classList.contains(BODY_CLASS)) {
      document.body.classList.add(BODY_CLASS);
    }

    elements.root.classList.add('is-visible');
    elements.root.setAttribute('aria-hidden', 'false');

    state.lastActive = document.activeElement && typeof document.activeElement.focus === 'function'
      ? document.activeElement
      : null;

    if (typeof requestAnimationFrame === 'function') {
      requestAnimationFrame(() => focusPrimary());
    } else {
      focusPrimary();
    }
  };

  const hide = (reason) => {
    const elements = ensureElements();
    elements.root.classList.remove('is-visible');
    elements.root.setAttribute('aria-hidden', 'true');
    state.onPrimary = null;
    setStatus('', false);
    if (document && document.body) {
      document.body.classList.remove(BODY_CLASS);
    }
    if (state.lastActive && typeof state.lastActive.focus === 'function') {
      try {
        if (document.contains(state.lastActive)) {
          state.lastActive.focus();
        }
      } catch (error) {}
    }
    state.lastActive = null;
    if (typeof reason === 'string' && reason) {
      elements.root.dataset.closeReason = reason;
    } else {
      delete elements.root.dataset.closeReason;
    }
  };

  const controller = {
    version: VERSION,
    ensure: () => {
      ensureElements();
      return controller;
    },
    show,
    hide,
    update: updateContent,
    setStatus,
    setPending,
    setPrimaryDisabled,
  };

  Object.defineProperty(controller, 'elements', {
    enumerable: false,
    configurable: false,
    get() {
      return ensureElements();
    },
  });

  global.HelenNotifications = controller;
})(typeof window !== 'undefined' ? window : globalThis);
