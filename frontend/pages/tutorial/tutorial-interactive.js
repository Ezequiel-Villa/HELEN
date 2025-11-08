(function (globalScope, isModule) {
  'use strict';

  const DEFAULT_CONFIRMATIONS_REQUIRED = 2;
  const DEFAULT_STEP_TIMEOUT_MS = 12000;
  const PROGRESS_MESSAGE_TEMPLATE = ({ confirmations, required }) =>
    `Confirmación ${Math.min(confirmations, required)} de ${required}. Mantén la posición.`;
  const DEFAULT_COMPLETION_MESSAGE = 'Ya dominaste esta guía. Puedes repetirla o volver al menú.';

  const NOOP = () => {};

  const sanitizeGesture = (value) => {
    if (typeof value === 'string') {
      return value.trim();
    }
    if (value == null) {
      return '';
    }
    return String(value).trim();
  };

  const normalizeGesture = (gesture) => {
    if (!gesture) return '';
    try {
      return gesture.normalize('NFKC');
    } catch (error) {
      return gesture;
    }
  };

  const collapseGesture = (gesture) => {
    if (!gesture) {
      return '';
    }
    return normalizeGesture(gesture)
      .normalize('NFD')
      .replace(/[\u0300-\u036f]/g, '')
      .replace(/\s+/g, '')
      .toLowerCase();
  };

  const resolveGestureFromPayload = (payload = {}) => {
    const gestureKeys = ['gesture', 'character', 'label', 'command', 'action', 'key'];
    for (const key of gestureKeys) {
      if (Object.prototype.hasOwnProperty.call(payload, key)) {
        const candidate = sanitizeGesture(payload[key]);
        if (candidate) {
          return candidate;
        }
      }
    }
    if (payload.active === true) {
      return 'Start';
    }
    if (payload.state && typeof payload.state === 'string') {
      return payload.state;
    }
    return '';
  };

  const createStepDefinition = (step, index) => {
    const id = step.id || `step-${index}`;
    const expectedList = Array.isArray(step.expected) && step.expected.length > 0
      ? step.expected
      : Array.isArray(step.gestures) ? step.gestures : [];
    const confirmationsRequired = Number.isFinite(step.confirmationsRequired)
      ? Math.max(1, Math.floor(step.confirmationsRequired))
      : null;
    const timeoutMs = Number.isFinite(step.timeoutMs)
      ? Math.max(1000, Math.floor(step.timeoutMs))
      : null;

    const normalizedGestures = new Set(expectedList.map((gesture) => collapseGesture(gesture)));

    return {
      id,
      text: step.text,
      hint: step.hint,
      complete: step.complete || step.success,
      retry: step.retry,
      expected: normalizedGestures,
      confirmationsRequired,
      timeoutMs,
    };
  };

  const modules = {
    weather: {
      key: 'weather',
      title: 'Clima',
      subtitle: 'Secuencia guiada para consultar el clima.',
      icon: 'bi-cloud-sun',
      footnote: 'Secuencia de práctica: H → C → Salir.',
      steps: [
        {
          id: 'activate',
          text: 'Haz la seña de la letra H para activar a HELEN.',
          hint: 'Esta seña despierta al asistente antes de cualquier comando.',
          retry: 'Levanta la mano con la forma de H frente a la cámara.',
          complete: 'Seña H reconocida. HELEN está lista para ayudarte.',
          expected: ['start', 'h'],
        },
        {
          id: 'ask-weather',
          text: 'Ahora haz la seña de la letra C para consultar el clima.',
          hint: 'Mantén la mano centrada frente a la cámara para una detección clara.',
          retry: 'Extiende la mano en forma de C y mantén la postura dos veces.',
          complete: '¡Perfecto! HELEN obtiene el clima actual.',
          expected: ['clima', 'c'],
        },
        {
          id: 'return-home',
          text: 'Para salir, practica la seña de cierre (H + I) o confirma el paso para continuar.',
          hint: 'Simula la seña de salir para finalizar la sesión de práctica.',
          retry: 'Haz la seña de salir o presiona “Confirmar paso” para avanzar.',
          complete: 'Tutorial de clima completado.',
          expected: ['inicio', 'i', 'salir', 'salida'],
        },
      ],
    },
    clock: {
      key: 'clock',
      title: 'Hora',
      subtitle: 'Aprende a pedir la hora con confirmaciones guiadas.',
      icon: 'bi-clock-history',
      footnote: 'Secuencia de práctica: H → R → Salir.',
      steps: [
        {
          id: 'activate',
          text: 'Haz la seña de la letra H para activar a HELEN.',
          hint: 'Asegúrate de que HELEN esté atenta antes de continuar.',
          retry: 'Activa a HELEN nuevamente con la letra H.',
          complete: 'Activación lista. HELEN está escuchando.',
          expected: ['start', 'h'],
        },
        {
          id: 'ask-time',
          text: 'Luego haz la seña de la letra R para consultar la hora.',
          hint: 'Forma la letra R con claridad y sostenla un instante.',
          retry: 'Junta índice y medio para formar la letra R.',
          complete: 'Hora solicitada correctamente.',
          expected: ['reloj', 'r'],
        },
        {
          id: 'return-home',
          text: 'Para terminar la guía, practica la seña de salir (H + I) o confirma manualmente el paso.',
          hint: 'Cierra con la seña de salir para volver a la pantalla principal.',
          retry: 'Realiza la seña de salir o utiliza el botón “Confirmar paso”.',
          complete: 'Tutorial de hora completado.',
          expected: ['inicio', 'i', 'salir', 'salida'],
        },
      ],
    },
    devices: {
      key: 'devices',
      title: 'Dispositivos',
      subtitle: 'Simula cómo encender o apagar dispositivos conectados.',
      icon: 'bi-lightbulb',
      footnote: 'Secuencia de práctica: H → Dispositivo → Salir.',
      steps: [
        {
          id: 'activate',
          text: 'Haz la seña de la letra H para activar a HELEN.',
          hint: 'Activa siempre a HELEN antes de controlar dispositivos.',
          retry: 'Repite la letra H hasta ver la confirmación.',
          complete: 'HELEN está lista para recibir la seña del dispositivo.',
          expected: ['start', 'h'],
        },
        {
          id: 'select-device',
          text: 'Realiza la seña de activación del dispositivo que necesites.',
          hint: 'Imagina la seña asignada al dispositivo para practicarla.',
          retry: 'Puedes usar la seña de Dispositivos para continuar.',
          complete: 'Comando del dispositivo confirmado.',
          expected: ['dispositivos', 'foco', 'd'],
        },
        {
          id: 'return-home',
          text: 'Para salir del tutorial, realiza la seña de salir (H + I) o confirma el paso de manera manual.',
          hint: 'Cierra la sesión con la seña de salir para volver al menú principal.',
          retry: 'Completa la seña de salir o utiliza el botón “Confirmar paso”.',
          complete: 'Tutorial de dispositivos completado.',
          expected: ['inicio', 'i', 'salir', 'salida'],
        },
      ],
    },
  };

  class TutorialStateMachine {
    constructor(options) {
      this.steps = Array.isArray(options.steps)
        ? options.steps.map((step, index) => createStepDefinition(step, index))
        : [];
      this.confirmationsRequired = Math.max(1, options.confirmationsRequired || DEFAULT_CONFIRMATIONS_REQUIRED);
      this.stepTimeoutMs = Math.max(1000, options.stepTimeoutMs || DEFAULT_STEP_TIMEOUT_MS);
      this.logger = typeof options.logger === 'function' ? options.logger : NOOP;
      this.hooks = {
        onStepStarted: typeof options.onStepStarted === 'function' ? options.onStepStarted : NOOP,
        onStepPassed: typeof options.onStepPassed === 'function' ? options.onStepPassed : NOOP,
        onStepTimeout: typeof options.onStepTimeout === 'function' ? options.onStepTimeout : NOOP,
        onStateChange: typeof options.onStateChange === 'function' ? options.onStateChange : NOOP,
        onProgress: typeof options.onProgress === 'function' ? options.onProgress : NOOP,
      };
      this.index = -1;
      this.state = 'idle';
      this.confirmations = 0;
      this.timeoutId = null;
      this.timeoutArmed = false;
    }
    _currentStep() {
      if (this.index < 0 || this.index >= this.steps.length) {
        return null;
      }
      return this.steps[this.index];
    }

    _requiredConfirmations(step) {
      if (step && Number.isFinite(step.confirmationsRequired)) {
        return Math.max(1, step.confirmationsRequired);
      }
      return this.confirmationsRequired;
    }

    _applyState(state, meta) {
      if (this.state !== state) {
        this.state = state;
        this.hooks.onStateChange({
          state,
          stepIndex: this.index,
          totalSteps: this.steps.length,
          confirmations: this.confirmations,
          meta: meta || null,
        });
      }
    }

    _clearTimeout() {
      if (this.timeoutId) {
        clearTimeout(this.timeoutId);
        this.timeoutId = null;
      }
      this.timeoutArmed = false;
    }

    _scheduleTimeout(step) {
      this._clearTimeout();
      const timeoutMs = step && Number.isFinite(step.timeoutMs) ? step.timeoutMs : this.stepTimeoutMs;
      this.timeoutArmed = true;
      this.timeoutId = setTimeout(() => {
        this.timeoutId = null;
        this.timeoutArmed = false;
        this._handleTimeout();
      }, timeoutMs);
    }

    _handleTimeout() {
      const step = this._currentStep();
      if (!step) {
        return;
      }
      this.logger('step-timeout', { stepId: step.id, index: this.index });
      this.confirmations = 0;
      this._applyState('waiting_retry');
      this.hooks.onStepTimeout({
        step,
        index: this.index,
        required: this._requiredConfirmations(step),
      });
    }

    start() {
      if (!this.steps.length) {
        return false;
      }
      this.index = 0;
      this.confirmations = 0;
      const step = this._currentStep();
      this._applyState('waiting_gesture');
      this._scheduleTimeout(step);
      this.logger('step-started', { stepId: step.id, index: this.index });
      this.hooks.onStepStarted({ step, index: this.index, total: this.steps.length });
      return true;
    }

    handleGesture(payload) {
      if (this.index < 0 || this.index >= this.steps.length) {
        return false;
      }
      const step = this._currentStep();
      if (!step || !step.expected.size) {
        return false;
      }

      const gesture = collapseGesture(resolveGestureFromPayload(payload));
      if (!gesture) {
        return false;
      }

      if (!step.expected.has(gesture)) {
        if (this.confirmations > 0) {
          this.logger('gesture-mismatch', { gesture, expected: Array.from(step.expected) });
        }
        this.confirmations = 0;
        if (this.state !== 'waiting_retry') {
          this._applyState('waiting_gesture');
        }
        return false;
      }

      if (!this.timeoutArmed) {
        this._scheduleTimeout(step);
      }

      const required = this._requiredConfirmations(step);
      this.confirmations += 1;
      this.hooks.onProgress({
        step,
        index: this.index,
        confirmations: this.confirmations,
        required,
      });

      this.logger('gesture-match', {
        gesture,
        confirmations: this.confirmations,
        required,
        stepId: step.id,
      });

      if (this.confirmations >= required) {
        this._completeStep(step, payload);
        return true;
      }

      this._applyState('confirming');
      return true;
    }

    simulateStep(payload) {
      if (this.index < 0 || this.index >= this.steps.length) {
        return false;
      }
      const step = this._currentStep();
      if (!step) {
        return false;
      }
      if (this.state === 'passed' || this.state === 'completed') {
        return false;
      }

      this.logger('step-simulated', { stepId: step.id, index: this.index });
      const basePayload = payload && typeof payload === 'object' ? payload : {};
      this.confirmations = this._requiredConfirmations(step);
      this._completeStep(step, { ...basePayload, simulated: true });
      return true;
    }

    _completeStep(step, payload) {
      const isLast = this.index >= this.steps.length - 1;
      this._clearTimeout();
      this.logger('step-confirmed', { stepId: step.id, index: this.index, isLast });
      this._applyState(isLast ? 'completed' : 'passed');
      this.hooks.onStepPassed({
        step,
        index: this.index,
        total: this.steps.length,
        isLast,
        payload,
        required: this._requiredConfirmations(step),
      });
    }

    advance() {
      if (this.index < 0 || this.index >= this.steps.length) {
        return false;
      }
      if (this.state !== 'passed') {
        return false;
      }
      if (this.index >= this.steps.length - 1) {
        this._applyState('completed');
        return false;
      }
      this.index += 1;
      this.confirmations = 0;
      const step = this._currentStep();
      this.logger('step-started', { stepId: step.id, index: this.index });
      this._applyState('waiting_gesture');
      this._scheduleTimeout(step);
      this.hooks.onStepStarted({ step, index: this.index, total: this.steps.length });
      return true;
    }

    retry() {
      const step = this._currentStep();
      if (!step) {
        return false;
      }
      this.confirmations = 0;
      this._applyState('waiting_gesture');
      this._scheduleTimeout(step);
      this.logger('step-retry', { stepId: step.id, index: this.index });
      this.hooks.onStepStarted({ step, index: this.index, total: this.steps.length });
      return true;
    }

    destroy() {
      this._clearTimeout();
      this.index = -1;
      this.confirmations = 0;
      this._applyState('idle');
    }

    snapshot() {
      return {
        stepIndex: this.index,
        totalSteps: this.steps.length,
        confirmations: this.confirmations,
        state: this.state,
      };
    }
  }
  class TutorialController {
    constructor(globalObject) {
      this.global = globalObject;
      this.document = globalObject.document;
      this.root = this.document && this.document.querySelector('[data-tutorial]');
      this.hooks = {
        stepStarted: NOOP,
        stepPassed: NOOP,
        stepTimeout: NOOP,
        stateChange: NOOP,
        sessionStart: NOOP,
        sessionEnd: NOOP,
      };
      this.config = {
        confirmationsRequired: DEFAULT_CONFIRMATIONS_REQUIRED,
        stepTimeoutMs: DEFAULT_STEP_TIMEOUT_MS,
      };
      this.machine = null;
      this.activeKey = null;
      this.activeModuleTitle = '';
      this.moduleGestures = new Set();
      this.socketHandler = (event) => this._handleSocketEvent(event);
      this.socketAttached = false;

      if (!this.root) {
        return;
      }

      this.menuView = this.root.querySelector('[data-view="menu"]');
      this.flowView = this.root.querySelector('[data-view="flow"]');
      this.moduleList = this.root.querySelector('[data-module-list]');
      this.cardTemplate = this.root.querySelector('template[data-module-card-template]');
      this.flowIcon = this.flowView.querySelector('[data-flow-icon]');
      this.flowTitle = this.flowView.querySelector('[data-flow-title]');
      this.flowSubtitle = this.flowView.querySelector('[data-flow-subtitle]');
      this.footnote = this.flowView.querySelector('[data-flow-footnote]');
      this.stepVisual = this.flowView.querySelector('[data-step-visual]');
      this.stepNumber = this.flowView.querySelector('[data-step-number]');
      this.stepDigit = this.flowView.querySelector('[data-step-digit]');
      this.stepText = this.flowView.querySelector('[data-step-text]');
      this.stepStatus = this.flowView.querySelector('[data-step-status]');
      this.dotsContainer = this.flowView.querySelector('[data-step-dots]');
      this.flowControls = this.flowView.querySelector('[data-flow-controls]');
      this.nextButton = this.flowView.querySelector('[data-action="next-step"]');
      this.closeButton = this.flowView.querySelector('[data-action="close-flow"]');
      this.returnButton = this.flowView.querySelector('[data-action="return-menu"]');
      this.successBanner = this.flowView.querySelector('[data-flow-success]');
      this.successTitle = this.flowView.querySelector('[data-success-title]');
      this.successDetail = this.flowView.querySelector('[data-success-detail]');
      this.reducedMotionQuery = this.global.matchMedia ? this.global.matchMedia('(prefers-reduced-motion: reduce)') : null;
      this.progress = [];

      this._buildMenu();
      this._attachEvents();
      this._ensureSocketListener();
      this._setView('menu');
    }

    _buildMenu() {
      if (!this.moduleList) {
        return;
      }
      this.moduleList.innerHTML = '';
      const order = ['weather', 'clock', 'devices'];
      const fragment = this.document.createDocumentFragment();
      const templateButton = this.cardTemplate && this.cardTemplate.content
        ? this.cardTemplate.content.querySelector('.tutorial-card')
        : null;

      const createCard = (key, module) => {
        let button;
        if (templateButton) {
          button = templateButton.cloneNode(true);
        } else {
          button = this.document.createElement('button');
          button.className = 'tutorial-card';
          button.type = 'button';
          button.innerHTML = `
            <span class="tutorial-card__icon" aria-hidden="true"><i class="bi" aria-hidden="true"></i></span>
            <span class="tutorial-card__title"></span>
            <span class="tutorial-card__subtitle"></span>
            <span class="tutorial-card__cta">Iniciar guía</span>
          `;
        }
        button.dataset.moduleTrigger = key;
        const icon = button.querySelector('[data-card-icon] i') || button.querySelector('.tutorial-card__icon i');
        if (icon) {
          icon.className = `bi ${module.icon}`;
        }
        const title = button.querySelector('[data-card-title]') || button.querySelector('.tutorial-card__title');
        if (title) {
          title.textContent = module.title;
        }
        const subtitle = button.querySelector('[data-card-subtitle]') || button.querySelector('.tutorial-card__subtitle');
        if (subtitle) {
          subtitle.textContent = module.subtitle;
        }
        button.addEventListener('click', () => this.startModule(key));
        fragment.appendChild(button);
      };

      order.forEach((key) => {
        const module = modules[key];
        if (module) {
          createCard(key, module);
        }
      });

      this.moduleList.appendChild(fragment);
    }

    _attachEvents() {
      if (this.nextButton) {
        this.nextButton.addEventListener('click', () => this._handleNext());
      }
      if (this.closeButton) {
        this.closeButton.addEventListener('click', () => this.exitFlow());
      }
      if (this.returnButton) {
        this.returnButton.addEventListener('click', () => this.exitFlow());
      }
      if (this.document) {
        this.document.addEventListener('keydown', (event) => {
          if (event.key === 'Escape' && this.flowView && this.flowView.classList.contains('is-active')) {
            this.exitFlow();
          }
        });
      }
      if (this.reducedMotionQuery && typeof this.reducedMotionQuery.addEventListener === 'function') {
        this.reducedMotionQuery.addEventListener('change', () => {
          if (this.flowView && this.flowView.classList.contains('is-active') && this.machine) {
            this._renderStep(this.machine._currentStep(), this.machine.index);
          }
        });
      }
    }

    _ensureSocketListener(retries = 0) {
      if (this.socketAttached) {
        return;
      }
      const socket = this.global.socket;
      if (socket && typeof socket.on === 'function') {
        socket.on('message', this.socketHandler);
        this.socketAttached = true;
        return;
      }
      if (retries < 10) {
        this.global.setTimeout(() => this._ensureSocketListener(retries + 1), 500);
      }
    }

    _handleSocketEvent(event) {
      if (!this.machine) {
        return;
      }
      const consumed = this.machine.handleGesture(event);
      if (consumed) {
        this._log('gesture consumed', { event });
      }
    }

    _handleNext() {
      if (!this.machine) {
        return;
      }
      if (this.machine.state === 'passed') {
        const advanced = this.machine.advance();
        if (!advanced && this.machine.state === 'completed') {
          this._showCompletion();
        }
        return;
      }
      if (this.machine.state === 'completed') {
        this._showCompletion();
        return;
      }
      this.machine.simulateStep({ source: 'manual' });
    }

    configure(options = {}) {
      if (options.confirmationsRequired != null && Number.isFinite(options.confirmationsRequired)) {
        this.config.confirmationsRequired = Math.max(1, Math.floor(options.confirmationsRequired));
      }
      if (options.stepTimeoutMs != null && Number.isFinite(options.stepTimeoutMs)) {
        this.config.stepTimeoutMs = Math.max(1000, Math.floor(options.stepTimeoutMs));
      }
    }

    setHook(name, handler) {
      if (this.hooks && Object.prototype.hasOwnProperty.call(this.hooks, name) && typeof handler === 'function') {
        this.hooks[name] = handler;
      }
    }
    startModule(key) {
      const module = modules[key];
      if (!module || !Array.isArray(module.steps) || !module.steps.length) {
        return false;
      }
      this.exitFlow(true);
      this.activeKey = key;
      this.activeModuleTitle = module.title || '';
      this.progress = module.steps.map(() => 'pending');
      this.moduleGestures = new Set();
      module.steps.forEach((step, index) => {
        const prepared = createStepDefinition(step, index);
        prepared.expected.forEach((gesture) => {
          if (gesture) {
            this.moduleGestures.add(gesture);
          }
        });
      });

      const machine = new TutorialStateMachine({
        steps: module.steps,
        confirmationsRequired: module.confirmationsRequired || this.config.confirmationsRequired,
        stepTimeoutMs: module.stepTimeoutMs || this.config.stepTimeoutMs,
        logger: (message, meta) => this._log(message, meta),
        onStepStarted: ({ step, index, total }) => {
          this.progress[index] = 'active';
          this.hooks.stepStarted({ module: key, step, index, total });
          this._renderStep(step, index);
        },
        onProgress: ({ step, index, confirmations, required }) => {
          if (this.stepStatus) {
            this.stepStatus.textContent = PROGRESS_MESSAGE_TEMPLATE({ confirmations, required });
          }
        },
        onStepPassed: ({ step, index, total, isLast }) => {
          this.progress[index] = 'passed';
          this.hooks.stepPassed({ module: key, step, index, total, isLast });
          this._markStepComplete(step, index, isLast);
        },
        onStepTimeout: ({ step, index, required }) => {
          this.progress[index] = 'waiting_retry';
          this.hooks.stepTimeout({ module: key, step, index, required });
          if (this.stepStatus) {
            const retryText = step.retry
              || step.hint
              || 'No detectamos la seña. Intenta nuevamente o utiliza “Confirmar paso”.';
            this.stepStatus.textContent = retryText;
          }
          if (this.stepVisual) {
            this.stepVisual.classList.remove('is-complete');
          }
        },
        onStateChange: (snapshot) => {
          this.hooks.stateChange({ module: key, snapshot });
        },
      });

      this.machine = machine;
      this._renderHeader(module);
      this._toggleFootnote(module.footnote);
      this._buildDots(module.steps.length);
      this._setView('flow');
      this.hooks.sessionStart({ module: key, definition: module });
      machine.start();
      return true;
    }

    exitFlow(skipMenu = false) {
      if (this.machine) {
        this.machine.destroy();
        this.machine = null;
      }
      if (this.activeKey && !skipMenu) {
        this.hooks.sessionEnd({ module: this.activeKey });
      }
      this.activeKey = null;
      this.progress = [];
      this.moduleGestures.clear();
      this.activeModuleTitle = '';
      this._toggleFootnote('');
      this._toggleSuccess(false);
      if (!skipMenu) {
        this._setView('menu');
      }
      if (this.stepStatus) {
        this.stepStatus.textContent = '';
      }
      if (this.stepVisual) {
        this.stepVisual.classList.remove('is-complete');
      }
      if (this.nextButton) {
        this.nextButton.disabled = true;
        this.nextButton.classList.remove('is-visible');
        this.nextButton.setAttribute('aria-hidden', 'true');
      }
      if (this.returnButton) {
        this.returnButton.classList.remove('is-visible');
        this.returnButton.setAttribute('aria-hidden', 'true');
      }
    }

    interceptNavigation(collapsedGesture) {
      if (!this.machine || !this.activeKey) {
        return false;
      }
      const gesture = collapseGesture(collapsedGesture);
      if (!gesture) {
        return false;
      }
      if (this.moduleGestures.has(gesture)) {
        this._log('navigation-intercepted', { gesture });
        return true;
      }
      return false;
    }

    getStateSnapshot() {
      if (!this.machine) {
        return null;
      }
      return {
        module: this.activeKey,
        progress: [...this.progress],
        machine: this.machine.snapshot(),
      };
    }

    _renderHeader(module) {
      if (this.flowIcon) {
        this.flowIcon.className = `flow-heading__icon bi ${module.icon}`;
      }
      if (this.flowTitle) {
        this.flowTitle.textContent = module.title;
      }
      if (this.flowSubtitle) {
        this.flowSubtitle.textContent = module.subtitle;
      }
      if (this.successTitle) {
        this.successTitle.textContent = `¡${module.title} completado!`;
      }
    }

    _toggleFootnote(text) {
      if (!this.footnote) {
        return;
      }
      if (text) {
        this.footnote.textContent = text;
        this.footnote.classList.remove('is-hidden');
      } else {
        this.footnote.textContent = '';
        this.footnote.classList.add('is-hidden');
      }
    }

    _toggleSuccess(show, { title, detail } = {}) {
      if (!this.successBanner) {
        return;
      }
      const isVisible = Boolean(show);
      this.successBanner.classList.toggle('is-visible', isVisible);
      this.successBanner.setAttribute('aria-hidden', isVisible ? 'false' : 'true');
      if (!isVisible) {
        return;
      }
      if (this.successTitle && typeof title === 'string') {
        this.successTitle.textContent = title;
      }
      if (this.successDetail) {
        this.successDetail.textContent = typeof detail === 'string' && detail.trim()
          ? detail
          : DEFAULT_COMPLETION_MESSAGE;
      }
    }

    _buildDots(count) {
      if (!this.dotsContainer) {
        return;
      }
      this.dotsContainer.innerHTML = '';
      this.dots = [];
      for (let index = 0; index < count; index += 1) {
        const dot = this.document.createElement('li');
        dot.className = 'step-dot';
        dot.dataset.stepIndex = String(index);
        this.dotsContainer.appendChild(dot);
        this.dots.push(dot);
      }
      this._updateDots(0);
    }

    _updateDots(activeIndex) {
      if (!this.dots) {
        return;
      }
      this.dots.forEach((dot, index) => {
        const state = this.progress[index];
        dot.classList.toggle('is-active', state === 'passed');
        dot.classList.toggle('is-current', index === activeIndex);
      });
    }

    _renderStep(step, index) {
      if (!step) {
        return;
      }
      this._toggleSuccess(false);
      if (this.stepVisual) {
        this.stepVisual.classList.remove('is-complete');
      }
      if (this.stepNumber) {
        this.stepNumber.textContent = `Paso ${index + 1} de ${this.progress.length}`;
      }
      if (this.stepDigit) {
        this.stepDigit.textContent = String(index + 1);
      }
      if (this.stepText) {
        this.stepText.textContent = step.text;
        this._animateCopy(this.stepText);
      }
      if (this.stepStatus) {
        const fallbackHint = 'Cuando HELEN confirme la seña o presiones “Confirmar paso”, verás esta marca verde.';
        let hintText = fallbackHint;
        if (step.hint) {
          const trimmedHint = String(step.hint).trim();
          const needsPeriod = trimmedHint && !/[.!?¡¿]$/.test(trimmedHint);
          hintText = `${trimmedHint}${needsPeriod ? '.' : ''} También puedes usar “Confirmar paso”.`;
        }
        this.stepStatus.textContent = hintText;
        this._animateCopy(this.stepStatus);
      }
      if (this.nextButton) {
        this.nextButton.disabled = false;
        this.nextButton.textContent = 'Confirmar paso';
        this.nextButton.classList.add('is-visible');
        this.nextButton.setAttribute('aria-hidden', 'false');
      }
      if (this.returnButton) {
        this.returnButton.classList.remove('is-visible');
        this.returnButton.setAttribute('aria-hidden', 'true');
      }
      if (this.flowControls) {
        this.flowControls.classList.toggle('is-hidden', false);
      }
      this._updateDots(index);
    }

    _markStepComplete(step, index, isLast) {
      if (this.stepVisual) {
        this.stepVisual.classList.add('is-complete');
      }
      if (this.stepStatus) {
        this.stepStatus.textContent = step.complete || '¡Paso completado!';
      }
      if (isLast) {
        if (this.flowControls) {
          this.flowControls.classList.add('is-hidden');
        }
        if (this.nextButton) {
          this.nextButton.classList.remove('is-visible');
          this.nextButton.setAttribute('aria-hidden', 'true');
        }
        const successTitle = this.activeModuleTitle ? `¡${this.activeModuleTitle} completado!` : undefined;
        this._toggleSuccess(true, { title: successTitle, detail: step.complete });
        if (this.returnButton) {
          this.returnButton.classList.add('is-visible');
          this.returnButton.setAttribute('aria-hidden', 'false');
          try {
            this.returnButton.focus({ preventScroll: true });
          } catch (error) {
            this.returnButton.focus();
          }
        }
      } else {
        if (this.flowControls) {
          this.flowControls.classList.remove('is-hidden');
        }
        if (this.nextButton) {
          this.nextButton.disabled = false;
          this.nextButton.textContent = 'Siguiente paso';
          this.nextButton.classList.add('is-visible');
          this.nextButton.setAttribute('aria-hidden', 'false');
          try {
            this.nextButton.focus({ preventScroll: true });
          } catch (error) {
            this.nextButton.focus();
          }
        }
      }
      this._updateDots(index);
    }

    _showCompletion() {
      if (this.successBanner && !this.successBanner.classList.contains('is-visible')) {
        const successTitle = this.activeModuleTitle ? `¡${this.activeModuleTitle} completado!` : undefined;
        this._toggleSuccess(true, { title: successTitle });
      }
      if (this.returnButton) {
        this.returnButton.classList.add('is-visible');
        this.returnButton.setAttribute('aria-hidden', 'false');
      }
    }
    _animateCopy(node) {
      if (!node) {
        return;
      }
      node.classList.remove('is-appear');
      void node.offsetWidth;
      node.classList.add('is-appear');
    }

    _setView(viewName) {
      const isFlow = viewName === 'flow';
      if (this.menuView) {
        this.menuView.classList.toggle('is-active', !isFlow);
        this.menuView.setAttribute('aria-hidden', isFlow ? 'true' : 'false');
      }
      if (this.flowView) {
        this.flowView.classList.toggle('is-active', isFlow);
        this.flowView.setAttribute('aria-hidden', isFlow ? 'false' : 'true');
        if (isFlow) {
          try {
            this.flowView.focus({ preventScroll: true });
          } catch (error) {
            this.flowView.focus();
          }
        }
      }
    }

    _log(message, meta) {
      try {
        console.info('[Tutorial]', message, meta || '');
      } catch (error) {
        // Ignorar ambientes sin consola
      }
    }
  }

  const createNoopApi = () => ({
    version: '2.0.0',
    interceptNavigation: () => false,
    startModule: () => false,
    exit: () => {},
    configure: () => {},
    setHook: () => {},
    getState: () => null,
  });

  const attachController = () => {
    if (!globalScope || !globalScope.document) {
      return createNoopApi();
    }
    const controller = new TutorialController(globalScope);
    if (!controller || !controller.root) {
      return createNoopApi();
    }
    return {
      version: '2.0.0',
      interceptNavigation: (collapsedGesture) => controller.interceptNavigation(collapsedGesture),
      startModule: (key) => controller.startModule(key),
      exit: () => controller.exitFlow(),
      configure: (options) => controller.configure(options),
      setHook: (name, handler) => controller.setHook(name, handler),
      getState: () => controller.getStateSnapshot(),
    };
  };

  const exportsObject = {
    modules,
    TutorialStateMachine,
    TutorialController,
    collapseGesture,
    resolveGestureFromPayload,
  };

  if (isModule) {
    module.exports = exportsObject;
    return;
  }

  const api = attachController();
  globalScope.helenTutorial = api;
})(typeof window !== 'undefined' ? window : globalThis, typeof module === 'object' && typeof module.exports === 'object');
