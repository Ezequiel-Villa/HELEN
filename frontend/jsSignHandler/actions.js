// actions.js - Control de activación y ruteo de comandos (SSE)

(function actionsController() {
    // ==============================
    // Estado y configuración
    // ==============================
    let isActive = false;
    let timeoutId = null;
    const DEACTIVATION_DELAY = 6000;

    /** Reinicia y programa el apagado por inactividad del modo activo. */
    function resetDeactivationTimer() {
        if (timeoutId) clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
            isActive = false;
            if (typeof window.setActivationRingState === 'function') {
                window.setActivationRingState('idle');
            }
        }, DEACTIVATION_DELAY);
    }

    /** Activa el modo gesto y muestra el ring persistente. */
    function setActive() {
        isActive = true;
        if (typeof window.triggerActivationAnimation === 'function') {
            window.triggerActivationAnimation({ persist: true });
        }
        resetDeactivationTimer();
    }

    // ==============================
    // Normalización y ruteo
    // ==============================

    /** Devuelve el comando normalizado desde el payload. */
    function getCommand(data) {
        if (typeof data === 'string') return data.trim();
        if (data && typeof data === 'object') {
            if (data.status === 'ok' && !data.character && !data.gesture && !data.key) return ''; // ping/health
            const c = (data.character || data.gesture || data.key || data.label || data.command || data.action || '')
                .toString();
            return c.trim();
        }
        return '';
    }

    /** Tabla de comandos -> acciones. (Definida una vez) */
    const COMMANDS = {
        Start: () => setActive(),
        Clima: () => window.goToWeather && window.goToWeather(),
        Ajustes: () => window.goToSettings && window.goToSettings(),
        Inicio: () => window.goToHome && window.goToHome(),
        Dispositivos: () => window.goToDevices && window.goToDevices(),
        Reloj: () => window.goToClock && window.goToClock(),
        Wifi: () => window.goToSettings && window.goToSettings(),
        Alarma: () => window.goToAlarm && window.goToAlarm(),
        Tutorial: () => window.goToHome && window.goToHome(),
        Anterior: () => { if (history.length > 1) history.back(); },
        Agregar: () => { /* placeholder */ },
    };

    /** Ejecuta la navegación asociada al comando. */
    function handleCommand(cmd) {
        const fn = COMMANDS[cmd];
        if (typeof fn === 'function') {
            fn();
            return true;
        }
        return false;
    }

    // ==============================
    // Listener SSE principal
    // ==============================

    /**
     * - Si llega "Start" => activa sin error rojo.
     * - Si llega otro comando y no está activo => ring rojo breve.
     * - Si está activo => ejecuta comando y da feedback de ring.
     */
    function onSseMessage(data) {
        const cmd = getCommand(data);
        if (!cmd) return;

        if (cmd === 'Start') {
            setActive();
            return;
        }

        if (!isActive) {
            if (typeof window.triggerRingError === 'function') {
                window.triggerRingError();
            }
            return;
        }

        resetDeactivationTimer();
        if (typeof window.triggerActivationAnimation === 'function') {
            window.triggerActivationAnimation();
        }

        if (!handleCommand(cmd)) {
            if (typeof window.triggerRingError === 'function') {
                window.triggerRingError();
            }
        }
    }

    // Suscripción al “socket” SSE global
    if (window.socket && typeof window.socket.on === 'function') {
        window.socket.on('message', onSseMessage);
    } else {
        console.warn('[actions] socket SSE no inicializado todavía.');
    }
})();
