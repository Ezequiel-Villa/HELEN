// eventConnector.js - Versión SSE organizada
// Conexión SSE, ring de activación y mejoras de navegación

// ==============================
// Conexión SSE (fachada tipo socket)
// ==============================

/** Devuelve la URL base del backend (protocolo, host y puerto). */
function getDefaultBackendUrl() {
    try {
        const p = location.protocol.startsWith('http') ? location.protocol : 'http:';
        const h = location.hostname || '127.0.0.1';
        const port = location.port || '5000';
        return `${p}//${h}:${port}`;
    } catch {
        return 'http://127.0.0.1:5000';
    }
}

/** Crea un “socket” basado en SSE con API .on/.close/.emit(no-op). */
function createSseSocket(url) {
    let es = null;
    const handlers = { connect: [], disconnect: [], message: [] };

    /** Registra un handler para 'connect' | 'disconnect' | 'message'. */
    function on(event, fn) {
        if (handlers[event]) handlers[event].push(fn);
        return api;
    }

    /** No-op: SSE es solo recepción. */
    function emit() { }

    /** Cierra la conexión SSE si está abierta. */
    function close() {
        if (es) es.close();
    }

    /** Dispara los handlers registrados. */
    function fire(event, payload) {
        handlers[event].forEach((fn) => {
            try { fn(payload); } catch (e) { console.warn(e); }
        });
    }

    /** Abre la conexión SSE y enruta mensajes. */
    function connect() {
        try {
            es = new EventSource(url, { withCredentials: false });
            es.onopen = () => fire('connect');
            es.onerror = () => fire('disconnect');
            es.onmessage = (evt) => {
                try {
                    const data = evt && evt.data ? JSON.parse(evt.data) : {};
                    fire('message', data);
                } catch {
                    fire('message', {});
                }
            };
        } catch (err) {
            console.error('[SSE] error creating EventSource:', err);
            fire('disconnect');
        }
    }

    const api = { on, emit, close };
    connect();
    return api;
}

/** Inicializa y expone el socket global basado en SSE. */
function initGlobalSocket() {
    const BASE = getDefaultBackendUrl();
    const EVENTS_URL = `${BASE}/events`;
    // Igual que io('http://127.0.0.1:5000'), pero usando SSE
    window.socket = createSseSocket(EVENTS_URL);
}
initGlobalSocket();


// ==============================
// Ring de activación (idle | active | error)
// ==============================

/** Crea o devuelve el nodo del ring. */
function ensureActivationRing() {
    let ring = document.querySelector('.activation-ring');
    if (!ring) {
        ring = document.createElement('div');
        ring.className = 'activation-ring';
        ring.setAttribute('aria-hidden', 'true');
        const halo = document.createElement('div');
        halo.className = 'activation-ring__halo';
        ring.appendChild(halo);
        document.addEventListener('DOMContentLoaded', () => document.body.appendChild(ring));
        if (document.readyState !== 'loading') document.body.appendChild(ring);
    }
    return ring;
}

let __ringFadeTimer;
let __ringErrorTimer;

/** Limpia timers activos del ring. */
function clearRingTimers() {
    if (__ringFadeTimer) clearTimeout(__ringFadeTimer);
    if (__ringErrorTimer) clearTimeout(__ringErrorTimer);
    __ringFadeTimer = __ringErrorTimer = undefined;
}

/** Cambia el estado visual del ring. */
function setActivationRing(next, { linger = 2000, persist = false } = {}) {
    const ring = ensureActivationRing();
    clearRingTimers();

    if (next === 'idle') {
        ring.classList.remove('is-active', 'is-error', 'is-visible', 'is-persistent');
        return;
    }

    ring.classList.add('is-visible');

    if (next === 'error') {
        ring.classList.remove('is-active', 'is-persistent');
        ring.classList.add('is-error');
        __ringErrorTimer = setTimeout(() => setActivationRing('idle'), 900);
        return;
    }

    // active
    ring.classList.remove('is-error');
    ring.classList.add('is-active');
    ring.classList.toggle('is-persistent', !!persist);
    if (!persist) {
        __ringFadeTimer = setTimeout(() => setActivationRing('idle'), Math.max(linger, 0));
    }
}

/** Muestra animación de activación breve o persistente. */
function triggerActivationAnimation(opts) { setActivationRing('active', opts); }

/** Muestra feedback de error breve (rojo). */
function triggerRingError() { setActivationRing('error'); }

/** Fija el estado del ring explícitamente. */
function setActivationRingState(s) { setActivationRing(s || 'idle'); }

window.triggerActivationAnimation = triggerActivationAnimation;
window.triggerRingError = triggerRingError;
window.setActivationRingState = setActivationRingState;


// ==============================
// Estado/temporizador global simples (opcional para acciones.js)
// ==============================

let isActive = false;
let timeoutId = null;
let lastNotification = '';
const DEACTIVATION_DELAY = 6000;


// ==============================
// Utilidades visuales y de ruta
// ==============================

/** Determina la ruta base a usar para enlaces relativos. */
function getBasePath() {
    const path = window.location.pathname;
    if (path.includes('/pages/')) {
        return '../';
    }
    return '../';
}

/** Muestra notificación toast (evita repeticiones consecutivas). */
function showPopup(message, type) {
    if (!window.Swal) return;
    if (message === lastNotification) return;
    lastNotification = message;
    window.Swal.fire({
        title: message,
        icon: type,
        showConfirmButton: false,
        timer: 3000,
        toast: true,
        position: 'top-end',
    });
}

/** Reinicia el temporizador de desactivación por inactividad. */
function resetDeactivationTimer() {
    if (timeoutId) clearTimeout(timeoutId);
    timeoutId = setTimeout(() => {
        isActive = false;
        setActivationRingState('idle');
        showPopup('Sistema desactivado por inactividad.', 'warning');
    }, DEACTIVATION_DELAY);
}

/** Navega mostrando loader si existe (loadingScreen.showAndExecute). */
async function goToPageWithLoading(targetUrl, pageName) {
    const currentUrl = window.location.href;
    if (currentUrl.includes(targetUrl)) return;

    const basePath = getBasePath();
    const fullTargetUrl = targetUrl.startsWith('/') ? targetUrl : basePath + targetUrl;

    function performNavigation() {
        if (window.myAPI && window.myAPI.navigate) {
            window.myAPI.navigate(fullTargetUrl);
        } else {
            window.location.href = fullTargetUrl;
        }
    }

    if (window.loadingScreen && typeof window.loadingScreen.showAndExecute === 'function') {
        await new Promise(resolve => setTimeout(resolve, 800));
        window.loadingScreen.showAndExecute(performNavigation, `Cargando ${pageName || ''}...`);
    } else {
        performNavigation();
    }
}


// ==============================
// Atajos de navegación semánticos
// ==============================

function enhancedGoToWeather() { goToPageWithLoading('pages/weather/weather.html', 'Clima'); }
function enhancedGoToSettings() { goToPageWithLoading('pages/settings/settings.html', 'Ajustes'); }
function enhancedGoToHome() { goToPageWithLoading('index.html', 'Inicio'); }
function enhancedGoToDevices() { goToPageWithLoading('pages/devices/devices.html', 'Dispositivos'); }
function enhancedGoToClock() { goToPageWithLoading('pages/clock/clock.html', 'Reloj'); }
function enhancedGoToWifi() { goToPageWithLoading('pages/settings/wifi.html', 'Wifi'); }
function enhancedGoToAlarm() { goToPageWithLoading('pages/clock/alarm.html', 'Alarma'); }
function enhancedGoToTutorial() { goToPageWithLoading('pages/tutorial/tutorial.html', 'Tutorial'); }

window.goToWeather = enhancedGoToWeather;
window.goToSettings = enhancedGoToSettings;
window.goToHome = enhancedGoToHome;
window.goToDevices = enhancedGoToDevices;
window.goToClock = enhancedGoToClock;
window.goToWifi = enhancedGoToWifi;
window.goToAlarm = enhancedGoToAlarm;
window.goToTutorial = enhancedGoToTutorial;
