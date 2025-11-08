(() => {
  const EVENT_CONNECT = 'connect';
  const EVENT_DISCONNECT = 'disconnect';
  const EVENT_MESSAGE = 'message';

  const listenersMap = new WeakMap();

  class HelenEventSocket {
    constructor(url) {
      this.url = url;
      this.source = null;
      this.retryDelay = 1500;
      this._connect();
    }

    _connect() {
      if (this.source) {
        try {
          this.source.close();
        } catch (error) {
          console.warn('[Helen] No se pudo cerrar el EventSource anterior.', error);
        }
      }

      try {
        this.source = new EventSource(this.url);
      } catch (connectionError) {
        console.error('[Helen] Error creando EventSource:', connectionError);
        this._scheduleReconnect();
        return;
      }

      this.source.addEventListener('open', () => {
        this._emit(EVENT_CONNECT);
      });

      this.source.addEventListener('message', (event) => {
        let payload = null;
        try {
          payload = event && typeof event.data === 'string' ? JSON.parse(event.data) : null;
        } catch (error) {
          console.warn('[Helen] No se pudo parsear el mensaje SSE:', error, event);
        }
        if (payload) {
          this._emit(EVENT_MESSAGE, payload);
        }
      });

      this.source.addEventListener('error', (event) => {
        console.warn('[Helen] Conexión SSE interrumpida.', event);
        this._emit(EVENT_DISCONNECT);
        this._scheduleReconnect();
      });
    }

    _scheduleReconnect() {
      if (this.source) {
        try {
          this.source.close();
        } catch (error) {
          console.warn('[Helen] Falló el cierre del EventSource:', error);
        }
      }
      window.setTimeout(() => this._connect(), this.retryDelay);
    }

    on(eventName, handler) {
      if (typeof handler !== 'function') {
        return this;
      }
      let listeners = listenersMap.get(this);
      if (!listeners) {
        listeners = new Map();
        listenersMap.set(this, listeners);
      }
      if (!listeners.has(eventName)) {
        listeners.set(eventName, new Set());
      }
      listeners.get(eventName).add(handler);
      return this;
    }

    off(eventName, handler) {
      const listeners = listenersMap.get(this);
      if (!listeners || !listeners.has(eventName)) {
        return this;
      }
      if (handler) {
        listeners.get(eventName).delete(handler);
      } else {
        listeners.get(eventName).clear();
      }
      return this;
    }

    emit(eventName, payload) {
      // Solo soportamos eco local para compatibilidad con la API antigua.
      this._emit(eventName, payload);
      return this;
    }

    close() {
      if (this.source) {
        this.source.close();
        this.source = null;
      }
    }

    _emit(eventName, payload) {
      const listeners = listenersMap.get(this);
      if (!listeners || !listeners.has(eventName)) {
        return;
      }
      for (const handler of listeners.get(eventName)) {
        try {
          handler(payload);
        } catch (error) {
          console.error('[Helen] Error en listener de', eventName, error);
        }
      }
    }
  }

  window.createHelenSocket = (url) => new HelenEventSocket(url);
})();
