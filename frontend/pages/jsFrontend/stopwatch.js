// =========================
// Configuración global
// =========================
const CONFIG = {
  DIAS: ['Domingo', 'Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado'],
  MESES: ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'],
};

const STOPWATCH_WORKER_URL = (() => {
  if (typeof document !== 'undefined') {
    const current = document.currentScript;
    if (current && current.src) {
      return new URL('./stopwatch-worker.js', current.src).toString();
    }
    const scripts = document.getElementsByTagName('script');
    const last = scripts[scripts.length - 1];
    if (last && last.src) {
      return new URL('./stopwatch-worker.js', last.src).toString();
    }
  }
  return 'stopwatch-worker.js';
})();

class StopwatchEngine {
  constructor(onTick) {
    this.onTick = typeof onTick === 'function' ? onTick : () => {};
    this.worker = null;
    this.ready = false;
    this.pendingMessages = [];
    this.elapsed = 0;
    this.supportsWorker = typeof Worker !== 'undefined';
    this.fallbackTimer = null;
    this.fallbackStart = 0;
    this.fallbackElapsed = 0;

    if (this.supportsWorker) {
      try {
        this._initWorker();
        return;
      } catch (error) {
        console.warn('[Stopwatch] No se pudo iniciar el worker dedicado:', error);
        this.supportsWorker = false;
      }
    }

    this._initFallback();
  }

  _initWorker() {
    const worker = new Worker(STOPWATCH_WORKER_URL);
    worker.onmessage = (event) => {
      const data = event.data || {};
      switch (data.type) {
        case 'ready':
          this.ready = true;
          while (this.pendingMessages.length) {
            worker.postMessage(this.pendingMessages.shift());
          }
          break;
        case 'tick':
          this.elapsed = Math.max(0, Number(data.elapsedMs) || 0);
          this.onTick(this.elapsed);
          break;
        default:
          break;
      }
    };
    worker.onerror = (error) => {
      console.warn('[Stopwatch] Error en worker, usando modo fallback.', error);
      this.supportsWorker = false;
      this._initFallback();
      this.start(this.elapsed);
    };
    this.worker = worker;
  }

  _initFallback() {
    if (this.fallbackTimer) {
      clearInterval(this.fallbackTimer);
    }
    this.fallbackTimer = null;
    this.fallbackElapsed = 0;
    this.fallbackStart = 0;
  }

  _post(message) {
    if (!this.supportsWorker || !this.worker) {
      return;
    }
    if (!this.ready) {
      this.pendingMessages.push(message);
      return;
    }
    try {
      this.worker.postMessage(message);
    } catch (error) {
      console.warn('[Stopwatch] No se pudo enviar mensaje al worker:', error);
    }
  }

  _startFallback(initialElapsed) {
    if (this.fallbackTimer) {
      clearInterval(this.fallbackTimer);
    }
    this.fallbackElapsed = initialElapsed;
    this.fallbackStart = (typeof performance !== 'undefined' && typeof performance.now === 'function')
      ? performance.now()
      : Date.now();
    this.fallbackTimer = setInterval(() => {
      const now = (typeof performance !== 'undefined' && typeof performance.now === 'function')
        ? performance.now()
        : Date.now();
      const delta = now - this.fallbackStart;
      this.elapsed = Math.max(0, this.fallbackElapsed + delta);
      this.onTick(this.elapsed);
    }, 40);
  }

  start(initialElapsed = 0) {
    const safeElapsed = Math.max(0, Number(initialElapsed) || 0);
    this.elapsed = safeElapsed;
    if (this.supportsWorker && this.worker) {
      this._post({ type: 'start', elapsedMs: safeElapsed });
    } else {
      this._startFallback(safeElapsed);
    }
    this.onTick(this.elapsed);
  }

  stop() {
    if (this.supportsWorker && this.worker) {
      this._post({ type: 'stop' });
      return;
    }
    if (this.fallbackTimer) {
      clearInterval(this.fallbackTimer);
      this.fallbackTimer = null;
    }
    const now = (typeof performance !== 'undefined' && typeof performance.now === 'function')
      ? performance.now()
      : Date.now();
    const delta = now - this.fallbackStart;
    this.elapsed = Math.max(0, this.fallbackElapsed + delta);
    this.fallbackElapsed = this.elapsed;
    this.onTick(this.elapsed);
  }

  reset() {
    if (this.supportsWorker && this.worker) {
      this._post({ type: 'reset' });
    } else {
      if (this.fallbackTimer) {
        clearInterval(this.fallbackTimer);
        this.fallbackTimer = null;
      }
      this.fallbackElapsed = 0;
      this.fallbackStart = (typeof performance !== 'undefined' && typeof performance.now === 'function')
        ? performance.now()
        : Date.now();
      this.elapsed = 0;
      this.onTick(0);
    }
  }

  dispose() {
    if (this.worker) {
      try {
        this.worker.terminate();
      } catch (error) {
        console.warn('[Stopwatch] Error cerrando worker:', error);
      }
      this.worker = null;
    }
    if (this.fallbackTimer) {
      clearInterval(this.fallbackTimer);
      this.fallbackTimer = null;
    }
  }
}

// =========================
// Clase principal TimerApp
// =========================
class TimerApp {
  constructor() {
    this.elements = {
      dateSection: $('.date-section'),
      timerDisplay: $('.timer-display'),
      milliseconds: $('.milliseconds'),
      timerIcon: $('.timer-icon'),

      totalTime: $('.total-time'),
      lapCount: $('.lap-count'),
      avgLap: $('.avg-lap'),
      bestLap: $('.best-lap'),
      worstLap: $('.worstLap'),

      startBtn: $('.start-btn'),
      resetBtn: $('.reset-btn'),
      lapBtn: $('.lap-btn'),

      lapsList: $('.laps-list'),
      lapsCountBadge: $('.laps-count-badge'),

      statsCard: $('.stats-card'),
      loadingIndicator: $('.timer-loading'),
      backBtn: $('.back-btn'),
    };

    this.elapsedTime = 0;
    this.running = false;
    this.laps = [];
    this.engine = new StopwatchEngine((elapsed) => {
      this.elapsedTime = elapsed;
      this.updateTimerDisplay();
    });

    this.init();
  }

  init() {
    this.setupEventListeners();
    this.startDateTimer();
    this.resetTimer();

    setTimeout(() => {
      if (this.elements.loadingIndicator && this.elements.loadingIndicator.length) {
        this.elements.loadingIndicator.fadeOut();
      }
    }, 500);

    $(window).on('beforeunload', () => {
      if (this.engine) {
        this.engine.dispose();
      }
    });
  }

  setupEventListeners() {
    this.elements.startBtn.on('click', () => {
      if (!this.running) this.startTimer();
      else this.stopTimer();
    });

    this.elements.resetBtn.on('click', () => this.resetTimer());
    this.elements.lapBtn.on('click', () => this.addLap());

    if (this.elements.backBtn && this.elements.backBtn.length) {
      this.elements.backBtn.on('click', () => history.back());
    }

    $(document).on('keydown', (e) => {
      if (e.key === ' ' || e.code === 'Space') {
        e.preventDefault();
        if (!this.running) this.startTimer();
        else this.stopTimer();
      }
      if ((e.key === 'l' || e.key === 'L') && this.running) this.addLap();
      if ((e.key === 'r' || e.key === 'R') && !this.running) this.resetTimer();
    });
  }

  startDateTimer() {
    this.updateDate();
    setInterval(() => this.updateDate(), 60000);
  }

  updateDate() {
    const d = new Date();
    const s = `${CONFIG.DIAS[d.getDay()]}, ${d.getDate()} de ${CONFIG.MESES[d.getMonth()]} de ${d.getFullYear()}`;
    this.elements.dateSection.html(s);
  }

  startTimer() {
    if (this.running) return;
    this.running = true;

    this.engine.start(this.elapsedTime);

    this.elements.startBtn.html('<i class="bi bi-pause-fill"></i><span>Detener</span>');
    this.elements.startBtn.addClass('stop-btn');
    this.elements.resetBtn.prop('disabled', true);
    this.elements.lapBtn.prop('disabled', false);
    this.elements.timerIcon.removeClass('bi-stopwatch').addClass('bi-stopwatch-fill');
  }

  stopTimer() {
    if (!this.running) return;
    this.running = false;

    this.engine.stop();

    this.elements.startBtn.html('<i class="bi bi-play-fill"></i><span>Continuar</span>');
    this.elements.startBtn.removeClass('stop-btn');
    this.elements.resetBtn.prop('disabled', false);
    this.elements.lapBtn.prop('disabled', true);
    this.elements.timerIcon.removeClass('bi-stopwatch-fill').addClass('bi-stopwatch');
  }

  resetTimer() {
    if (this.running) this.stopTimer();

    this.engine.reset();
    this.elapsedTime = 0;
    this.laps = [];

    this.updateTimerDisplay();
    this.elements.startBtn.html('<i class="bi bi-play-fill"></i><span>Iniciar</span>');
    this.elements.resetBtn.prop('disabled', true);
    this.elements.lapBtn.prop('disabled', true);
    this.elements.lapsList.empty();

    this.updateLapStats();
    this.updateLapsCountBadge();
    this.toggleStatsVisibility();
  }

  updateTimerDisplay() {
    const t = this.formatTime(this.elapsedTime);
    this.elements.timerDisplay.text(t.main);
    this.elements.milliseconds.text(t.ms);
    this.elements.totalTime.text(`Tiempo Total: ${t.main}`);
  }

  addLap() {
    if (!this.running) return;

    const lapTime = this.elapsedTime;
    const prevCumulative = this.laps.length > 0 ? this.laps[this.laps.length - 1].time : 0;
    const lapDuration = lapTime - prevCumulative;

    this.laps.push({
      number: this.laps.length + 1,
      time: lapTime,
      lapTime: lapDuration,
    });

    this.updateLapsList();
    this.updateLapStats();
    this.updateLapsCountBadge();
    this.toggleStatsVisibility();
  }

  updateLapsList() {
    this.elements.lapsList.empty();

    [...this.laps].reverse().forEach((lap) => {
      const lapTimeFormatted = this.formatTime(lap.lapTime);
      const totalFormatted = this.formatTime(lap.time);

      const item = `
        <div class="lap-item">
          <div class="lap-number">#${lap.number}</div>
          <div class="lap-time">${lapTimeFormatted.main}.${lapTimeFormatted.ms}</div>
          <div class="lap-delta">${totalFormatted.main}</div>
        </div>
      `;
      this.elements.lapsList.append(item);
    });
  }

  updateLapStats() {
    const n = this.laps.length;
    this.elements.lapCount.text(`Vueltas: ${n}`);

    if (n > 0) {
      const avg = this.calculateAverageLapTime();
      const avgF = this.formatTime(avg);
      this.elements.avgLap.text(`Promedio: ${avgF.main}`);

      const lapsOnly = this.laps.map((l) => l.lapTime);
      const best = Math.min(...lapsOnly);
      const worst = Math.max(...lapsOnly);

      if (this.elements.bestLap && this.elements.bestLap.length) {
        this.elements.bestLap.text(`Más Rápida: ${this.formatTime(best).main}`);
      }
      if (this.elements.worstLap && this.elements.worstLap.length) {
        this.elements.worstLap.text(`Más Lenta: ${this.formatTime(worst).main}`);
      }
    } else {
      this.elements.avgLap.text('Promedio: 00:00:00');
      if (this.elements.bestLap && this.elements.bestLap.length) {
        this.elements.bestLap.text('Más Rápida: 00:00:00');
      }
      if (this.elements.worstLap && this.elements.worstLap.length) {
        this.elements.worstLap.text('Más Lenta: 00:00:00');
      }
    }
  }

  updateLapsCountBadge() {
    const n = this.laps.length;
    if (this.elements.lapsCountBadge && this.elements.lapsCountBadge.length) {
      this.elements.lapsCountBadge.text(`(${n})`);
    }
  }

  toggleStatsVisibility() {
    if (!this.elements.statsCard || !this.elements.statsCard.length) return;
    if (this.laps.length > 0) {
      this.elements.statsCard.removeClass('is-hidden');
    } else {
      this.elements.statsCard.addClass('is-hidden');
    }
  }

  formatTime(timeInMs) {
    const totalSeconds = Math.floor(timeInMs / 1000);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    const centiseconds = Math.floor((timeInMs % 1000) / 10);

    return {
      main: `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`,
      ms: centiseconds.toString().padStart(2, '0'),
    };
  }

  calculateAverageLapTime() {
    if (this.laps.length === 0) return 0;
    const total = this.laps.reduce((sum, lap) => sum + lap.lapTime, 0);
    return total / this.laps.length;
  }
}

$(document).ready(() => {
  window.timerApp = new TimerApp();
});
