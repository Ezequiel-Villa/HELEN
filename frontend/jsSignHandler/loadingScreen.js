// loadingScreen.js - Sistema centralizado de pantalla de carga

class LoadingScreen {
  constructor() {
    this.loadingOverlay = null;
    this.videoElement = null;
    this.messageElement = null;
    this.initialized = false;
    this.webmPath = './assets/videos/carga-avion.webm';
    this.cssPath = './css/loadingScreen.css';
  }

  initialize() {
    if (this.initialized) return;

    this.adjustPathForContext();

    this.loadingOverlay = document.createElement('div');
    this.loadingOverlay.className = 'loading-overlay';
    
    const loadingContent = document.createElement('div');
    loadingContent.className = 'loading-content';
    
    this.videoElement = document.createElement('video');
    this.videoElement.className = 'loading-video';
    this.videoElement.loop = true;
    this.videoElement.muted = true;
    this.videoElement.playsInline = true;
    this.videoElement.src = this.webmPath;
    
    this.messageElement = document.createElement('div');
    this.messageElement.className = 'loading-message';
    this.messageElement.textContent = 'Cargando...';
    
    loadingContent.appendChild(this.videoElement);
    loadingContent.appendChild(this.messageElement);
    this.loadingOverlay.appendChild(loadingContent);
    document.body.appendChild(this.loadingOverlay);
    
    if (!document.getElementById('loading-screen-styles')) {
      const styleLink = document.createElement('link');
      styleLink.id = 'loading-screen-styles';
      styleLink.rel = 'stylesheet';
      styleLink.href = this.cssPath;
      document.head.appendChild(styleLink);
    }
    
    this.initialized = true;
    this.hide();
  }

  show(message = 'Cargando...') {
    if (!this.initialized) this.initialize();
    
    this.messageElement.textContent = message;
    this.loadingOverlay.style.display = 'flex';
    this.videoElement.play().catch(err => console.error('Error al reproducir el video:', err));
  }

  hide() {
    if (!this.initialized) return;
    
    this.loadingOverlay.style.display = 'none';
    this.videoElement.pause();
  }

  async showAndExecute(processingFunction, message = 'Cargando...') {
    this.show(message);
    
    try {
      if (processingFunction && typeof processingFunction === 'function') {
        await processingFunction();
      }
      
      await new Promise(resolve => setTimeout(resolve, 800));
      
      this.hide();
    } catch (error) {
      console.error('Error durante el procesamiento:', error);
      this.hide();
    }
  }

  adjustPathForContext() {
    const path = window.location.pathname;
    
    if (path.includes('/pages/')) {
      this.webmPath = '../../assets/videos/carga-avion.webm';
      this.cssPath = '../../css/loadingScreen.css';
    } else {
      this.webmPath = './assets/videos/carga-avion.webm';
      this.cssPath = './css/loadingScreen.css';
    }
    
    if (this.videoElement) {
      this.videoElement.src = this.webmPath;
    }
  }
}

const loadingScreen = new LoadingScreen();

window.loadingScreen = loadingScreen;

document.addEventListener('DOMContentLoaded', () => {
  loadingScreen.initialize();
});