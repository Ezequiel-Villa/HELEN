// loadingScreenLoader.js
// Este script debe incluirse en TODAS las páginas y el index.html

// Función para determinar la ruta correcta según el contexto
function getRelativePath() {
    const path = window.location.pathname;
    
    if (path.includes('/pages/')) {
        return {
            js: '../../jsSignHandler/loadingScreen.js',
            css: '../../css/loadingScreen.css',
            video: '../../assets/videos/carga-avion.webm'
        };
    } else {
        return {
            js: './jsSignHandler/loadingScreen.js',
            css: './css/loadingScreen.css',
            video: './assets/videos/carga-avion.webm'
        };
    }
}

function loadResources() {
    const paths = getRelativePath();
    
    if (!document.getElementById('loading-screen-styles')) {
        const styleLink = document.createElement('link');
        styleLink.id = 'loading-screen-styles';
        styleLink.rel = 'stylesheet';
        styleLink.href = paths.css;
        document.head.appendChild(styleLink);
    }
    
    if (!window.loadingScreen) {
        const script = document.createElement('script');
        script.src = paths.js;
        script.async = true;
        script.onload = function() {
            if (window.loadingScreen) {
                window.loadingScreen.webmPath = paths.video;
                if (window.loadingScreen.initialized && window.loadingScreen.videoElement) {
                    window.loadingScreen.videoElement.src = paths.video;
                }
            }
        };
        document.body.appendChild(script);
    }
}

document.addEventListener('DOMContentLoaded', loadResources);