const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { exec } = require('child_process');
const fs = require('fs');

let mainWindow;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1024,
        height: 600,
        minHeight: 600,
        minWidth: 1024,
        maxHeight: 600,
        maxWidth: 1024,
        fullscreen: true,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false
        }
    });

    mainWindow.loadFile('index.html');
    // mainWindow.webContents.openDevTools();
}

// Manejar eventos de navegación
ipcMain.on('navigate', (event, relativePath) => {
    const fullPath = path.join(__dirname, relativePath);
    mainWindow.loadFile(fullPath).catch(err => {
        console.error('Error al cargar la página:', err);
    });
});

app.whenReady().then(() => {
    setupWifiIpcHandlers();
    createWindow();
    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit();
});

function setupWifiIpcHandlers() {
    // Escanear redes
    ipcMain.handle('wifi:scan', async () => {
        return new Promise((resolve, reject) => {
            exec('sudo iwlist wlan0 scan', (error, stdout) => {
                if (error) return reject('Error al escanear redes WiFi');

                const networks = [];
                const blocks = stdout.split('Cell');
                blocks.forEach(block => {
                    const ssidMatch = block.match(/ESSID:"(.+?)"/);
                    const qualityMatch = block.match(/Quality=(\d+)\/(\d+)/);
                    const encryption = /Encryption key:on/.test(block);

                    if (ssidMatch) {
                        networks.push({
                            ssid: ssidMatch[1],
                            quality: qualityMatch ? Math.round((parseInt(qualityMatch[1]) / parseInt(qualityMatch[2])) * 100) : 0,
                            security: encryption ? 'WPA/WEP' : ''
                        });
                    }
                });
                resolve(networks);
            });
        });
    });

    // Obtener conexión actual
    ipcMain.handle('wifi:getCurrentConnections', async () => {
        return new Promise((resolve) => {
            exec('iwgetid -r', (error, stdout) => {
                if (error || !stdout.trim()) return resolve([]);
                resolve([{ ssid: stdout.trim() }]);
            });
        });
    });

    // Conectar a red WiFi
    ipcMain.handle('wifi:connect', async (event, config) => {
        return new Promise((resolve, reject) => {

            exec(`sudo /usr/local/bin/connect-wifi.sh "${config.ssid}" "${config.password}"`, (error) => {
                if (error) {
                    return reject({ success: false, error: 'Error al ejecutar el script de conexión WiFi' });
                }
                resolve({ success: true });
            });
        });
    });

    // Olvidar una red wifi
    ipcMain.handle('wifi:forget', async (event, ssid) => {
        return new Promise((resolve, reject) => {
            exec(`sudo /usr/local/bin/forget-wifi.sh "${ssid}"`, (error, stdout, stderr) => {
                if (error) {
                    return reject({ success: false, error: stderr.toString() });
                }
                resolve({ success: true, message: stdout.toString() });
            });
        });
    });

    ipcMain.handle('wifi:disable', async () => execPromise('sudo ifconfig wlan0 down'));
    ipcMain.handle('wifi:enable', async () => execPromise('sudo ifconfig wlan0 up'));
    ipcMain.handle('wifi:disconnect', async () => execPromise('sudo iwconfig wlan0 essid off'));
}

function execPromise(cmd) {
    return new Promise((resolve, reject) => {
        exec(cmd, (error) => {
            if (error) reject({ success: false, error: error.message });
            else resolve({ success: true });
        });
    });
}

async function checkAndNotifyWifiStatus() {
    exec('iwgetid -r', (error, stdout) => {
        const isConnected = !error && stdout.trim() !== '';
        const currentConnection = isConnected ? { ssid: stdout.trim() } : null;

        BrowserWindow.getAllWindows().forEach(win => {
            win.webContents.send('wifi:statusChange', { isConnected, currentConnection });
        });
    });
}

module.exports = {
    checkAndNotifyWifiStatus
};
