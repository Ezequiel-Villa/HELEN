const { ipcRenderer, contextBridge } = require('electron');

// Exponer APIs seguras al renderer process
contextBridge.exposeInMainWorld('wifiAPI', {
    scanNetworks: () => ipcRenderer.invoke('wifi:scan'),
    getCurrentConnections: () => ipcRenderer.invoke('wifi:getCurrentConnections'),
    connect: (config) => ipcRenderer.invoke('wifi:connect', config),
    disconnect: () => ipcRenderer.invoke('wifi:disconnect'),
    disableWifi: () => ipcRenderer.invoke('wifi:disable'),
    enableWifi: () => ipcRenderer.invoke('wifi:enable'),
    forget: (ssid) => ipcRenderer.invoke('wifi:forget', ssid)
});

// Exponer funciones al contexto del frontend
contextBridge.exposeInMainWorld('myAPI', {
    navigate: (relativePath) => ipcRenderer.send('navigate', relativePath) // Enviar mensaje al proceso principal
});

// Otras APIs que puedas necesitar
contextBridge.exposeInMainWorld('electronAPI', {
    onWifiStatusChange: (callback) => ipcRenderer.on('wifi:statusChange', callback)
});