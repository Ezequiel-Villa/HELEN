/* =========================================================
   HELEN – Wi-Fi Panel (REST backend)
   Reutiliza el componente existente, sin modales adicionales
   ========================================================= */

(function () {
  'use strict';

  const API_BASE = (typeof window.HELEN_API_BASE === 'string' && window.HELEN_API_BASE.trim())
    ? window.HELEN_API_BASE.trim().replace(/\/$/, '')
    : '';

  const refreshWifiButton = document.getElementById('refresh-wifi');
  const scanLabel = document.getElementById('scan-label');
  const networksEmpty = document.getElementById('networks-empty');
  const networksEmptyMessage = networksEmpty ? networksEmpty.querySelector('p') : null;
  const wifiList = document.getElementById('wifi-list');
  const passwordContainer = document.getElementById('password-container');
  const passwordInput = document.getElementById('wifi-password');
  const togglePassword = document.getElementById('toggle-password');
  const passwordToggleIcon = document.getElementById('password-toggle-icon');
  const connectWifiButton = document.getElementById('connect-wifi-button');
  const connectText = document.getElementById('connect-text');
  const connectingText = document.getElementById('connecting-text');
  const connectFeedback = document.getElementById('connect-feedback');

  const statusCard = document.getElementById('wifi-status');
  const statusPrimary = document.getElementById('wifi-status-primary');
  const statusSecondary = document.getElementById('wifi-status-secondary');
  const statusMeta = document.getElementById('wifi-status-extra');

  const pollingIntervalMs = 12000;
  const AUTO_RECONNECT_DELAY_MS = 7000;
  const AUTO_RECONNECT_MAX_ATTEMPTS = 3;

  let networks = [];
  let selectedNetwork = null;
  let lastStatus = null;
  let pollingHandle = null;
  let scanning = false;
  let connecting = false;
  let autoReconnectTimer = null;
  let autoReconnectAttempts = 0;
  let lastSuccessfulCredentials = null;

  const SECURE_WORDS = ['wpa', 'wpa2', 'wpa3', 'sae', 'wep'];

  const httpErrorCopy = {
    default: 'Ocurrió un error inesperado. Intenta nuevamente.',
    timeout: 'El servidor de red tardó demasiado en responder.',
    network: 'No se pudo contactar el servicio de red.',
    invalidPassword: 'Contraseña incorrecta o red rechazada.',
  };

  const isSecureNetwork = (value) => {
    if (!value) return false;
    const normalized = String(value).toLowerCase();
    if (!normalized.trim()) return false;
    return SECURE_WORDS.some((token) => normalized.includes(token)) && !normalized.includes('abiert');
  };

  const normalizeSignal = (value) => {
    const number = Number(value);
    if (!Number.isFinite(number)) return null;
    return Math.max(0, Math.min(100, Math.round(number)));
  };

  const formatSignal = (value) => {
    const normalized = normalizeSignal(value);
    if (normalized == null) return 'Sin dato';
    return `${normalized}%`;
  };

  const getApiUrl = (path) => {
    if (!path) return API_BASE || '';
    if (API_BASE) {
      return `${API_BASE}${path}`;
    }
    return path;
  };

  const clearSelection = () => {
    selectedNetwork = null;
    if (wifiList) {
      wifiList.querySelectorAll('.wifi-item.selected').forEach((node) => node.classList.remove('selected'));
    }
    updatePasswordPanel('idle');
    if (connectWifiButton) {
      connectWifiButton.disabled = true;
    }
    if (connectText) {
      connectText.textContent = 'Conectar';
    }
    if (connectingText) {
      connectingText.textContent = 'Conectando...';
    }
    setFeedback('', '');
  };

  const renderMetaLines = (lines) => {
    if (!statusMeta) return;
    statusMeta.innerHTML = '';
    if (!lines || !lines.length) {
      statusMeta.style.display = 'none';
      return;
    }
    lines.forEach((line) => {
      const span = document.createElement('span');
      span.textContent = line;
      statusMeta.appendChild(span);
    });
    statusMeta.style.display = 'block';
  };

  const setStatusTexts = (primary, secondary, metaLines, stateClass) => {
    if (statusCard) {
      statusCard.classList.remove('is-online', 'is-offline', 'is-connecting');
      if (stateClass) {
        statusCard.classList.add(stateClass);
      }
    }
    if (statusPrimary) {
      statusPrimary.textContent = primary || 'Sin conexión';
    }
    if (statusSecondary) {
      statusSecondary.textContent = secondary || '';
    }
    renderMetaLines(metaLines);
  };

  const extractErrorMessage = (error, fallback) => {
    if (!error) return fallback || httpErrorCopy.default;
    if (error.code === 'ECONNABORTED') {
      return httpErrorCopy.timeout;
    }
    if (error.message && /timeout|network error/i.test(error.message)) {
      return httpErrorCopy.timeout;
    }
    if (error.response && error.response.data) {
      const data = error.response.data;
      if (typeof data === 'string' && data.trim()) {
        return data;
      }
      if (typeof data.reason === 'string' && data.reason.trim()) {
        return data.reason;
      }
      if (typeof data.error === 'string' && data.error.trim()) {
        return data.error;
      }
    }
    if (error.message && typeof error.message === 'string' && error.message.trim()) {
      return error.message;
    }
    return fallback || httpErrorCopy.default;
  };

  const setFeedback = (message, tone) => {
    if (!connectFeedback) return;
    connectFeedback.textContent = message || '';
    connectFeedback.classList.remove('connect-feedback--error', 'connect-feedback--success');
    if (!message) {
      connectFeedback.setAttribute('data-visible', 'false');
      return;
    }
    connectFeedback.setAttribute('data-visible', 'true');
    if (tone === 'error') {
      connectFeedback.classList.add('connect-feedback--error');
    } else if (tone === 'success') {
      connectFeedback.classList.add('connect-feedback--success');
    }
  };

  const updatePasswordPanel = (mode, savedPassword = '') => {
    if (!passwordContainer) {
      return;
    }

    const normalizedMode = typeof mode === 'string' ? mode : 'idle';
    passwordContainer.setAttribute('data-mode', normalizedMode);

    if (passwordInput) {
      const requiresPassword = normalizedMode === 'secure';
      passwordInput.disabled = !requiresPassword;
      if (requiresPassword) {
        passwordInput.placeholder = 'Introduce la contraseña';
        if (savedPassword) {
          passwordInput.value = savedPassword;
        }
      } else if (normalizedMode === 'open') {
        passwordInput.value = '';
        passwordInput.placeholder = 'Esta red no requiere contraseña';
      } else {
        passwordInput.value = '';
        passwordInput.placeholder = 'Selecciona una red para continuar';
      }
    }

    if (togglePassword) {
      const disabled = normalizedMode !== 'secure';
      togglePassword.setAttribute('aria-disabled', disabled ? 'true' : 'false');
    }
  };

  const setButtonLoading = (loading) => {
    if (!connectWifiButton) return;
    connectWifiButton.classList.toggle('is-loading', Boolean(loading));
    if (loading) {
      connectWifiButton.disabled = true;
    } else if (selectedNetwork) {
      connectWifiButton.disabled = false;
    }
  };

  updatePasswordPanel('idle');
  if (connectFeedback) {
    connectFeedback.setAttribute('data-visible', 'false');
  }

  function clearAutoReconnectTimer () {
    if (autoReconnectTimer) {
      window.clearTimeout(autoReconnectTimer);
      autoReconnectTimer = null;
    }
  }

  function scheduleAutoReconnect () {
    if (!lastSuccessfulCredentials) return;
    if (connecting) return;
    if (autoReconnectTimer) return;
    if (autoReconnectAttempts >= AUTO_RECONNECT_MAX_ATTEMPTS) return;
    if (document.hidden) return;
    if (lastStatus && lastStatus.connected_ssid) {
      autoReconnectAttempts = 0;
      return;
    }

    autoReconnectTimer = window.setTimeout(() => {
      autoReconnectTimer = null;
      if (connecting) {
        scheduleAutoReconnect();
        return;
      }
      if (lastStatus && lastStatus.connected_ssid) {
        autoReconnectAttempts = 0;
        return;
      }
      autoReconnectAttempts += 1;
      performConnectionAttempt(lastSuccessfulCredentials, { auto: true }).then((success) => {
        if (success) {
          autoReconnectAttempts = 0;
        } else {
          scheduleAutoReconnect();
        }
      });
    }, AUTO_RECONNECT_DELAY_MS);
  }

  const setNetworksEmptyState = (visible, message) => {
    if (!networksEmpty) return;
    networksEmpty.style.display = visible ? 'flex' : 'none';
    if (networksEmptyMessage) {
      networksEmptyMessage.textContent = message || '';
    }
  };

  const setScanningUI = (value) => {
    scanning = Boolean(value);
    if (refreshWifiButton) {
      refreshWifiButton.classList.toggle('scanning', scanning);
    }
    if (scanLabel) {
      scanLabel.textContent = scanning ? 'Buscando...' : 'Buscar Redes';
    }
  };

  const normalizeNetworks = (items) => {
    if (!Array.isArray(items)) return [];
    return items
      .map((entry) => {
        const ssid = typeof entry.ssid === 'string' ? entry.ssid.trim() : '';
        const signal = normalizeSignal(entry.signal ?? entry.quality);
        const security = typeof entry.security === 'string' ? entry.security : '';
        return {
          ssid,
          signal,
          security,
          secure: isSecureNetwork(security),
        };
      })
      .filter((item) => item.ssid);
  };

  const barsIconFromQuality = (value) => {
    const percent = normalizeSignal(value);
    if (percent == null) return 'bi-reception-0';
    if (percent >= 80) return 'bi-reception-4';
    if (percent >= 60) return 'bi-reception-3';
    if (percent >= 40) return 'bi-reception-2';
    if (percent >= 20) return 'bi-reception-1';
    return 'bi-reception-0';
  };

  const highlightConnected = (ssid) => {
    if (!wifiList) return;
    const target = (ssid || '').toLowerCase();
    wifiList.querySelectorAll('.wifi-item').forEach((node) => {
      const nodeSsid = (node.getAttribute('data-ssid') || '').toLowerCase();
      node.classList.toggle('active', target && nodeSsid === target);
    });
  };

  const renderNetworkItem = (network, isActive) => {
    const item = document.createElement('div');
    item.className = 'wifi-item';
    item.setAttribute('role', 'option');
    item.tabIndex = 0;
    item.dataset.ssid = network.ssid;
    if (isActive) {
      item.classList.add('active');
    }

    const signalWrap = document.createElement('div');
    signalWrap.className = 'wifi-signal';
    const signalIcon = document.createElement('i');
    signalIcon.className = isActive ? 'bi bi-wifi' : 'bi bi-wifi';
    signalWrap.appendChild(signalIcon);

    const info = document.createElement('div');
    info.className = 'wifi-info';

    const titleRow = document.createElement('div');
    titleRow.className = 'wifi-row';
    const nameEl = document.createElement('div');
    nameEl.className = 'wifi-name';
    nameEl.textContent = network.ssid || '(SSID desconocido)';
    titleRow.appendChild(nameEl);
    if (isActive) {
      const badge = document.createElement('span');
      badge.className = 'badge';
      badge.textContent = 'Conectado';
      titleRow.appendChild(badge);
    }

    const meta = document.createElement('div');
    meta.className = 'meta';
    const lock = document.createElement('span');
    lock.className = 'lock';
    lock.innerHTML = network.secure
      ? '<i class="bi bi-lock"></i> Segura'
      : '<i class="bi bi-unlock"></i> Abierta';
    const strength = document.createElement('span');
    strength.className = 'signal';
    strength.textContent = formatSignal(network.signal);
    const bars = document.createElement('span');
    bars.className = 'bars';
    const barsIcon = document.createElement('i');
    barsIcon.className = `bi ${barsIconFromQuality(network.signal)}`;
    bars.appendChild(barsIcon);

    meta.append(lock, strength, bars);
    info.append(titleRow, meta);

    const aside = document.createElement('div');
    aside.className = 'wifi-strength';

    item.append(signalWrap, info, aside);

    const select = () => {
      if (!connectWifiButton) return;
      wifiList.querySelectorAll('.wifi-item').forEach((el) => el.classList.remove('selected'));
      item.classList.add('selected');
      selectedNetwork = network;
      clearAutoReconnectTimer();
      autoReconnectAttempts = 0;
      if (lastSuccessfulCredentials && lastSuccessfulCredentials.ssid !== network.ssid) {
        lastSuccessfulCredentials = null;
      }
      setFeedback('', '');
      const isCurrent = Boolean(lastStatus && lastStatus.connected_ssid && lastStatus.connected_ssid.toLowerCase() === network.ssid.toLowerCase());
      const savedPassword = (lastSuccessfulCredentials && lastSuccessfulCredentials.ssid === network.ssid)
        ? lastSuccessfulCredentials.password
        : '';
      updatePasswordPanel(network.secure ? 'secure' : 'open', savedPassword);
      if (network.secure && passwordInput) {
        passwordInput.focus({ preventScroll: true });
        if (savedPassword) {
          const length = savedPassword.length;
          passwordInput.setSelectionRange(length, length);
        }
      } else if (!network.secure) {
        setFeedback('Red abierta: no requiere contraseña.', '');
      }
      connectWifiButton.disabled = false;
      if (connectText) {
        connectText.textContent = isCurrent ? 'Reconectar' : 'Conectar';
      }
      if (connectingText) {
        connectingText.textContent = isCurrent ? 'Reconectando...' : 'Conectando...';
      }
    };

    item.addEventListener('click', select);
    item.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        select();
      }
    });

    return item;
  };

  const renderNetworks = (list) => {
    if (!wifiList) return;
    wifiList.innerHTML = '';
    if (!list.length) {
      setNetworksEmptyState(true, scanning ? 'Buscando redes disponibles...' : 'No se encontraron redes.');
      return;
    }
    setNetworksEmptyState(false);
    const current = lastStatus && lastStatus.connected_ssid ? lastStatus.connected_ssid.toLowerCase() : '';
    list.forEach((net) => {
      const isActive = current && net.ssid.toLowerCase() === current;
      const element = renderNetworkItem(net, isActive);
      wifiList.appendChild(element);
    });
  };

  const updateStatusUI = (status, fallbackMessage) => {
    lastStatus = status || null;
    if (!status) {
      setStatusTexts('Sin conexión', fallbackMessage || 'Escanea para buscar redes disponibles.', [] , 'is-offline');
      highlightConnected('');
      scheduleAutoReconnect();
      return;
    }

    const online = Boolean(status.online || status.connected_ssid);
    const primary = online
      ? `Conectado a ${status.connected_ssid || 'Wi-Fi'}`
      : 'Sin conexión';
    const secondary = online
      ? `Señal ${formatSignal(status.signal)}${status.ipv4 ? ` · IP ${status.ipv4}` : ''}`
      : (fallbackMessage || 'Escanea para buscar redes disponibles.');

    const metaLines = [];
    if (status.iface) {
      metaLines.push(`Interfaz: ${status.iface}`);
    }
    if (status.gateway) {
      metaLines.push(`Puerta de enlace: ${status.gateway}`);
    }
    if (!online && typeof status.message === 'string' && status.message.trim()) {
      metaLines.push(status.message.trim());
    }

    const stateClass = online ? 'is-online' : (status.connecting ? 'is-connecting' : 'is-offline');
    setStatusTexts(primary, secondary, metaLines, stateClass);
    highlightConnected(status.connected_ssid || '');

    if (online) {
      clearAutoReconnectTimer();
      autoReconnectAttempts = 0;
    } else {
      scheduleAutoReconnect();
    }
  };

  const refreshStatus = async () => {
    try {
      const onlineRequest = axios.get(getApiUrl('/net/online'), { timeout: 4000 }).catch(() => null);
      const statusRequest = axios.get(getApiUrl('/net/status'), { timeout: 5000 }).catch(() => null);
      const [onlineResponse, statusResponse] = await Promise.all([onlineRequest, statusRequest]);
      const onlineData = onlineResponse && onlineResponse.data ? onlineResponse.data : null;
      const statusData = statusResponse && statusResponse.data ? statusResponse.data : null;
      const merged = Object.assign({}, onlineData || {}, statusData || {});
      updateStatusUI(merged, onlineData && onlineData.message);
      return merged;
    } catch (error) {
      console.error('[WiFi] refreshStatus:', error);
      updateStatusUI(null, httpErrorCopy.network);
      return null;
    }
  };

  const schedulePolling = () => {
    if (pollingHandle) {
      window.clearInterval(pollingHandle);
    }
    pollingHandle = window.setInterval(() => {
      if (!document.hidden) {
        refreshStatus();
      }
    }, pollingIntervalMs);
  };

  const stopPolling = () => {
    if (pollingHandle) {
      window.clearInterval(pollingHandle);
      pollingHandle = null;
    }
  };

  const scanNetworks = async () => {
    if (scanning) return;
    setScanningUI(true);
    setFeedback('', '');
    try {
      const response = await axios.get(getApiUrl('/net/scan'), { timeout: 12000 });
      const list = normalizeNetworks(response && response.data && response.data.networks);
      networks = list;
      renderNetworks(list);
      if (selectedNetwork) {
        const stillExists = list.some((net) => net.ssid === selectedNetwork.ssid);
        if (!stillExists) {
          clearSelection();
        }
      }
      if (!list.length) {
        setNetworksEmptyState(true, 'No se encontraron redes disponibles.');
      }
    } catch (error) {
      console.error('[WiFi] scanNetworks:', error);
      networks = [];
      renderNetworks([]);
      setNetworksEmptyState(true, extractErrorMessage(error, 'No se pudo buscar redes.'));
      setFeedback('No se pudieron obtener redes Wi-Fi.', 'error');
    } finally {
      setScanningUI(false);
    }
  };

  async function performConnectionAttempt (credentials, options = {}) {
    if (!credentials || !credentials.ssid || connecting) {
      return false;
    }

    const auto = Boolean(options.auto);
    const passwordValue = credentials.password || '';
    const secure = Boolean(credentials.secure);

    if (!auto) {
      setFeedback('', '');
    } else if (connectFeedback && credentials.ssid) {
      setFeedback(`Intentando reconectar a ${credentials.ssid}...`, '');
    }

    connecting = true;
    if (!auto) {
      setButtonLoading(true);
    }

    const payload = { ssid: credentials.ssid };
    if (secure && passwordValue) {
      payload.password = passwordValue;
    }

    try {
      const response = await axios.post(getApiUrl('/net/connect'), payload, { timeout: 20000 });
      const data = response && response.data ? response.data : {};
      if (data.connected) {
        setFeedback(`Conectado a ${credentials.ssid}.`, 'success');
        lastSuccessfulCredentials = {
          ssid: credentials.ssid,
          password: passwordValue,
          secure
        };
        autoReconnectAttempts = 0;
        clearAutoReconnectTimer();
        await refreshStatus();
        window.setTimeout(scanNetworks, 500);
        return true;
      }

      const reason = extractErrorMessage({ response: { data } }, httpErrorCopy.invalidPassword);
      if (!auto) {
        setFeedback(reason, 'error');
      } else {
        console.warn('[WiFi] Reconexión automática fallida:', reason);
      }
      return false;
    } catch (error) {
      const message = extractErrorMessage(error, httpErrorCopy.default);
      if (!auto) {
        setFeedback(message, 'error');
      } else {
        console.error('[WiFi] Reconexión automática fallida:', message);
      }
      return false;
    } finally {
      connecting = false;
      if (!auto) {
        setButtonLoading(false);
      }
    }
  }

  const connectToNetwork = async () => {
    if (!selectedNetwork || connecting) {
      return;
    }

    const requiresPassword = Boolean(selectedNetwork.secure);
    const passwordValue = requiresPassword && passwordInput ? passwordInput.value : '';

    if (requiresPassword && !passwordValue) {
      setFeedback('Introduce la contraseña de la red seleccionada.', 'error');
      if (passwordInput) {
        passwordInput.focus();
      }
      return;
    }

    clearAutoReconnectTimer();
    autoReconnectAttempts = 0;

    const credentials = {
      ssid: selectedNetwork.ssid,
      password: passwordValue,
      secure: requiresPassword
    };

    await performConnectionAttempt(credentials);
  };

  if (togglePassword && passwordInput) {
    togglePassword.addEventListener('click', () => {
      if (togglePassword.getAttribute('aria-disabled') === 'true') {
        return;
      }
      const nextType = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
      passwordInput.setAttribute('type', nextType);
      if (passwordToggleIcon) {
        passwordToggleIcon.className = nextType === 'password' ? 'bi bi-eye-slash' : 'bi bi-eye';
      }
    });
  }

  if (refreshWifiButton) {
    refreshWifiButton.addEventListener('click', () => {
      scanNetworks();
      refreshStatus();
    });
  }

  if (connectWifiButton) {
    connectWifiButton.addEventListener('click', connectToNetwork);
  }

  if (passwordInput) {
    passwordInput.addEventListener('keyup', (event) => {
      if (event.key === 'Enter') {
        connectToNetwork();
      }
    });
  }

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopPolling();
      clearAutoReconnectTimer();
    } else {
      refreshStatus();
      schedulePolling();
    }
  });

  document.addEventListener('DOMContentLoaded', async () => {
    await refreshStatus();
    await scanNetworks();
    schedulePolling();
  });
})();
