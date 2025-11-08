// ================== Estado global ==================
let selectedIcon = '';
let selectedAddIcon = '';
let deviceToDeleteId = null;
let deviceToEditId = null;
let nextDeviceId = 1;
let selectedRoom = '';
let selectedType = '';
let devices = [];
let currentPage = 1;

const DEVICES_PER_PAGE = 4;

const STORAGE_KEY = 'helen:devices:v1';

const qs  = (s) => document.querySelector(s);
const qsa = (s) => document.querySelectorAll(s);

const safeStorage = {
  get(key) {
    try {
      return window.localStorage.getItem(key);
    } catch (error) {
      console.warn('[Dispositivos] No se pudo leer localStorage:', error);
      return null;
    }
  },
  set(key, value) {
    try {
      window.localStorage.setItem(key, value);
    } catch (error) {
      console.warn('[Dispositivos] No se pudo escribir localStorage:', error);
    }
  },
};

function loadDevicesFromStorage() {
  const raw = safeStorage.get(STORAGE_KEY);
  if (!raw) {
    devices = [];
    nextDeviceId = 1;
    return;
  }
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      devices = [];
      nextDeviceId = 1;
      return;
    }
    const mapped = [];
    let maxId = 0;
    parsed.forEach((item) => {
      if (!item) return;
      const numericId = Number(item.id);
      if (!Number.isFinite(numericId)) return;
      const room = typeof item.room === 'string' ? item.room : '';
      const type = typeof item.type === 'string' ? item.type : '';
      const baseName = typeof item.name === 'string' ? item.name : [room, type].filter(Boolean).join(' - ');
      const name = baseName || 'Dispositivo';
      const icon = typeof item.icon === 'string' ? item.icon : '';
      const isOn = Boolean(item.isOn);
      mapped.push({
        id: numericId,
        room,
        type,
        name,
        icon,
        isOn,
      });
      if (numericId > maxId) {
        maxId = numericId;
      }
    });
    devices = mapped;
    nextDeviceId = maxId + 1 || 1;
    currentPage = 1;
  } catch (error) {
    console.warn('[Dispositivos] No se pudo parsear dispositivos almacenados:', error);
    devices = [];
    nextDeviceId = 1;
    currentPage = 1;
  }
}

function saveDevicesToStorage() {
  safeStorage.set(STORAGE_KEY, JSON.stringify(devices));
}

function getDeviceById(id) {
  const numericId = Number(id);
  if (!Number.isFinite(numericId)) return null;
  return devices.find((device) => device.id === numericId) || null;
}

function renderDevices() {
  const list = qs('#deviceList');
  if (!list) return;
  list.innerHTML = '';
  const total = Math.max(1, Math.ceil(devices.length / DEVICES_PER_PAGE));
  if (currentPage > total) {
    currentPage = total;
  }

  const start = (currentPage - 1) * DEVICES_PER_PAGE;
  const visible = devices.slice(start, start + DEVICES_PER_PAGE);

  visible.forEach((device) => {
    const name = device.name || [device.room, device.type].filter(Boolean).join(' - ') || 'Dispositivo';
    const statusText = device.isOn ? 'Encendido' : 'Apagado';
    const badgeClass = device.isOn ? 'status-on' : 'status-off';
    const badgeLabel = device.isOn ? 'ON' : 'OFF';
    const card = document.createElement('div');
    card.className = `device-card${device.isOn ? '' : ' is-off'}`;
    card.id = `device-${device.id}`;
    card.innerHTML = `
      <div class="device-header">
        <div class="device-icon">${device.icon || '💡'}</div>
        <div class="device-name">${name}</div>
      </div>
      <div class="device-status">
        <label class="switch">
          <input type="checkbox" ${device.isOn ? 'checked' : ''} onchange="toggleDeviceStatus(this, ${device.id})">
          <span class="slider"></span>
        </label>
        <span class="device-status-text">${statusText}</span>
        <span class="status-badge ${badgeClass}">${badgeLabel}</span>
      </div>
      <div class="device-actions">
        <button class="action-button" onclick="showEditModal(${device.id})" aria-label="Editar">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </button>
        <button class="action-button" onclick="showDeleteModal(${device.id})" aria-label="Eliminar">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2M10 11v6M14 11v6" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </button>
      </div>
    `;
    list.appendChild(card);
  });

  updateUIState();
  renderPagination();
  if (window.HelenRaspberryFit && typeof window.HelenRaspberryFit.refresh === 'function') {
    window.HelenRaspberryFit.refresh();
  }
}

// ================== Init ==================
document.addEventListener('DOMContentLoaded', () => {
  ensureEmptyStateNode();
  loadDevicesFromStorage();
  renderDevices();
});

// ================== Helpers UI ==================
function ensureEmptyStateNode() {
  const list = qs('#deviceList');
  if (!list) return;

  // Crea contenedor de vacío si no existe
  if (!qs('#emptyState')) {
    const empty = document.createElement('section');
    empty.id = 'emptyState';
    empty.className = 'empty-state';
    empty.style.display = 'none';
    empty.innerHTML = `
      <div class="empty-hero">
        <div class="empty-icon">🏠</div>
        <h3 class="empty-title">Sin dispositivos configurados</h3>
        <p class="empty-sub">Conecta tus dispositivos inteligentes para controlarlos desde HELEN</p>
        <div class="empty-perks">
          <div class="perk"><i class="bi bi-wifi"></i><span>Conexión Automática</span></div>
          <div class="perk"><i class="bi bi-houses"></i><span>Control Centralizado</span></div>
          <div class="perk"><i class="bi bi-plus-circle"></i><span>Fácil Configuración</span></div>
        </div>
        <button class="add-button" onclick="showAddModal()">+ Agregar dispositivo</button>
      </div>
    `;
    // Inserta el vacío justo antes de la lista para mantener layout
    list.parentNode.insertBefore(empty, list);
  }
}

function updateUIState() {
  const list     = qs('#deviceList');
  const empty    = qs('#emptyState');
  const subtitle = qs('#devices-subtitle');

  if (!list) return;

  const count = Array.isArray(devices) ? devices.length : list.querySelectorAll('.device-card').length;

  if (count > 0) {
    list.style.display = 'grid';
    if (empty) empty.style.display = 'none';
    if (subtitle) subtitle.textContent = 'Tus dispositivos conectados';
  } else {
    list.style.display = 'none';
    if (empty) empty.style.display = 'block';
    if (subtitle) subtitle.textContent = 'No hay dispositivos conectados aún';
    currentPage = 1;
  }
}

function renderPagination() {
  const container = qs('#devicePagination');
  if (!container) return;

  const total = Math.max(1, Math.ceil(devices.length / DEVICES_PER_PAGE));

  if (devices.length <= DEVICES_PER_PAGE) {
    container.innerHTML = '';
    container.setAttribute('hidden', 'hidden');
    return;
  }

  container.removeAttribute('hidden');
  container.innerHTML = '';

  const prev = document.createElement('button');
  prev.type = 'button';
  prev.textContent = 'Anterior';
  prev.disabled = currentPage <= 1;
  prev.addEventListener('click', () => {
    if (currentPage > 1) {
      currentPage -= 1;
      renderDevices();
    }
  });

  const next = document.createElement('button');
  next.type = 'button';
  next.textContent = 'Siguiente';
  next.disabled = currentPage >= total;
  next.addEventListener('click', () => {
    if (currentPage < total) {
      currentPage += 1;
      renderDevices();
    }
  });

  const indicator = document.createElement('span');
  indicator.className = 'page-indicator';
  indicator.textContent = `Página ${currentPage} de ${total}`;

  container.appendChild(prev);
  container.appendChild(indicator);
  container.appendChild(next);
}

// ================== Eliminar ==================
function showDeleteModal(deviceId) {
  deviceToDeleteId = Number(deviceId);
  const m = qs('#deleteModal');
  if (m) m.style.display = 'grid';
}
function hideDeleteModal() {
  const m = qs('#deleteModal');
  if (m) m.style.display = 'none';
  deviceToDeleteId = null;
}
function confirmDelete() {
  if (deviceToDeleteId == null) return;
  devices = devices.filter((device) => device.id !== deviceToDeleteId);
  saveDevicesToStorage();
  renderDevices();
  hideDeleteModal();
}

// ================== Editar ==================
function showEditModal(deviceId) {
  deviceToEditId = Number(deviceId);

  const device = getDeviceById(deviceToEditId);
  if (!device) return;

  selectedRoom = device.room || '';
  selectedType = device.type || '';
  selectedIcon = device.icon || '';

  if ((!selectedRoom || !selectedType) && device.name) {
    const [roomFromName, typeFromName] = device.name.split(' - ');
    if (!selectedRoom && roomFromName) selectedRoom = roomFromName;
    if (!selectedType && typeFromName) selectedType = typeFromName;
  }

  qsa('#editModal .room-option').forEach((opt) => opt.classList.remove('selected'));
  if (selectedRoom) {
    const roomEl = qs(`#editModal .room-option[data-room="${selectedRoom}"]`);
    if (roomEl) {
      selectRoom(roomEl, selectedRoom);
    }
  }

  qsa('#editModal .type-option').forEach((opt) => opt.classList.remove('selected'));
  if (selectedType) {
    const typeEl = qs(`#editModal .type-option[data-type="${selectedType}"]`);
    if (typeEl) {
      selectDeviceType(typeEl, selectedType);
    }
  }

  qsa('#editModal .icon-option').forEach((opt) => {
    const match = opt.textContent.trim() === (selectedIcon || '').trim();
    opt.classList.toggle('selected', match);
  });

  const m = qs('#editModal');
  if (m) m.style.display = 'grid';
  updateDeviceNamePreview('edit');
}

function hideEditModal() {
  const m = qs('#editModal');
  if (m) m.style.display = 'none';
  deviceToEditId = null;
  selectedRoom = '';
  selectedType = '';
  selectedIcon = '';
}

function selectIcon(el, icon) {
  qsa('#editModal .icon-option').forEach(o => o.classList.remove('selected'));
  el.classList.add('selected');
  selectedIcon = icon;
  suggestDeviceType(icon, 'edit');
}

function selectRoom(el, room) {
  qsa('#editModal .room-option').forEach(o => o.classList.remove('selected'));
  el.classList.add('selected');
  selectedRoom = room;
  updateDeviceNamePreview('edit');
}

function selectDeviceType(el, type) {
  qsa('#editModal .type-option').forEach(o => o.classList.remove('selected'));
  el.classList.add('selected');
  selectedType = type;
  updateDeviceNamePreview('edit');
}

function updateDeviceNamePreview(mode) {
  const span = qs(`#${mode === 'edit' ? 'deviceNamePreview' : 'newDeviceNamePreview'}`);
  if (!span) return;

  if (selectedRoom && selectedType) {
    span.textContent = `${selectedRoom} - ${selectedType}`;
    span.parentElement.classList.remove('name-preview-empty');
  } else if (selectedRoom) {
    span.textContent = `${selectedRoom} - ...`;
    span.parentElement.classList.remove('name-preview-empty');
  } else if (selectedType) {
    span.textContent = `... - ${selectedType}`;
    span.parentElement.classList.remove('name-preview-empty');
  } else {
    span.textContent = 'Selecciona ubicación y tipo';
    span.parentElement.classList.add('name-preview-empty');
  }
}

function suggestDeviceType(icon, mode) {
  let suggested = '';
  switch (icon) {
    case '💡': suggested = 'Luz'; break;
    case '🔌': suggested = 'Enchufe'; break;
    case '🌡️': suggested = 'Termostato'; break;
    case '📺': suggested = 'TV'; break;
    case '🔊': suggested = 'Altavoz'; break;
    case '🎮': suggested = 'Consola'; break;
    case '🚪': suggested = 'Puerta'; break;
    case '🔒': suggested = 'Cerradura'; break;
    case '💧': suggested = 'Riego'; break;
    case '☕': suggested = 'Cafetera'; break;
    case '💻': suggested = 'Ordenador'; break;
    case '⏰': suggested = 'Alarma'; break;
    case '📱': suggested = 'Móvil'; break;
    case '🎛️': suggested = 'Control'; break;
    case '🖨️': suggested = 'Impresora'; break;
    default: return;
  }
  const scope = mode === 'edit' ? '#editModal' : '#addModal';
  const el = qs(`${scope} .type-option[data-type="${suggested}"]`);
  if (!el) return;
  mode === 'edit' ? selectDeviceType(el, suggested) : selectAddDeviceType(el, suggested);
}

function saveDeviceChanges() {
  if (!selectedRoom || !selectedType) { alert('Selecciona ubicación y tipo'); return; }
  if (!selectedIcon) { alert('Selecciona un ícono'); return; }

  const device = getDeviceById(deviceToEditId);
  if (!device) return;

  device.room = selectedRoom;
  device.type = selectedType;
  device.name = `${selectedRoom} - ${selectedType}`;
  device.icon = selectedIcon;

  saveDevicesToStorage();
  renderDevices();
  hideEditModal();
}

// ================== Agregar ==================
function showAddModal() {
  selectedAddIcon = '';
  selectedRoom = '';
  selectedType = '';

  qsa('#addModal .icon-option').forEach(o => o.classList.remove('selected'));
  qsa('#addModal .room-option').forEach(o => o.classList.remove('selected'));
  qsa('#addModal .type-option').forEach(o => o.classList.remove('selected'));

  updateDeviceNamePreview('add');
  const m = qs('#addModal');
  if (m) m.style.display = 'grid';
}
function hideAddModal() {
  const m = qs('#addModal');
  if (m) m.style.display = 'none';
  selectedRoom = '';
  selectedType = '';
  selectedAddIcon = '';
}

function selectAddIcon(el, icon) {
  qsa('#addModal .icon-option').forEach(o => o.classList.remove('selected'));
  el.classList.add('selected');
  selectedAddIcon = icon;
  suggestDeviceType(icon, 'add');
}
function selectAddRoom(el, room) {
  qsa('#addModal .room-option').forEach(o => o.classList.remove('selected'));
  el.classList.add('selected');
  selectedRoom = room;
  updateDeviceNamePreview('add');
}
function selectAddDeviceType(el, type) {
  qsa('#addModal .type-option').forEach(o => o.classList.remove('selected'));
  el.classList.add('selected');
  selectedType = type;
  updateDeviceNamePreview('add');
}

function addNewDevice() {
  if (!selectedRoom || !selectedType) { alert('Por favor selecciona una ubicación y un tipo para el dispositivo'); return; }
  if (!selectedAddIcon) { alert('Por favor selecciona un ícono para el dispositivo'); return; }

  const deviceName = `${selectedRoom} - ${selectedType}`;
  const id = nextDeviceId++;

  devices.push({
    id,
    room: selectedRoom,
    type: selectedType,
    name: deviceName,
    icon: selectedAddIcon,
    isOn: false,
  });

  saveDevicesToStorage();
  currentPage = Math.max(1, Math.ceil(devices.length / DEVICES_PER_PAGE));
  renderDevices();
  hideAddModal();
}

// ================== Encendido/Apagado ==================
function toggleDeviceStatus(checkbox, deviceId) {
  const card       = qs(`#device-${deviceId}`);
  if (!card) return;

  const statusText  = card.querySelector('.device-status-text');
  const statusBadge = card.querySelector('.status-badge');
  const isOn = checkbox.checked;

  if (statusText)  statusText.textContent  = isOn ? 'Encendido' : 'Apagado';
  if (statusBadge) {
    statusBadge.textContent = isOn ? 'ON' : 'OFF';
    statusBadge.className   = 'status-badge ' + (isOn ? 'status-on' : 'status-off');
  }
  card.classList.toggle('is-off', !isOn);

  const device = getDeviceById(deviceId);
  if (device) {
    device.isOn = isOn;
    saveDevicesToStorage();
  }

  // Hook opcional para backend
  if (typeof window.onDeviceToggle === 'function') {
    try { window.onDeviceToggle(deviceId, isOn); } catch (_) {}
  }
}

// ================== Vacío (compatibilidad antigua) ==================
function checkEmptyState() {
  // Compat: algunas vistas llamaban a esta función
  ensureEmptyStateNode();
  updateUIState();
}

// ========== Exponer para los onclick del HTML ==========
window.showDeleteModal = showDeleteModal;
window.hideDeleteModal = hideDeleteModal;
window.confirmDelete   = confirmDelete;

window.showEditModal   = showEditModal;
window.hideEditModal   = hideEditModal;
window.selectIcon      = selectIcon;
window.selectRoom      = selectRoom;
window.selectDeviceType= selectDeviceType;
window.saveDeviceChanges = saveDeviceChanges;

window.showAddModal    = showAddModal;
window.hideAddModal    = hideAddModal;
window.selectAddIcon   = selectAddIcon;
window.selectAddRoom   = selectAddRoom;
window.selectAddDeviceType = selectAddDeviceType;
window.addNewDevice    = addNewDevice;

window.toggleDeviceStatus = toggleDeviceStatus;
window.checkEmptyState    = checkEmptyState;
