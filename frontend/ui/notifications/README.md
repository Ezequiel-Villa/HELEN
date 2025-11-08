# Módulo de notificaciones globales

Este módulo proporciona el popup único que se muestra cuando un temporizador termina o una alarma se activa. Se apoya en el motor `alarm-core.js`/`HelenScheduler` y garantiza un único botón principal **Detener**, audio en bucle controlado y cola de eventos para múltiples disparos.

## Carga rápida
1. Importa la hoja `helen/css/globals.css` (ya enlaza `ui/notifications/notifications.css`).
2. Inserta `<script src="../../ui/notifications/notifications.js" defer></script>` **antes** de cargar `alarm-core.js` en cada plantilla que necesite escuchar alarmas o temporizadores.
3. Incluye `alarm-core.js` como siempre: el motor detecta automáticamente el módulo y administra el popup.

No es necesario instanciar elementos en el DOM: el popup se crea dinámicamente la primera vez que se necesita.

## API JavaScript
El archivo `notifications.js` expone el objeto global `window.HelenNotifications` (versión 1). Está pensado para uso interno de `HelenScheduler`, pero se puede consultar desde otras integraciones si hace falta personalización.

### `HelenNotifications.ensure()`
Crea los nodos del popup si aún no existen y devuelve el propio controlador. Todas las funciones siguientes lo invocan internamente.

### `HelenNotifications.show(entry, options)`
Muestra el popup con la información del disparo actual. Parámetros principales:
- `entry`: objeto opcional con claves `title`, `label`, `detail`, `meta`, `tone` (`"timer"`/`"alarm"`). Si falta alguna clave se usan textos predeterminados.
- `options.pendingCount`: número entero ≥0 que alimenta la pastilla `+N` cuando hay eventos en cola.
- `options.onPrimary(source)`: callback que se ejecuta cuando el usuario pulsa **Detener** o cierra con `Escape`. El argumento `source` identifica el origen (`"primary"`, `"escape"`).
- `options.disablePrimary`: si es `true`, bloquea temporalmente el botón.
- `options.statusMessage`: texto opcional para la línea auxiliar (por ejemplo, “Activa el sonido tocando la pantalla.”).

### `HelenNotifications.hide(reason)`
Oculta el popup y limpia clases auxiliares en `<body>`. Si se pasa una cadena en `reason`, queda almacenada en `data-close-reason` para diagnóstico.

### `HelenNotifications.update(entry)`
Permite refrescar título/etiquetas del popup sin reabrirlo.

### `HelenNotifications.setPending(count)`
Actualiza manualmente el contador `+N` de la cola.

### `HelenNotifications.setStatus(message, visible?)`
Controla la franja de estado bajo la descripción (p. ej. mensajes de desbloqueo de audio). Cuando `message` es vacío oculta el elemento. El parámetro `visible` es `true` por defecto.

### `HelenNotifications.setPrimaryDisabled(disabled)`
Habilita o deshabilita el botón **Detener**.

## Eventos globales disponibles
`alarm-core.js` expone un bus único `window.HelenTimekeeperBus` (versión 1) y emite eventos tanto por el bus como por `CustomEvent` en `window` para mantener compatibilidad:

- `helen:timekeeper:fired`: evento genérico para cualquier disparo.
- `timer:finished`: específico de temporizadores.
- `alarm:triggered`: específico de alarmas.

Cada evento entrega el mismo *payload*:

```json
{
  "id": "timer-123",
  "eventId": "timer-123:1717000000000",
  "type": "timer",
  "tone": "timer",
  "title": "Temporizador finalizado",
  "label": "Temporizador",
  "detail": "Duración: 25:00",
  "meta": "Finalizó a las 10:15"
}
```

Puedes suscribirte usando `window.HelenTimekeeperBus.on(...)` o `window.addEventListener('timer:finished', handler)`.

## Flujo con `HelenScheduler`
- Cuando un temporizador o alarma finaliza, `HelenScheduler` coloca el snapshot en una cola y reproduce audio.
- Sólo se muestra un popup a la vez. Si llegan varios disparos, el badge `+N` refleja los pendientes y se atienden en orden.
- Al pulsar **Detener** se detiene el audio activo, se limpia el evento atendido (cancelando el temporizador en caso de serlo) y se avanza al siguiente de la cola.
- También se soporta cierre con `Escape`, preservando la accesibilidad del modal.

## Accesibilidad y foco
- El contenedor usa `role="alertdialog"`, `aria-modal="true"` y `aria-live="assertive"`.
- El botón **Detener** recibe el foco inicial y se mantiene el foco atrapado mediante la tecla `Tab`.
- Escape invoca la misma acción primaria.
- El sonido respeta las políticas de *autoplay*: si el navegador bloquea la reproducción, aparece un mensaje pidiendo interacción.

## Estilos
Los estilos viven en `ui/notifications/notifications.css` y se importan desde `globals.css`. El popup utiliza tokens compartidos (`--blue-a`, `--r-lg`, `--t-med`, etc.) para integrarse con HELEN. Si necesitas variaciones visuales, añade selectores derivados como `.helen-global-modal[data-tone="alarm"]` sin modificar clases existentes.
