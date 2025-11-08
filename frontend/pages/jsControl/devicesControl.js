const client = mqtt.connect("wss://teamrocket.serveminecraft.net:8883/mqtt");

const triggerActivationRing = (options = {}) => {
  if (typeof window.triggerActivationAnimation === 'function') {
    window.triggerActivationAnimation(options);
  }
};

const resetRingToIdle = () => {
  if (typeof window.setActivationRingState === 'function') {
    window.setActivationRingState('idle');
  }
};

client.on("connect", () => {
  console.log("Conectado al broker MQTT");
  client.publish("general/light", "true", (err) => {
    if (!err) {
      console.log('Mensaje "true" enviado al topic general/light');
    } else {
      console.error("Error al enviar el mensaje:", err);
    }
  });
});

client.on("error", (err) => {
  console.error("Error de conexión:", err);
});

socket.on("message", (data) => {
  console.debug("Mensaje recibido del servidor:", data);

  if (data.character === "Start") {
    isActive = true;
    triggerActivationRing({ persist: true });
    resetDeactivationTimer();
    return;
  }

  if (!isActive) {
    resetRingToIdle();
    return;
  }
  resetDeactivationTimer();
  switch (data.character) {
    case "0":
      client.publish("general/light", "false", (err) => {
        if (!err) {
          console.log('Mensaje "false" enviado al topic general/light');
        } else {
          console.error("Error al enviar el mensaje:", err);
        }
      });
      break;

    case "1":
      client.publish("general/light", "1", (err) => {
        if (!err) {
          console.log('Mensaje "1" enviado al topic general/light');
        } else {
          console.error("Error al enviar el mensaje:", err);
        }
      });
      break;

    case "2":
      client.publish("general/light", "2", (err) => {
        if (!err) {
          console.log('Mensaje "2" enviado al topic general/light');
        } else {
          console.error("Error al enviar el mensaje:", err);
        }
      });
      break;

    case "3":
      client.publish("general/light", "3", (err) => {
        if (!err) {
          console.log('Mensaje "3" enviado al topic general/light');
        } else {
          console.error("Error al enviar el mensaje:", err);
        }
      });
      break;

    case "Ajustes":
      console.log("Navegando a los ajustes...");
      goToAlarm();
      break;

    case "Clima":
      console.log("Navegando al clima...");
      goToWeather();
      break;

    case "Inicio":
      console.log("Navegando al inicio...");
      goToHome();
      break;

    case "Reloj":
      console.log("Navegando al reloj...");
      goToClock();
      break;

    case "Dispositivos":
      console.log("Navegando a los dispositivos...");
      goToDevices();
      break;
  }
});
