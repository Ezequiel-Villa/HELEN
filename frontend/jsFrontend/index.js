$(function () {
  const clock = $('.clock');
  const dateSection = $('.date-section');
  const locationSection = $('.location');
  const weatherIcon = $('.weather-icon');
  const temperature = $('.temperature');
  const weatherDescription = $('.status');
  const activeSensor = $('.active-sensor');

  const feelsLike = $('#feels-like-value');
  const humidityValue = $('#humidity-value');
  const windValue = $('#wind-value');
  const pressureValue = $('#pressure-value');

  const DAYS = ['Domingo', 'Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado'];
  const MONTHS = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'];

  const WEATHER_REFRESH_SUCCESS_MS = 10 * 60 * 1000;
  const WEATHER_REFRESH_ERROR_MS = 90 * 1000;
  const WEATHER_ICON_CLASSES = {
    '01d': 'bi-brightness-high', '01n': 'bi-moon',
    '02d': 'bi-cloud-sun', '02n': 'bi-cloud-moon',
    '03d': 'bi-cloud', '03n': 'bi-cloud',
    '04d': 'bi-clouds', '04n': 'bi-clouds',
    '09d': 'bi-cloud-drizzle', '09n': 'bi-cloud-drizzle',
    '10d': 'bi-cloud-rain', '10n': 'bi-cloud-rain',
    '11d': 'bi-cloud-lightning', '11n': 'bi-cloud-lightning',
    '13d': 'bi-cloud-snow', '13n': 'bi-cloud-snow',
    '50d': 'bi-cloud-haze', '50n': 'bi-cloud-haze'
  };

  const WEATHER_ENDPOINT = (() => {
    const key = '6266f75957014a7de4ae0ded34d1e7cc';
    const lat = 32.43347;
    const lon = -116.67447;
    return `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${key}&units=metric&lang=es`;
  })();

  let clockTimerId = null;
  let weatherTimerId = null;
  let lastWeatherSnapshot = null;

  const setTextIfChanged = (element, value) => {
    if (!element || !element.length) {
      return;
    }
    const node = element[0];
    const next = String(value);
    if (node.textContent !== next) {
      element.text(next);
    }
  };

  const scheduleClockTick = () => {
    const now = new Date();
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const day = now.getDate();
    const month = now.getMonth();
    const year = now.getFullYear();
    const weekDay = now.getDay();

    setTextIfChanged(clock, `${hours}:${minutes}`);
    setTextIfChanged(dateSection, `${DAYS[weekDay]}, ${day} de ${MONTHS[month]} de ${year}`);

    const millisUntilNextMinute = 60000 - (now.getSeconds() * 1000 + now.getMilliseconds());
    const delay = Math.max(500, millisUntilNextMinute);
    clockTimerId = window.setTimeout(scheduleClockTick, delay);
  };

  const startClock = () => {
    if (clockTimerId) {
      window.clearTimeout(clockTimerId);
    }
    scheduleClockTick();
  };

  const updateLocation = () => {
    setTextIfChanged(locationSection, 'Tijuana, MX');
  };

  const applyWeatherSnapshot = (snapshot) => {
    if (!snapshot) {
      return;
    }

    if (!lastWeatherSnapshot || lastWeatherSnapshot.icon !== snapshot.icon) {
      const iconClass = WEATHER_ICON_CLASSES[snapshot.icon] || '';
      weatherIcon.attr('class', iconClass ? `weather-icon bi ${iconClass}` : 'weather-icon');
    }

    if (!lastWeatherSnapshot || lastWeatherSnapshot.description !== snapshot.description) {
      const capitalized = snapshot.description
        ? `${snapshot.description.charAt(0).toUpperCase()}${snapshot.description.slice(1)}`
        : 'No disponible';
      setTextIfChanged(weatherDescription, capitalized);
    }

    if (!lastWeatherSnapshot || lastWeatherSnapshot.temperature !== snapshot.temperature) {
      setTextIfChanged(temperature, `${snapshot.temperature}°C`);
    }

    if (feelsLike.length && (!lastWeatherSnapshot || lastWeatherSnapshot.feelsLike !== snapshot.feelsLike)) {
      setTextIfChanged(feelsLike, `${snapshot.feelsLike}°`);
    }

    if (humidityValue.length && (!lastWeatherSnapshot || lastWeatherSnapshot.humidity !== snapshot.humidity)) {
      setTextIfChanged(humidityValue, `${snapshot.humidity}%`);
    }

    if (windValue.length && (!lastWeatherSnapshot || lastWeatherSnapshot.wind !== snapshot.wind)) {
      setTextIfChanged(windValue, `${snapshot.wind} km/h`);
    }

    if (pressureValue.length && (!lastWeatherSnapshot || lastWeatherSnapshot.pressure !== snapshot.pressure)) {
      setTextIfChanged(pressureValue, `${snapshot.pressure} hPa`);
    }

    lastWeatherSnapshot = snapshot;
  };

  const fetchWeather = async () => {
    if (weatherTimerId) {
      window.clearTimeout(weatherTimerId);
      weatherTimerId = null;
    }

    let nextRefresh = WEATHER_REFRESH_SUCCESS_MS;

    try {
      const response = await axios.get(WEATHER_ENDPOINT, { timeout: 8000 });
      const data = response.data || {};

      const snapshot = {
        icon: data.weather && data.weather[0] ? data.weather[0].icon || '' : '',
        description: data.weather && data.weather[0] ? String(data.weather[0].description || '').trim() : '',
        temperature: Math.round(data.main?.temp ?? 0),
        feelsLike: Math.round(data.main?.feels_like ?? data.main?.temp ?? 0),
        humidity: Math.round(data.main?.humidity ?? 0),
        wind: Math.max(0, Math.round((data.wind?.speed ?? 0) * 3.6)),
        pressure: Math.round(data.main?.pressure ?? 0)
      };

      applyWeatherSnapshot(snapshot);
    } catch (error) {
      console.error('[Helen] No se pudo actualizar el clima:', error);
      nextRefresh = WEATHER_REFRESH_ERROR_MS;
      if (!lastWeatherSnapshot) {
        setTextIfChanged(weatherDescription, 'No disponible');
        setTextIfChanged(temperature, '--°C');
      }
    } finally {
      weatherTimerId = window.setTimeout(fetchWeather, nextRefresh);
    }
  };

  const cleanup = () => {
    if (clockTimerId) {
      window.clearTimeout(clockTimerId);
      clockTimerId = null;
    }
    if (weatherTimerId) {
      window.clearTimeout(weatherTimerId);
      weatherTimerId = null;
    }
  };

  if (activeSensor.length) {
    activeSensor.on('click', () => {
      console.info('Sensor activado');
    });
  }

  $(window).on('beforeunload pagehide', cleanup);

  startClock();
  updateLocation();
  fetchWeather();
});
