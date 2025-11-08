/* ===========================
   HELEN – WEATHER (JS)
   =========================== */

// ---------- Config ----------
const CONFIG = {
  API_KEY: '6266f75957014a7de4ae0ded34d1e7cc',
  STORAGE_KEY: 'selectedCity',
  DEFAULT_CITY: 'tijuana',
  UPDATE_INTERVAL: 60_000,
  WEATHER_UPDATE_INTERVAL: 300_000, // 5 min
  WEATHER_UNITS: 'metric',
  WEATHER_LANG: 'es',
};

// ---------- Municipios BC ----------
const BC_MUNICIPIOS = {
  mexicali: { lat: 32.6633, lon: -115.4678, name: 'Mexicali' },
  tijuana:  { lat: 32.5027, lon: -117.00371, name: 'Tijuana' },
  ensenada: { lat: 31.8667, lon: -116.5967, name: 'Ensenada' },
  tecate:   { lat: 32.5667, lon: -116.6333, name: 'Tecate' },
  rosarito: { lat: 32.3614, lon: -117.0553, name: 'Playas de Rosarito' },
  chiapas:  { lat: 16.7519, lon: -93.1167, name: 'Tuxtla Gutiérrez' },
};

// ---------- Iconos ----------
const WEATHER_ICONS = {
  '01d':'bi-brightness-high','01n':'bi-moon',
  '02d':'bi-cloud-sun','02n':'bi-cloud-moon',
  '03d':'bi-cloud','03n':'bi-cloud',
  '04d':'bi-clouds','04n':'bi-clouds',
  '09d':'bi-cloud-drizzle','09n':'bi-cloud-drizzle',
  '10d':'bi-cloud-rain','10n':'bi-cloud-rain',
  '11d':'bi-cloud-lightning','11n':'bi-cloud-lightning',
  '13d':'bi-cloud-snow','13n':'bi-cloud-snow',
  '50d':'bi-cloud-haze','50n':'bi-cloud-haze'
};

// ---------- Utilidades ----------
const DIAS   = ['Domingo','Lunes','Martes','Miércoles','Jueves','Viernes','Sábado'];
const DIAS_C = ['Dom','Lun','Mar','Mié','Jue','Vie','Sáb'];
const MESES  = ['Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre'];

class WeatherApp {
  constructor(){
    this.elements = {
      loading: $('.weather-loading'),
      citySelectWrap: $('.change-city'),
      hourlyContainer: $('.hourly-forecast'),
      dailyRow: $('.daily-forecast .forecast-row'),
      // Panel actual:
      icon: $('.weather-icon'),
      temp: $('.temperature'),
      status: $('.status'),
      wind: $('.wind-speed'),
      hum: $('.humidity'),
      vis: $('.visibility'),
      press: $('.pressure'),
      feelSlot: $('.feel-temp'),
      maxSlot: $('.max-temp'),
      minSlot: $('.min-temp')
    };

    this.currentLocation = null;
    this.autoTimer = null;

    this.init();
  }

  // --- helpers de tiempo ---
  static toLocalDate(dt, tzSeconds){
    const d = new Date((dt + tzSeconds) * 1000);
    // construimos en hora local real (no UTC) para poder comparar por día local
    return new Date(
      d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate(),
      d.getUTCHours(), d.getUTCMinutes(), 0, 0
    );
  }

  static hhmm(date){
    return date.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
  }

  // --- boot ---
  init(){
    this.injectCitySelector();
    this.loadSavedCity();
    this.bindEvents();

    this.refreshAll();

    // autorefresco
    this.autoTimer = setInterval(()=> this.refreshAll(), CONFIG.WEATHER_UPDATE_INTERVAL);
  }

  debug(msg){ console.log('[WeatherApp] ' + msg); }

  injectCitySelector(){
    this.elements.citySelectWrap.html(`
      <div class="custom-select-container">
        <div class="select-icon"><i class="bi bi-geo-alt-fill"></i></div>
        <span class="select-label">Cambiar Ciudad</span>
        <select id="citySelector" class="custom-select" aria-label="Cambiar ciudad">
          <option value="" selected>Selecciona un municipio</option>
          <option value="tijuana">Tijuana</option>
          <option value="mexicali">Mexicali</option>
          <option value="ensenada">Ensenada</option>
          <option value="tecate">Tecate</option>
          <option value="rosarito">Playas de Rosarito</option>
          <option value="chiapas">Chiapas</option>
        </select>
      </div>
    `);
    this.citySelect = $('#citySelector');
  }

  loadSavedCity(){
    const saved = localStorage.getItem(CONFIG.STORAGE_KEY) || CONFIG.DEFAULT_CITY;
    this.currentLocation = BC_MUNICIPIOS[saved] || BC_MUNICIPIOS[CONFIG.DEFAULT_CITY];
    this.citySelect.val(saved);
  }

  bindEvents(){
    this.citySelect.on('change', e=>{
      const key = $(e.target).val();
      if(!key || !BC_MUNICIPIOS[key]) return;
      this.currentLocation = BC_MUNICIPIOS[key];
      localStorage.setItem(CONFIG.STORAGE_KEY, key);
      this.refreshAll();
    });
  }

  async refreshAll(){
    try{
      this.elements.loading.fadeIn(120);
      const {lat, lon} = this.currentLocation;
      await this.getWeather(lat, lon);           // actual + header
      await this.getTodayHourlyForecast(lat, lon); // próximas horas (h/h)
    }catch(e){
      console.error(e);
      Swal.fire({icon:'error', title:'Error', text:'No se pudo obtener el clima. Intenta más tarde.'});
    }finally{
      this.elements.loading.fadeOut(160);
    }
  }

  // --- Header (título, ciudad, fecha) ---
  renderHeader(city, country){
    const now = new Date();
    const dateStr = `${DIAS[now.getDay()]}, ${now.getDate()} de ${MESES[now.getMonth()]} de ${now.getFullYear()}`;
    $('.title-col').html(`
      <h1>Pronóstico del Tiempo</h1>
      <div class="location-name"><i class="bi bi-geo-alt-fill"></i> ${city}, ${country}</div>
      <div class="date-section">${dateStr}</div>
    `);
  }

  // --- API: actual + daily (5d/3h) ---
  async getWeather(lat, lon){
    // actual
    const urlNow = `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${CONFIG.API_KEY}&units=${CONFIG.WEATHER_UNITS}&lang=${CONFIG.WEATHER_LANG}`;
    const res = await axios.get(urlNow);
    const d = res.data;

    // ciudad
    let city = d.name, country = d.sys?.country || '';
    if(!city){
      const rev = await this.getReverseGeocoding(lat, lon);
      city = rev.name; country = rev.country;
    }

    // pintar panel actual
    const nowData = {
      icon: d.weather?.[0]?.icon || '',
      description: d.weather?.[0]?.description || '—',
      temperature: Math.round(d.main?.temp ?? 0),
      feelsLike: Math.round(d.main?.feels_like ?? 0),
      humidity: d.main?.humidity ?? null,
      windSpeed: d.wind?.speed != null ? Math.round(d.wind.speed * 3.6) : null, // m/s -> km/h
      visibility: d.visibility != null ? Math.round(d.visibility/1000) : null,   // m -> km
      pressure: d.main?.pressure ?? null,
      tempMax: d.main?.temp_max != null ? Math.round(d.main.temp_max) : null,
      tempMin: d.main?.temp_min != null ? Math.round(d.main.temp_min) : null,
      city, country
    };
    this.updateWeatherUI(nowData);

    // daily (derivado de 5d/3h)
    await this.getDailyFrom3h(lat, lon);
  }

  async getReverseGeocoding(lat, lon){
    try{
      const url = `https://api.openweathermap.org/geo/1.0/reverse?lat=${lat}&lon=${lon}&limit=1&appid=${CONFIG.API_KEY}`;
      const {data} = await axios.get(url);
      if(data && data.length){
        return {name: data[0].name, country: data[0].country};
      }
    }catch{}
    return {name:'Desconocido', country:'—'};
  }

  updateWeatherUI(data){
    // icono
    this.elements.icon.removeClass().addClass('weather-icon bi ' + (WEATHER_ICONS[data.icon] || 'bi-question-circle'));
    // textos
    this.elements.status.text(data.description);
    this.elements.temp.text(`${data.temperature}°`);
    this.elements.wind.text(data.windSpeed != null ? `${data.windSpeed} km/h` : '— km/h');
    this.elements.hum.text(data.humidity != null ? `${data.humidity}%` : '—%');
    this.elements.vis.text(data.visibility != null ? `${data.visibility} km` : '— km');
    this.elements.press.text(data.pressure != null ? `${data.pressure} hPa` : '— hPa');
    // slots
    this.elements.feelSlot.text(data.feelsLike != null ? `${data.feelsLike}°` : '—°');
    this.elements.maxSlot.text(data.tempMax != null ? `${data.tempMax}°` : '—°');
    this.elements.minSlot.text(data.tempMin != null ? `${data.tempMin}°` : '—°');

    // header
    this.renderHeader(data.city, data.country);
  }

  // --- Daily usando feed 5d/3h (1 muestra por día, ~mediodía) ---
  async getDailyFrom3h(lat, lon){
    const url = `https://api.openweathermap.org/data/2.5/forecast?lat=${lat}&lon=${lon}&appid=${CONFIG.API_KEY}&units=${CONFIG.WEATHER_UNITS}&lang=${CONFIG.WEATHER_LANG}`;
    const {data} = await axios.get(url);

    const tz = data.city?.timezone ?? 0;
    const items = data.list || [];
    const byDay = new Map();

    for(const it of items){
      const ld = WeatherApp.toLocalDate(it.dt, tz);
      const key = ld.getFullYear()+'-'+(ld.getMonth()+1)+'-'+ld.getDate();

      const pack = {
        date: ld,
        hour: ld.getHours(),
        temp: Math.round(it.main.temp),
        icon: it.weather?.[0]?.icon || ''
      };

      if(!byDay.has(key)) byDay.set(key, []);
      byDay.get(key).push(pack);
    }

    const today = WeatherApp.toLocalDate(Math.floor(Date.now()/1000), tz);
    const todayKey = today.getFullYear()+'-'+(today.getMonth()+1)+'-'+today.getDate();

    const days = [];
    for(const [key, arr] of byDay){
      if(key === todayKey) continue; // empezamos desde mañana
      // elegir cercano a 15:00
      arr.sort((a,b)=> Math.abs(a.hour-15) - Math.abs(b.hour-15));
      days.push(arr[0]);
    }

    days.sort((a,b)=> a.date - b.date);
    const next7 = days.slice(0,7);

    // pintar
    const row = this.elements.dailyRow;
    row.empty();
    next7.forEach(d=>{
      row.append(`
        <div class="forecast-item">
          <i class="weather-icon bi ${WEATHER_ICONS[d.icon] || 'bi-question-circle'}"></i>
          <p class="day">${DIAS[d.date.getDay()]}</p>
          <p class="temperature">${d.temp}°C</p>
        </div>
      `);
    });
  }

  // --- Próximas horas (One Call 3.0 -> fallback 3h) ---
  async getTodayHourlyForecast(lat, lon){
    // 1) intentar One Call 3.0
    try{
      const url = `https://api.openweathermap.org/data/3.0/onecall?lat=${lat}&lon=${lon}&appid=${CONFIG.API_KEY}&units=${CONFIG.WEATHER_UNITS}&lang=${CONFIG.WEATHER_LANG}&exclude=minutely,daily,alerts`;
      const {data} = await axios.get(url);

      const tz = data.timezone_offset ?? 0;
      const now = WeatherApp.toLocalDate(Math.floor(Date.now()/1000), tz);
      const end = new Date(now); end.setHours(23,59,59,999);

      const hours = (data.hourly || [])
        .map(h=>({
          date: WeatherApp.toLocalDate(h.dt, tz),
          temp: Math.round(h.temp),
          icon: h.weather?.[0]?.icon || ''
        }))
        .filter(h=> h.date >= now && h.date <= end);

      if(hours.length){
        this.updateHourlyForecastDisplay(hours);
        return;
      }
      throw new Error('One Call sin horas restantes de hoy');
    }catch(e){
      // 2) fallback feed 3h
      try{
        const url = `https://api.openweathermap.org/data/2.5/forecast?lat=${lat}&lon=${lon}&appid=${CONFIG.API_KEY}&units=${CONFIG.WEATHER_UNITS}&lang=${CONFIG.WEATHER_LANG}`;
        const {data} = await axios.get(url);

        const tz = data.city?.timezone ?? 0;
        const now = WeatherApp.toLocalDate(Math.floor(Date.now()/1000), tz);
        const y=now.getFullYear(), m=now.getMonth(), d=now.getDate();

        const hours = (data.list || [])
          .map(it=>{
            const ld = WeatherApp.toLocalDate(it.dt, tz);
            return {
              date: ld,
              sameDay: (ld.getFullYear()===y && ld.getMonth()===m && ld.getDate()===d),
              temp: Math.round(it.main.temp),
              icon: it.weather?.[0]?.icon || ''
            };
          })
          .filter(x=> x.sameDay && x.date > now);

        this.updateHourlyForecastDisplay(hours);
      }catch(err2){
        console.error(err2);
        this.updateHourlyForecastDisplay([]);
      }
    }
  }

  updateHourlyForecastDisplay(hourItems){
    const wrap = this.elements.hourlyContainer;
    wrap.empty();

    if(!hourItems.length){
      wrap.append('<p>No hay datos disponibles para las próximas horas</p>');
      return;
    }

    hourItems.forEach(h=>{
      const ic = WEATHER_ICONS[h.icon] || 'bi-brightness-high';
      wrap.append(`
        <div class="forecast-item">
          <p class="day">${WeatherApp.hhmm(h.date)}</p>
          <i class="weather-icon bi ${ic}"></i>
          <p class="temperature">${h.temp}°C</p>
        </div>
      `);
    });
  }
}

// ---------- Boot ----------
$(function(){
  new WeatherApp();
});
