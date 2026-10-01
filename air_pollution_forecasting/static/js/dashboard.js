/* ─── AirSense Dashboard JS ─── */

const CHART_DEFAULTS = {
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 600, easing: 'easeInOutQuart' },
  plugins: {
    legend: { display: false },
    tooltip: {
      backgroundColor: '#0d1219',
      borderColor: '#1e2d3d',
      borderWidth: 1,
      titleColor: '#00e5ff',
      bodyColor: '#c8d8e8',
      padding: 10,
      titleFont: { family: 'Space Mono', size: 11 },
      bodyFont:  { family: 'Space Mono', size: 11 },
    }
  },
  scales: {
    x: {
      ticks: { color: '#5a7a90', font: { family: 'Space Mono', size: 10 }, maxTicksLimit: 10 },
      grid:  { color: '#1e2d3d' },
    },
    y: {
      ticks: { color: '#5a7a90', font: { family: 'Space Mono', size: 10 } },
      grid:  { color: '#1e2d3d' },
    }
  }
};

const CAT_COLORS = {
  'Good': '#00e676', 'Satisfactory': '#aeea00',
  'Moderate': '#ffd740', 'Poor': '#ff6d00',
  'Very Poor': '#f44336', 'Severe': '#880e4f'
};

let histChart = null;
let forecastChart = null;

// ─── Clock ────────────────────────────────────────────
function updateClock() {
  const el = document.getElementById('current-time');
  if (el) el.textContent = new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}
setInterval(updateClock, 1000);
updateClock();

// ─── Fetch current readings ───────────────────────────
async function loadCurrentReadings() {
  try {
    const res = await fetch('/api/current');
    const data = await res.json();
    updateCards(data);
    updateWeather(data.weather);
  } catch (e) {
    console.error('Failed to load current readings', e);
  }
}

function aqiFromValue(pollutant, value) {
  const bp = {
    'PM2.5': [[0,30,0,50],[30,60,51,100],[60,90,101,200],[90,120,201,300],[120,250,301,400],[250,500,401,500]],
    'PM10':  [[0,50,0,50],[50,100,51,100],[100,250,101,200],[250,350,201,300],[350,430,301,400],[430,600,401,500]],
    'NO2':   [[0,40,0,50],[40,80,51,100],[80,180,101,200],[180,280,201,300],[280,400,301,400],[400,800,401,500]],
    'CO':    [[0,1,0,50],[1,2,51,100],[2,10,101,200],[10,17,201,300],[17,34,301,400],[34,50,401,500]],
  };
  for (const [clo,chi,ilo,ihi] of (bp[pollutant]||[])) {
    if (value >= clo && value <= chi)
      return Math.round(ilo + (ihi-ilo)*(value-clo)/(chi-clo));
  }
  return value > 0 ? 500 : 0;
}

function aqiCategory(aqi) {
  if (aqi <= 50)  return 'Good';
  if (aqi <= 100) return 'Satisfactory';
  if (aqi <= 200) return 'Moderate';
  if (aqi <= 300) return 'Poor';
  if (aqi <= 400) return 'Very Poor';
  return 'Severe';
}

function updateCards(data) {
  const pollutants = data.pollutants || {};
  document.querySelectorAll('.aqi-card').forEach(card => {
    const key = card.dataset.pollutant;
    const info = pollutants[key];
    if (!info) return;

    const aqi = aqiFromValue(key, info.value);
    const cat = aqiCategory(aqi);
    const color = CAT_COLORS[cat] || '#c8d8e8';
    const pct = Math.min(100, (aqi / 500) * 100);

    card.querySelector('.card-value').textContent = info.value;
    card.querySelector('.card-unit').textContent  = info.unit;
    card.querySelector('.card-category').textContent = cat;
    card.querySelector('.card-category').className = `card-category cat-${cat.replace(' ','')}`;
    card.querySelector('.card-bar-fill').style.width = pct + '%';
    card.querySelector('.card-bar-fill').style.background = color;
    card.querySelector('.card-value').style.color = color;
    card.style.borderColor = color + '44';
    card.classList.remove('loading');
  });
}

function updateWeather(weather) {
  if (!weather) return;
  document.getElementById('w-temp').textContent = weather.temperature;
  document.getElementById('w-humi').textContent = weather.humidity;
  document.getElementById('w-wind').textContent = weather.wind_speed;
}

// ─── Historical Chart ─────────────────────────────────
async function loadHistorical(pollutant) {
  try {
    const res  = await fetch(`/api/historical?pollutant=${encodeURIComponent(pollutant)}`);
    const data = await res.json();
    renderHistChart(data);
    updateStats(data.stats, data.unit);
  } catch (e) {
    console.error('Failed to load historical data', e);
  }
}

function renderHistChart(data) {
  const labels = data.timestamps.filter((_, i) => i % 6 === 0 || i === data.timestamps.length - 1)
    .map(t => t.slice(11, 16));
  const fullLabels = data.timestamps.map(t => t.slice(5, 16));

  const ctx = document.getElementById('histChart').getContext('2d');
  if (histChart) histChart.destroy();

  const gradient = ctx.createLinearGradient(0, 0, 0, 240);
  gradient.addColorStop(0, 'rgba(0,229,255,0.25)');
  gradient.addColorStop(1, 'rgba(0,229,255,0.01)');

  histChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: fullLabels,
      datasets: [{
        data: data.values,
        borderColor: '#00e5ff',
        borderWidth: 1.5,
        backgroundColor: gradient,
        fill: true,
        tension: 0.35,
        pointRadius: 0,
        pointHoverRadius: 4,
        pointHoverBackgroundColor: '#00e5ff',
      }]
    },
    options: {
      ...CHART_DEFAULTS,
      scales: {
        x: {
          ...CHART_DEFAULTS.scales.x,
          ticks: {
            ...CHART_DEFAULTS.scales.x.ticks,
            callback: function(val, i) {
              return i % 12 === 0 ? this.getLabelForValue(val) : '';
            }
          }
        },
        y: {
          ...CHART_DEFAULTS.scales.y,
          title: {
            display: true,
            text: data.unit,
            color: '#5a7a90',
            font: { family: 'Space Mono', size: 10 }
          }
        }
      }
    }
  });
}

function updateStats(stats, unit) {
  document.getElementById('s-mean').textContent = `${stats.mean} ${unit}`;
  document.getElementById('s-max').textContent  = `${stats.max} ${unit}`;
  document.getElementById('s-min').textContent  = `${stats.min} ${unit}`;
  document.getElementById('s-std').textContent  = `±${stats.std}`;
}

// ─── Tabs ─────────────────────────────────────────────
document.querySelectorAll('#hist-tabs .tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#hist-tabs .tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    loadHistorical(btn.dataset.p);
  });
});

// ─── Forecast ─────────────────────────────────────────
document.getElementById('run-forecast').addEventListener('click', runForecast);

async function runForecast() {
  const pollutant = document.getElementById('fc-pollutant').value;
  const steps     = parseInt(document.getElementById('fc-steps').value);
  const btn       = document.getElementById('run-forecast');

  btn.disabled = true;
  btn.querySelector('.btn-text').style.display = 'none';
  btn.querySelector('.btn-loader').style.display = 'inline';

  try {
    const res  = await fetch('/api/forecast', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pollutant, steps })
    });
    const data = await res.json();
    renderForecastChart(data);
    updateForecastSummary(data);
  } catch (e) {
    console.error('Forecast failed', e);
  } finally {
    btn.disabled = false;
    btn.querySelector('.btn-text').style.display = 'inline';
    btn.querySelector('.btn-loader').style.display = 'none';
  }
}

function renderForecastChart(data) {
  const predictions = data.predictions;
  const labels      = predictions.map(p => p.timestamp.slice(11, 16));
  const values      = predictions.map(p => p.value);
  const unit        = predictions[0]?.unit || '';
  const colors      = predictions.map(p => CAT_COLORS[p.category] || '#00e5ff');

  const ctx = document.getElementById('forecastChart').getContext('2d');
  if (forecastChart) forecastChart.destroy();

  const gradient = ctx.createLinearGradient(0, 0, 0, 300);
  gradient.addColorStop(0, 'rgba(255,107,53,0.3)');
  gradient.addColorStop(1, 'rgba(255,107,53,0.01)');

  forecastChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        data: values,
        borderColor: '#ff6b35',
        borderWidth: 2,
        backgroundColor: gradient,
        fill: true,
        tension: 0.4,
        pointRadius: 3,
        pointBackgroundColor: colors,
        pointBorderColor: colors,
        pointHoverRadius: 6,
      }]
    },
    options: {
      ...CHART_DEFAULTS,
      plugins: {
        ...CHART_DEFAULTS.plugins,
        tooltip: {
          ...CHART_DEFAULTS.plugins.tooltip,
          titleColor: '#ff6b35',
          callbacks: {
            label: (ctx) => {
              const p = predictions[ctx.dataIndex];
              return ` ${ctx.parsed.y} ${unit}  |  AQI ${p.aqi}  |  ${p.category}`;
            }
          }
        }
      },
      scales: {
        x: { ...CHART_DEFAULTS.scales.x },
        y: {
          ...CHART_DEFAULTS.scales.y,
          title: {
            display: true, text: unit,
            color: '#5a7a90', font: { family: 'Space Mono', size: 10 }
          }
        }
      }
    }
  });
}

function updateForecastSummary(data) {
  const predictions = data.predictions;
  const values = predictions.map(p => p.value);
  const aqis   = predictions.map(p => p.aqi);
  const peak   = Math.max(...values);
  const avg    = (values.reduce((a,b) => a+b, 0) / values.length).toFixed(1);
  const worst  = aqiCategory(Math.max(...aqis));
  const unit   = predictions[0]?.unit || '';

  document.getElementById('fc-peak').textContent  = `${peak} ${unit}`;
  document.getElementById('fc-avg').textContent   = `${avg} ${unit}`;
  document.getElementById('fc-worst').textContent = worst;
  document.getElementById('fc-worst').className   = `summary-val cat-${worst.replace(' ','')}`;
  document.getElementById('fc-summary').style.display = 'flex';
}

// ─── Init ─────────────────────────────────────────────
(async function init() {
  await loadCurrentReadings();
  await loadHistorical('PM2.5');
  await runForecast();

  // Refresh current readings every 60s
  setInterval(loadCurrentReadings, 60_000);
})();
