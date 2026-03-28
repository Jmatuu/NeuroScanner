const API_BASE = 'http://localhost:8000';
if (document.getElementById('apiBaseDisplay')) {
  document.getElementById('apiBaseDisplay').textContent = API_BASE;
}

// ── State ──────────────────────────────────────────────────────────────────
let currentSessionId = null;
let blinkCount       = 0;
let recordCount      = 0;
let pollInterval     = null;
let gazeHistory      = [];
const MAX_GAZE_HISTORY = 60;
// Estadísticas de sesión
let sesionInicio     = null;
let duraciones       = [];
let zonaConteoStats  = { OJOS: 0, NARIZ: 0, BOCA: 0, OTRO: 0 };
let timerSesion      = null;
let duracionesArray = [];

// ── Zona config ────────────────────────────────────────────────────────────
const ZONA_ICONS = {
  OJOS:   '👁️',
  NARIZ:  '👃',
  BOCA:   '👄',
  OTRO:   '❓'
};

const ZONA_COLORS = {
  OJOS:  '#4fffb0',
  NARIZ: '#00ffff',
  BOCA:  '#ff6b6b',
  OTRO:  '#4a4f6a'
};

let zonaConteo = { OJOS: 0, NARIZ: 0, BOCA: 0, OTRO: 0 };
// ── Modal ──────────────────────────────────────────────────────────────────
function abrirModal() {
  document.getElementById('modal').classList.add('open');
  document.getElementById('modalOverlay').classList.add('open');
  document.getElementById('patientId').focus();
}

function cerrarModal() {
  document.getElementById('modal').classList.remove('open');
  document.getElementById('modalOverlay').classList.remove('open');
}

// ── Theme ──────────────────────────────────────────────────────────────────
function toggleTheme() {
  const html  = document.documentElement;
  const icon  = document.getElementById('themeIcon');
  const atual = html.getAttribute('data-theme');

  if (atual === 'light') {
    html.setAttribute('data-theme', 'dark');
    icon.textContent = '☀️';
    localStorage.setItem('theme', 'dark');
  } else {
    html.setAttribute('data-theme', 'light');
    icon.textContent = '🌙';
    localStorage.setItem('theme', 'light');
  }
}

// Cargar tema guardado al iniciar
(function() {
  const saved = localStorage.getItem('theme') || 'dark';
  document.documentElement.setAttribute('data-theme', saved);
  const icon = document.getElementById('themeIcon');
  if (icon) icon.textContent = saved === 'light' ? '🌙' : '☀️';
})();

function buildZonaList() {
  const el = document.getElementById('zonaList');
  el.innerHTML = ['OJOS', 'NARIZ', 'BOCA'].map(z => `
    <div class="zona-row">
      <div class="zona-label-row">
        <span>${z}</span>
        <span id="zpct-${z}">0%</span>
      </div>
      <div class="bar-bg"><div class="bar-fill" id="zbar-${z}" style="width:0%"></div></div>
    </div>
  `).join('');
}
buildZonaList();

// Auto-calcular grupo cuando cambia la edad ← fuera de updateZonaBars
document.getElementById('patientAge').addEventListener('input', function() {
  const grupo  = calcularGrupo(this.value);
  const badge  = document.getElementById('grupoBadge');
  badge.textContent = grupo.nombre;
  badge.className   = 'grupo-badge ' + grupo.clase;
});

function updateZonaBars(zona) {
  if (zona in zonaConteo) zonaConteo[zona]++;
  const total = Object.values(zonaConteo).reduce((a, b) => a + b, 0);
  ['OJOS', 'NARIZ', 'BOCA'].forEach(z => {
    const pct = total > 0 ? Math.round((zonaConteo[z] / total) * 100) : 0;
    document.getElementById(`zbar-${z}`).style.width = pct + '%';
    document.getElementById(`zpct-${z}`).textContent = pct + '%';
    const bar = document.getElementById(`zbar-${z}`);
    bar.classList.toggle('active', z === zona);
    bar.style.background = z === zona ? ZONA_COLORS[z] : '';
  });
}

// ── Canvas setup ───────────────────────────────────────────────────────────
const canvas = document.getElementById('gazeCanvas');
const ctx    = canvas.getContext('2d');

function resizeCanvas() {
  canvas.width  = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

// ── Configuración de las 9 zonas ──────────────────────────────────────────
const ZONAS_9 = [
  { id: 'ARRIBA_IZQ', label: 'ARRIBA\nIZQ',  col: 0, row: 0 },
  { id: 'ARRIBA',     label: 'ARRIBA',        col: 1, row: 0 },
  { id: 'ARRIBA_DER', label: 'ARRIBA\nDER',   col: 2, row: 0 },
  { id: 'IZQUIERDA',  label: 'IZQ',           col: 0, row: 1 },
  { id: 'CENTRO',     label: 'CENTRO',        col: 1, row: 1 },
  { id: 'DERECHA',    label: 'DER',           col: 2, row: 1 },
  { id: 'ABAJO_IZQ',  label: 'ABAJO\nIZQ',   col: 0, row: 2 },
  { id: 'ABAJO',      label: 'ABAJO',         col: 1, row: 2 },
  { id: 'ABAJO_DER',  label: 'ABAJO\nDER',   col: 2, row: 2 },
];

const ZONA_COLORES_9 = {
  ARRIBA_IZQ: 'rgba(124,111,255,0.35)',
  ARRIBA:     'rgba(255,107,107,0.35)',
  ARRIBA_DER: 'rgba(124,111,255,0.35)',
  IZQUIERDA:  'rgba(0,255,255,0.35)',
  CENTRO:     'rgba(79,255,176,0.35)',
  DERECHA:    'rgba(255,255,0,0.35)',
  ABAJO_IZQ:  'rgba(124,111,255,0.35)',
  ABAJO:      'rgba(0,255,128,0.35)',
  ABAJO_DER:  'rgba(124,111,255,0.35)',
};

let ultimaDireccion = 'CENTRO';

function drawGaze(x, y, direccion) {
  const w  = canvas.width;
  const h  = canvas.height;
  const cw = w / 3;   // ancho de cada celda
  const ch = h / 3;   // alto de cada celda

  ctx.clearRect(0, 0, w, h);

  // ── Dibujar las 9 celdas ────────────────────────────────────────────────
  ZONAS_9.forEach(zona => {
    const x0 = zona.col * cw;
    const y0 = zona.row * ch;

    // Fondo — iluminar si es la zona activa
    if (zona.id === direccion) {
      ctx.fillStyle = ZONA_COLORES_9[zona.id] || 'rgba(79,255,176,0.2)';
    } else {
      ctx.fillStyle = 'rgba(10,11,15,0.8)';
    }
    ctx.fillRect(x0, y0, cw, ch);

    // Borde de celda
    ctx.strokeStyle = zona.id === direccion
      ? 'rgba(79,255,176,0.6)'
      : 'rgba(30,33,51,0.8)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x0 + 0.5, y0 + 0.5, cw - 1, ch - 1);

    // Texto de la zona
    ctx.fillStyle = zona.id === direccion
      ? '#ffffff'
      : 'rgba(74,79,106,0.8)';
    ctx.font = '10px Space Mono, monospace';
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'middle';

    // Soporte para texto con salto de línea
    const lineas = zona.label.split('\n');
    if (lineas.length === 1) {
      ctx.fillText(zona.label, x0 + cw / 2, y0 + ch / 2);
    } else {
      ctx.fillText(lineas[0], x0 + cw / 2, y0 + ch / 2 - 8);
      ctx.fillText(lineas[1], x0 + cw / 2, y0 + ch / 2 + 8);
    }
  });

  // ── Punto de mirada ─────────────────────────────────────────────────────
  // Convertir -1..1 a píxeles
  const px = (x + 1) / 2 * w;
  const py = (1 - (y + 1) / 2) * h;

  // Trail
  gazeHistory.push({x: px, y: py});
  if (gazeHistory.length > MAX_GAZE_HISTORY) gazeHistory.shift();

  if (gazeHistory.length > 1) {
    for (let i = 1; i < gazeHistory.length; i++) {
      const alpha = i / gazeHistory.length;
      ctx.beginPath();
      ctx.strokeStyle = `rgba(79,255,176,${alpha * 0.3})`;
      ctx.lineWidth   = 1.5;
      ctx.moveTo(gazeHistory[i-1].x, gazeHistory[i-1].y);
      ctx.lineTo(gazeHistory[i].x,   gazeHistory[i].y);
      ctx.stroke();
    }
  }

  // Punto principal con glow
  const grad = ctx.createRadialGradient(px, py, 0, px, py, 16);
  grad.addColorStop(0, 'rgba(79,255,176,0.7)');
  grad.addColorStop(1, 'rgba(79,255,176,0)');
  ctx.beginPath();
  ctx.arc(px, py, 16, 0, Math.PI * 2);
  ctx.fillStyle = grad;
  ctx.fill();

  ctx.beginPath();
  ctx.arc(px, py, 5, 0, Math.PI * 2);
  ctx.fillStyle = '#4fffb0';
  ctx.fill();

  ultimaDireccion = direccion;
} 

// ── Update UI ──────────────────────────────────────────────────────────────
function updateUI(data) {
  const dir = (data.emotion || 'CENTRO').toUpperCase();
  drawGaze(data.gaze_x, data.gaze_y, dir);
  document.getElementById('gazeX').textContent = data.gaze_x.toFixed(3);
  document.getElementById('gazeY').textContent = data.gaze_y.toFixed(3);

  const now = new Date();
  document.getElementById('lastUpdate').textContent =
    now.toLocaleTimeString('es-EC', {hour12: false});

  // Dirección
  const zona    = (data.emotion || 'OTRO').toUpperCase();
  const dirText = document.getElementById('dirBadge');
  if (dirText) dirText.textContent = zona;

  // Zona de atención
  document.getElementById('zonaName').textContent = zona;
  updateZonaBars(zona);

  // Ojos
  const left  = document.getElementById('leftEyeStatus');
  const right = document.getElementById('rightEyeStatus');
  left.textContent  = data.left_eye_open  ? 'abierto' : 'cerrado';
  left.className    = 'eye-status ' + (data.left_eye_open  ? 'open' : 'closed');
  right.textContent = data.right_eye_open ? 'abierto' : 'cerrado';
  right.className   = 'eye-status ' + (data.right_eye_open ? 'open' : 'closed');

  // Parpadeos
  if (data.blink_detected) {
    blinkCount++;
    const flash = document.getElementById('blinkFlash');
    flash.style.width      = '100%';
    flash.style.background = '#4fffb0';
    setTimeout(() => { flash.style.width = '0%'; }, 300);
}

if (data.total_parpadeos !== undefined) {
    document.getElementById('blinkCount').textContent = data.total_parpadeos;
} else {
    document.getElementById('blinkCount').textContent = blinkCount;
}

  // Status
  document.getElementById('statusDot').classList.add('live');
  document.getElementById('statusText').textContent = 'en vivo';
  // Status dot en header
  document.getElementById('sqDot').classList.add('live');
  document.getElementById('sqText').textContent = document.getElementById('infoPatient').textContent;
// Calcular duración entre frames para promedio
  if (currentSessionId) {
    const ahora = Date.now();
    if (updateUI._ultimo) {
      duracionesArray.push(ahora - updateUI._ultimo);
      if (duracionesArray.length > 200) duracionesArray.shift();
    }
    updateUI._ultimo = ahora;
  }
  // Log row
  if (currentSessionId) {
    recordCount++;
    document.getElementById('infoRecords').textContent = recordCount;
    addLogRow(data, now);
  }
  actualizarStats(data);
}

function addLogRow(data, now) {
  const tbody = document.getElementById('logBody');
  const empty = tbody.querySelector('.empty');
  if (empty) empty.parentElement.remove();

  const zona = (data.emotion || 'OTRO').toUpperCase();
  const tr   = document.createElement('tr');
  tr.innerHTML = `
    <td>${now.toLocaleTimeString('es-EC', {hour12:false})}</td>
    <td>${data.gaze_x.toFixed(3)}</td>
    <td>${data.gaze_y.toFixed(3)}</td>
    <td><span class="tag ${data.left_eye_open?'tag-open':'tag-closed'}">${data.left_eye_open?'abierto':'cerrado'}</span></td>
    <td><span class="tag ${data.right_eye_open?'tag-open':'tag-closed'}">${data.right_eye_open?'abierto':'cerrado'}</span></td>
    <td><span class="tag tag-zona">${zona}</span></td>
    <td>${Math.round((data.emotion_confidence||0)*100)}%</td>
    <td>${data.blink_detected?'✓':''}</td>
  `;
  tbody.insertBefore(tr, tbody.firstChild);
  while (tbody.rows.length > 200) tbody.deleteRow(tbody.rows.length - 1);
}

// ── Poll ───────────────────────────────────────────────────────────────────
async function pollData() {
  try {
    const res  = await fetch(`${API_BASE}/data/current`);
    if (!res.ok) return;
    const data = await res.json();
    if (data.status === 'no_data') return;
    updateUI(data);
  } catch (e) {
    document.getElementById('statusDot').classList.remove('live');
    document.getElementById('statusText').textContent = 'sin conexión';
  }
}

// ── Session ────────────────────────────────────────────────────────────────
async function startSession() {
  const pid = document.getElementById('patientId').value.trim();
  if (!pid) { showToast('Ingresa un ID de paciente', 'error'); return; }
  try {
    const res  = await fetch(`${API_BASE}/session/start`, {
      method:  'POST',
      headers: {'Content-Type': 'application/json'},
      body:    JSON.stringify({
        patient_id: pid,
        notes: document.getElementById('sessionNotes').value.trim() || null
      })
    });
    const data = await res.json();
    currentSessionId = data.session_id;
    cerrarModal();
    const edad  = document.getElementById('patientAge').value;
    const grupo = calcularGrupo(edad);
    document.getElementById('infoName').textContent  = document.getElementById('patientName').value || '—';
    document.getElementById('infoAge').textContent   = edad ? edad + ' años' : '—';
    document.getElementById('infoGrupo').textContent = grupo.nombre;
    recordCount = 0;
    blinkCount  = 0;
    zonaConteo  = { OJOS: 0, NARIZ: 0, BOCA: 0, OTRO: 0 };
    document.getElementById('blinkCount').textContent = '0';
    document.getElementById('infoSessionId').textContent = currentSessionId.slice(0,8) + '...';
    document.getElementById('infoPatient').textContent   = pid;
    document.getElementById('infoStart').textContent     = new Date(data.started_at).toLocaleTimeString('es-EC');
    document.getElementById('infoRecords').textContent   = '0';
    document.getElementById('btnStart').disabled = true;
    document.getElementById('btnNuevaSesion').textContent = '+ Nueva sesión';
    document.getElementById('btnEnd').disabled   = false;
    document.getElementById('btnExport').disabled = false;
    pollInterval = setInterval(pollData, 500);
    zonaConteoStats = { OJOS: 0, NARIZ: 0, BOCA: 0, OTRO: 0 };
    duracionesArray  = [];          // ← agregar aquí
    updateUI._ultimo = null;        // ← y aquí
    iniciarTimerSesion();
    showToast('Sesión iniciada', 'success');
  } catch (e) {
    showToast('No se pudo conectar al servidor', 'error');
  }
}

async function endSession() {
  if (!currentSessionId) return;
  try {
    clearInterval(pollInterval);
    clearInterval(timerSesion);
    await fetch(`${API_BASE}/session/${currentSessionId}/end`, {method: 'POST'});
    showToast(`Sesión finalizada · ${recordCount} registros guardados`, 'success');
    document.getElementById('btnStart').disabled = false;
    document.getElementById('btnEnd').disabled   = true;
    currentSessionId = null;
    document.getElementById('statusDot').classList.remove('live');
    document.getElementById('statusText').textContent = 'sesión terminada';
    document.getElementById('btnExport').disabled = false;
    document.getElementById('sqDot').classList.remove('live');
    document.getElementById('sqText').textContent = 'Sin sesión';
  } catch(e) {
    showToast('Error al terminar sesión', 'error');
  }
}

function actualizarStats(data) {
  // Conteo por zona usando zona_detectada
  const zona = (data.zona_cara || 'OTRO').toUpperCase();
  const zonasValidas = ['OJOS', 'NARIZ', 'BOCA'];
  const zonaStats = zonasValidas.includes(zona) ? zona : 'OTRO';
  zonaConteoStats[zonaStats]++;

  const total = Object.values(zonaConteoStats).reduce((a, b) => a + b, 0);

  // Actualizar barras
  ['OJOS', 'NARIZ', 'BOCA', 'OTRO'].forEach(z => {
    const pct = total > 0 ? Math.round((zonaConteoStats[z] / total) * 100) : 0;
    document.getElementById(`statBar${z.charAt(0) + z.slice(1).toLowerCase()}`).style.width = pct + '%';
    document.getElementById(`statPct${z.charAt(0) + z.slice(1).toLowerCase()}`).textContent = pct + '%';
  });

  // Duración promedio — calculamos entre registros consecutivos
  if (duracionesArray.length > 1) {
    const sum  = duracionesArray.reduce((a, b) => a + b, 0);
    const prom = Math.round(sum / duracionesArray.length);
    document.getElementById('statDurProm').textContent = prom + ' ms';
  }

  // Total fijaciones y parpadeos
  document.getElementById('statFijaciones').textContent = recordCount;
  document.getElementById('statParpadeos').textContent  = blinkCount;
}

function iniciarTimerSesion() {
  sesionInicio = Date.now();
  timerSesion  = setInterval(() => {
    const seg  = Math.floor((Date.now() - sesionInicio) / 1000);
    const min  = Math.floor(seg / 60);
    const s    = seg % 60;
    document.getElementById('statDurSesion').textContent =
      `${min}:${s.toString().padStart(2, '0')}`;
  }, 1000);
}

function calcularGrupo(edad) {
  edad = parseInt(edad);
  if (isNaN(edad) || edad <= 0) return { nombre: '—', clase: '' };
  if (edad <= 11)  return { nombre: 'Niño',        clase: 'nino' };
  if (edad <= 17)  return { nombre: 'Adolescente', clase: 'adolescente' };
  if (edad <= 25)  return { nombre: 'Joven',       clase: 'joven' };
  return           { nombre: 'Adulto',             clase: 'adulto' };
}

async function exportarReporte() {
  if (!currentSessionId) {
    showToast('No hay sesión activa', 'error');
    return;
  }

  // Calcular estadísticas finales
  const total = Object.values(zonaConteoStats).reduce((a, b) => a + b, 0);
  const pctOjos  = total > 0 ? Math.round((zonaConteoStats.OJOS  / total) * 100) : 0;
  const pctNariz = total > 0 ? Math.round((zonaConteoStats.NARIZ / total) * 100) : 0;
  const pctBoca  = total > 0 ? Math.round((zonaConteoStats.BOCA  / total) * 100) : 0;
  const pctOtro  = total > 0 ? Math.round((zonaConteoStats.OTRO  / total) * 100) : 0;

  const durProm = document.getElementById('statDurProm').textContent;
  const durSesion = document.getElementById('statDurSesion').textContent;

  // Construir CSV
  const fecha = new Date().toLocaleString('es-EC');
  let csv = '====== REPORTE NEUROSCAN - EYE TRACKING ======\n\n';
  csv += `Fecha:,${fecha}\n`;
  csv += `ID Sesion:,${currentSessionId}\n`;
  csv += `ID Paciente:,${document.getElementById('infoPatient').textContent}\n`;
  csv += `Nombre:,${document.getElementById('infoName').textContent}\n`;
  csv += `Edad:,${document.getElementById('infoAge').textContent}\n`;
  csv += `Grupo:,${document.getElementById('infoGrupo').textContent}\n`;
  csv += `Duracion sesion:,${durSesion}\n\n`;
  csv += '====== ESTADISTICAS ======\n\n';
  csv += `Total fijaciones:,${recordCount}\n`;
  csv += `Duracion promedio:,${durProm}\n`;
  csv += `Total parpadeos:,${blinkCount}\n\n`;
  csv += '====== DISTRIBUCION POR ZONA ======\n\n';
  csv += `OJOS:,${zonaConteoStats.OJOS},${pctOjos}%\n`;
  csv += `NARIZ:,${zonaConteoStats.NARIZ},${pctNariz}%\n`;
  csv += `BOCA:,${zonaConteoStats.BOCA},${pctBoca}%\n`;
  csv += `OTRO:,${zonaConteoStats.OTRO},${pctOtro}%\n\n`;
  csv += '====== REGISTRO DE DATOS ======\n\n';
  csv += 'Hora,Gaze X,Gaze Y,Ojo Izq,Ojo Der,Zona,Parpadeo\n';

  // Agregar registros de la tabla
  const filas = document.querySelectorAll('#logBody tr');
  filas.forEach(fila => {
    const celdas = fila.querySelectorAll('td');
    if (celdas.length > 1) {
      csv += `${celdas[0].textContent},${celdas[1].textContent},${celdas[2].textContent},`;
      csv += `${celdas[3].textContent},${celdas[4].textContent},`;
      csv += `${celdas[5].textContent},${celdas[7].textContent}\n`;
    }
  });

  // Descargar
  const blob = new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8;' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `reporte_${document.getElementById('infoPatient').textContent}_${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);

  showToast('Reporte descargado ✓', 'success');
}

// ── Toast ──────────────────────────────────────────────────────────────────
let toastTimer;
function showToast(msg, type='') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className   = 'show ' + type;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.className = '', 3000);
}