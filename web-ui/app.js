/* ═══════════════════════════════════════════════════════════════
   ta-LBFGS Web UI — Application Logic
   ═══════════════════════════════════════════════════════════════ */

// ── Background Particles ────────────────────────────────────────
(function initBgParticles() {
  const canvas = document.getElementById('bgCanvas');
  const ctx = canvas.getContext('2d');
  let particles = [];
  const COUNT = 60;

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  for (let i = 0; i < COUNT; i++) {
    particles.push({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      r: Math.random() * 1.5 + 0.5,
      alpha: Math.random() * 0.15 + 0.02,
    });
  }

  function drawBg() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach(p => {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0) p.x = canvas.width;
      if (p.x > canvas.width) p.x = 0;
      if (p.y < 0) p.y = canvas.height;
      if (p.y > canvas.height) p.y = 0;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(99,102,241,${p.alpha})`;
      ctx.fill();
    });
    // Draw subtle connections
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 150) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(99,102,241,${0.03 * (1 - dist / 150)})`;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(drawBg);
  }
  drawBg();
})();

// ── Hero 3D Loss Landscape Canvas ──────────────────────────────
(function initHeroCanvas() {
  const canvas = document.getElementById('heroCanvas');
  const ctx = canvas.getContext('2d');
  let time = 0;

  function rosenbrock(x, y) {
    return (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x);
  }

  function render() {
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    time += 0.008;

    const gridSize = 40;
    const scale = 6;
    const offsetX = W / 2, offsetY = H * 0.65;

    // Draw wireframe landscape
    for (let i = -gridSize / 2; i < gridSize / 2; i++) {
      ctx.beginPath();
      for (let j = -gridSize / 2; j <= gridSize / 2; j++) {
        const x = i / (gridSize / 4), y = j / (gridSize / 4);
        const z = Math.log1p(rosenbrock(x, y)) * 0.15;

        // Isometric projection with rotation
        const angle = time * 0.3;
        const rx = x * Math.cos(angle) - y * Math.sin(angle);
        const ry = x * Math.sin(angle) + y * Math.cos(angle);
        const px = offsetX + rx * scale * 20 - ry * scale * 12;
        const py = offsetY + ry * scale * 8 + rx * scale * 5 - z * scale * 12;

        if (j === -gridSize / 2) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      const hue = 250 + i * 3;
      ctx.strokeStyle = `hsla(${hue}, 70%, 60%, 0.15)`;
      ctx.lineWidth = 0.8;
      ctx.stroke();
    }

    // Cross-lines
    for (let j = -gridSize / 2; j < gridSize / 2; j++) {
      ctx.beginPath();
      for (let i = -gridSize / 2; i <= gridSize / 2; i++) {
        const x = i / (gridSize / 4), y = j / (gridSize / 4);
        const z = Math.log1p(rosenbrock(x, y)) * 0.15;
        const angle = time * 0.3;
        const rx = x * Math.cos(angle) - y * Math.sin(angle);
        const ry = x * Math.sin(angle) + y * Math.cos(angle);
        const px = offsetX + rx * scale * 20 - ry * scale * 12;
        const py = offsetY + ry * scale * 8 + rx * scale * 5 - z * scale * 12;
        if (i === -gridSize / 2) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      const hue = 290 + j * 2;
      ctx.strokeStyle = `hsla(${hue}, 60%, 50%, 0.12)`;
      ctx.lineWidth = 0.6;
      ctx.stroke();
    }

    // Animated trajectory point
    const t = time * 0.5;
    const tx = 0.8 * Math.cos(t) * Math.exp(-t * 0.02);
    const ty = tx * tx + 0.1 * Math.sin(t * 2);
    const tz = Math.log1p(rosenbrock(tx, ty)) * 0.15;
    const angle = time * 0.3;
    const trx = tx * Math.cos(angle) - ty * Math.sin(angle);
    const tryy = tx * Math.sin(angle) + ty * Math.cos(angle);
    const tpx = offsetX + trx * scale * 20 - tryy * scale * 12;
    const tpy = offsetY + tryy * scale * 8 + trx * scale * 5 - tz * scale * 12;

    // Glow
    const grd = ctx.createRadialGradient(tpx, tpy, 0, tpx, tpy, 20);
    grd.addColorStop(0, 'rgba(168,85,247,0.6)');
    grd.addColorStop(1, 'transparent');
    ctx.fillStyle = grd;
    ctx.fillRect(tpx - 20, tpy - 20, 40, 40);

    // Point
    ctx.beginPath();
    ctx.arc(tpx, tpy, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#a855f7';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    requestAnimationFrame(render);
  }
  render();
})();

// ── Nav scroll effect ──────────────────────────────────────────
window.addEventListener('scroll', () => {
  document.getElementById('topnav').classList.toggle('scrolled', window.scrollY > 20);

  // Active nav link
  const sections = ['hero', 'features', 'algorithm', 'dashboard', 'benchmark'];
  let current = 'hero';
  sections.forEach(id => {
    const el = document.getElementById(id);
    if (el && el.getBoundingClientRect().top < 200) current = id;
  });
  document.querySelectorAll('.nav-link').forEach(link => {
    link.classList.toggle('active', link.dataset.section === current);
  });
});

// ── Simple Canvas Chart Renderer ────────────────────────────────
class MiniChart {
  constructor(canvasId, options = {}) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.data = [];
    this.datasets = options.datasets || [{ color: '#6366f1', data: [] }];
    this.options = options;
    this._resizeCanvas();
  }

  _resizeCanvas() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.canvas.width = rect.width - 40;
    this.canvas.height = this.options.height || 180;
  }

  update(datasets) {
    this.datasets = datasets;
    this.draw();
  }

  draw() {
    const { ctx, canvas } = this;
    const W = canvas.width, H = canvas.height;
    const pad = { top: 10, right: 10, bottom: 24, left: 50 };
    ctx.clearRect(0, 0, W, H);

    // Find global min/max
    let allVals = [];
    this.datasets.forEach(ds => allVals.push(...ds.data));
    if (allVals.length === 0) return;
    let min = Math.min(...allVals), max = Math.max(...allVals);
    if (max - min < 1e-8) { min -= 0.5; max += 0.5; }

    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (plotH / 4) * i;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
      const val = max - (i / 4) * (max - min);
      ctx.fillStyle = 'rgba(255,255,255,0.3)';
      ctx.font = '10px "JetBrains Mono"';
      ctx.textAlign = 'right';
      ctx.fillText(val.toFixed(val < 1 ? 4 : 1), pad.left - 6, y + 3);
    }

    // Lines
    this.datasets.forEach(ds => {
      if (ds.data.length < 2) return;
      ctx.beginPath();
      ds.data.forEach((v, i) => {
        const x = pad.left + (i / Math.max(ds.data.length - 1, 1)) * plotW;
        const y = pad.top + (1 - (v - min) / (max - min)) * plotH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = ds.color;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Area fill
      if (ds.fill) {
        const lastX = pad.left + plotW;
        ctx.lineTo(lastX, pad.top + plotH);
        ctx.lineTo(pad.left, pad.top + plotH);
        ctx.closePath();
        const grd = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
        grd.addColorStop(0, ds.color.replace(')', ',0.15)').replace('rgb', 'rgba'));
        grd.addColorStop(1, 'transparent');
        ctx.fillStyle = grd;
        ctx.fill();
      }

      // Markers for events
      if (ds.markers) {
        ds.markers.forEach(m => {
          const x = pad.left + (m.idx / Math.max(ds.data.length - 1, 1)) * plotW;
          ctx.beginPath();
          ctx.moveTo(x, pad.top);
          ctx.lineTo(x, pad.top + plotH);
          ctx.strokeStyle = m.color || '#ef4444';
          ctx.lineWidth = 1;
          ctx.setLineDash([3, 3]);
          ctx.stroke();
          ctx.setLineDash([]);
        });
      }
    });
  }
}

// ── Simulation Engine ──────────────────────────────────────────
const SIM = {
  running: false,
  step: 0,
  maxSteps: 40,
  interval: null,
  lossHistory: [],
  kappaHistories: [[], [], [], []],
  lrHistories: [[], [], [], []],
  wdHistories: [[], [], [], []],
  events: [],
  evasions: 0,
  layerData: [{}, {}, {}, {}],
};

let lossChart, kappaChart, hpChart;

function initCharts() {
  lossChart = new MiniChart('lossChart', { height: 180 });
  kappaChart = new MiniChart('kappaChart', { height: 180 });
  hpChart = new MiniChart('hpChart', { height: 180 });
}

function simulateStep() {
  const step = SIM.step;

  // Simulate loss (exponential decay + noise + occasional bumps)
  const baseLoss = 2.8 * Math.exp(-step * 0.08) + 0.3 + Math.random() * 0.12;
  const bump = (step === 12 || step === 24) ? 0.3 : 0;
  const loss = baseLoss + bump;
  SIM.lossHistory.push(loss);

  // Per-layer topology
  const layerNames = ['layers.0', 'layers.1', 'layers.2', 'layers.3'];
  const conditionScales = [10, 20, 30, 40];

  layerNames.forEach((name, i) => {
    const baseKappa = conditionScales[i] * (1 + 0.5 * Math.sin(step * 0.3 + i));
    const kappa = Math.max(1, baseKappa + (Math.random() - 0.5) * 8);
    SIM.kappaHistories[i].push(kappa);

    const lr = 1e-4 * Math.exp(-step * 0.02) * (1 + 0.1 * Math.random());
    const wd = 1e-2 * (1 + 0.05 * Math.sin(step * 0.2 + i));
    SIM.lrHistories[i].push(lr);
    SIM.wdHistories[i].push(wd);

    let landscape, cellClass;
    const secant = 0.5 + Math.random() * 0.5 - (kappa > 30 ? 0.8 : 0);
    if (secant <= 0) {
      landscape = 'Saddle Point'; cellClass = 'saddle';
    } else if (kappa > 30) {
      landscape = 'Narrow Ravine'; cellClass = 'ravine';
    } else if (kappa > 10) {
      landscape = 'Ill-Conditioned'; cellClass = 'ill';
    } else {
      landscape = 'Convex Bowl'; cellClass = 'convex';
    }

    const memSize = Math.max(3, Math.min(20, Math.ceil(Math.log2(Math.max(1, kappa)))));

    SIM.layerData[i] = {
      name, kappa: kappa.toFixed(1), landscape, cellClass, memSize,
      gradNorm: (Math.random() * 0.5).toFixed(4),
    };

    // Generate events
    if (secant <= 0) {
      SIM.evasions++;
      SIM.events.unshift({
        type: 'pivot',
        text: `Step ${step + 1}: Saddle detected in ${name} — eigenvector perturbation applied`,
      });
    }
  });

  // Spectral guard events
  if (step === 8 || step === 22) {
    SIM.events.unshift({
      type: 'spectral',
      text: `Step ${step + 1}: Spectral guard fired — κ exceeded 10,000`,
    });
  }

  // Convergence event
  if (step === SIM.maxSteps - 1) {
    SIM.events.unshift({
      type: 'converge',
      text: `Optimization complete — final loss: ${loss.toFixed(4)}`,
    });
  }

  // Info events
  if (step % 10 === 0 && step > 0) {
    SIM.events.unshift({
      type: 'info',
      text: `Step ${step + 1}: Adaptive memory resized for ${layerNames[Math.floor(Math.random() * 4)]}`,
    });
  }

  SIM.step++;
  updateDashboardUI();
  
  // Update 3D topology visualization with synthetic data
  if (window.Topo3D && window.Topo3D.simulateStep) {
    window.Topo3D.simulateStep(SIM.step);
  }
}

function updateDashboardUI() {
  const step = SIM.step;
  const loss = SIM.lossHistory[SIM.lossHistory.length - 1];
  const prevLoss = SIM.lossHistory.length > 1 ? SIM.lossHistory[SIM.lossHistory.length - 2] : loss;

  // Metrics
  document.getElementById('dValLoss').textContent = loss.toFixed(4);
  const delta = loss - prevLoss;
  const deltaEl = document.getElementById('dValDelta');
  deltaEl.textContent = `${delta >= 0 ? '+' : ''}${delta.toFixed(4)}`;
  deltaEl.className = `dash-metric-delta ${delta < 0 ? 'negative' : 'positive'}`;

  document.getElementById('dStep').textContent = `${step} / ${SIM.maxSteps}`;
  document.getElementById('dStepFill').style.width = `${(step / SIM.maxSteps) * 100}%`;

  const avgKappa = SIM.kappaHistories.reduce((s, h) => s + (h[h.length - 1] || 0), 0) / 4;
  document.getElementById('dKappa').textContent = avgKappa.toFixed(1);
  const kappaTag = document.getElementById('dKappaTag');
  if (avgKappa < 10) { kappaTag.textContent = 'Well-conditioned'; kappaTag.style.color = '#22c55e'; kappaTag.style.background = 'rgba(34,197,94,0.1)'; }
  else if (avgKappa < 30) { kappaTag.textContent = 'Moderate'; kappaTag.style.color = '#f59e0b'; kappaTag.style.background = 'rgba(245,158,11,0.1)'; }
  else { kappaTag.textContent = 'Ill-conditioned'; kappaTag.style.color = '#ef4444'; kappaTag.style.background = 'rgba(239,68,68,0.1)'; }

  document.getElementById('dEvasions').textContent = SIM.evasions;

  // Loss trend
  const trend = document.getElementById('lossTrend');
  if (SIM.lossHistory.length >= 5) {
    const recent = SIM.lossHistory.slice(-5);
    const d = recent[recent.length - 1] - recent[0];
    trend.textContent = d < -0.05 ? '↓ Converging' : (d > 0.05 ? '↑ Diverging' : '→ Plateau');
    trend.style.color = d < -0.05 ? '#22c55e' : (d > 0.05 ? '#ef4444' : '#f59e0b');
  }

  // Charts
  lossChart.update([{
    color: 'rgb(99,102,241)', data: SIM.lossHistory, fill: true,
    markers: SIM.events.filter(e => e.type === 'pivot').map((e, i) => ({ idx: parseInt(e.text.match(/\d+/)[0]) - 1, color: '#ef4444' })),
  }]);

  const kappaColors = ['#6366f1', '#a855f7', '#ec4899', '#14b8a6'];
  kappaChart.update(SIM.kappaHistories.map((h, i) => ({
    color: kappaColors[i], data: h,
  })));

  hpChart.update([
    ...SIM.lrHistories.map((h, i) => ({ color: kappaColors[i], data: h })),
  ]);

  // Topo grid
  const topoGrid = document.getElementById('topoGrid');
  topoGrid.innerHTML = SIM.layerData.map(d => `
    <div class="topo-cell ${d.cellClass || ''}">
      <div class="topo-cell-name">${d.name || '—'}</div>
      <div class="topo-cell-kappa">${d.kappa || '—'}</div>
      <div class="topo-cell-status">${d.landscape || '—'}</div>
      <div class="topo-cell-mem">m = ${d.memSize || '—'}</div>
    </div>
  `).join('');

  // Events
  const eventFeed = document.getElementById('eventFeed');
  eventFeed.innerHTML = SIM.events.slice(0, 20).map(e => `
    <div class="event-item">
      <span class="event-dot ${e.type}"></span>
      ${e.text}
    </div>
  `).join('');
  document.getElementById('eventCount').textContent = `${SIM.events.length} events`;

  // Sim status
  document.getElementById('simStatus').textContent =
    SIM.step >= SIM.maxSteps ? '✓ Complete' : `Step ${SIM.step}/${SIM.maxSteps}`;
}

function startSimulation() {
  if (SIM.running) return;
  SIM.running = true;
  document.getElementById('btnStartSim').disabled = true;
  document.getElementById('simStatus').textContent = 'Running...';

  SIM.interval = setInterval(() => {
    if (SIM.step >= SIM.maxSteps) {
      clearInterval(SIM.interval);
      SIM.running = false;
      document.getElementById('btnStartSim').disabled = false;
      document.getElementById('simStatus').textContent = '✓ Complete';
      return;
    }
    simulateStep();
  }, 300);
}

function resetSimulation() {
  clearInterval(SIM.interval);
  SIM.running = false;
  SIM.step = 0;
  SIM.lossHistory = [];
  SIM.kappaHistories = [[], [], [], []];
  SIM.lrHistories = [[], [], [], []];
  SIM.wdHistories = [[], [], [], []];
  SIM.events = [];
  SIM.evasions = 0;
  SIM.layerData = [{}, {}, {}, {}];
  document.getElementById('btnStartSim').disabled = false;
  document.getElementById('simStatus').textContent = 'Ready';
  document.getElementById('dValLoss').textContent = '—';
  document.getElementById('dValDelta').textContent = '';
  document.getElementById('dStep').textContent = '0 / 40';
  document.getElementById('dStepFill').style.width = '0%';
  document.getElementById('dKappa').textContent = '—';
  document.getElementById('dKappaTag').textContent = 'Initializing';
  document.getElementById('dEvasions').textContent = '0';
  document.getElementById('lossTrend').textContent = '—';
  document.getElementById('topoGrid').innerHTML = '';
  document.getElementById('eventFeed').innerHTML = '<div class="event-empty">Run a simulation to see optimizer events</div>';
  document.getElementById('eventCount').textContent = '0 events';
  if (lossChart) lossChart.update([{ color: 'rgb(99,102,241)', data: [] }]);
  if (kappaChart) kappaChart.update([]);
  if (hpChart) hpChart.update([]);
  
  // Reset 3D topology visualization
  if (window.Topo3D && window.Topo3D.simulateStep) {
    window.Topo3D.simulateStep(0);
  }
}

// ── SSE Connection (try connecting to real optimizer) ───────────
function trySSEConnection() {
  const dot = document.querySelector('.status-dot');
  const text = document.querySelector('.status-text');

  try {
    const es = new EventSource('http://127.0.0.1:7860/events');
    es.onopen = () => {
      dot.className = 'status-dot online';
      text.textContent = 'Connected';
    };
    es.onerror = () => {
      // Keep standalone mode — don't show alarming "Offline"
      dot.className = 'status-dot standalone';
      text.textContent = 'Standalone';
      es.close();
    };
    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data.run) {
          dot.className = 'status-dot online';
          text.textContent = `Live — Step ${data.run.outer_step}`;
        }
        // Wire topology_3d data to Topo3D visualization
        if (data.topology_3d && window.Topo3D && window.Topo3D.update) {
          window.Topo3D.update(data.topology_3d);
        }
      } catch(e) {}
    };
  } catch(e) {
    // Server not running — stay in standalone mode
  }
}

// ── Init ────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  initCharts();
  trySSEConnection();

  // Intersection observer for animations
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
      }
    });
  }, { threshold: 0.1 });

  document.querySelectorAll('.feature-card, .bench-card, .pipeline-step').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(el);
  });
});
