/* ═══════════════════════════════════════
   TinyCNNCANNet — JavaScript
   Particle canvas · Counters · Tabs ·
   Scroll reveals · Demo classifier
   Real API: POST /api/predict (Flask)
   Fallback: local heuristic simulation
   ═══════════════════════════════════════ */

'use strict';

// ── API config ────────────────────────────
// When served from server.py (port 5000) same-origin requests work automatically.
// When opened via file:// or port 7823, we point to the Flask server.
const API_BASE = window.location.port === '5000'
  ? ''                         // same origin — server.py serves the frontend too
  : 'http://localhost:5000';   // cross-origin fallback

let serverOnline = false; // updated by health check below

// ── Health check ──────────────────────────
async function checkServer() {
  try {
    const res = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(2000) });
    if (res.ok) {
      serverOnline = true;
      addLogInfo('✅ Connected to real model server (Flask API).');
      updateServerBadge(true);
    }
  } catch {
    serverOnline = false;
    addLogInfo('⚠️ Server offline — using local simulation. Run server.py to use the real model.');
    updateServerBadge(false);
  }
}

function updateServerBadge(online) {
  const badge = document.getElementById('serverBadge');
  if (!badge) return;
  badge.textContent  = online ? '🟢 Real Model Connected' : '🟡 Simulation Mode';
  badge.style.color  = online ? 'var(--green)' : 'var(--amber)';
  badge.style.borderColor = online ? 'rgba(16,185,129,0.4)' : 'rgba(245,158,11,0.4)';
}

// ── Helpers ────────────────────────────────
const $ = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

// ── Navbar scroll ────────────────────────
const navbar = $('#navbar');
window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 20);
}, { passive: true });

// ── Hamburger ────────────────────────────
const hamburger = $('#hamburger');
const navLinks  = $('#navLinks');
hamburger.addEventListener('click', () => navLinks.classList.toggle('open'));

// ── Particle Canvas ──────────────────────
(function initParticles() {
  const canvas = document.getElementById('particleCanvas');
  const ctx = canvas.getContext('2d');
  let W, H, particles = [];

  const resize = () => {
    W = canvas.width  = canvas.offsetWidth;
    H = canvas.height = canvas.offsetHeight;
  };

  class Particle {
    constructor() { this.reset(true); }
    reset(initial = false) {
      this.x  = Math.random() * W;
      this.y  = initial ? Math.random() * H : H + 10;
      this.vx = (Math.random() - 0.5) * 0.3;
      this.vy = -(Math.random() * 0.5 + 0.2);
      this.r  = Math.random() * 1.5 + 0.5;
      this.alpha = Math.random() * 0.6 + 0.2;
      this.color = Math.random() > 0.5 ? '0,245,255' : '124,58,237';
    }
    update() {
      this.x += this.vx;
      this.y += this.vy;
      if (this.y < -10) this.reset();
    }
    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${this.color},${this.alpha})`;
      ctx.fill();
    }
  }

  const init = () => {
    resize();
    particles = Array.from({ length: 80 }, () => new Particle());
  };

  const loop = () => {
    ctx.clearRect(0, 0, W, H);
    particles.forEach(p => { p.update(); p.draw(); });
    // Draw connecting lines for close particles
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 80) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(0,245,255,${0.08 * (1 - dist / 80)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(loop);
  };

  window.addEventListener('resize', resize, { passive: true });
  init();
  loop();
})();

// ── Counter Animation ─────────────────────
(function initCounters() {
  const counters = $$('.counter');
  let started = false;

  const run = () => {
    if (started) return;
    started = true;
    counters.forEach(el => {
      const target = +el.dataset.target;
      let current = 0;
      const step = target / 40;
      const tick = () => {
        current = Math.min(current + step, target);
        el.textContent = Math.round(current);
        if (current < target) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });
  };

  const obs = new IntersectionObserver(entries => {
    if (entries.some(e => e.isIntersecting)) { run(); obs.disconnect(); }
  }, { threshold: 0.5 });
  obs.observe($('#heroStats'));
})();

// ── Scroll Reveal ─────────────────────────
(function initReveal() {
  // Add data-reveal to cards/items
  const targets = [
    '#absCard1', '#absCard2', '#absCard3',
    '#contrib1', '#contrib2', '#contrib3',
    '#ds1', '#ds2', '#ds3', '#ds4',
    '#rm1', '#rm2', '#rm3', '#rm4',
    '#pipe1', '#pipe2', '#pipe3', '#pipe4', '#pipe5',
  ];
  targets.forEach((sel, i) => {
    const el = $(sel);
    if (el) {
      el.setAttribute('data-reveal', '');
      el.style.transitionDelay = `${(i % 4) * 0.1}s`;
    }
  });

  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) { e.target.classList.add('revealed'); obs.unobserve(e.target); }
    });
  }, { threshold: 0.15 });
  $$('[data-reveal]').forEach(el => obs.observe(el));

  // Bar chart animation
  const barObs = new IntersectionObserver(entries => {
    if (entries.some(e => e.isIntersecting)) {
      $$('.bar-fill').forEach(bar => {
        bar.style.width = bar.style.getPropertyValue('--pct');
      });
      barObs.disconnect();
    }
  }, { threshold: 0.3 });
  const chart = $('#barChart');
  if (chart) barObs.observe(chart);
})();

// ── Architecture Tabs ─────────────────────
(function initTabs() {
  $$('.arch-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      $$('.arch-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const target = tab.dataset.tab;
      $$('.arch-panel').forEach(panel => {
        panel.classList.toggle('hidden', panel.dataset.panel !== target);
      });
    });
  });
})();

// ── CAN Frame Scenarios ───────────────────
const scenarios = {
  normal: {
    timestamp: '0.001200', id: '0244', dlc: '8',
    d: ['00', '00', '3C', '00', '00', '00', '00', '00'],
    label: 'Normal', isAttack: false, conf: 0.97,
    probs: { Normal: 0.97, DoS: 0.01, Fuzzy: 0.01, Gear: 0.005, RPM: 0.005 }
  },
  dos: {
    timestamp: '0.000010', id: '0244', dlc: '8',
    d: ['FF', 'FF', 'FF', 'FF', 'FF', 'FF', 'FF', 'FF'],
    label: 'DoS Attack', isAttack: true, conf: 0.995,
    probs: { Normal: 0.002, DoS: 0.995, Fuzzy: 0.002, Gear: 0.0005, RPM: 0.0005 }
  },
  fuzzy: {
    timestamp: '0.000800', id: 'A1B2', dlc: '8',
    d: ['3F', '7C', 'D1', '0E', '9A', 'F4', '22', 'BB'],
    label: 'Fuzzy Attack', isAttack: true, conf: 0.983,
    probs: { Normal: 0.008, DoS: 0.004, Fuzzy: 0.983, Gear: 0.003, RPM: 0.002 }
  },
  rpm: {
    timestamp: '0.002100', id: '0316', dlc: '8',
    d: ['FF', 'FF', '00', '00', '00', '00', '01', 'F4'],
    label: 'RPM Spoof', isAttack: true, conf: 0.961,
    probs: { Normal: 0.015, DoS: 0.008, Fuzzy: 0.012, Gear: 0.004, RPM: 0.961 }
  }
};

// Scenario buttons
$$('.scenario-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    $$('.scenario-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const sc = scenarios[btn.dataset.scenario];
    if (!sc) return;
    $('#fieldTimestamp').value = sc.timestamp;
    $('#fieldId').value = sc.id;
    $('#fieldDlc').value = sc.dlc;
    sc.d.forEach((v, i) => { $('#fieldD' + i).value = v; });
  });
});

// ── Classify ─────────────────────────────
const COLORS = {
  Normal: '#10b981', DoS: '#ef4444', Fuzzy: '#f59e0b', Gear: '#7c3aed', RPM: '#00f5ff'
};

$('#btnClassify').addEventListener('click', async () => {
  const btn = $('#btnClassify');
  btn.disabled = true;
  btn.querySelector('.btn-classify-text').textContent = 'Running…';

  const timestamp = parseFloat($('#fieldTimestamp').value) || 0;
  const canId     = $('#fieldId').value.trim().toUpperCase() || '0000';
  const dlc       = parseInt($('#fieldDlc').value) || 0;
  const bytes     = [0,1,2,3,4,5,6,7].map(i => parseInt($('#fieldD' + i).value, 16) || 0);

  const tStart = performance.now();

  try {
    let result;
    if (serverOnline) {
      // ── Real model via Flask API ──
      const payload = {
        timestamp, id: canId.toLowerCase(), dlc,
        d0: bytes[0].toString(16).padStart(2,'0'),
        d1: bytes[1].toString(16).padStart(2,'0'),
        d2: bytes[2].toString(16).padStart(2,'0'),
        d3: bytes[3].toString(16).padStart(2,'0'),
        d4: bytes[4].toString(16).padStart(2,'0'),
        d5: bytes[5].toString(16).padStart(2,'0'),
        d6: bytes[6].toString(16).padStart(2,'0'),
        d7: bytes[7].toString(16).padStart(2,'0'),
        prev_timestamp: 0.0,
      };
      const res  = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(8000),
      });
      const json = await res.json();
      if (json.error) throw new Error(json.error);
      result = {
        label:    json.label,
        isAttack: json.is_attack,
        conf:     json.confidence,
        probs:    json.probs,
        real:     true,
        serverMs: json.inference_ms,
      };
    } else {
      // ── Local heuristic fallback ──
      result = localSimulate(canId, timestamp, dlc, bytes);
    }

    const elapsed = (performance.now() - tStart).toFixed(1);
    showResult(result, elapsed, canId, bytes);

  } catch (err) {
    addLogInfo(`❌ API error: ${err.message} — falling back to simulation.`);
    serverOnline = false;
    updateServerBadge(false);
    const result  = localSimulate(canId, timestamp, dlc, bytes);
    const elapsed = (performance.now() - tStart).toFixed(1);
    showResult(result, elapsed, canId, bytes);
  }

  btn.disabled = false;
  btn.querySelector('.btn-classify-text').textContent = 'Run Classification';
});

/** Local heuristic simulation (offline fallback). */
function localSimulate(canId, timestamp, dlc, bytes) {
  const byteSum     = bytes.reduce((a,b) => a+b, 0);
  const allFF       = bytes.every(b => b === 0xFF);
  const highEntropy = new Set(bytes).size > 6;
  const isHighFreq  = timestamp < 0.0005;

  if (allFF || (isHighFreq && byteSum > 1800)) {
    return { label:'DoS Attack',  isAttack:true,  conf:0.993, probs:{Normal:0.003, DoS:0.993, Fuzzy:0.002, Gear:0.001, RPM:0.001} };
  }
  if (highEntropy && byteSum > 500 && byteSum < 1500) {
    return { label:'Fuzzy Attack', isAttack:true,  conf:0.978, probs:{Normal:0.010, DoS:0.005, Fuzzy:0.978, Gear:0.004, RPM:0.003} };
  }
  if (canId.startsWith('03') && bytes[0] === 0xFF && bytes[1] === 0xFF) {
    return { label:'RPM Spoof',   isAttack:true,  conf:0.965, probs:{Normal:0.015, DoS:0.006, Fuzzy:0.010, Gear:0.004, RPM:0.965} };
  }
  return { label:'Normal', isAttack:false, conf:0.977, probs:{Normal:0.977, DoS:0.009, Fuzzy:0.008, Gear:0.003, RPM:0.003} };
}

function showResult(result, elapsed, canId, bytes) {
  const idle   = $('#resultIdle');
  const active = $('#resultActive');
  const badge  = $('#resultBadge');
  const confBar = $('#confBar');
  const confPct = $('#confPct');
  const probsEl = $('#resultProbs');
  const timing  = $('#timingVal');

  idle.classList.add('hidden');
  active.classList.remove('hidden');

  // Badge
  badge.textContent = result.label;
  badge.className   = 'result-label-badge ' + (result.isAttack ? 'label-attack' : 'label-normal');

  // Confidence bar
  const pct = Math.round(result.conf * 100);
  confPct.textContent = pct + '%';
  setTimeout(() => { confBar.style.width = pct + '%'; }, 50);

  // Prob bars
  probsEl.innerHTML = '';
  Object.entries(result.probs).forEach(([cls, prob]) => {
    const pctStr = (prob * 100).toFixed(1) + '%';
    const color  = COLORS[cls] || '#fff';
    probsEl.innerHTML += `
      <div class="prob-row">
        <span class="prob-name">${cls}</span>
        <div class="prob-bar-outer">
          <div class="prob-bar-fill" style="width:${pctStr}; background:${color}"></div>
        </div>
        <span class="prob-pct">${pctStr}</span>
      </div>`;
  });

  // Timing
  timing.textContent = elapsed + ' ms';

  // Log
  addLog(result, canId, bytes);
}

// ── Log Panel ─────────────────────────────
const logBody = $('#logBody');
let logCount  = 0;

function addLog(result, canId, bytes) {
  const now  = new Date().toLocaleTimeString('en-GB');
  const cls  = result.isAttack ? 'attack' : 'normal';
  const byteStr = bytes.map(b => b.toString(16).toUpperCase().padStart(2,'0')).join(' ');
  const entry = document.createElement('div');
  entry.className = 'log-entry';
  entry.innerHTML = `
    <span class="log-time">[${now}]</span>
    <span class="log-msg ${cls}">▶ ID:${canId} [${byteStr}] → ${result.label} (${(result.conf*100).toFixed(1)}%)</span>`;
  logBody.prepend(entry);
  logCount++;
  if (logCount > 20) logBody.lastElementChild?.remove();
}

$('#logClearBtn').addEventListener('click', () => {
  logBody.innerHTML = '';
  logCount = 0;
  addLogInfo('Log cleared.');
});

function addLogInfo(msg) {
  const now = new Date().toLocaleTimeString('en-GB');
  const entry = document.createElement('div');
  entry.className = 'log-entry';
  entry.innerHTML = `<span class="log-time">[${now}]</span><span class="log-msg info">${msg}</span>`;
  logBody.prepend(entry);
}

// Startup
setTimeout(() => {
  addLogInfo('TinyCNNCANNet loaded · Model: ~13K params · Checking API server…');
  checkServer();
}, 800);

// ── Smooth Nav Links ──────────────────────
$$('.nav-link').forEach(link => {
  link.addEventListener('click', (e) => {
    const href = link.getAttribute('href');
    if (href.startsWith('#')) {
      e.preventDefault();
      document.querySelector(href)?.scrollIntoView({ behavior: 'smooth' });
      navLinks.classList.remove('open');
    }
  });
});
