import initSonaWasm, { WasmSonaEngine, WasmEphemeralAgent, WasmFederatedCoordinator, wasm_init } from './sonawasm/ruvector_sona.js';

// --- Constants ---
const MOVES = ['rock', 'paper', 'scissors'];
const COUNTER = { rock: 'paper', paper: 'scissors', scissors: 'rock' };
const EMOJI = { rock: '\u{1FAA8}', paper: '\u{1F4C4}', scissors: '\u2702\uFE0F' };
const DIM = 15; // 5 moves * 3 one-hot
const ROLLING_WINDOW = 20;
const AGGREGATE_EVERY = 5; // aggregate agent → coordinator every N rounds

// --- DOM ---
const statusEl = document.getElementById('status');
const humanScoreEl = document.getElementById('human-score');
const drawScoreEl = document.getElementById('draw-score');
const aiScoreEl = document.getElementById('ai-score');
const roundResultEl = document.getElementById('round-result');
const btnRock = document.getElementById('btn-rock');
const btnPaper = document.getElementById('btn-paper');
const btnScissors = document.getElementById('btn-scissors');
const accuracyCanvas = document.getElementById('accuracy-chart');
const cumulativeCanvas = document.getElementById('cumulative-chart');

// --- State ---
let engine = null;       // WasmSonaEngine (kept for stats)
let agent = null;        // WasmEphemeralAgent (trajectory collection)
let coordinator = null;  // WasmFederatedCoordinator (learning + prediction)
let moveHistory = [];
let scores = { human: 0, ai: 0, draw: 0 };
let round = 0;
let learnCycles = 0;

// Per-model tracking
const models = {
  ngram:    { predictions: [], accHistory: [], correct: 0, total: 0, lastPick: null },
  sona:     { predictions: [], accHistory: [], correct: 0, total: 0, lastPick: null },
  combined: { predictions: [], accHistory: [], correct: 0, total: 0, lastPick: null },
};

// --- N-gram pattern predictor ---

const ngramCounts = new Map();

function ngramKey(history, windowSize) {
  if (history.length < windowSize) return null;
  return history.slice(-windowSize).join(',');
}

function recordNgram(history, move) {
  for (let w = 1; w <= 5; w++) {
    if (history.length < w) continue;
    const key = `${w}:${ngramKey(history, w)}`;
    if (!ngramCounts.has(key)) ngramCounts.set(key, { rock: 0, paper: 0, scissors: 0 });
    ngramCounts.get(key)[move]++;
  }
}

function predictFromNgrams(history) {
  for (let w = 5; w >= 1; w--) {
    if (history.length < w) continue;
    const key = `${w}:${ngramKey(history, w)}`;
    const counts = ngramCounts.get(key);
    if (!counts) continue;
    const total = counts.rock + counts.paper + counts.scissors;
    if (total < 2) continue;
    let best = 'rock', bestCount = counts.rock;
    if (counts.paper > bestCount) { best = 'paper'; bestCount = counts.paper; }
    if (counts.scissors > bestCount) { best = 'scissors'; bestCount = counts.scissors; }
    if (bestCount / total > 0.4) return best;
  }
  return null;
}

// --- Encoding for SONA ---

function encodeHistory(history) {
  const vec = new Float32Array(DIM);
  const start = Math.max(0, history.length - 5);
  for (let i = start; i < history.length; i++) {
    const slot = i - start;
    const moveIdx = MOVES.indexOf(history[i]);
    if (moveIdx >= 0) vec[slot * 3 + moveIdx] = 1.0;
  }
  let norm = 0;
  for (let i = 0; i < DIM; i++) norm += vec[i] * vec[i];
  if (norm > 0) {
    norm = Math.sqrt(norm);
    for (let i = 0; i < DIM; i++) vec[i] /= norm;
  } else {
    for (let i = 0; i < DIM; i++) vec[i] = 1.0 / Math.sqrt(DIM);
  }
  return vec;
}

// --- JS-side pattern store (SONA learning) ---
// Uses WASM agent for trajectory collection, JS for pattern matching + LoRA

const patternStore = [];   // { embedding: Float32Array, moveIdx: number, quality: number }
const MAX_PATTERNS = 200;

// Simple cosine similarity
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const d = Math.sqrt(na) * Math.sqrt(nb);
  return d > 0 ? dot / d : 0;
}

// Simple LoRA-like weight matrix (DIM x DIM), updated from patterns
const loraWeights = new Float32Array(DIM * DIM);  // starts at zero

function updateLoraFromPatterns() {
  // Build a simple projection from input patterns → move embeddings
  // Gradient: for each pattern, nudge weights so W*embedding points toward move one-hot
  const lr = 0.005;
  for (const pat of patternStore.slice(-50)) { // use last 50 patterns
    const target = new Float32Array(DIM);
    // Set target: one-hot for the move in the last slot position
    const slotBase = Math.min(4, Math.floor(pat.embedding.length / 3) - 1) * 3;
    target[slotBase + pat.moveIdx] = 1.0;

    // Simple outer-product update: W += lr * quality * (target - W*input) * input^T
    const Wx = new Float32Array(DIM);
    for (let i = 0; i < DIM; i++)
      for (let j = 0; j < DIM; j++)
        Wx[i] += loraWeights[i * DIM + j] * pat.embedding[j];

    for (let i = 0; i < DIM; i++) {
      const err = (target[i] - Wx[i]) * lr * pat.quality;
      for (let j = 0; j < DIM; j++)
        loraWeights[i * DIM + j] += err * pat.embedding[j];
    }
  }
}

function applyLoraJS(input) {
  const out = new Float32Array(DIM);
  for (let i = 0; i < DIM; i++)
    for (let j = 0; j < DIM; j++)
      out[i] += loraWeights[i * DIM + j] * input[j];
  return out;
}

// --- SONA-based prediction (JS pattern matching + LoRA) ---

function predictFromSona(history) {
  if (history.length < 2 || patternStore.length < 3) return null;
  try {
    const contextVec = encodeHistory(history);

    // Strategy 1: k-NN pattern search
    const scored = patternStore.map((pat, idx) => ({
      idx,
      sim: cosine(contextVec, pat.embedding),
      moveIdx: pat.moveIdx,
      quality: pat.quality,
    }));
    scored.sort((a, b) => b.sim - a.sim);
    const topK = scored.slice(0, Math.min(10, scored.length));

    // Weighted vote
    const votes = [0, 0, 0];
    for (const s of topK) {
      if (s.sim > 0.5) { // only count reasonably similar patterns
        votes[s.moveIdx] += s.sim * s.quality;
      }
    }
    const maxVote = Math.max(...votes);
    if (maxVote > 0) return MOVES[votes.indexOf(maxVote)];

    // Strategy 2: LoRA transformation
    const transformed = applyLoraJS(contextVec);
    const sims = MOVES.map((move) => {
      const hypo = [...history.slice(-4), move];
      const hypoVec = encodeHistory(hypo);
      return cosine(transformed, hypoVec);
    });
    const bestIdx = sims.indexOf(Math.max(...sims));
    if (Math.max(...sims) > 0) return MOVES[bestIdx];
  } catch (e) {
    console.warn('SONA prediction error:', e);
  }
  return null;
}

// --- Rolling accuracy helper ---

function rollingAcc(preds) {
  const recent = preds.slice(-ROLLING_WINDOW);
  if (recent.length === 0) return 0;
  return recent.filter(p => p).length / recent.length;
}

// --- Game logic ---

function determineWinner(playerMove, aiMove) {
  if (playerMove === aiMove) return 'draw';
  if (COUNTER[playerMove] === aiMove) return 'lose';
  return 'win';
}

function playRound(playerMove) {
  round++;

  // Get predictions from BOTH models independently
  const ngramPick = predictFromNgrams(moveHistory);
  const sonaPick = predictFromSona(moveHistory);

  // Combined: n-gram first, SONA fallback, then random
  let combinedPick;
  let source;
  if (ngramPick) {
    combinedPick = ngramPick;
    source = 'ngram';
  } else if (sonaPick) {
    combinedPick = sonaPick;
    source = 'sona';
  } else {
    combinedPick = MOVES[Math.floor(Math.random() * 3)];
    source = 'random';
  }

  // Store last picks for display
  models.ngram.lastPick = ngramPick;
  models.sona.lastPick = sonaPick;
  models.combined.lastPick = combinedPick;

  const aiMove = COUNTER[combinedPick];
  const result = determineWinner(playerMove, aiMove);

  if (result === 'win') scores.human++;
  else if (result === 'lose') scores.ai++;
  else scores.draw++;

  // Score each model independently
  const ngramCorrect = ngramPick === playerMove;
  const sonaCorrect = sonaPick === playerMove;
  const combinedCorrect = combinedPick === playerMove;

  models.ngram.predictions.push(ngramPick != null ? ngramCorrect : null);
  models.sona.predictions.push(sonaPick != null ? sonaCorrect : null);
  models.combined.predictions.push(combinedCorrect);

  if (ngramPick != null) { models.ngram.total++; if (ngramCorrect) models.ngram.correct++; }
  if (sonaPick != null) { models.sona.total++; if (sonaCorrect) models.sona.correct++; }
  models.combined.total++;
  if (combinedCorrect) models.combined.correct++;

  // Rolling accuracy (only count rounds where model had a prediction)
  const ngramRecent = models.ngram.predictions.slice(-ROLLING_WINDOW).filter(v => v !== null);
  const sonaRecent = models.sona.predictions.slice(-ROLLING_WINDOW).filter(v => v !== null);
  const combinedRecent = models.combined.predictions.slice(-ROLLING_WINDOW);

  const ngramRolling = ngramRecent.length > 0 ? ngramRecent.filter(Boolean).length / ngramRecent.length : NaN;
  const sonaRolling = sonaRecent.length > 0 ? sonaRecent.filter(Boolean).length / sonaRecent.length : NaN;
  const combinedRolling = combinedRecent.filter(Boolean).length / combinedRecent.length;

  models.ngram.accHistory.push(ngramRolling);
  models.sona.accHistory.push(sonaRolling);
  models.combined.accHistory.push(combinedRolling);

  // Record n-gram patterns
  recordNgram(moveHistory, playerMove);

  // Feed SONA: WASM agent for trajectory collection + JS pattern store for learning
  const contextVec = encodeHistory(moveHistory);
  const moveIdx = MOVES.indexOf(playerMove);
  const quality = combinedCorrect ? 1.0 : 0.2;

  try {
    // Record trajectory in WASM agent
    agent.processTask(contextVec, quality);
  } catch (e) {
    console.warn('WASM agent error:', e);
  }

  // Store pattern in JS pattern store (this is where learning happens)
  patternStore.push({ embedding: new Float32Array(contextVec), moveIdx, quality });
  if (patternStore.length > MAX_PATTERNS) patternStore.shift();

  // Update LoRA weights every AGGREGATE_EVERY rounds
  if (round % AGGREGATE_EVERY === 0) {
    updateLoraFromPatterns();
    learnCycles++;

    // Also aggregate in WASM coordinator (for stats/debugging)
    try {
      const state = agent.exportState();
      coordinator.aggregate(state);
      coordinator.consolidate();
      agent.clear();
    } catch (e) {
      console.warn('Coordinator sync error:', e);
    }
  }

  moveHistory.push(playerMove);

  // Stats
  let sonaStats = {
    trajectories_buffered: patternStore.length,
    patterns_stored: patternStore.length,
  };
  try {
    sonaStats.trajectories_buffered = agent.trajectoryCount() + patternStore.length;
    sonaStats.total_trajectories = coordinator.totalTrajectories();
  } catch (_) {}

  updateScoreboard();
  showRoundResult(playerMove, aiMove, combinedPick, result, combinedCorrect, source);
  updateMetrics(sonaStats);
  drawCharts();
}

// --- UI updates ---

function updateScoreboard() {
  humanScoreEl.textContent = scores.human;
  aiScoreEl.textContent = scores.ai;
  drawScoreEl.textContent = scores.draw;
}

function showRoundResult(playerMove, aiMove, predicted, result, correct, source) {
  const outcomeClass = result === 'win' ? 'win' : result === 'lose' ? 'lose' : 'draw';
  const outcomeText = result === 'win' ? 'You win!' : result === 'lose' ? 'AI wins!' : 'Draw!';
  const sourceLabel = source === 'ngram' ? 'n-gram' : source === 'sona' ? 'SONA LoRA' : 'random';
  roundResultEl.innerHTML = `
    <div class="outcome ${outcomeClass}">${EMOJI[playerMove]} vs ${EMOJI[aiMove]} — ${outcomeText}</div>
    <div class="detail">Predicted <strong>${predicted}</strong> via ${sourceLabel} ${correct ? '(correct!)' : '(wrong)'}</div>
  `;
}

function pct(n, d) { return d > 0 ? `${(n / d * 100).toFixed(1)}%` : '-'; }

function updateMetrics(sonaStats) {
  document.getElementById('m-round').textContent = round;

  // N-gram metrics
  document.getElementById('m-ngram-acc').textContent = pct(models.ngram.correct, models.ngram.total);
  document.getElementById('m-ngram-hits').textContent = `${models.ngram.correct}/${models.ngram.total}`;
  document.getElementById('m-ngram-coverage').textContent = pct(models.ngram.total, round);
  document.getElementById('m-ngram-pick').textContent = models.ngram.lastPick ?? '-';

  // SONA metrics
  document.getElementById('m-sona-acc').textContent = pct(models.sona.correct, models.sona.total);
  document.getElementById('m-sona-hits').textContent = `${models.sona.correct}/${models.sona.total}`;
  document.getElementById('m-sona-coverage').textContent = pct(models.sona.total, round);
  document.getElementById('m-sona-pick').textContent = models.sona.lastPick ?? '-';

  // Combined / engine metrics
  document.getElementById('m-combined-acc').textContent = pct(models.combined.correct, models.combined.total);
  document.getElementById('m-learn-cycles').textContent = learnCycles;
  document.getElementById('m-trajectories').textContent = sonaStats.trajectories_buffered ?? 0;
  document.getElementById('m-patterns').textContent = sonaStats.patterns_stored ?? 0;
}

// --- Chart drawing ---

function drawCharts() {
  drawMultiLineChart(accuracyCanvas, [
    { data: models.ngram.accHistory, color: '#22d3ee', label: 'N-gram' },
    { data: models.sona.accHistory, color: '#a78bfa', label: 'SONA LoRA' },
    { data: models.combined.accHistory, color: '#f8fafc', label: 'Combined', dash: [4, 3] },
  ], { min: 0, max: 1, baseline: 1/3 });

  // Cumulative accuracy (lifetime, not rolling)
  const ngramCum = cumulativeAcc(models.ngram.predictions);
  const sonaCum = cumulativeAcc(models.sona.predictions);
  const combinedCum = cumulativeAcc(models.combined.predictions);
  drawMultiLineChart(cumulativeCanvas, [
    { data: ngramCum, color: '#22d3ee', label: 'N-gram' },
    { data: sonaCum, color: '#a78bfa', label: 'SONA LoRA' },
    { data: combinedCum, color: '#f8fafc', label: 'Combined', dash: [4, 3] },
  ], { min: 0, max: 1, baseline: 1/3 });
}

function cumulativeAcc(preds) {
  const result = [];
  let correct = 0, total = 0;
  for (const p of preds) {
    if (p !== null) { total++; if (p) correct++; }
    result.push(total > 0 ? correct / total : NaN);
  }
  return result;
}

function drawMultiLineChart(canvas, series, opts) {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width;
  const H = rect.height;
  ctx.clearRect(0, 0, W, H);

  const maxLen = Math.max(...series.map(s => s.data.length));
  if (maxLen < 2) {
    ctx.fillStyle = '#334155';
    ctx.font = '11px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('Play a few rounds...', W / 2, H / 2);
    return;
  }

  const { min, max, baseline } = opts;
  const pad = { top: 16, right: 8, bottom: 16, left: 32 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  // Y-axis labels
  ctx.fillStyle = '#475569';
  ctx.font = '9px system-ui';
  ctx.textAlign = 'right';
  ctx.fillText('100%', pad.left - 4, pad.top + 4);
  ctx.fillText('0%', pad.left - 4, pad.top + plotH + 2);
  ctx.fillText('50%', pad.left - 4, pad.top + plotH / 2 + 2);

  // Baseline
  if (baseline !== undefined) {
    const by = pad.top + plotH * (1 - (baseline - min) / (max - min));
    ctx.strokeStyle = '#334155';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(pad.left, by);
    ctx.lineTo(pad.left + plotW, by);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#475569';
    ctx.textAlign = 'left';
    ctx.fillText('random', pad.left + 4, by - 3);
  }

  // Draw each series
  const maxPoints = 100;
  for (const s of series) {
    const raw = s.data.length > maxPoints ? s.data.slice(-maxPoints) : s.data;
    const safe = raw.map(v => (isNaN(v) || v === null) ? null : Math.max(min, Math.min(max, v)));

    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.dash ? 1.2 : 1.5;
    ctx.lineJoin = 'round';
    if (s.dash) ctx.setLineDash(s.dash);
    else ctx.setLineDash([]);

    ctx.beginPath();
    let started = false;
    for (let i = 0; i < safe.length; i++) {
      if (safe[i] === null) { started = false; continue; }
      const x = pad.left + (i / (safe.length - 1)) * plotW;
      const y = pad.top + plotH * (1 - (safe[i] - min) / (max - min));
      if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Legend
  ctx.font = '9px system-ui';
  let lx = pad.left + 2;
  for (const s of series) {
    ctx.fillStyle = s.color;
    ctx.fillRect(lx, pad.top - 12, 12, 2);
    if (s.dash) {
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(lx + 4, pad.top - 12, 3, 2);
    }
    ctx.fillStyle = s.color;
    ctx.textAlign = 'left';
    ctx.fillText(s.label, lx + 15, pad.top - 8);
    lx += ctx.measureText(s.label).width + 28;
  }
}

// --- Init ---

async function main() {
  try {
    await initSonaWasm();
    try { wasm_init(); } catch (_) {}

    const sonaConfig = {
      hidden_dim: DIM,
      embedding_dim: DIM,
      micro_lora_rank: 2,
      base_lora_rank: 4,
      micro_lora_lr: 0.01,
      base_lora_lr: 0.001,
      ewc_lambda: 100.0,
      pattern_clusters: 8,
      trajectory_capacity: 1000,
      background_interval_ms: 1000,
      quality_threshold: 0.1,
      enable_simd: true,
    };

    // Create EphemeralAgent (trajectory collection)
    try {
      agent = WasmEphemeralAgent.withConfig('rps-agent', JSON.stringify(sonaConfig));
    } catch (e) {
      console.warn('Agent withConfig failed, using defaults:', e);
      agent = new WasmEphemeralAgent('rps-agent');
    }
    console.log('Agent created, stats:', agent.getStats());

    // Create FederatedCoordinator (learning + prediction)
    try {
      coordinator = WasmFederatedCoordinator.withConfig('rps-coord', JSON.stringify(sonaConfig));
    } catch (e) {
      console.warn('Coordinator withConfig failed, using defaults:', e);
      coordinator = new WasmFederatedCoordinator('rps-coord');
    }
    coordinator.setQualityThreshold(0.1); // accept most trajectories for RPS
    console.log('Coordinator created, stats:', coordinator.getStats());

    // Keep engine for reference/comparison
    try {
      engine = WasmSonaEngine.withConfig(JSON.stringify(sonaConfig));
    } catch (e) {
      engine = new WasmSonaEngine(DIM);
    }

    window.__sonaAgent = agent;
    window.__sonaCoordinator = coordinator;
    window.__sonaEngine = engine;

    btnRock.disabled = false;
    btnPaper.disabled = false;
    btnScissors.disabled = false;

    statusEl.textContent = 'Ready — pick your move! (keys: 1/r, 2/p, 3/s)';
    drawCharts();
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
    console.error(err);
  }
}

btnRock.addEventListener('click', () => playRound('rock'));
btnPaper.addEventListener('click', () => playRound('paper'));
btnScissors.addEventListener('click', () => playRound('scissors'));

document.addEventListener('keydown', (e) => {
  if (btnRock.disabled) return;
  if (e.key === '1' || e.key.toLowerCase() === 'r') playRound('rock');
  if (e.key === '2' || e.key.toLowerCase() === 'p') playRound('paper');
  if (e.key === '3' || e.key.toLowerCase() === 's') playRound('scissors');
});

main();
