import initWasm, {
  init as initRuntime,
  ChatTemplateWasm,
  ChatMessageWasm,
  MicroLoraConfigWasm,
  MicroLoraWasm,
  AdaptFeedbackWasm,
  SonaConfigWasm,
  SonaInstantWasm,
  KvCacheConfigWasm,
  KvCacheWasm,
  BufferPoolWasm,
  InferenceArenaWasm,
  healthCheck,
  isReady,
  getVersion,
  feature_summary,
} from '@ruvector/ruvllm-wasm';

const statusEl = document.getElementById('status');
const healthOutput = document.getElementById('health-output');
const templateSelect = document.getElementById('template-select');
const systemMsg = document.getElementById('system-msg');
const userMsg = document.getElementById('user-msg');
const formatBtn = document.getElementById('format-btn');
const templateOutput = document.getElementById('template-output');
const loraOutput = document.getElementById('lora-output');
const sonaOutput = document.getElementById('sona-output');

function log(msg) {
  console.log(`[ruvllm] ${msg}`);
}

// --- Chat Templates ---

function getTemplate(name) {
  switch (name) {
    case 'llama3': return ChatTemplateWasm.llama3();
    case 'mistral': return ChatTemplateWasm.mistral();
    case 'chatml': return ChatTemplateWasm.chatml();
    case 'phi': return ChatTemplateWasm.phi();
    case 'gemma': return ChatTemplateWasm.gemma();
    default: return ChatTemplateWasm.llama3();
  }
}

function formatChat() {
  try {
    const template = getTemplate(templateSelect.value);
    const messages = [];
    const sys = systemMsg.value.trim();
    if (sys) messages.push(ChatMessageWasm.system(sys));
    const usr = userMsg.value.trim();
    if (usr) messages.push(ChatMessageWasm.user(usr));
    templateOutput.textContent = template.format(messages);
  } catch (err) {
    templateOutput.textContent = `Error: ${err.message}`;
    console.error('format error:', err);
  }
}

// --- Feature probes ---

function probeMicroLoRA() {
  const dim = 64;
  const config = new MicroLoraConfigWasm();
  config.rank = 2;
  config.inFeatures = dim;
  config.outFeatures = dim;

  const lora = new MicroLoraWasm(config);

  // Create a simple input vector
  const input = new Float32Array(dim);
  for (let i = 0; i < dim; i++) input[i] = Math.sin(i * 0.1);

  const adapted = lora.apply(input);

  // Send quality feedback: adapt(input, feedback)
  const feedback = new AdaptFeedbackWasm(0.85);
  lora.adapt(input, feedback);
  lora.applyUpdates(0.01);

  // stats() returns MicroLoraStatsWasm object, not JSON
  const stats = lora.stats();

  return {
    ok: true,
    inputSample: Array.from(input.slice(0, 4)).map(v => v.toFixed(4)),
    outputSample: Array.from(adapted.slice(0, 4)).map(v => v.toFixed(4)),
    paramCount: stats.paramCount,
    samplesSeen: stats.samplesSeen,
    memoryBytes: stats.memoryBytes,
  };
}

function probeSona() {
  const dim = 64;
  const config = new SonaConfigWasm();
  config.hiddenDim = dim;
  const sona = new SonaInstantWasm(config);

  const input = new Float32Array(dim);
  for (let i = 0; i < dim; i++) input[i] = Math.cos(i * 0.15);

  const result = sona.instantAdapt(input, 0.9);

  // Record a pattern
  const pattern = new Float32Array(dim);
  for (let i = 0; i < dim; i++) pattern[i] = Math.sin(i * 0.2);
  sona.recordPattern(pattern, "test-agent", true);

  // stats() returns SonaStatsWasm object
  const stats = sona.stats();

  return {
    ok: true,
    applied: result.applied,
    qualityEma: result.qualityEma.toFixed(4),
    currentRank: result.currentRank,
    adaptations: stats.adaptations.toString(),
    patternsRecorded: stats.patternsRecorded.toString(),
    avgQuality: stats.avgQuality.toFixed(4),
  };
}

function probeKvCache() {
  const config = new KvCacheConfigWasm();
  const cache = new KvCacheWasm(config);

  // stats() returns KvCacheStatsWasm object
  const stats = cache.stats();

  return {
    ok: true,
    totalTokens: stats.totalTokens,
    tailTokens: stats.tailTokens,
    storeTokens: stats.storeTokens,
    compressionRatio: stats.compressionRatio,
  };
}

function probeMemory() {
  const results = {};

  try {
    const pool = new BufferPoolWasm();
    pool.prewarmAll(2);
    const poolStats = JSON.parse(pool.statsJson());
    results.bufferPool = { ok: true, hitRate: pool.hitRate, stats: poolStats };
  } catch (err) {
    results.bufferPool = { ok: false, error: err.message };
  }

  try {
    const arena = new InferenceArenaWasm(1024);
    results.arena = { ok: true, size: 1024 };
  } catch (err) {
    results.arena = { ok: false, error: err.message };
  }

  return results;
}

// --- Init ---

async function main() {
  const results = {};

  // Step 1: Load WASM
  try {
    statusEl.textContent = 'Loading WASM binary...';
    await initWasm();
    initRuntime();
    log('WASM loaded + runtime initialized');
  } catch (err) {
    statusEl.textContent = `WASM load failed: ${err.message}`;
    console.error(err);
    return;
  }

  // Step 2: Health check
  try {
    const ok = healthCheck();
    const ready = isReady();
    let version = ''; try { version = getVersion(); } catch (_) {}
    let summary = ''; try { summary = feature_summary(); } catch (_) {}

    results.health = ok;
    healthOutput.innerHTML = [
      ok ? '<span class="badge badge-ok">WASM OK</span>' : '<span class="badge badge-err">FAIL</span>',
      `ready=${ready}`,
      version ? `v${version}` : '',
      summary ? `<br><pre style="margin-top:0.5rem;font-size:0.7rem">${summary}</pre>` : '',
    ].filter(Boolean).join(' ');
    log(`healthCheck=${ok}, ready=${ready}, version=${version}`);
  } catch (err) {
    healthOutput.innerHTML = `<span class="badge badge-err">FAIL</span> ${err.message}`;
    console.error(err);
  }

  // Step 3: Chat templates
  try {
    templateSelect.disabled = false;
    systemMsg.disabled = false;
    userMsg.disabled = false;
    formatBtn.disabled = false;
    formatChat();
    results.chat = true;
    log('Chat templates OK');
  } catch (err) {
    templateOutput.textContent = `Error: ${err.message}`;
    console.error(err);
  }

  // Step 4: MicroLoRA
  try {
    const lora = probeMicroLoRA();
    results.lora = true;
    loraOutput.innerHTML = `<span class="badge badge-ok">OK</span> params=${lora.paramCount}, samples=${lora.samplesSeen}, mem=${lora.memoryBytes}B
input:  [${lora.inputSample.join(', ')}, ...]
output: [${lora.outputSample.join(', ')}, ...]`;
    log('MicroLoRA OK');
  } catch (err) {
    results.lora = false;
    loraOutput.innerHTML = `<span class="badge badge-err">FAIL</span> ${err.message}`;
    console.error('MicroLoRA error:', err);
  }

  // Step 5: SONA
  try {
    const sona = probeSona();
    results.sona = true;
    sonaOutput.innerHTML = `<span class="badge badge-ok">OK</span> adapted=${sona.applied}, quality=${sona.qualityEma}, rank=${sona.currentRank}
adaptations=${sona.adaptations}, patterns=${sona.patternsRecorded}, avgQuality=${sona.avgQuality}`;
    log('SONA OK');
  } catch (err) {
    results.sona = false;
    sonaOutput.innerHTML = `<span class="badge badge-err">FAIL</span> ${err.message}`;
    console.error('SONA error:', err);
  }

  // Step 6: KV Cache + memory
  try {
    const kv = probeKvCache();
    log(`KV Cache OK: ${JSON.stringify(kv.stats)}`);
  } catch (err) {
    log(`KV Cache error: ${err.message}`);
  }
  try {
    const mem = probeMemory();
    log(`Memory probes: ${JSON.stringify(mem)}`);
  } catch (err) {
    log(`Memory probe error: ${err.message}`);
  }

  const working = Object.entries(results).filter(([,v]) => v === true).map(([k]) => k);
  statusEl.textContent = `Ready — ${working.length} features loaded: ${working.join(', ')}`;
}

// --- Event listeners ---
formatBtn.addEventListener('click', formatChat);
templateSelect.addEventListener('change', formatChat);

main();
