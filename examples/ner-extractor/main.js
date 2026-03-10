import { pipeline, env } from '@huggingface/transformers';

// Use remote models from HuggingFace Hub (ONNX)
env.allowLocalModels = false;

const inputEl = document.getElementById('input');
const extractBtn = document.getElementById('extract-btn');
const statusEl = document.getElementById('status');
const annotatedEl = document.getElementById('annotated');
const entityListEl = document.getElementById('entity-list');
const progressBar = document.getElementById('progress');
const progressFill = document.getElementById('progress-fill');

let nerPipeline = null;

// --- Model loading ---

async function loadModel() {
  statusEl.textContent = 'Henter model (~180 MB, caches efter f\u00f8rste gang)...';
  progressBar.classList.add('active');
  progressFill.style.width = '10%';

  try {
    nerPipeline = await pipeline(
      'token-classification',
      'thomasbeste/modernbert-da-ner-base-onnx-int8',
      {
        progress_callback: (progress) => {
          if (progress.status === 'progress' && progress.progress) {
            const pct = Math.round(progress.progress);
            progressFill.style.width = `${pct}%`;
            statusEl.textContent = `Henter: ${progress.file} (${pct}%)`;
          } else if (progress.status === 'ready') {
            progressFill.style.width = '100%';
          }
        },
      },
    );

    progressBar.classList.remove('active');
    statusEl.textContent = 'Model klar. Inds\u00e6t tekst og klik Find entiteter.';
    extractBtn.disabled = false;
  } catch (err) {
    progressBar.classList.remove('active');
    statusEl.textContent = `Fejl ved indl\u00e6sning: ${err.message}`;
    console.error(err);
  }
}

// --- Entity extraction ---

async function extractEntities() {
  const text = inputEl.value.trim();
  if (!text || !nerPipeline) return;

  extractBtn.disabled = true;
  statusEl.textContent = 'Analyserer tekst...';

  const start = performance.now();

  try {
    const rawEntities = await nerPipeline(text);

    const elapsed = performance.now() - start;

    // Debug: log first 3 raw entities
    for (let i = 0; i < Math.min(3, rawEntities.length); i++) {
      console.log(`Raw entity [${i}]:`, JSON.stringify(rawEntities[i]));
    }

    // Normalize + aggregate entities
    const entities = aggregateEntities(rawEntities, text);

    renderAnnotated(text, entities);
    renderEntityList(entities);
    statusEl.textContent = `Fandt ${entities.length} entiteter p\u00e5 ${elapsed.toFixed(0)}ms`;
  } catch (err) {
    statusEl.textContent = `Fejl: ${err.message}`;
    console.error(err);
  }

  extractBtn.disabled = false;
}

// --- Entity aggregation (BIO tags → grouped spans) ---

function aggregateEntities(rawEntities, text) {
  const result = [];
  let current = null;

  for (const tok of rawEntities) {
    const label = tok.entity || tok.entity_group || '';
    const score = tok.score || 0;
    const word = tok.word || '';

    // Skip O (outside), low-confidence, or special tokens
    if (label === 'O' || label === '' || score < 0.5) {
      if (current) { result.push(current); current = null; }
      continue;
    }

    const isBegin = label.startsWith('B-');
    const isCont = label.startsWith('I-');
    const entityType = label.replace(/^[BI]-/, '');

    // Determine the token text — handle subword tokens (##prefix)
    const isSubword = word.startsWith('##');
    const cleanWord = isSubword ? word.slice(2) : word;

    // Use character offsets if available, otherwise build from words
    const hasOffsets = typeof tok.start === 'number' && typeof tok.end === 'number';

    if (isBegin || (!isCont && !current) || (current && current.entity_group !== entityType && !isSubword)) {
      if (current) result.push(current);
      current = {
        entity_group: entityType,
        word: hasOffsets ? text.slice(tok.start, tok.end) : cleanWord,
        start: hasOffsets ? tok.start : -1,
        end: hasOffsets ? tok.end : -1,
        score: score,
        tokenCount: 1,
      };
    } else if (current) {
      // Continue: append word to current entity
      if (hasOffsets) {
        current.word = text.slice(current.start, tok.end);
        current.end = tok.end;
      } else {
        // No offsets: concatenate words, subwords without space
        current.word += isSubword ? cleanWord : (' ' + cleanWord);
      }
      current.score = (current.score * current.tokenCount + score) / (current.tokenCount + 1);
      current.tokenCount++;
    }
  }

  if (current) result.push(current);

  // If we don't have character offsets, find them by searching in the source text
  if (result.length > 0 && result[0].start === -1) {
    resolveOffsets(result, text);
  }

  return result;
}

function resolveOffsets(entities, text) {
  let searchFrom = 0;
  for (const ent of entities) {
    const idx = text.indexOf(ent.word, searchFrom);
    if (idx !== -1) {
      ent.start = idx;
      ent.end = idx + ent.word.length;
      searchFrom = ent.end;
    } else {
      // Try case-insensitive or partial match
      const lowerIdx = text.toLowerCase().indexOf(ent.word.toLowerCase(), searchFrom);
      if (lowerIdx !== -1) {
        ent.start = lowerIdx;
        ent.end = lowerIdx + ent.word.length;
        ent.word = text.slice(lowerIdx, lowerIdx + ent.word.length); // Use original casing
        searchFrom = ent.end;
      }
    }
  }
}

// --- Rendering ---

function renderAnnotated(text, entities) {
  if (entities.length === 0) {
    annotatedEl.innerHTML = escapeHtml(text);
    return;
  }

  // Sort by start position
  const sorted = [...entities].sort((a, b) => a.start - b.start);

  let html = '';
  let lastEnd = 0;

  for (const ent of sorted) {
    // Add text before this entity
    if (ent.start > lastEnd) {
      html += escapeHtml(text.slice(lastEnd, ent.start));
    }

    const label = ent.entity_group;
    const word = text.slice(ent.start, ent.end);

    html += `<span class="entity entity-${label}" title="${label}: ${(ent.score * 100).toFixed(1)}%">`
      + escapeHtml(word)
      + `<span class="entity-label">${label}</span></span>`;

    lastEnd = ent.end;
  }

  // Remaining text
  if (lastEnd < text.length) {
    html += escapeHtml(text.slice(lastEnd));
  }

  annotatedEl.innerHTML = html;
}

function renderEntityList(entities) {
  if (entities.length === 0) {
    entityListEl.innerHTML = '<div style="color:#4a4570;font-size:0.85rem;padding:0.5rem;">Ingen entiteter fundet</div>';
    return;
  }

  entityListEl.innerHTML = entities
    .map(e => {
      const label = e.entity_group;
      return `<div class="entity-row">
        <span class="tag tag-${label}">${label}</span>
        <span class="word">${escapeHtml(e.word)}</span>
        <span class="conf">${(e.score * 100).toFixed(1)}%</span>
      </div>`;
    })
    .join('');
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// --- Event handlers ---

extractBtn.addEventListener('click', extractEntities);

inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && e.ctrlKey) {
    extractEntities();
  }
});

// Example buttons
document.querySelectorAll('.btn-example').forEach(btn => {
  btn.addEventListener('click', () => {
    inputEl.value = btn.dataset.text;
    if (nerPipeline) extractEntities();
  });
});

// Start loading
loadModel();
