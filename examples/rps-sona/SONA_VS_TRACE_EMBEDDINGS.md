# SONA vs. Trace Embedding / Representation Learning Approaches

A comparison of the SONA adaptive learning library (as used in this RPS demo) with established trace embedding and representation learning methods from process mining and NLP.

## What SONA does in this codebase

1. **Encoding**: The last 5 discrete events (moves) are one-hot encoded into a 15-dim vector (5 slots x 3 categories), then L2-normalized. This is the "trace embedding."
2. **Pattern store**: Each (embedding, move_index, quality) tuple is stored in a bounded buffer (max 200). This is the memory.
3. **Prediction via k-NN**: At inference, cosine similarity finds the 10 nearest stored patterns; a quality-weighted vote over their follow-up moves yields the prediction.
4. **LoRA weight matrix**: A 15x15 learned linear projection (outer-product gradient updates, lr=0.005) maps context embeddings toward one-hot move targets. Trained every 5 rounds on the last 50 patterns. Acts as a generalization layer when k-NN fails.
5. **WASM trajectory collection**: The Rust/WASM side (`WasmEphemeralAgent` + `WasmFederatedCoordinator`) buffers trajectories and does federated aggregation with quality filtering + EWC++ to prevent catastrophic forgetting, but the actual prediction in this demo is JS-side k-NN + LoRA.

## Contrast with established trace embedding methods

| Dimension | **SONA (rps-sona)** | **Trace2Vec** (De Koninck et al. 2018) | **Act2Vec / Event2Vec** | **Doc2Vec** (Le & Mikolov 2014) |
|---|---|---|---|---|
| **Unit of embedding** | Fixed-length sliding window (last *k* events) | Entire trace (variable-length activity sequence) | Individual activities | Entire document |
| **Embedding method** | Hand-crafted one-hot concatenation + L2 norm | Neural: Word2Vec/Doc2Vec adapted to event logs (PV-DM or PV-DBOW) | Neural: Skip-gram or CBOW over activity co-occurrences | Neural: Paragraph vector (PV-DM / PV-DBOW) |
| **Dimensionality** | Deterministic: `window * |alphabet|` (here 15) | Learned: typically 50-200 | Learned: typically 50-100 | Learned: typically 100-400 |
| **Training regime** | **Online / incremental** — updates every N rounds | **Batch / offline** — trained on a corpus of completed traces | **Batch / offline** | **Batch / offline** |
| **What co-occurrence means** | Not used — positions are absolute slots in the window | Activities that appear in similar trace contexts | Activities that co-occur within a trace window | Words that co-occur within document context |
| **Similarity metric** | Cosine similarity over hand-crafted vectors | Cosine similarity over learned dense vectors | Cosine similarity over learned dense vectors | Cosine similarity over learned dense vectors |
| **Generalization** | Limited — identical positions must align; LoRA adds a linear transform | Strong — similar traces get nearby vectors even with activity reordering or substitution | Strong — semantically related activities cluster | Strong — documents with similar content cluster |
| **Catastrophic forgetting** | EWC++ (Elastic Weight Consolidation) in WASM layer | N/A (retrain on full corpus) | N/A | N/A |
| **Adaptation speed** | Immediate (k-NN) to gradual (LoRA) | Requires retraining or incremental PV-DM | Requires retraining | Requires retraining or inference-time optimization |
| **Computational model** | k-NN retrieval + linear transform | Embedding lookup (after training) | Embedding lookup | Embedding lookup |

## Key architectural differences

### 1. Learned vs. hand-crafted representations

SONA's encoding is a *deterministic function* — there is no unsupervised representation learning. The 15-dim vector is a flattened one-hot window. This is closer to a **feature engineering** approach (like a bag-of-n-grams one-hot) than to representation learning. Trace2Vec and Doc2Vec learn a *distributed representation* where the dimensions have no predetermined meaning but capture statistical regularities. The consequence: SONA cannot discover that two sequences are semantically similar if they don't share positional alignment. Doc2Vec/Trace2Vec can.

The LoRA matrix partially compensates — it learns a linear projection from the hand-crafted space. But it's a 15x15 linear map, not a nonlinear embedding. It can rotate and scale the space but cannot discover latent structure the way a neural embedding does.

### 2. Online vs. batch learning

This is SONA's genuine differentiator. Trace2Vec, Act2Vec, and Doc2Vec all assume a fixed corpus. You train once (or periodically retrain), and inference uses frozen embeddings. SONA learns continuously: every round adds to the pattern store and periodically updates the LoRA weights. For non-stationary processes (a player changing strategy, a business process evolving), this matters. The process mining literature addresses this via concept drift detection + retraining, but it's not built into the embedding layer.

### 3. Trajectory vs. trace

In process mining, a *trace* is a complete case execution (start to finish). In SONA, a "trajectory" is a sliding window — the last *k* observations plus a quality score. This is closer to **prefix-based prediction** (as in LSTM next-event prediction, e.g., Tax et al. 2017) than to full-trace embedding. SONA never represents a complete trace; it only sees a local window.

### 4. Federated / multi-agent architecture

SONA's `WasmEphemeralAgent` → `WasmFederatedCoordinator` pattern has no analogue in trace2vec/doc2vec. Those are single-model approaches. SONA's architecture is designed for distributed learning where multiple agents collect trajectories independently and a coordinator aggregates. This is relevant for multi-tenant or privacy-preserving settings, but orthogonal to the representation quality question.

### 5. Quality-weighted learning

SONA attaches a quality score to each trajectory and uses it to weight both k-NN votes and gradient updates. This is a form of **reward-weighted regression** — closer to reinforcement learning than to the unsupervised objectives of Doc2Vec/Trace2Vec. In process mining, the analogue would be cost-aware trace clustering or outcome-oriented predictive monitoring.

## Where each approach is stronger

| Scenario | Best fit |
|---|---|
| Small alphabet, short sequences, real-time adaptation | **SONA** — online learning, immediate k-NN, no training pipeline |
| Large event logs, offline analysis, trace clustering | **Trace2Vec** — learned embeddings capture semantic similarity across variable-length traces |
| Understanding which activities are interchangeable | **Act2Vec** — activity-level embeddings reveal substitution patterns |
| Rich textual attributes on events | **Doc2Vec** — handles arbitrary text, not just categorical activities |
| Non-stationary process, concept drift | **SONA** — continuous learning with EWC++ prevents forgetting while adapting |
| Scalability to thousands of distinct activities | **Trace2Vec / Act2Vec** — embedding dimension is independent of alphabet size; SONA's one-hot scales as O(window x alphabet) |

## The fundamental tradeoff

SONA trades **representational power** for **online adaptability**. It uses a shallow, deterministic encoding where trace2vec/doc2vec use deep learned representations. But it can update in-browser, in real-time, with no training pipeline, no GPU, and no batch of historical data. The LoRA layer is an attempt to bridge this gap — adding a learned linear transform on top of hand-crafted features — but it remains limited compared to a full neural embedding.

For the RPS use case (3 symbols, short memory), this tradeoff is favorable. For process mining at scale (hundreds of activities, thousands of trace variants, complex control flow), the hand-crafted encoding would break down and learned embeddings would dominate.
