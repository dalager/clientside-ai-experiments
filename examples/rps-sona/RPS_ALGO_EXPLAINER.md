# How the AI Predicts Your Moves

This game uses two independent prediction systems running side-by-side. Both try to guess what you'll play next, but they work in fundamentally different ways.

## N-gram Predictor

**The idea:** Remember what you did after specific sequences of moves, then bet you'll do it again.

The n-gram predictor keeps a tally sheet. Every time you play, it records what move followed what sequence. If you've played rock-paper three times and followed it with scissors twice and rock once, then next time you play rock-paper, it predicts scissors.

It checks sequences of different lengths (1 to 5 moves) and trusts longer matches more. A 5-move match is more specific than a 1-move match, so it checks the longest patterns first.

**When it stays quiet:** It only speaks up when it has seen a sequence at least twice and the most common follow-up accounts for more than 40% of observations. Otherwise it abstains — it would rather say nothing than guess poorly.

**Think of it as:** A card counter at a poker table. It doesn't understand *why* you play a certain way — it just tracks frequencies and spots repetition.

## SONA LoRA Predictor

**The idea:** Convert game situations into points in a geometric space, then find similar past situations and learn a transformation that maps contexts to likely moves.

Each round, the last five moves are encoded as a 15-dimensional vector (5 slots x 3 possible moves, one-hot encoded and normalized). This gives each game state a unique "fingerprint" — a point in space where similar game situations land near each other.

Two mechanisms work together:

1. **Pattern search (k-NN):** When predicting, it finds the 10 stored patterns most similar to the current situation (using cosine similarity). If past situations that look like this one were followed by rock, it predicts rock. This is a "memory recall" step — finding analogies in experience.

2. **LoRA weight matrix:** A 15x15 weight matrix is trained to transform context embeddings toward the move that actually followed. Every 5 rounds, the matrix is updated using a simple gradient step across recent patterns. Over time, this matrix learns a general mapping from "what the game looks like" to "what the player will do." This is a "generalization" step — learning abstract rules from data.

**When it stays quiet:** Only when fewer than 3 patterns are stored (the first couple of rounds).

**Think of it as:** A chess player who both remembers specific positions they've seen before *and* develops intuition about what kinds of positions lead to what kinds of moves.

## How They Combine

The game uses both predictions through a simple priority:

1. If the n-gram predictor has a confident prediction, use it
2. Otherwise, if SONA has a prediction, use it
3. Otherwise, pick randomly

The AI then plays the counter to the predicted move (if it predicts you'll play rock, it plays paper).

## Comparison

| Dimension | N-gram | SONA LoRA |
|-----------|--------|-----------|
| **How it learns** | Exact sequence counting | Geometric similarity + weight training |
| **Memory model** | Lookup table of sequences | Vector store + learned transformation |
| **Coverage** | Low-medium (needs exact pattern matches) | High (generalizes to unseen situations) |
| **Accuracy on repetitive play** | Very high — catches loops fast | High but slower to converge |
| **Accuracy on varied play** | Drops off — needs repeated sequences | Holds better — finds partial similarities |
| **Speed to first prediction** | 2-3 rounds | 3 rounds |
| **Adapts to strategy change** | Fast (old counts diluted by new ones) | Gradual (weight matrix retains old learning) |
| **Computational cost** | Negligible (hash map lookups) | Moderate (vector math, matrix multiply) |

### Where N-gram Wins

- **Simple, repetitive opponents.** If someone plays rock-paper-scissors-rock-paper-scissors in a loop, the n-gram predictor catches on in about 6 moves and holds near-perfect accuracy. It's purpose-built for this.
- **Speed.** A hash map lookup is effectively instant. No matrix math, no distance calculations.
- **Transparency.** You can inspect the tally sheet and understand exactly why it predicted what it did.

### Where SONA Would Win

- **High input variance.** When the player mixes strategies and rarely repeats exact sequences, n-gram runs out of data. SONA can still find *similar* (not identical) past situations and extrapolate. In a 3-move game this advantage is modest, but in a game with 10+ possible actions it would be significant.

- **Complex state spaces.** Rock-paper-scissors has only 3 moves and short meaningful sequences. In a game with richer state — say, a card game with hundreds of possible hands, or a strategy game with spatial positioning — the combinatorial explosion makes exact sequence matching impractical. SONA's embedding space compresses this complexity into fixed-size vectors where similarity still works.

- **Transfer learning.** The LoRA weight matrix captures *general* patterns about how contexts relate to moves. In principle, weights trained on one player could partially transfer to another player with a similar style. N-gram counts are player-specific and don't transfer at all.

- **Scaling with data.** N-gram prediction quality plateaus once common sequences are well-counted. SONA's weight matrix continues to improve with more data — the gradient updates refine the transformation. With thousands of rounds, SONA's generalization advantage compounds.

- **Multi-dimensional inputs.** If the game included timing data, mouse movement, hesitation duration, or other signals beyond just which button was pressed, SONA can encode all of it into the embedding vector. N-gram can only work with discrete move sequences.

### The Practical Takeaway

For rock-paper-scissors specifically, the n-gram predictor is the better tool. The state space is tiny (3 moves), sequences are short, and exact matching works. SONA's machinery is overkill here.

But this demo exists to show *how* the SONA approach works in a setting simple enough to understand and verify. The same pattern-search-plus-learned-transformation architecture scales to problems where n-gram would drown in combinatorial complexity — and that's where it earns its 227 KB of WebAssembly.
