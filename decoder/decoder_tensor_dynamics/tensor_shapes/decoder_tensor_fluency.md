## Model Context

* Model type: Transformer decoder-only
* Task: Next-token prediction
* Mode: Autoregressive generation
* Focus: Mechanics, not semantics

The objective is simple and strict: Given an input sentence, I can mentally trace how data flows through the decoder from start to end, including every tensor shape, without running code.

## Fixed Reference Setup

To avoid ambiguity, all reasoning is done with the following fixed assumptions:

* Batch size: 1
* Sequence length: 7
* Maximum sequence length: 32
* Model dimension: 64
* Vocabulary size: 20,000
* Number of attention heads: 4
* Head dimension: 16

## 1. Input Tensor

The input consists of token IDs.

* Shape: `(batch, sequence)`
* Shape used here: `1 × 7`

Each value is an integer index into the vocabulary.

## 2. Token Embeddings

* Embedding table shape: `(vocab_size, model_dim)` = `20,000 × 64`
* After lookup, embeddings shape: `1 × 7 × 64`

Each token ID selects a 64-dimensional vector from the embedding table.

## 3. Positional Embeddings

* Positional embedding table shape: `(max_seq_len, model_dim)` = `32 × 64`
* Positional embeddings used: `7 × 64`
* After addition to token embeddings: `1 × 7 × 64`

Positional information is injected without changing tensor shape.

## 4. Multi-Head Self-Attention (Decoder)

### Q, K, V Projections

* Projection weight shapes: `64 × 64`
* Query, key, value tensors: `1 × 7 × 64`

### Head Splitting

* After reshaping: `1 × 7 × 4 × 16`
* After transpose: `1 × 4 × 7 × 16`

Each head operates independently on a 16-dimensional subspace.

### Attention Scores

* Score computation per head: `(7 × 16) × (16 × 7) = 7 × 7`
* Full attention score tensor: `1 × 4 × 7 × 7`

Rows correspond to query tokens.
Columns correspond to key tokens.

### Causal Masking

* Mask shape: `1 × 1 × 7 × 7`
* Masked positions: all future tokens where `j > i`

This ensures autoregressive behavior and prevents future information leakage.

### Attention Weights

* Softmax applied along the last dimension
* Attention weights shape: `1 × 4 × 7 × 7`
* Each row sums to 1
* Future positions receive zero probability

### Attention Output

* Single head output: `7 × 16`
* All heads combined: `1 × 4 × 7 × 16`
* After concatenation: `1 × 7 × 64`
* Output projection: `64 × 64`
* Final attention output: `1 × 7 × 64`

## 5. Layer Normalization

Layer normalization is applied independently to each token.

* Normalized dimension: 64
* Learnable parameters:

  * Gamma: `(64)`
  * Beta: `(64)`

Formula:

* x̂ = (x − mean) / sqrt(variance + epsilon)
* y = gamma × x̂ + beta

The tensor shape remains `1 × 7 × 64`.

## 6. Feed-Forward Network (FFN)

The FFN is applied to each token independently.

* Expansion layer weight matrix: `64 × 256`
* Projection layer weight matrix: `256 × 64`

Shapes:

* Expansion: `(1 × 7 × 64) × (64 × 256) = 1 × 7 × 256`
* Projection: `(1 × 7 × 256) × (256 × 64) = 1 × 7 × 64`

Another layer normalization follows, preserving shape.

## 7. Output Projection (Logits)

* Logit weight matrix: `64 × 20,000`
* Final decoder output: `1 × 7 × 20,000`

Each position now has a score for every vocabulary token.

## 8. Attention Matrix Interpretation

Given an attention matrix:

* Each row corresponds to a query token
* Each column corresponds to a key token
* The matrix is square because attention operates within the same sequence
* Rows sum to 1 due to softmax
* The decoder differs from the encoder by enforcing causal masking

## 9. Attention Tensor Indexing (PyTorch)

Assume attention weights shape: `(batch, heads, sequence, sequence)`.

* `weights[0, h]`  attention matrix for one head
* `weights[0, :, -1]`  all heads for the last query token
* `weights[0, h, i]`  attention distribution of token `i`
* `weights[0, h, i, j]`  attention from token `i` to token `j`

## 10. Masking Logic

* All future positions must be masked
* A single off-by-one error causes information leakage
* Masking must be applied before softmax
* Padding masks ignore non-content tokens
* Causal masks enforce temporal order
