## Model Context

The code in this folder implements a **decoder-only Transformer** with a **shallow architecture (2 layers)**.

The model contains all essential components of a standard decoder:

* token embeddings
* positional encoding
* self-attention
* feed-forward network (FFN)
* residual connections
* layer normalization
* final logit projection

The model is trained on **WikiText**, using the **GPT-2 tokenizer only for token ID conversion**.
At its current stage, the model has a loss of approximately **6.2**, which reflects an **early learning phase**. This makes it suitable for studying *mechanical behavior* rather than language quality.

## Aim of This Study

The high-level task of the decoder is next-token prediction. However, the goal here is **not** to evaluate predictions or semantics. Instead, the focus is narrowed to a **single token inside a sentence**.

For a given input sentence, I track **one token** and observe:

* how its representation changes at each layer
* how attention, FFN, residuals, and normalization affect it
* how much it drifts from its original embedding

This drift is measured using **cosine similarity** between:

* embedding → layer 1
* layer 1 → layer 2
* embedding → final layer
(observe cells 17 and 18)

## Why This Matters

By isolating one token and following it through the decoder, the layers stop feeling abstract or mysterious.

Each decoder layer can be treated as a **geometric transformation** applied to a vector:

* attention reshapes context
* FFN transforms features
* residuals preserve structure
* layer normalization stabilizes scale

After this exercise, it becomes possible to:

* inspect any token in a sentence
* see which layer affected it most
* compare how far it moved from its original representation

This builds intuition for how decoder layers operate **before** thinking about meaning or language understanding.
