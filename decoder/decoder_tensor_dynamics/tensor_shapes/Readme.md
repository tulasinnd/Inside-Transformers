# Decoder Tensor Fluency & Mental Simulation

## Why This Document Exists

Before thinking about meaning, language, or semantics, the decoder needs to become **mentally transparent**.

This document focuses on building fluency with:

* tensor shapes
* data flow from input to output
* where each operation happens inside the decoder

The goal is simple:
given an input sentence, I should be able to **mentally trace** how tensors move through the decoder layers and produce the final output — without running the code.

To build this fluency, I start by following the tensors step by step through the decoder, focusing only on structure and flow, **not meaning**.
