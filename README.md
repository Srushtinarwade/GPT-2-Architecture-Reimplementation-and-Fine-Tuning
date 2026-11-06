# GPT-2 Architecture Reimplementation and Fine-Tuning

A hands-on implementation of the GPT-2 architecture using PyTorch, exploring the fundamental building blocks of decoder-only Transformers, attention mechanisms, language modeling, and transfer learning through fine-tuning.

This project focuses on understanding how GPT-2 works internally by implementing its core components and applying a pretrained GPT-2 model to a downstream binary text-classification task.

---

## Project Overview

GPT-2 is an autoregressive Transformer-based language model that predicts the next token based on previously observed tokens.

Rather than treating GPT-2 as a black-box model, this project explores its architecture from the ground up, covering:

- Tokenization and token embeddings.
- Positional embeddings.
- Scaled dot-product attention.
- Causal masking and multi-head attention.
- Transformer blocks and residual connections.
- Layer normalization and GELU activation.
- Autoregressive language modeling.
- Pretrained GPT-2 weight loading.
- Transfer learning and binary classification.
- Model evaluation and inference.

The implementation uses the GPT-2 Small (124M) architecture for the fine-tuning experiment.

## Architecture

The implementation follows the decoder-only Transformer architecture.

```text
                 Input Text
                     |
                     v
              GPT-2 Tokenizer
                     |
                     v
               Token IDs
                     |
                     v
             Token Embeddings
                     +
             Positional Embeddings
                     |
                     v
              Embedding Dropout
                     |
                     v
          +------------------------+
          |   Transformer Block 1  |
          |                        |
          | Layer Normalization    |
          | Causal Self-Attention   |
          | Residual Connection     |
          | Layer Normalization    |
          | Feed-Forward Network   |
          | Residual Connection     |
          +------------------------+
                     |
                    ...
                     |
                     v
          +------------------------+
          |  Transformer Block 12  |
          +------------------------+
                     |
                     v
             Final Layer Norm
                     |
                     v
             Linear Output Head
                     |
                     v
                  Logits
