# Project Structure

## Overview

This project implements core components of the **LLaMA 2** language model and applies it to sentiment classification on two datasets: **SST-5** (5-class) and **CFIMDB** (binary). The pretrained backbone is `stories42M.pt` — an 8-layer, 42M-parameter model trained on the TinyStories dataset.

---

## File Descriptions

### Core Model

| File | Description |
|------|-------------|
| `llama.py` | Main LLaMA 2 model. Implements **RMSNorm**, **Attention** (Grouped-Query Attention with Scaled Dot-Product), **FeedForward** (SwiGLU activation), **LlamaLayer** (pre-norm transformer block with residual connections), and **Llama** (full model with weight tying and temperature-based text generation). |
| `base_llama.py` | Base class `LlamaPreTrainedModel` — wraps `nn.Module` with config handling and weight initialization utilities. |
| `rope.py` | **Rotary Positional Embeddings (RoPE)**. Implements `apply_rotary_emb` which computes per-position rotation frequencies and applies them to query/key tensors via complex-number-style rotation. |
| `config.py` | Configuration classes. `PretrainedConfig` handles generic transformer config (generation params, tokenizer settings, etc.); `LlamaConfig` extends it with LLaMA-specific hyperparameters (dim, n_layers, n_heads, n_kv_heads, max_seq_len, etc.). |

### Classification

| File | Description |
|------|-------------|
| `classifier.py` | Two classifier variants built on top of the LLaMA backbone: **`LlamaZeroShotClassifier`** — freezes LLaMA weights and uses next-token log-probabilities to score label strings (zero-shot prompting); **`LlamaEmbeddingClassifier`** — extracts the hidden state at the final token, applies dropout, and passes it through a linear head (supports `pretrain` mode with frozen weights or `finetune` mode with full gradient updates). |

### Optimization

| File | Description |
|------|-------------|
| `optimizer.py` | Custom **AdamW** optimizer implementation. Implements bias-corrected first/second moment estimates (Algorithm 1 in Kingma & Ba 2015) with decoupled weight decay applied separately from the gradient update step. |

### Training & Evaluation

| File | Description |
|------|-------------|
| `run_llama.py` | Main entry point. Handles three modes: (1) **generate** — temperature-controlled text completion; (2) **prompt** — zero-shot sentiment classification via `LlamaZeroShotClassifier`; (3) **finetune** — full training loop with AdamW, dev-set early stopping, and prediction export. |
| `run_hyperparam_search.py` | Grid search over `lr`, `epochs`, and `hidden_dropout_prob` for `LlamaEmbeddingClassifier`. Saves incremental results to CSV after each trial so no data is lost if interrupted. |

### Utilities & Testing

| File | Description |
|------|-------------|
| `tokenizer.py` | SentencePiece-based tokenizer wrapping LLaMA's `tokenizer.model`. Supports BOS/EOS tokens, max-length truncation, and padding. |
| `utils.py` | Shared utilities: HuggingFace-compatible model downloading/caching, HTTP helpers, and `get_parameter_dtype`. |
| `sanity_check.py` | Validates the full LLaMA forward pass against reference logits and hidden states saved in `sanity_check.data`. |
| `rope_test.py` | Unit test for `apply_rotary_emb` against reference tensors in `rotary_embedding_actual.data`. |
| `optimizer_test.py` | Unit test for the AdamW optimizer — trains a small linear model for 1000 steps and checks weights against a NumPy reference (`optimizer_test.npy`). |
| `setup.sh` | Environment setup script — creates a conda environment, installs dependencies (PyTorch 2.1.2, scikit-learn, sentencepiece, etc.), and downloads `stories42M.pt`. |

---

## Key Implementation Details

### RMSNorm (`llama.py`)
Implements Equation 4 from [Zhang & Sennrich 2019](https://arxiv.org/abs/1910.07467):
```
RMS(x) = x / sqrt(mean(x²) + ε)
```

### Rotary Embeddings (`rope.py`)
Follows Section 3 of [Su et al. 2021](https://arxiv.org/abs/2104.09864). Position-dependent rotation is applied to query/key pairs before attention, encoding relative position information without learned embeddings.

### Attention (`llama.py`)
Scaled Dot-Product Attention with GQA (Grouped-Query Attention): key/value heads are fewer than query heads and expanded via `repeat_interleave`. Causal masking is not applied (classification use case).

### SwiGLU FeedForward (`llama.py`)
Follows [Shazeer 2020](https://arxiv.org/abs/2204.02311):
```
SwiGLU(x) = SiLU(W₁x) ⊙ W₃x
output    = W₂ · SwiGLU(x)
```

### AdamW (`optimizer.py`)
Bias correction is computed as an efficient step-size scalar:
```
step_size = lr * sqrt(1 - β₂ᵗ) / (1 - β₁ᵗ)
```
Weight decay is applied **after** the gradient step using the base learning rate (decoupled formulation).

---

## How to Run

```bash
# 1. Setup
bash setup.sh

# 2. Text generation
python run_llama.py \
    --option generate \
    --pretrained_model_path stories42M.pt \
    --generated_sentence_low_temp_out generated-sentence-temp-0.txt \
    --generated_sentence_high_temp_out generated-sentence-temp-1.txt

# 3. Zero-shot prompting 
 # SST
python run_llama.py \
    --option prompt \
    --train data/sst-train.txt \
    --dev data/sst-dev.txt \
    --test data/sst-test.txt \
    --label-names data/sst-label-mapping.json \
    --pretrained_model_path stories42M.pt \
    --dev_out sst-dev-prompting-output.txt \
    --test_out sst-test-prompting-output.txt
 # CFIMDB
python run_llama.py \
    --option prompt \
    --train data/cfimdb-train.txt \
    --dev data/cfimdb-dev.txt \
    --test data/cfimdb-test.txt \
    --label-names data/cfimdb-label-mapping.json \
    --pretrained_model_path stories42M.pt \
    --dev_out cfimdb-dev-prompting-output.txt \
    --test_out cfimdb-test-prompting-output.txt

# 4. Finetuning 
 # SST-default hyperparameters (lr=2e-5, epochs=5, dropout=0.3, batch=64)
python run_llama.py \
    --option finetune \
    --train data/sst-train.txt \
    --dev data/sst-dev.txt \
    --test data/sst-test.txt \
    --pretrained_model_path stories42M.pt \
    --batch_size 64 \
    --dev_out sst-dev-finetuning-output.txt \
    --test_out sst-test-finetuning-output.txt \
--use_gpu
 # CFIMDB-default hyperparameters (lr=2e-5, epochs=5, dropout=0.3, batch=8)
python run_llama.py \
    --option finetune \
    --train data/cfimdb-train.txt \
    --dev data/cfimdb-dev.txt \
    --test data/cfimdb-test.txt \
    --pretrained_model_path stories42M.pt \
    --batch_size 8 \
    --dev_out cfimdb-dev-finetuning-output.txt \
    --test_out cfimdb-test-finetuning-output.txt \
    --use_gpu

# 5. Hyperparameter search
 # SST - save result at hyperparam_results_sst.csv
python run_hyperparam_search.py \
    --dataset sst \
    --train data/sst-train.txt \
    --dev data/sst-dev.txt \
    --test data/sst-test.txt \
    --pretrained_model_path stories42M.pt \
    --use_gpu
 # CFIMDB - save result at hyperparam_results_cfimdb.csv
python run_hyperparam_search.py \
    --dataset cfimdb \
    --train data/cfimdb-train.txt \
    --dev data/cfimdb-dev.txt \
    --test data/cfimdb-test.txt \
    --pretrained_model_path stories42M.pt \
```