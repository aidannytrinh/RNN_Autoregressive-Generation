# RNN Autoregressive Text Generation

A hands-on exercise building an RNN language model in PyTorch that generates new text **autoregressively**, trained on a short Vietnamese (unaccented) text about **playing sports / working out**.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Notebook Structure](#notebook-structure)
- [Technical Details](#technical-details)
  - [Text Preprocessing](#1-text-preprocessing)
  - [Training Data via Teacher Forcing](#2-training-data-via-teacher-forcing)
  - [Model Architecture](#3-model-architecture)
  - [Training (BPTT)](#4-training-bptt)
  - [Next-Word Prediction](#5-next-word-prediction)
  - [The `generate(seed_text, max_len)` Function](#6-the-generateseed_text-max_len-function)
  - [Teacher Forcing vs. Autoregression](#7-teacher-forcing-vs-autoregression)
- [Results](#results)
- [How to Run](#how-to-run)
- [Limitations and Possible Extensions](#limitations-and-possible-extensions)

## Overview

The `RNN_Autoregressive_Generation.ipynb` notebook builds on the standard RNN-based next-word predictor and focuses specifically on **autoregressive text generation**: repeatedly feeding the model's own predictions back in as input to generate a full sequence of new words from a short seed phrase.

Main goals covered in the notebook:

1. Use a short paragraph (~10-15 sentences) as the training corpus.
2. Build the RNN architecture (`nn.Embedding` → `nn.RNN` → `nn.Linear`).
3. Implement a `generate(seed_text, max_len)` function that:
   - Feeds `seed_text` into the model to predict the next word.
   - Picks the highest-probability word as the next input.
   - Repeats until `max_len` new words have been generated.
4. Compare what **Teacher Forcing** means during training versus what **Autoregression** means during generation.

## Requirements

- Python 3.8+
- PyTorch (`torch`)

```bash
pip install torch
```

The notebook automatically selects `cuda` if a GPU is available, otherwise it falls back to `cpu`:

```python
thiet_bi = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Notebook Structure

| Cell | Content |
|---|---|
| 1 | Import libraries, select device, set `torch.manual_seed(42)` |
| 2 | Define the source text — 12 short sentences about sports/exercise |
| 3 | Text preprocessing → build `tu_sang_so` / `so_sang_tu` vocabularies, encode text as integers |
| 4 | Build training data with Teacher Forcing (context length 6), wrap into `DataLoader` |
| 5 | Define `MoHinhRNNVanBan` (Embedding → RNN → Linear, no weight tying this time) |
| 6 | Training loop with `CrossEntropyLoss` + `Adam` (60 epochs), relying on BPTT |
| 7 | `du_doan_tu_tiep_theo`: predicts a single next word from a seed sentence |
| 8 | `generate(seed_text, max_len)`: the core autoregressive generation loop |
| 9 | Generation experiments with several different seed phrases |
| 10 | Side-by-side comparison of Teacher Forcing (training) vs. Autoregression (generation) |
| 11 | Short conclusion / recap of what was implemented |

## Technical Details

### 1. Text Preprocessing

`tien_xu_ly_van_ban` lowercases the text, strips out anything that isn't a letter or digit with a regex (`[^a-zA-Z0-9\s]`), and splits the result into a word list. From that, `tu_sang_so` (word → index) and its inverse `so_sang_tu` (index → word) are built.

For this dataset: **149 words** total, with a vocabulary of **94 unique words**.

### 2. Training Data via Teacher Forcing

Same sliding-window approach as before, but with a longer context window (`do_dai_ngu_canh = 6`): for each window, the label sequence is the input sequence shifted right by one word.

```text
Input : moi buoi sang em chay bo
Label : buoi sang em chay bo quanh
```

That is: given `moi`, predict `buoi`; given `buoi`, predict `sang`; and so on. Feeding the model the **true** word at every step (rather than its own prediction) is exactly what Teacher Forcing means, and it's what makes training converge quickly and stably.

This produces `X` and `y` tensors of shape `[143, 6]`, wrapped in a `TensorDataset` / `DataLoader` (`batch_size=4`, `shuffle=True`).

### 3. Model Architecture

```python
class MoHinhRNNVanBan(nn.Module):
    def __init__(self, kich_thuoc_tu_dien, kich_thuoc_embedding, kich_thuoc_an):
        super().__init__()
        self.embedding = nn.Embedding(kich_thuoc_tu_dien, kich_thuoc_embedding)
        self.rnn = nn.RNN(kich_thuoc_embedding, kich_thuoc_an, batch_first=True)
        self.fc = nn.Linear(kich_thuoc_an, kich_thuoc_tu_dien)

    def forward(self, dau_vao):
        vector_tu = self.embedding(dau_vao)
        dau_ra_rnn, trang_thai_an = self.rnn(vector_tu)
        du_doan = self.fc(dau_ra_rnn)
        return du_doan
```

Three components, same roles as in a standard word-level RNN LM:

1. **`nn.Embedding`** (94 → 64): maps each word index to a 64-dim dense vector.
2. **`nn.RNN`** (input 64, hidden 64, `batch_first=True`): processes the sequence step by step, carrying forward a hidden state that encodes prior context.
3. **`nn.Linear`** (64 → 94): projects each timestep's hidden output into a score (logit) for every word in the vocabulary.

The model's output has shape `(batch_size, do_dai_ngu_canh, kich_thuoc_tu_dien)` — one prediction distribution per timestep. Unlike the earlier next-word-prediction notebook, this one does **not** apply Weight Tying between the embedding and output layers.

### 4. Training (BPTT)

- **Loss:** `nn.CrossEntropyLoss()`.
- **Optimizer:** `torch.optim.Adam`, `lr=0.01`.
- **Epochs:** 60.

The prediction and label tensors are flattened before computing the loss (`(batch, T, V) → (batch*T, V)` and `(batch, T) → (batch*T,)`), since `CrossEntropyLoss` expects 2D logits and a 1D target.

Calling `loss.backward()` triggers **Backpropagation Through Time (BPTT)**: gradients are propagated backward across every timestep of the RNN to update its weights, since the same recurrent weight matrix was reused at every step of the forward pass.

### 5. Next-Word Prediction

`du_doan_tu_tiep_theo(mo_hinh, seed_text)` preprocesses `seed_text`, keeps only the last `do_dai_ngu_canh` words as context (truncating older ones if the seed is longer), runs the model in `eval()` mode, applies `softmax` to the last timestep's logits, and returns the `argmax` word (greedy decoding).

### 6. The `generate(seed_text, max_len)` Function

This is the heart of the notebook — the autoregressive generation loop:

```python
def generate(seed_text, max_len):
    cau_hien_tai = seed_text
    for _ in range(max_len):
        tu_moi = du_doan_tu_tiep_theo(mo_hinh, cau_hien_tai)
        if tu_moi is None:
            break
        cau_hien_tai = cau_hien_tai + " " + tu_moi
    return cau_hien_tai
```

The process:

1. Feed `seed_text` into the model.
2. Predict the next word.
3. Take the highest-probability word.
4. Append it to the end of the current sentence.
5. Use that updated sentence as the input for the next prediction.
6. Repeat until `max_len` new words have been generated.

The key difference from training: at generation time the model is **never** shown the true next word from the original corpus — it only ever sees words it generated itself. This is what "autoregressive" means here, and it's structurally different from Teacher Forcing.

### 7. Teacher Forcing vs. Autoregression

| | Teacher Forcing (training) | Autoregression (generation) |
|---|---|---|
| Input at each step | The **true** word from the corpus | The model's **own previous prediction** |
| Purpose | Fast, stable learning of next-word probabilities | Producing new, unseen text from a seed |
| Error propagation | Each step is corrected by ground truth, so one wrong prediction doesn't affect the next input | A wrong prediction becomes part of the context for every subsequent step, so errors can compound |

The notebook demonstrates this directly: it prints the true `(input, label)` pair used during training right next to a step-by-step trace of `generate` extending the seed `"em chay bo"` one word at a time, so you can see how each newly generated word gets folded back into the next prediction's input.

## Results

Training loss over 60 epochs:

```
Epoch  10 | Loss trung bình: 0.2133
Epoch  20 | Loss trung bình: 0.1935
Epoch  30 | Loss trung bình: 0.1762
Epoch  40 | Loss trung bình: 0.1730
Epoch  50 | Loss trung bình: 0.2179
Epoch  60 | Loss trung bình: 0.2076
```

Generation examples from several different seed phrases (`max_len=10`):

```
Seed: moi buoi sang
Result: moi buoi sang em chay bo quanh cong vien de ren suc ben

Seed: tap the thao
Result: tap the thao deu dan giup co the khoe manh hon moi ngay

Seed: sau gio hoc
Result: sau gio hoc em danh cau long voi ban be trong san truong

Seed: uong du nuoc
Result: uong du nuoc trong luc tap giup co the khong bi met moi

Seed: the thao giup
Result: the thao giup tinh than vui ve va hoc tap tap trung hon
```

Because the model has essentially memorized this very small corpus, each seed regenerates the exact sentence it came from in the training text — a useful, honest illustration of how autoregressive generation works, though not yet a demonstration of generalization to unseen phrasing.

## How to Run

1. Clone the repo and open the notebook with Jupyter or Google Colab:

   ```bash
   jupyter notebook RNN_Autoregressive_Generation.ipynb
   ```

2. Run the cells in order from top to bottom.
3. Hyperparameters you can experiment with:
   - `do_dai_ngu_canh` (context length, default 6)
   - `kich_thuoc_embedding`, `kich_thuoc_an` (default 64)
   - `batch_size`, `lr`, `so_epoch`
4. Try generating text from your own seed phrases:

   ```python
   generate("tap gym can", max_len=10)
   ```

## Limitations and Possible Extensions

- **Tiny dataset** (149 words, 94 unique) means the model mostly memorizes the training sentences rather than learning to generalize — generated text from a seen seed will usually just reproduce the original sentence.
- **Greedy decoding** (`argmax`) is deterministic and can't explore alternative continuations; sampling strategies like temperature scaling, top-k, or nucleus (top-p) sampling would produce more varied output.
- **No weight tying** in this version, unlike the earlier next-word-prediction notebook — trying it here would be a good comparison exercise.
- **Vanilla `nn.RNN`** is limited in how much context it can retain over long sequences; `nn.LSTM` or `nn.GRU` would likely help, especially with a larger corpus.
- No validation/test split exists to measure generalization beyond the training text.
- Could be extended with a larger, more diverse corpus, or by comparing against a small Transformer-based generator.
