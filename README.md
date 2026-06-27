# BLAIR — Video Games Domain Specialization

> **B**ridging **L**anguage and **I**tems for **R**etrieval  
> Amazon Reviews 2023 · Video Games · From Scratch Implementation

A domain-specialized implementation of [BLAIR](https://arxiv.org/abs/2403.03952) for the Video Games category, extending the original paper with multi-aspect positive sampling and aspect-divergent hard negative mining.

---

## Files in This Repo

| File | Description |
|---|---|
| `download_data.py` | Downloads Video Games reviews and metadata from Amazon Reviews 2023 |
| `preprocess.py` | Parses raw `.jsonl.gz` files into train/val/test parquet splits |
| `dataset.py` | `ContrastivePairDataset` — PyTorch Dataset for (review, metadata) pairs |
| `model.py` | `BLAIR` model — RoBERTa encoder + projection head + contrastive loss |
| `build_training_data.py` | Builds multi-aspect training pairs from `products_rich.parquet` |
| `train_blair.py` | Main training script — roberta-large, NT-Xent loss, early stopping |

---

## Model Architecture (`model.py`)

```python
class BLAIR(nn.Module):
    # Base: roberta-base encoder
    # Projection head: Linear(768→768) → ReLU → Linear(768→1024)
    # Output: 1024-dim L2-normalized embeddings via CLS token
```

- **Base**: `roberta-base` (123M parameters)
- **Projection head**: two-layer MLP projecting to 1024-dim
- **Embedding**: CLS token → projection → L2 normalize
- **Loss**: Symmetric NT-Xent (in-batch negatives, temperature=0.07)
- Loads saved projection head from `proj_head.pt` if present in checkpoint

---

## Training (`train_blair.py`)

- **Base model**: `roberta-large` (355M) — raw CLS token, no projection head
- **Output**: 1024-dim L2-normalized directly from CLS
- **Loss**: NT-Xent + hard negative hinge loss (weight 0.3)
- **Hard negatives**: aspect-divergent same-category peers (50% probability)
- **Early stopping**: patience=3, max epochs=5
- **Batch size**: 8 | **Max length**: 128 | **LR**: 2e-5
- **Optimizer**: AdamW (weight_decay=0.01) + linear warmup scheduler
- **AMP**: FP16 via `torch.amp.GradScaler`
- **Gradient checkpointing**: enabled to fit 8GB VRAM
- Saves best model to `blair-videogames-multiaspect/` and epoch checkpoints to `checkpoints/epoch_N/`

---

## Data Pipeline

### Step 1 — Download (`download_data.py`)
Downloads directly from UCSD McAuley Lab servers:
- `Video_Games.jsonl.gz` — 4.6M reviews
- `meta_Video_Games.jsonl.gz` — 137k product metadata entries

### Step 2 — Preprocess (`preprocess.py`)
- Splits by absolute timestamp: train < `1_640_000_000` (Jan 2022), val < `1_672_000_000` (Jan 2023), rest = test
- Handles ms/s timestamp normalization
- Builds metadata string from title + features + description
- Saves parquet splits + user interaction histories

```
Train: 3,570,172 pairs
Val:     381,256 pairs
Test:    192,174 pairs
Items:   137,269
```

### Step 3 — Build Training Pairs (`build_training_data.py`)
Requires 3 files from teammate (Yogendra):
- `products_nlp.parquet` — 137,269 products with 95 NLP columns
- `reviews_nlp.parquet` — 814,586 reviews
- `products_rich.parquet` — rich text documents per product

Builds 7 types of (anchor=rich_doc, positive=?) pairs per product:

| Type | Positive source |
|---|---|
| `aspect_gameplay` | Top 5 helpful reviews mentioning gameplay keywords |
| `aspect_graphics` | Top 5 helpful reviews mentioning graphics keywords |
| `aspect_story` | Top 5 helpful reviews mentioning story keywords |
| `aspect_controls` | Top 5 helpful reviews mentioning controls keywords |
| `aspect_value` | Top 5 helpful reviews mentioning value keywords |
| `all_reviews` | Top 10 most helpful reviews concatenated (1500 chars) |
| `metadata` | Title + full_category_path + brand |

Produces `training_pairs.jsonl` (~107k pairs total).

**Key optimization**: reviews are pre-grouped by ASIN once before the loop — avoids scanning 814k rows per product.

### Step 4 — Hard Negatives (`build_hard_negatives.py`)
For each product, finds peers in the same `leaf_category` with:
- Rating difference ≤ 0.5 stars
- Aspect vector distance > 0.3

Produces `hard_negatives.json` — up to 10 hard negatives per product.

---

## Setup

```bash
# Windows
python -m venv blair_env
blair_env\Scripts\activate

# Install
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate pandas pyarrow tqdm scikit-learn pyyaml huggingface-hub
```

## Running

```bash
# 1. Download raw data (~2GB)
python download_data.py

# 2. Preprocess into parquet splits
python preprocess.py

# 3. Build multi-aspect training pairs (needs Yogendra's files)
python build_training_data.py

# 4. Build hard negatives
python build_hard_negatives.py

# 5. Train (6-10 hours on RTX 4060)
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train_blair.py

# 6. Generate embeddings
python generate_embeddings.py
```

---

## Hardware

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop (8.6GB VRAM) |
| CPU | AMD Ryzen 9 7845HX (12 cores / 24 threads) |
| RAM | 16GB |
| OS | Windows 11 |

---

## Key Design Decisions

**Why two model files?**  
`model.py` is the original BLAIR implementation (roberta-base + projection head) used for initial experiments. `train_blair.py` contains the final `BLAIRModel` class (roberta-large, raw CLS, no projection head) as specified by the team pipeline.

**Why multi-aspect positives?**  
Standard BLAIR uses one positive per anchor. Using 7 types forces the model to learn aspect-level gaming semantics — a model that can distinguish "good gameplay, bad controls" from "bad gameplay, good controls" is much more useful for recommendation.

**Why hard negatives?**  
In-batch negatives alone are often too easy (random games from different categories). Hard negatives within the same category with divergent aspect profiles force the model to discriminate fine-grained quality signals.

---

## Reference

```bibtex
@article{hou2024blair,
  title   = {Bridging Language and Items for Retrieval and Recommendation},
  author  = {Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and
             Chen, Xiusi and McAuley, Julian},
  journal = {arXiv preprint arXiv:2403.03952},
  year    = {2024}
}
```
