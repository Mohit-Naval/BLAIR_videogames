# train_blair.py
"""
Multi-Aspect BLAIR Training
Base: roberta-large (1024-dim CLS token)
Objective: NT-Xent contrastive with hard negatives
Early stopping: patience=3, max epochs=5
Windows-compatible: num_workers=0, collate_fn at module level
"""

import json
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


MODEL_NAME = "roberta-large"
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 5
PATIENCE = 3
LR = 2e-5
TEMPERATURE = 0.07
HARD_NEG_PROB = 0.5
SAVE_DIR = "./blair-videogames-multiaspect"
CHECKPOINT_DIR = "./checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"VRAM: {round(torch.cuda.get_device_properties(0).total_memory/1e9, 1)} GB")


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class BLAIRModel(nn.Module):
    """
    BLAIR-style encoder.
    Raw CLS token, L2 normalized → 1024-dim output (roberta-large).
    No projection head as per teammate spec.
    """

    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, **kwargs):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls = out.last_hidden_state[:, 0]
        return F.normalize(cls, dim=-1)


class BLAIRDataset(Dataset):
    def __init__(self, pairs_file, hard_neg_file, rich_lookup):
        with open(pairs_file, encoding="utf-8") as f:
            self.pairs = [json.loads(l) for l in f]
        with open(hard_neg_file, encoding="utf-8") as f:
            self.hard_negs = json.load(f)
        self.rich_lookup = rich_lookup
        random.shuffle(self.pairs)
        print(f"Dataset: {len(self.pairs):,} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        anchor = pair["anchor"]
        positive = pair["positive"]
        asin = pair["asin"]

        hard_neg = ""
        if (asin in self.hard_negs
                and random.random() < HARD_NEG_PROB
                and len(self.hard_negs[asin]) > 0):
            neg_asin = random.choice(self.hard_negs[asin])
            if neg_asin in self.rich_lookup:
                hard_neg = self.rich_lookup[neg_asin]

        return {"anchor": anchor, "positive": positive, "hard_neg": hard_neg}


def collate_fn(batch):
    anchors = [b["anchor"] for b in batch]
    positives = [b["positive"] for b in batch]
    hard_negs = [b["hard_neg"] for b in batch]
    has_hard_neg = [hn != "" for hn in hard_negs]

    anchor_enc = tokenizer(
        anchors, padding=True, truncation=True,
        max_length=MAX_LENGTH, return_tensors="pt"
    )
    pos_enc = tokenizer(
        positives, padding=True, truncation=True,
        max_length=MAX_LENGTH, return_tensors="pt"
    )

    hard_neg_enc = None
    if any(has_hard_neg):
        hard_neg_texts = [hn if hn else "placeholder" for hn in hard_negs]
        hard_neg_enc = tokenizer(
            hard_neg_texts, padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt"
        )

    return anchor_enc, pos_enc, hard_neg_enc, has_hard_neg


def nt_xent_loss(anchor_emb, pos_emb, temperature=0.07):
    B = anchor_emb.size(0)
    sim = torch.mm(anchor_emb, pos_emb.T) / temperature
    labels = torch.arange(B, device=anchor_emb.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


def hard_neg_loss(anchor_emb, hard_neg_emb, temperature=0.07):
    sim = (anchor_emb * hard_neg_emb).sum(dim=1) / temperature
    margin = 0.3
    return F.relu(sim - margin).mean()


def train():
    import pandas as pd

    print("Loading rich documents...")
    rich_docs = pd.read_parquet("products_rich.parquet")
    rich_lookup = dict(zip(rich_docs.parent_asin, rich_docs.rich_text))
    print(f"  {len(rich_lookup):,} rich documents loaded")

    dataset = BLAIRDataset(
        "training_pairs.jsonl",
        "hard_negatives.json",
        rich_lookup,
    )

    model = BLAIRModel(MODEL_NAME).to(device)

    model.encoder.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled.")

    scaler = torch.amp.GradScaler("cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.01
    )

    total_steps = len(dataset) // BATCH_SIZE * EPOCHS
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=total_steps // 10,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    baseline = torch.log(torch.tensor(float(BATCH_SIZE))).item()
    print(f"\nTraining config:")
    print(f"  Max epochs:       {EPOCHS}")
    print(f"  Early stopping:   patience={PATIENCE}")
    print(f"  Batches/epoch:    {len(dataloader):,}")
    print(f"  Batch size:       {BATCH_SIZE}")
    print(f"  Random baseline:  {baseline:.4f}")
    print(f"  Hard neg prob:    {HARD_NEG_PROB}")
    print("-" * 60)

    best_loss = float("inf")
    epochs_no_improve = 0
    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_steps_epoch = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for anchor_enc, pos_enc, hard_neg_enc, has_hard_neg in pbar:
            anchor_enc = {k: v.to(device) for k, v in anchor_enc.items()}
            pos_enc = {k: v.to(device) for k, v in pos_enc.items()}

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                anchor_emb = model(**anchor_enc)
                pos_emb = model(**pos_enc)
                loss = nt_xent_loss(anchor_emb, pos_emb, TEMPERATURE)

                if hard_neg_enc is not None:
                    hard_neg_enc_dev = {
                        k: v.to(device) for k, v in hard_neg_enc.items()
                    }
                    hard_neg_emb = model(**hard_neg_enc_dev)
                    mask = torch.tensor(
                        has_hard_neg, dtype=torch.bool, device=device
                    )
                    if mask.any():
                        hn_loss = hard_neg_loss(
                            anchor_emb[mask],
                            hard_neg_emb[mask],
                            TEMPERATURE,
                        )
                        loss = loss + 0.3 * hn_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            total_steps_epoch += 1
            global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg":  f"{total_loss/total_steps_epoch:.4f}",
                "lr":   f"{scheduler.get_last_lr()[0]:.2e}",
            })

            # Free VRAM periodically
            if global_step % 500 == 0 and device.type == "cuda":
                vram = torch.cuda.memory_allocated(0) / 1e9
                print(f"\n  step {global_step} | VRAM used: {vram:.1f} GB")
                torch.cuda.empty_cache()

        avg_loss = total_loss / total_steps_epoch
        print(f"\nEpoch {epoch+1} — avg loss: {avg_loss:.4f} | "
              f"best so far: {best_loss:.4f}")

        ckpt_path = f"{CHECKPOINT_DIR}/epoch_{epoch+1}"
        model.encoder.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        print(f"  Checkpoint: {ckpt_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            model.encoder.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print(f"  ✓ New best! Saved to {SAVE_DIR}")
        else:
            epochs_no_improve += 1
            print(
                f"  No improvement — patience {epochs_no_improve}/{PATIENCE}")
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping after epoch {epoch+1}.")
                print(f"No improvement for {PATIENCE} consecutive epochs.")
                break

        print("-" * 60)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best loss:     {best_loss:.4f}")
    print(f"Baseline loss: {baseline:.4f}")
    print(f"Improvement:   {(baseline-best_loss)/baseline*100:.1f}%")
    print(f"Best model:    {SAVE_DIR}")
    print(f"{'='*60}")
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = train()
