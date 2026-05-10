
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel


class BLAIR(nn.Module):
    def __init__(self, model_name="roberta-base", tau=0.07, lambda_mlm=0.1):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(model_name)
        self.tau = tau
        self.lambda_mlm = lambda_mlm
        hidden = self.encoder.config.hidden_size

        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1024),
        )

        import os
        proj_path = os.path.join(model_name, "proj_head.pt")
        if os.path.exists(proj_path):
            self.proj.load_state_dict(
                torch.load(proj_path, map_location="cpu")
            )
            print(f"Loaded projection head from {proj_path}")

    def encode(self, input_ids, attention_mask, **kwargs):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls_emb = out.last_hidden_state[:, 0, :]
        proj = self.proj(cls_emb)
        return F.normalize(proj, dim=-1)

    def contrastive_loss(self, c_emb, m_emb):
        B = c_emb.size(0)
        logits = torch.matmul(c_emb, m_emb.T) / self.tau
        labels = torch.arange(B, device=c_emb.device)
        return (F.cross_entropy(logits, labels) +
                F.cross_entropy(logits.T, labels)) / 2

    def forward(self, enc_review, enc_meta):
        c_emb = self.encode(**enc_review)
        m_emb = self.encode(**enc_meta)
        loss = self.contrastive_loss(c_emb, m_emb)
        return loss, c_emb, m_emb
