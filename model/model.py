import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


INTENT_LABELS = ["create", "complete", "delete", "list", "unknown"]
BIO_LABELS = ["O", "B", "I"]


class NanoNLU(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64,
                 num_intents=5, num_bio_tags=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.intent_head = nn.Linear(hidden_dim * 2, num_intents)
        self.entity_head = nn.Linear(hidden_dim * 2, num_bio_tags)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.gru(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)

        mask = torch.arange(output.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask_f = mask.unsqueeze(2).float()

        pooled = (output * mask_f).sum(dim=1) / lengths.unsqueeze(1).float()
        intent_logits = self.intent_head(pooled)

        bio_logits = self.entity_head(output)

        return intent_logits, bio_logits


class NanoExtractor(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64, num_bio_tags=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.bio_head = nn.Linear(hidden_dim * 2, num_bio_tags)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.gru(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        return self.bio_head(output)
