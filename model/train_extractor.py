"""Shared training loop for BIO-only extractor models (priority, time)."""

import json
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tokenizer import Tokenizer
from model import NanoExtractor, BIO_LABELS

BIO2IDX = {label: i for i, label in enumerate(BIO_LABELS)}


class BIODataset(Dataset):
    def __init__(self, data, tokenizer):
        self.examples = []
        for ex in data:
            ids = tokenizer.encode(ex["text"])
            bio = [BIO2IDX[t] for t in ex["bio"]]
            assert len(ids) == len(bio), f"length mismatch: {ex['text']!r}"
            self.examples.append((ids, bio))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate(batch):
    ids_list, bios_list = zip(*batch)
    lengths = torch.tensor([len(ids) for ids in ids_list])
    max_len = lengths.max().item()

    padded_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_bios = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, (ids, bio) in enumerate(zip(ids_list, bios_list)):
        padded_ids[i, :len(ids)] = torch.tensor(ids)
        padded_bios[i, :len(bio)] = torch.tensor(bio)

    return padded_ids, lengths, padded_bios


def check_accuracy(model, loader):
    model.train(False)
    correct = 0
    total = 0

    with torch.no_grad():
        for ids, lengths, bios in loader:
            bio_logits = model(ids, lengths)
            for i in range(len(lengths)):
                seq_len = lengths[i].item()
                preds = bio_logits[i, :seq_len].argmax(dim=1)
                true = bios[i, :seq_len]
                correct += (preds == true).sum().item()
                total += seq_len

    model.train(True)
    return correct / total if total > 0 else 0


def train_extractor(name, train_file, val_file, out_dir):
    base = os.path.dirname(__file__)

    with open(os.path.join(base, "data", train_file)) as f:
        train_data = json.load(f)
    with open(os.path.join(base, "data", val_file)) as f:
        val_data = json.load(f)

    tokenizer = Tokenizer()
    tokenizer.build_vocab([ex["text"] for ex in train_data])
    print(f"[{name}] Vocabulary size: {tokenizer.vocab_size}")

    train_ds = BIODataset(train_data, tokenizer)
    val_ds = BIODataset(val_data, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=64, collate_fn=collate)

    model = NanoExtractor(tokenizer.vocab_size)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[{name}] Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    for epoch in range(50):
        model.train(True)
        total_loss = 0
        for ids, lengths, bios in train_loader:
            bio_logits = model(ids, lengths)

            mask = torch.arange(bio_logits.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
            bios_masked = bios.clone()
            bios_masked[~mask] = -1

            loss = loss_fn(bio_logits.view(-1, 3), bios_masked.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            acc = check_accuracy(model, val_loader)
            print(f"[{name}] Epoch {epoch+1:3d}  loss={total_loss/len(train_loader):.4f}  bio_acc={acc:.4f}")

    acc = check_accuracy(model, val_loader)
    print(f"[{name}] Final bio_acc={acc:.4f}")

    save_dir = os.path.join(base, "trained", out_dir)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    tokenizer.save(os.path.join(save_dir, "vocab.json"))

    meta = {
        "vocab_size": tokenizer.vocab_size,
        "embed_dim": 128,
        "hidden_dim": 64,
        "bio_labels": BIO_LABELS,
    }
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[{name}] Saved to {save_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: train_extractor.py <priority|time>")
        sys.exit(1)

    kind = sys.argv[1]
    if kind == "priority":
        train_extractor("priority", "priority_train.json", "priority_val.json", "priority")
    elif kind == "time":
        train_extractor("time", "time_train.json", "time_val.json", "time")
    else:
        print(f"Unknown extractor: {kind}")
        sys.exit(1)
