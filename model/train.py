import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tokenizer import Tokenizer
from model import NanoNLU, INTENT_LABELS, BIO_LABELS

BASE_DIR = os.path.dirname(__file__)

INTENT2IDX = {label: i for i, label in enumerate(INTENT_LABELS)}
BIO2IDX = {label: i for i, label in enumerate(BIO_LABELS)}


class NLUDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.examples = []
        for ex in data:
            ids = tokenizer.encode(ex["text"])
            intent = INTENT2IDX[ex["intent"]]
            bio = [BIO2IDX[t] for t in ex["bio"]]
            assert len(ids) == len(bio), f"length mismatch: {ex['text']!r} ids={len(ids)} bio={len(bio)}"
            self.examples.append((ids, intent, bio))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate(batch):
    ids_list, intents, bios_list = zip(*batch)
    lengths = torch.tensor([len(ids) for ids in ids_list])
    max_len = lengths.max().item()

    padded_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_bios = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, (ids, bio) in enumerate(zip(ids_list, bios_list)):
        padded_ids[i, :len(ids)] = torch.tensor(ids)
        padded_bios[i, :len(bio)] = torch.tensor(bio)

    intents = torch.tensor(intents, dtype=torch.long)
    return padded_ids, lengths, intents, padded_bios


def check_accuracy(model, loader):
    model.eval()
    correct_intent = 0
    total = 0
    correct_bio = 0
    total_bio = 0

    with torch.no_grad():
        for ids, lengths, intents, bios in loader:
            intent_logits, bio_logits = model(ids, lengths)

            preds = intent_logits.argmax(dim=1)
            correct_intent += (preds == intents).sum().item()
            total += len(intents)

            for i in range(len(lengths)):
                seq_len = lengths[i].item()
                bio_preds = bio_logits[i, :seq_len].argmax(dim=1)
                bio_true = bios[i, :seq_len]
                correct_bio += (bio_preds == bio_true).sum().item()
                total_bio += seq_len

    intent_acc = correct_intent / total if total > 0 else 0
    bio_acc = correct_bio / total_bio if total_bio > 0 else 0
    return intent_acc, bio_acc


torch.manual_seed(25)
def main():
    with open(os.path.join(BASE_DIR, "data", "train.json")) as f:
        train_data = json.load(f)
    with open(os.path.join(BASE_DIR, "data", "val.json")) as f:
        val_data = json.load(f)

    tokenizer = Tokenizer()
    tokenizer.build_vocab([ex["text"] for ex in train_data])
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    train_ds = NLUDataset(train_data, tokenizer)
    val_ds = NLUDataset(val_data, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=64, collate_fn=collate)

    model = NanoNLU(tokenizer.vocab_size, embed_dim=160, hidden_dim=80)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    intent_loss_fn = nn.CrossEntropyLoss()
    bio_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    for epoch in range(50):
        model.train()
        total_loss = 0
        for ids, lengths, intents, bios in train_loader:
            intent_logits, bio_logits = model(ids, lengths)

            loss_intent = intent_loss_fn(intent_logits, intents)

            mask = torch.arange(bio_logits.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
            bios_masked = bios.clone()
            bios_masked[~mask] = -1
            loss_bio = bio_loss_fn(bio_logits.view(-1, 3), bios_masked.view(-1))

            loss = loss_intent + loss_bio
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            intent_acc, bio_acc = check_accuracy(model, val_loader)
            print(f"Epoch {epoch+1:3d}  loss={total_loss/len(train_loader):.4f}  "
                  f"intent_acc={intent_acc:.4f}  bio_acc={bio_acc:.4f}")

    intent_acc, bio_acc = check_accuracy(model, val_loader)
    print(f"\nFinal — intent_acc={intent_acc:.4f}  bio_acc={bio_acc:.4f}")

    out_dir = os.path.join(BASE_DIR, "trained")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "nano_nlu.pt"))
    tokenizer.save(os.path.join(out_dir, "vocab.json"))

    meta = {
        "vocab_size": tokenizer.vocab_size,
        "intent_labels": INTENT_LABELS,
        "bio_labels": BIO_LABELS,
        "embed_dim": 160,
        "hidden_dim": 80,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
