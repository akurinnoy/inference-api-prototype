import json
import os
import torch
from flask import Flask, request, jsonify

from tokenizer import Tokenizer
from model import NanoNLU, NanoExtractor, INTENT_LABELS, BIO_LABELS

BASE_DIR = os.path.dirname(__file__)
TRAINED_DIR = os.path.join(BASE_DIR, "trained")
EXTRACTION_MODE = os.environ.get("EXTRACTION_MODE", "parallel")

app = Flask(__name__)

intent_model = None
intent_tokenizer = None
priority_model = None
priority_tokenizer = None
time_model = None
time_tokenizer = None


def load_extractor(subdir):
    path = os.path.join(TRAINED_DIR, subdir)
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)

    tok = Tokenizer()
    tok.load(os.path.join(path, "vocab.json"))

    mdl = NanoExtractor(
        vocab_size=meta["vocab_size"],
        embed_dim=meta["embed_dim"],
        hidden_dim=meta["hidden_dim"],
    )
    mdl.load_state_dict(torch.load(os.path.join(path, "model.pt"), weights_only=True))
    mdl.train(False)
    params = sum(p.numel() for p in mdl.parameters())
    print(f"  [{subdir}] {params:,} parameters")
    return mdl, tok


def load_all():
    global intent_model, intent_tokenizer
    global priority_model, priority_tokenizer
    global time_model, time_tokenizer

    with open(os.path.join(TRAINED_DIR, "meta.json")) as f:
        meta = json.load(f)

    intent_tokenizer = Tokenizer()
    intent_tokenizer.load(os.path.join(TRAINED_DIR, "vocab.json"))

    intent_model = NanoNLU(
        vocab_size=meta["vocab_size"],
        embed_dim=meta["embed_dim"],
        hidden_dim=meta["hidden_dim"],
    )
    intent_model.load_state_dict(torch.load(
        os.path.join(TRAINED_DIR, "nano_nlu.pt"), weights_only=True,
    ))
    intent_model.train(False)
    params = sum(p.numel() for p in intent_model.parameters())
    print(f"  [intent] {params:,} parameters")

    priority_model, priority_tokenizer = load_extractor("priority")
    time_model, time_tokenizer = load_extractor("time")

    total = params + sum(p.numel() for p in priority_model.parameters()) + sum(p.numel() for p in time_model.parameters())
    print(f"Pipeline loaded: {total:,} total parameters (mode={EXTRACTION_MODE})")


def bio_extract(mdl, tok, text):
    ids = tok.encode(text)
    if not ids:
        return "", text

    ids_tensor = torch.tensor([ids], dtype=torch.long)
    lengths = torch.tensor([len(ids)])

    with torch.no_grad():
        bio_logits = mdl(ids_tensor, lengths)

    preds = bio_logits[0, :len(ids)].argmax(dim=1).tolist()
    words = text.lower().split()

    extracted = []
    remaining = []
    for word, tag_idx in zip(words, preds):
        if BIO_LABELS[tag_idx] in ("B", "I"):
            extracted.append(word)
        else:
            remaining.append(word)

    return " ".join(extracted), " ".join(remaining)


def bio_tag(mdl, tok, text):
    ids = tok.encode(text)
    if not ids:
        return []
    ids_tensor = torch.tensor([ids], dtype=torch.long)
    lengths = torch.tensor([len(ids)])
    with torch.no_grad():
        bio_logits = mdl(ids_tensor, lengths)
    return bio_logits[0, :len(ids)].argmax(dim=1).tolist()


def reconcile(words, prio_tags, time_tags):
    title_words, prio_words, time_words = [], [], []
    for word, pt, tt in zip(words, prio_tags, time_tags):
        p = BIO_LABELS[pt] in ("B", "I")
        t = BIO_LABELS[tt] in ("B", "I")
        if p and t:
            title_words.append(word)
        elif p:
            prio_words.append(word)
        elif t:
            time_words.append(word)
        else:
            title_words.append(word)
    return " ".join(title_words), " ".join(prio_words), " ".join(time_words)


def predict(text):
    ids = intent_tokenizer.encode(text)
    if not ids:
        return "unknown", "", "", "", 0.0

    ids_tensor = torch.tensor([ids], dtype=torch.long)
    lengths = torch.tensor([len(ids)])

    with torch.no_grad():
        intent_logits, bio_logits = intent_model(ids_tensor, lengths)

    intent_probs = torch.softmax(intent_logits, dim=1)
    intent_idx = intent_probs.argmax(dim=1).item()
    confidence = intent_probs[0, intent_idx].item()
    intent = INTENT_LABELS[intent_idx]

    bio_preds = bio_logits[0, :len(ids)].argmax(dim=1).tolist()
    words = text.lower().split()
    entity_words = []
    for word, tag_idx in zip(words, bio_preds):
        if BIO_LABELS[tag_idx] in ("B", "I"):
            entity_words.append(word)
    entity = " ".join(entity_words)

    if EXTRACTION_MODE == "sequential":
        priority, after_priority = bio_extract(priority_model, priority_tokenizer, entity)
        time_expr, title = bio_extract(time_model, time_tokenizer, after_priority)
    else:
        prio_tags = bio_tag(priority_model, priority_tokenizer, entity)
        time_tags = bio_tag(time_model, time_tokenizer, entity)
        e_words = entity.split()
        title, priority, time_expr = reconcile(e_words, prio_tags, time_tags)

    return intent, title, priority, time_expr, confidence


@app.route("/classify", methods=["POST"])
def classify():
    text = request.get_data(as_text=True).strip()
    if not text:
        return jsonify({"error": "empty request"}), 400

    intent, entity, priority, time_expr, confidence = predict(text)
    return jsonify({
        "intent": intent,
        "entity": entity,
        "priority": priority,
        "time": time_expr,
        "confidence": round(confidence, 4),
    })


if __name__ == "__main__":
    load_all()
    port = int(os.environ.get("MODEL_PORT", 5001))
    print(f"Sidecar listening on :{port}")
    app.run(host="127.0.0.1", port=port)
