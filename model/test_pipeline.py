import json
import os
import torch

from tokenizer import Tokenizer
from model import NanoNLU, NanoExtractor, INTENT_LABELS, BIO_LABELS

BASE_DIR = os.path.dirname(__file__)
TRAINED_DIR = os.path.join(BASE_DIR, "trained")

PASS = 0
FAIL = 0


def load_intent_model():
    with open(os.path.join(TRAINED_DIR, "meta.json")) as f:
        meta = json.load(f)
    tok = Tokenizer()
    tok.load(os.path.join(TRAINED_DIR, "vocab.json"))
    mdl = NanoNLU(meta["vocab_size"], meta["embed_dim"], meta["hidden_dim"])
    mdl.load_state_dict(torch.load(os.path.join(TRAINED_DIR, "nano_nlu.pt"), weights_only=True))
    mdl.train(False)
    return mdl, tok


def load_extractor(subdir):
    path = os.path.join(TRAINED_DIR, subdir)
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    tok = Tokenizer()
    tok.load(os.path.join(path, "vocab.json"))
    mdl = NanoExtractor(meta["vocab_size"], meta["embed_dim"], meta["hidden_dim"])
    mdl.load_state_dict(torch.load(os.path.join(path, "model.pt"), weights_only=True))
    mdl.train(False)
    return mdl, tok


def bio_extract(mdl, tok, text):
    ids = tok.encode(text)
    if not ids:
        return "", text
    ids_tensor = torch.tensor([ids], dtype=torch.long)
    lengths = torch.tensor([len(ids)])
    with torch.no_grad():
        logits = mdl(ids_tensor, lengths)
    preds = logits[0, :len(ids)].argmax(dim=1).tolist()
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
        logits = mdl(ids_tensor, lengths)
    return logits[0, :len(ids)].argmax(dim=1).tolist()


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


def predict_intent(mdl, tok, text):
    ids = tok.encode(text)
    if not ids:
        return "unknown", "", 0.0
    ids_tensor = torch.tensor([ids], dtype=torch.long)
    lengths = torch.tensor([len(ids)])
    with torch.no_grad():
        intent_logits, bio_logits = mdl(ids_tensor, lengths)
    probs = torch.softmax(intent_logits, dim=1)
    idx = probs.argmax(dim=1).item()
    intent = INTENT_LABELS[idx]
    confidence = probs[0, idx].item()
    bio_preds = bio_logits[0, :len(ids)].argmax(dim=1).tolist()
    words = text.lower().split()
    entity = " ".join(w for w, t in zip(words, bio_preds) if BIO_LABELS[t] in ("B", "I"))
    return intent, entity, confidence


def check(text, expected_intent, expected_title=None, expected_priority=None, expected_time=None,
          mode="parallel", intent_mdl=None, intent_tok=None, prio_mdl=None, prio_tok=None,
          time_mdl=None, time_tok=None):
    global PASS, FAIL

    intent, entity, _ = predict_intent(intent_mdl, intent_tok, text)

    if mode == "sequential":
        priority, after_priority = bio_extract(prio_mdl, prio_tok, entity)
        time_expr, title = bio_extract(time_mdl, time_tok, after_priority)
    else:
        prio_tags = bio_tag(prio_mdl, prio_tok, entity)
        time_tags = bio_tag(time_mdl, time_tok, entity)
        e_words = entity.split()
        title, priority, time_expr = reconcile(e_words, prio_tags, time_tags)

    errors = []
    if intent != expected_intent:
        errors.append(f"intent: got {intent!r}, want {expected_intent!r}")
    if expected_title is not None and title != expected_title:
        errors.append(f"title: got {title!r}, want {expected_title!r}")
    if expected_priority is not None and priority != expected_priority:
        errors.append(f"priority: got {priority!r}, want {expected_priority!r}")
    if expected_time is not None and time_expr != expected_time:
        errors.append(f"time: got {time_expr!r}, want {expected_time!r}")

    if errors:
        FAIL += 1
        print(f"  FAIL  [{mode}] {text!r}")
        for e in errors:
            print(f"        {e}")
    else:
        PASS += 1
        print(f"  PASS  [{mode}] {text!r}")


def run_tests(mode, **models):
    print(f"\n{'='*50}")
    print(f"Mode: {mode}")
    print(f"{'='*50}")

    print("\nFull extraction (title + priority + time):")
    check("add buy milk today at 6pm urgent",
          "create", "buy milk", "urgent", "today at 6pm", mode=mode, **models)
    check("remind me to call the dentist tomorrow asap",
          "create", "call the dentist", "asap", "tomorrow", mode=mode, **models)
    check("add meeting with boss next monday high priority",
          "create", "meeting with boss", "high priority", "next monday", mode=mode, **models)

    print("\nTitle + time, no priority:")
    check("add buy milk tomorrow",
          "create", "buy milk", "", "tomorrow", mode=mode, **models)
    check("remind me to call mom at 6pm",
          "create", "call mom", "", "at 6pm", mode=mode, **models)
    check("add meeting with boss next friday",
          "create", "meeting with boss", "", "next friday", mode=mode, **models)

    print("\nTitle + priority, no time:")
    check("add review PR urgent",
          "create", "review pr", "urgent", "", mode=mode, **models)
    check("add fix the login bug high priority",
          "create", "fix the login bug", "high priority", "", mode=mode, **models)
    check("add pay electricity bill asap",
          "create", "pay electricity bill", "asap", "", mode=mode, **models)

    print("\nTitle only (no priority, no time):")
    check("add buy milk",
          "create", "buy milk", "", "", mode=mode, **models)
    check("add send the invoice to accounting",
          "create", "send the invoice to accounting", "", "", mode=mode, **models)

    print("\nNon-create intents (pipeline shouldn't break):")
    check("mark milk as done", "complete", mode=mode, **models)
    check("delete the meeting", "delete", mode=mode, **models)
    check("show all", "list", mode=mode, **models)
    check("fly to the moon", "unknown", mode=mode, **models)

    if mode == "parallel":
        print("\nParallel-specific (word order independence):")
        check("add buy milk urgent tomorrow",
              "create", "buy milk", "urgent", "tomorrow", mode=mode, **models)
        check("add meeting high priority next monday",
              "create", "meeting", "high priority", "next monday", mode=mode, **models)


def main():
    intent_mdl, intent_tok = load_intent_model()
    prio_mdl, prio_tok = load_extractor("priority")
    time_mdl, time_tok = load_extractor("time")

    models = dict(
        intent_mdl=intent_mdl, intent_tok=intent_tok,
        prio_mdl=prio_mdl, prio_tok=prio_tok,
        time_mdl=time_mdl, time_tok=time_tok,
    )

    print("--- Pipeline: intent + priority + time ---")

    run_tests("sequential", **models)
    run_tests("parallel", **models)

    print(f"\n{'='*50}")
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")

    if FAIL > 0:
        exit(1)


if __name__ == "__main__":
    main()
