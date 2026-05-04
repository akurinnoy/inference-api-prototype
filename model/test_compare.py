#!/usr/bin/env python3
"""
Compare sequential vs parallel extraction on test cases from data/test_cases.json.

Loads the three models once, runs every test case in both modes,
prints a side-by-side comparison table and summary stats.
"""

import json
import os
import sys
import torch

from tokenizer import Tokenizer
from model import NanoNLU, NanoExtractor, INTENT_LABELS, BIO_LABELS

BASE_DIR = os.path.dirname(__file__)
TRAINED_DIR = os.path.join(BASE_DIR, "trained")
TEST_CASES = os.path.join(BASE_DIR, "data", "test_cases.json")


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
    extracted, remaining = [], []
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


def run_pipeline(text, mode, intent_mdl, intent_tok, prio_mdl, prio_tok, time_mdl, time_tok):
    intent, entity, confidence = predict_intent(intent_mdl, intent_tok, text)

    if mode == "sequential":
        priority, after_priority = bio_extract(prio_mdl, prio_tok, entity)
        time_expr, title = bio_extract(time_mdl, time_tok, after_priority)
    else:
        prio_tags = bio_tag(prio_mdl, prio_tok, entity)
        time_tags = bio_tag(time_mdl, time_tok, entity)
        e_words = entity.split()
        title, priority, time_expr = reconcile(e_words, prio_tags, time_tags)

    return {
        "intent": intent,
        "title": title,
        "priority": priority,
        "time": time_expr,
    }


def check_result(got, expected):
    errors = []
    if got["intent"] != expected["intent"]:
        errors.append(f"intent: {got['intent']!r} != {expected['intent']!r}")
    if expected.get("title") is not None and got["title"] != expected["title"]:
        errors.append(f"title: {got['title']!r} != {expected['title']!r}")
    if expected.get("priority") is not None and got["priority"] != expected["priority"]:
        errors.append(f"priority: {got['priority']!r} != {expected['priority']!r}")
    if expected.get("time") is not None and got["time"] != expected["time"]:
        errors.append(f"time: {got['time']!r} != {expected['time']!r}")
    return errors


def main():
    with open(TEST_CASES) as f:
        cases = json.load(f)

    intent_mdl, intent_tok = load_intent_model()
    prio_mdl, prio_tok = load_extractor("priority")
    time_mdl, time_tok = load_extractor("time")

    models = dict(
        intent_mdl=intent_mdl, intent_tok=intent_tok,
        prio_mdl=prio_mdl, prio_tok=prio_tok,
        time_mdl=time_mdl, time_tok=time_tok,
    )

    seq_pass, seq_fail = 0, 0
    par_pass, par_fail = 0, 0
    diffs = []

    print(f"Running {len(cases)} test cases in both modes...\n")

    for i, case in enumerate(cases):
        text = case["input"]
        seq = run_pipeline(text, "sequential", **models)
        par = run_pipeline(text, "parallel", **models)

        seq_errors = check_result(seq, case)
        par_errors = check_result(par, case)

        seq_ok = len(seq_errors) == 0
        par_ok = len(par_errors) == 0

        if seq_ok:
            seq_pass += 1
        else:
            seq_fail += 1
        if par_ok:
            par_pass += 1
        else:
            par_fail += 1

        seq_mark = "PASS" if seq_ok else "FAIL"
        par_mark = "PASS" if par_ok else "FAIL"

        if seq_mark != par_mark or seq != par:
            diffs.append(i)

        print(f"  {i+1:2d}. {text}")
        print(f"      seq={seq_mark}  par={par_mark}", end="")

        if seq == par:
            print("  (identical)")
        else:
            print("  << DIFFER >>")
            if case.get("title") is not None:
                print(f"      expected: title={case['title']!r}  prio={case.get('priority','')!r}  time={case.get('time','')!r}")
            print(f"      seq:      title={seq['title']!r}  prio={seq['priority']!r}  time={seq['time']!r}")
            print(f"      par:      title={par['title']!r}  prio={par['priority']!r}  time={par['time']!r}")

        if seq_errors and not par_errors:
            for e in seq_errors:
                print(f"      seq error: {e}")
        elif par_errors and not seq_errors:
            for e in par_errors:
                print(f"      par error: {e}")
        elif seq_errors and par_errors:
            for e in seq_errors:
                print(f"      seq error: {e}")
            for e in par_errors:
                print(f"      par error: {e}")

    total = len(cases)
    print(f"\n{'='*60}")
    print(f"Sequential: {seq_pass}/{total} passed, {seq_fail} failed")
    print(f"Parallel:   {par_pass}/{total} passed, {par_fail} failed")
    print(f"Differences: {len(diffs)} cases produced different results")
    print(f"{'='*60}")

    if diffs:
        print(f"\nDiffering cases: {', '.join(str(d+1) for d in diffs)}")

    if seq_fail > 0 or par_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
