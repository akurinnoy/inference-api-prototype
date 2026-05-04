import json
import os
import torch

from tokenizer import Tokenizer
from model import NanoNLU, INTENT_LABELS, BIO_LABELS

BASE_DIR = os.path.dirname(__file__)
TRAINED_DIR = os.path.join(BASE_DIR, "trained")

PASS = 0
FAIL = 0


def predict(model, tokenizer, text):
    ids = tokenizer.encode(text)
    if not ids:
        return "unknown", ""

    ids_tensor = torch.tensor([ids], dtype=torch.long)
    lengths = torch.tensor([len(ids)])

    with torch.no_grad():
        intent_logits, bio_logits = model(ids_tensor, lengths)

    intent_idx = intent_logits.argmax(dim=1).item()
    intent = INTENT_LABELS[intent_idx]

    bio_preds = bio_logits[0, :len(ids)].argmax(dim=1).tolist()
    words = text.lower().split()
    entity_words = []
    for word, tag_idx in zip(words, bio_preds):
        if BIO_LABELS[tag_idx] in ("B", "I"):
            entity_words.append(word)
    entity = " ".join(entity_words)

    return intent, entity


def check(model, tokenizer, text, expected_intent, expected_entity=None):
    global PASS, FAIL
    intent, entity = predict(model, tokenizer, text)

    intent_ok = intent == expected_intent
    entity_ok = expected_entity is None or entity == expected_entity

    if intent_ok and entity_ok:
        PASS += 1
        print(f"  PASS  {text!r}")
    else:
        FAIL += 1
        parts = []
        if not intent_ok:
            parts.append(f"intent: got {intent!r}, expected {expected_intent!r}")
        if not entity_ok:
            parts.append(f"entity: got {entity!r}, expected {expected_entity!r}")
        print(f"  FAIL  {text!r} — {', '.join(parts)}")


def main():
    with open(os.path.join(TRAINED_DIR, "meta.json")) as f:
        meta = json.load(f)

    tokenizer = Tokenizer()
    tokenizer.load(os.path.join(TRAINED_DIR, "vocab.json"))

    model = NanoNLU(
        vocab_size=meta["vocab_size"],
        embed_dim=meta["embed_dim"],
        hidden_dim=meta["hidden_dim"],
    )
    model.load_state_dict(torch.load(
        os.path.join(TRAINED_DIR, "nano_nlu.pt"),
        weights_only=True,
    ))
    model.train(False)

    print("--- Intent classification ---")

    print("\nCREATE intents:")
    check(model, tokenizer, "add buy milk", "create", "buy milk")
    check(model, tokenizer, "remind me to call the dentist", "create", "call the dentist")
    check(model, tokenizer, "new task: review PR #42", "create", "review pr #42")
    check(model, tokenizer, "I need to finish the report by Friday", "create", "finish the report by friday")
    check(model, tokenizer, "schedule meeting with boss tomorrow at 12:15", "create", "meeting with boss tomorrow at 12:15")
    check(model, tokenizer, "don't forget to pick up kids from school", "create", "pick up kids from school")

    print("\nCOMPLETE intents:")
    check(model, tokenizer, "mark buy milk as done", "complete", "buy milk")
    check(model, tokenizer, "finish the report", "complete", "finish the report")
    check(model, tokenizer, "I finished the meeting task", "complete")
    check(model, tokenizer, "buy milk is done", "complete", "buy milk")
    check(model, tokenizer, "check off call the dentist", "complete", "call the dentist")

    print("\nDELETE intents:")
    check(model, tokenizer, "delete buy milk", "delete", "buy milk")
    check(model, tokenizer, "remove the meeting", "delete", "meeting")
    check(model, tokenizer, "I don't need buy milk anymore", "delete", "buy milk")
    check(model, tokenizer, "scratch call the dentist", "delete", "call the dentist")
    check(model, tokenizer, "cancel the meeting", "delete", "meeting")

    print("\nLIST intents:")
    check(model, tokenizer, "show all", "list")
    check(model, tokenizer, "what do I have to do", "list")
    check(model, tokenizer, "list everything", "list")
    check(model, tokenizer, "show me my tasks", "list")
    check(model, tokenizer, "what's on my plate", "list")

    print("\nUNKNOWN intents:")
    check(model, tokenizer, "hello", "unknown")
    check(model, tokenizer, "how are you", "unknown")
    check(model, tokenizer, "what time is it", "unknown")
    check(model, tokenizer, "tell me a joke", "unknown")

    print("\n--- Unknown rejection (verb+object patterns that aren't todos) ---")
    check(model, tokenizer, "fly to the moon", "unknown")
    check(model, tokenizer, "cook dinner for 10 people", "unknown")
    check(model, tokenizer, "drive to the airport", "unknown")
    check(model, tokenizer, "open the door", "unknown")
    check(model, tokenizer, "deploy to production", "unknown")
    check(model, tokenizer, "order pizza", "unknown")
    check(model, tokenizer, "explain quantum physics", "unknown")
    check(model, tokenizer, "the server is down", "unknown")
    check(model, tokenizer, "I want to go home", "unknown")

    print("\n--- Entity preservation (stop words in entities) ---")
    check(model, tokenizer, "add buy milk for the cat", "create", "buy milk for the cat")
    check(model, tokenizer, "remind me to take the dog to the vet", "create", "take the dog to the vet")
    check(model, tokenizer, "add send the invoice to accounting", "create", "send the invoice to accounting")

    print(f"\n{'='*40}")
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")

    if FAIL > 0:
        exit(1)


if __name__ == "__main__":
    main()
