# NLU Model Pipeline

A pipeline of three nano models (365K total parameters) for natural language understanding. Each model is a BiGRU-based BIO sequence tagger trained from scratch on synthetic data.

## Architecture

```
"add buy milk tomorrow urgent"
  → Intent model (185K)  → intent: create, entity: "buy milk tomorrow urgent"
  → Priority model (91K) → priority: "urgent"       ─┐
  → Time model (89K)     → time: "tomorrow"          ─┤→ reconcile → title: "buy milk"
```

| Model | Architecture | Params | Task |
|---|---|---|---|
| Intent (`NanoNLU`) | Embedding(160) → BiGRU(80) → intent_head + bio_head | 185K | Intent classification (5 classes) + entity extraction |
| Priority (`NanoExtractor`) | Embedding(128) → BiGRU(64) → bio_head | 91K | BIO tagging for priority expressions |
| Time (`NanoExtractor`) | Embedding(128) → BiGRU(64) → bio_head | 89K | BIO tagging for time expressions |

Both extraction modes available:
- **Parallel** (default): both extractors tag the same entity text independently, reconciliation merges results
- **Sequential**: priority extracts first, time extracts from the remainder

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Training

```bash
# Generate training data
.venv/bin/python3 generate_data.py            # intent (3000 examples)
.venv/bin/python3 generate_priority_data.py   # priority (1600 examples)
.venv/bin/python3 generate_time_data.py       # time (1600 examples)

# Train models
.venv/bin/python3 train.py                    # intent → trained/nano_nlu.pt
.venv/bin/python3 train_extractor.py priority  # priority → trained/priority/model.pt
.venv/bin/python3 train_extractor.py time      # time → trained/time/model.pt
```

Trained weights are saved to `trained/`. The Go server's sidecar loads them at startup.

## Serving

```bash
.venv/bin/python3 serve.py                           # default: parallel mode on :5001
EXTRACTION_MODE=sequential .venv/bin/python3 serve.py # sequential mode
MODEL_PORT=5002 .venv/bin/python3 serve.py            # custom port
```

The sidecar exposes `POST /classify` which the Go server calls.

## Testing

```bash
.venv/bin/python3 test_model.py      # 37 tests — intent classification + entity extraction
.venv/bin/python3 test_pipeline.py   # 32 tests — full pipeline in both modes
.venv/bin/python3 test_compare.py    # 28 tests — side-by-side sequential vs parallel comparison
```

## Files

| File | Purpose |
|---|---|
| `model.py` | Model definitions: `NanoNLU` (intent+entity), `NanoExtractor` (BIO-only) |
| `tokenizer.py` | Word-level tokenizer with PAD/UNK tokens |
| `serve.py` | Flask sidecar — loads all models, exposes `/classify` |
| `train.py` | Training loop for the intent model |
| `train_extractor.py` | Shared training loop for priority/time extractors |
| `generate_data.py` | Synthetic training data for intent model |
| `generate_priority_data.py` | Synthetic training data for priority extractor |
| `generate_time_data.py` | Synthetic training data for time extractor |
| `test_model.py` | Intent model regression tests |
| `test_pipeline.py` | Full pipeline tests (both extraction modes) |
| `test_compare.py` | Sequential vs parallel comparison on `data/test_cases.json` |
| `data/test_cases.json` | Test cases with expected outputs (add your own here) |
