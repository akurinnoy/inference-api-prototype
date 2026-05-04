# TODO Recharged

A TODO app with two API surfaces: a classic REST API for the web frontend, and a natural language inference endpoint for AI agents. Both share the same store.

## The idea

Classical APIs are explicit contracts designed for human programmers. This prototype adds an inference layer on top — same capabilities, but agents interact via natural language instead of structured HTTP calls.

```bash
# Classic API (for the frontend)
curl localhost:8080/api/todos
curl -X POST localhost:8080/api/todos -H 'Content-Type: application/json' \
  -d '{"title":"buy milk","priority":"urgent","time":"today at 6pm"}'

# Recharged API (for agents)
curl -X POST localhost:8080/infer -d "remind me to buy milk today at 6pm urgent"
curl -X POST localhost:8080/infer -d "show all"
```

Both hit the same store. A todo created via REST is visible via inference and vice versa.

## How it works

A pipeline of three nano models (365K total parameters, trained from scratch) processes natural language:

```
"remind me to buy milk today at 6pm urgent"
  → Intent model (185K)    → intent: create, entity: "buy milk today at 6pm urgent"
  → Priority model (91K)   → priority: "urgent"      ─┐
  → Time model (89K)       → time: "today at 6pm"    ──┤→ reconcile → title: "buy milk"
```

Each model is a BIO sequence tagger. In parallel mode (default), both extractors tag the same entity text independently and a reconciliation step merges the results. In sequential mode (`EXTRACTION_MODE=sequential`), each passes remaining words downstream. New capabilities are added by adding models to the pipeline, not retraining existing ones.

## Agent discovery

AI agents discover the app via `GET /agents.md`:

```bash
curl localhost:8080/agents.md
```

This returns a plain-text description of capabilities with natural language examples. No OpenAPI spec, no SDK.

## Running

```bash
# First-time setup
cd model
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python3 generate_data.py
.venv/bin/python3 generate_priority_data.py
.venv/bin/python3 generate_time_data.py
.venv/bin/python3 train.py
.venv/bin/python3 train_extractor.py priority
.venv/bin/python3 train_extractor.py time

# Run (parallel extraction is default)
cd model && .venv/bin/python3 serve.py   # terminal 1: model sidecar on :5001
go run .                                  # terminal 2: app on :8080

# Or run with sequential extraction
EXTRACTION_MODE=sequential .venv/bin/python3 serve.py
```

Open http://localhost:8080 or use curl.

## Endpoints

### Classic REST API (for the web frontend)

```bash
# List all todos
curl localhost:8080/api/todos

# Create a todo
curl -X POST localhost:8080/api/todos \
  -H 'Content-Type: application/json' \
  -d '{"title":"buy milk","priority":"urgent","time":"today at 6pm"}'

# Complete a todo
curl -X PATCH localhost:8080/api/todos/<id>

# Delete a todo
curl -X DELETE localhost:8080/api/todos/<id>
```

### Inference API (for agents)

```bash
# Create — natural language, priority and time extracted automatically
curl -X POST localhost:8080/infer -d "remind me to buy milk today at 6pm urgent"
# → {"ok":true,"result":{"id":"...","title":"buy milk","priority":"urgent","time":"today at 6pm",...}}

# List
curl -X POST localhost:8080/infer -d "show all"

# Complete — reference by title
curl -X POST localhost:8080/infer -d "mark milk as done"

# Delete — reference by title
curl -X POST localhost:8080/infer -d "remove the meeting"

# Unknown input
curl -X POST localhost:8080/infer -d "fly to the moon"
# → {"ok":false,"error":"I didn't understand, see /agents.md for usage"}
```

### Agent discovery

```bash
curl localhost:8080/agents.md
```

## Testing

```bash
go test ./...                                    # 13 Go tests (REST API + inference, 2 require sidecar)
cd model && .venv/bin/python3 test_model.py       # 37 tests (intent + entity)
cd model && .venv/bin/python3 test_pipeline.py     # 32 tests (priority + time pipeline, both modes)
```

## Background

This prototype accompanies the article *"From API to API: The Rise of Agentic Programming Inference"* - an argument that software architecture is shifting from explicit contracts to inference over context.
