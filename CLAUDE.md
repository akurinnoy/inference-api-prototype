# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A prototype TODO web app that demonstrates both a classical REST API and a "recharged" natural language inference endpoint side by side. The web frontend uses the classic API (`/api/todos`). AI agents use the inference endpoint (`POST /infer`) with plain text. Both share the same in-memory store.

A pipeline of three nano models (365K total params) handles intent classification, priority extraction, and time extraction via BIO sequence tagging.


## Build & Run

```bash
# Start the model sidecar (terminal 1)
cd model && .venv/bin/python3 serve.py

# Start the Go server (terminal 2)
go run .                  # starts on :8080
PORT=9000 go run .        # custom port
```

The Go server calls the sidecar on `:5001` for classification. The sidecar must be running for the inference endpoint to work.

### First-time setup

```bash
cd model
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python3 generate_data.py
.venv/bin/python3 generate_priority_data.py
.venv/bin/python3 generate_time_data.py
.venv/bin/python3 train.py
.venv/bin/python3 train_extractor.py priority
.venv/bin/python3 train_extractor.py time
```

## Testing

```bash
go test ./...                                          # 13 Go tests (REST API + confirmation flow, 2 require sidecar)
cd model && .venv/bin/python3 test_model.py             # 37 tests (intent + entity extraction)
cd model && .venv/bin/python3 test_pipeline.py           # 32 tests (priority + time pipeline, both modes)
```

### Comparing sequential vs parallel extraction

Test cases live in `model/data/test_cases.json`. Format:

```json
[
  {
    "input": "add buy milk today at 6pm urgent",
    "intent": "create",
    "title": "buy milk",
    "priority": "urgent",
    "time": "today at 6pm"
  },
  {
    "input": "show all",
    "intent": "list",
    "title": null,
    "priority": null,
    "time": null
  }
]
```

Fields: `input` (the raw text), `intent` (create/complete/delete/list/unknown), `title`, `priority`, `time`. Use `null` for fields you don't want to assert (e.g., non-create intents).

Run the comparison:

```bash
cd model && .venv/bin/python3 test_compare.py           # side-by-side sequential vs parallel
```

Add new test cases by editing `model/data/test_cases.json` — no code changes needed.

### Manual testing via curl

```bash
curl -X POST http://localhost:8080/infer -d 'add buy milk'           # returns parse (action: confirm)
curl -X POST http://localhost:8080/infer -H 'Content-Type: application/json' \
  -d '{"title":"buy milk"}'                                         # JSON → creates todo
curl -X POST http://localhost:8080/infer -d 'show all'
curl -X POST http://localhost:8080/infer -d 'mark milk as done'
curl -X POST http://localhost:8080/infer -d 'delete milk'
curl http://localhost:8080/agents.md            # agent discovery
```

## Architecture

Two API surfaces share one store:

- **Classic REST** (`/api/todos`) — `GET`, `POST`, `PATCH /:id`, `DELETE /:id`. Used by the web frontend.
- **Inference** (`POST /infer`) — plain text in, JSON out. Used by AI agents. Routed through a pipeline of nano models.

Request flow for inference: **raw text → intent model → priority extractor + time extractor → dispatch → store → JSON response**. Extractors run in parallel by default (both receive the entity, a reconciliation step merges tags). Set `EXTRACTION_MODE=sequential` to chain them instead.

### Files

- `main.go` — HTTP server. Routes for both REST API and inference endpoint. `classifyInput()` calls the sidecar.
- `main_test.go` — Go tests for REST API CRUD and shared-store verification.
- `store.go` — Thread-safe in-memory todo store (`sync.RWMutex`). CRUD operations + `FindByTitle()` for fuzzy substring matching.
- `static/agents.md` — Discovery endpoint for AI agents. Served at `GET /agents.md`.
- `static/` — Vanilla HTML/CSS/JS frontend. Uses the classic REST API.
- `model/` — Python nano model pipeline (3 models, 365K total params):
  - Intent model (185K params) — joint intent classification (5 classes) + BIO entity extraction
  - Priority extractor (91K params) — BIO tagging for priority expressions
  - Time extractor (89K params) — BIO tagging for time expressions
  - `serve.py` — Flask sidecar on `:5001`, runs all three models (parallel by default, sequential via `EXTRACTION_MODE=sequential`)

## Key Design Decisions

- **Two APIs, one store** — classic REST for the UI, inference for agents. Demonstrates coexistence.
- **Pipeline of extractors** — each model does BIO tagging on its domain. In parallel mode (default), both extractors tag the same entity text independently and a reconciliation step merges results. In sequential mode, each passes remaining words downstream. New capabilities added by adding models, not retraining existing ones.
- **Reference resolution by title substring** — complete/delete resolve "milk" to the todo containing "milk" in its title. Ambiguous matches return an error.
- **Sidecar required** — Go server requires the Python sidecar for inference. Returns 503 if the sidecar is down. Logs show `via=model(0.99)`.
