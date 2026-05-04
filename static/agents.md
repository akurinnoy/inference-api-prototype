# TODO Recharged

A TODO app with a tiny NLU model. Follow these patterns exactly or parsing will fail.

## Creating tasks

Send plain text to `POST /infer`.

**Pattern:** `add {verb} the {noun} [time] [priority]`

IMPORTANT: always include "the" between verb and noun. Without "the", parsing often fails.

### Tested examples that work

```
add buy milk
add buy groceries today
add call the dentist tomorrow
add call mom tomorrow urgent
add fix the bug urgent
add fix the bug by friday
add fix the api today
add review the pr today
add merge the pr
add write the report asap
add write the report next week
add write the tests high priority
add read the book tomorrow
add send the report by friday
add pay the bill tomorrow urgent
add call the vet tomorrow
add call the vet today urgent
add clean the kitchen today urgent
add clean the kitchen next monday
add check the plants
add send the email
add update the api
add check the api
add run the deployment
add run the deployment today urgent
add finish the report next week
add check the meeting next monday
```

### Rephrasing technical tasks

The model has ~130 known words. Use this translation table:

| You want to say | Send this instead | Why |
|---|---|---|
| Deploy the service | `add run the deployment` | "deploy" as verb fails, use "deployment" as noun |
| Update the server | `add update the api` | "server" causes parsing errors |
| Fix the server | `add fix the api` | "server" causes parsing errors |
| Check server status | `add check the api` | "server" causes parsing errors |
| Debug production issue | `add fix the bug urgent` | "debug" drops from title |
| Run integration tests | `add run the tests` | avoid "integration" |
| Fix CI/CD pipeline | `add fix the tests` | "pipeline" and "ci/cd" are unknown, "build" drops the verb |
| Update Kubernetes | `add run the deployment` | "kubernetes" is unknown; "update deployment" drops verb |
| Review pull request | `add review the pr` | spell out as "pr" not "pull request" |
| Write documentation | `add write the report` | "documentation" misparses |
| Monitor the system | `add check the api today` | "monitor" as verb unreliable |

### Nouns that work well

bug, pr, tests, report, book, bill, dentist, kitchen, dog, milk, groceries, database, api, deployment, slides, demo, meeting, mom, file, data, backup, version, email, phone, team, task, note, page, plants, cat, kids, lunch, pizza, flight, invoice, vet, proposal

### Words to AVOID (cause parse errors)

Nouns: server, hotfix, production, staging, app, build. Verbs: walk, feed, deploy, debug, monitor, schedule.

### Priority words

urgent, asap, high priority, low priority, critical, top priority

### Time expressions

today, tomorrow, tonight, next monday, next week, next friday, by friday, at 6pm, in 20 minutes

### Combining time + priority

Put time FIRST, then priority LAST: `add call the vet today urgent`, `add run the deployment today critical`, `add write the report next week asap`

## Creating: two-step flow

Creating a task is a dialogue:

1. **Send natural language** (plain text) → the service parses it and returns what it understood:
   ```json
   {"ok": true, "action": "confirm", "result": {"title": "buy milk", "priority": "urgent", "time": "tomorrow"}}
   ```

2. **You decide:**
   - Parse is correct → send JSON to confirm:
     `POST /infer` with `Content-Type: application/json`
     ```json
     {"title": "buy milk", "priority": "urgent", "time": "tomorrow"}
     ```
   - Parse is wrong or partial → correct it and send JSON with the right values:
     ```json
     {"title": "buy almond milk", "priority": "urgent", "time": "next week"}
     ```
   Common corrections: if title is truncated, send the full intended title. If time lost a word (e.g. "week" instead of "next week"), add the correct time in your JSON.

The service creates the todo only when it receives JSON. Natural language always returns a parse for review.

## Other actions (single-step, no confirmation)

- **Complete:** `finish buy groceries` or `mark meeting done`
- **Delete:** `remove buy groceries` or `delete the task`
- **List:** `show all` or `list everything`

## Rules

1. Start every create request with **add**
2. Keep titles to 3-5 simple words
3. Never use words not listed above
4. Put time and priority at the end only
5. No colons or special punctuation
6. Prefer everyday words over technical jargon
