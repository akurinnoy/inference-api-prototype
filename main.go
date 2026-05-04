package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

type InferResponse struct {
	OK      bool   `json:"ok"`
	Result  any    `json:"result,omitempty"`
	Error   string `json:"error,omitempty"`
	Action  string `json:"action,omitempty"`
	Message string `json:"message,omitempty"`
}

type ParsedCreate struct {
	OriginalInput string `json:"original_input"`
	Title         string `json:"title"`
	Priority      string `json:"priority,omitempty"`
	Time          string `json:"time,omitempty"`
}

func main() {
	store := NewTodoStore()

	mux := http.NewServeMux()

	mux.HandleFunc("/infer", handleInfer(store))
	mux.HandleFunc("/agents.md", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/markdown; charset=utf-8")
		http.ServeFile(w, r, "static/agents.md")
	})

	mux.HandleFunc("/api/todos", handleTodos(store))
	mux.HandleFunc("/api/todos/", handleTodoByID(store))

	mux.Handle("/", http.FileServer(http.Dir("static")))

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("TODO Recharged listening on :%s", port)
	log.Fatal(http.ListenAndServe(":"+port, mux))
}

func handleInfer(store *TodoStore) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeJSON(w, http.StatusMethodNotAllowed, InferResponse{Error: "method not allowed"})
			return
		}

		body, err := io.ReadAll(r.Body)
		if err != nil || len(body) == 0 {
			writeJSON(w, http.StatusBadRequest, InferResponse{Error: "empty request"})
			return
		}

		ct := r.Header.Get("Content-Type")

		if strings.Contains(ct, "application/json") {
			handleJSONCreate(w, store, body)
			return
		}

		handleNaturalLanguage(w, store, body)
	}
}

func handleJSONCreate(w http.ResponseWriter, store *TodoStore, body []byte) {
	var input struct {
		Title    string `json:"title"`
		Priority string `json:"priority"`
		Time     string `json:"time"`
	}
	if err := json.Unmarshal(body, &input); err != nil || input.Title == "" {
		writeJSON(w, http.StatusBadRequest, InferResponse{Error: "JSON input requires a 'title' field"})
		return
	}

	todo := store.Create(input.Title, input.Priority, input.Time)
	log.Printf("[infer] JSON create → id=%s title=%q", todo.ID, todo.Title)
	writeJSON(w, http.StatusOK, InferResponse{OK: true, Result: todo})
}

func handleNaturalLanguage(w http.ResponseWriter, store *TodoStore, body []byte) {
	input := string(body)
	log.Printf("[infer] ← %q", input)

	c, source, err := classifyInput(input)
	if err != nil {
		log.Printf("[classify] %s", err)
		writeJSON(w, http.StatusServiceUnavailable, InferResponse{Error: "inference service unavailable"})
		return
	}
	log.Printf("[classify] intent=%s  entity=%q  via=%s", c.Intent, c.Entity, source)

	resp, err := dispatch(store, c)
	if err != nil {
		log.Printf("[error] %s", err)
		writeJSON(w, http.StatusOK, InferResponse{Error: err.Error()})
		return
	}

	if r, ok := resp.(InferResponse); ok {
		log.Printf("[ok] intent=%s → confirm", c.Intent)
		writeJSON(w, http.StatusOK, r)
		return
	}

	log.Printf("[ok] intent=%s completed", c.Intent)
	writeJSON(w, http.StatusOK, InferResponse{OK: true, Result: resp})
}

func dispatch(store *TodoStore, c Classification) (any, error) {
	switch c.Intent {
	case "create":
		if c.Entity == "" {
			return nil, fmt.Errorf("I didn't understand what to create. Try: add buy milk")
		}
		parsed := ParsedCreate{
			OriginalInput: c.OriginalInput,
			Title:         c.Entity,
			Priority:      c.Priority,
			Time:          c.Time,
		}
		log.Printf("[parse] title=%q priority=%q time=%q", parsed.Title, parsed.Priority, parsed.Time)
		return InferResponse{
			OK:      true,
			Action:  "confirm",
			Result:  parsed,
			Message: fmt.Sprintf("I parsed: title=%q, priority=%q, time=%q. Send JSON to /infer to confirm or correct.", parsed.Title, parsed.Priority, parsed.Time),
		}, nil

	case "complete":
		return resolveAndAct(store, c.Entity, "complete", func(id string) (any, error) {
			return store.Complete(id)
		})

	case "delete":
		return resolveAndAct(store, c.Entity, "delete", func(id string) (any, error) {
			return nil, store.Delete(id)
		})

	case "list":
		todos := store.List()
		log.Printf("[list] %d todos", len(todos))
		return todos, nil

	default:
		return nil, fmt.Errorf("I didn't understand, see /agents.md for usage")
	}
}

func resolveAndAct(store *TodoStore, query string, intent string, action func(string) (any, error)) (any, error) {
	if query == "" {
		return nil, fmt.Errorf("I didn't understand which todo you mean. Be more specific")
	}

	log.Printf("[resolve] searching for %q", query)
	matches, err := store.FindByTitle(query)
	if err != nil {
		log.Printf("[resolve] no match for %q", query)
		return nil, err
	}

	if len(matches) > 1 {
		titles := make([]string, len(matches))
		for i, m := range matches {
			titles[i] = m.Title
		}
		log.Printf("[resolve] ambiguous: %q matched %d todos: %v", query, len(matches), titles)
		return nil, fmt.Errorf("multiple todos match '%s': %v. Be more specific", query, titles)
	}

	log.Printf("[%s] resolved %q → id=%s title=%q", intent, query, matches[0].ID, matches[0].Title)
	return action(matches[0].ID)
}

// --- Classic REST API handlers ---

func handleTodos(store *TodoStore) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			log.Printf("[api] GET /api/todos")
			writeJSON(w, http.StatusOK, store.List())

		case http.MethodPost:
			var body struct {
				Title    string `json:"title"`
				Priority string `json:"priority"`
				Time     string `json:"time"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.Title == "" {
				writeJSON(w, http.StatusBadRequest, map[string]string{"error": "title is required"})
				return
			}
			todo := store.Create(body.Title, body.Priority, body.Time)
			log.Printf("[api] POST /api/todos → id=%s title=%q", todo.ID, todo.Title)
			writeJSON(w, http.StatusCreated, todo)

		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	}
}

func handleTodoByID(store *TodoStore) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id := strings.TrimPrefix(r.URL.Path, "/api/todos/")
		if id == "" {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "missing todo id"})
			return
		}

		switch r.Method {
		case http.MethodPatch:
			todo, err := store.Complete(id)
			if err != nil {
				writeJSON(w, http.StatusNotFound, map[string]string{"error": err.Error()})
				return
			}
			log.Printf("[api] PATCH /api/todos/%s → completed", id)
			writeJSON(w, http.StatusOK, todo)

		case http.MethodDelete:
			if err := store.Delete(id); err != nil {
				writeJSON(w, http.StatusNotFound, map[string]string{"error": err.Error()})
				return
			}
			log.Printf("[api] DELETE /api/todos/%s", id)
			w.WriteHeader(http.StatusNoContent)

		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	}
}

// --- Sidecar integration ---

type Classification struct {
	Intent        string
	Entity        string
	Priority      string
	Time          string
	OriginalInput string
}

var sidecarURL = "http://localhost:5001/classify"

var httpClient = &http.Client{Timeout: 5 * time.Second}

type sidecarResponse struct {
	Intent     string  `json:"intent"`
	Entity     string  `json:"entity"`
	Priority   string  `json:"priority"`
	Time       string  `json:"time"`
	Confidence float64 `json:"confidence"`
}

func classifyInput(input string) (Classification, string, error) {
	resp, err := httpClient.Post(sidecarURL, "text/plain", strings.NewReader(input))
	if err != nil {
		return Classification{}, "", fmt.Errorf("model sidecar unreachable: %v", err)
	}
	defer resp.Body.Close()

	var sr sidecarResponse
	if err := json.NewDecoder(resp.Body).Decode(&sr); err != nil {
		return Classification{}, "", fmt.Errorf("model sidecar response error: %v", err)
	}

	return Classification{
		Intent:        sr.Intent,
		Entity:        sr.Entity,
		Priority:      sr.Priority,
		Time:          sr.Time,
		OriginalInput: input,
	}, fmt.Sprintf("model(%.2f)", sr.Confidence), nil
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}
