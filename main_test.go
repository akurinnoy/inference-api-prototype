package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func requireSidecar(t *testing.T) {
	t.Helper()
	resp, err := http.Get(sidecarURL)
	if err != nil {
		t.Skip("model sidecar not running (start with: cd model && .venv/bin/python3 serve.py)")
	}
	resp.Body.Close()
}

func setupServer() (*http.ServeMux, *TodoStore) {
	store := NewTodoStore()
	mux := http.NewServeMux()
	mux.HandleFunc("/api/todos", handleTodos(store))
	mux.HandleFunc("/api/todos/", handleTodoByID(store))
	mux.HandleFunc("/infer", handleInfer(store))
	return mux, store
}

func TestAPICreateAndList(t *testing.T) {
	mux, _ := setupServer()

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/api/todos", strings.NewReader(`{"title":"buy milk","priority":"urgent","time":"today"}`))
	req.Header.Set("Content-Type", "application/json")
	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusCreated {
		t.Fatalf("create: got %d, want %d", rec.Code, http.StatusCreated)
	}

	var created Todo
	json.NewDecoder(rec.Body).Decode(&created)
	if created.Title != "buy milk" || created.Priority != "urgent" || created.Time != "today" {
		t.Fatalf("create: unexpected todo: %+v", created)
	}

	rec = httptest.NewRecorder()
	req = httptest.NewRequest("GET", "/api/todos", nil)
	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("list: got %d, want %d", rec.Code, http.StatusOK)
	}

	var todos []*Todo
	json.NewDecoder(rec.Body).Decode(&todos)
	if len(todos) != 1 {
		t.Fatalf("list: got %d todos, want 1", len(todos))
	}
	if todos[0].ID != created.ID {
		t.Fatalf("list: id mismatch: got %s, want %s", todos[0].ID, created.ID)
	}
}

func TestAPIComplete(t *testing.T) {
	mux, store := setupServer()
	todo := store.Create("test task", "", "")

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("PATCH", "/api/todos/"+todo.ID, nil)
	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("complete: got %d, want %d", rec.Code, http.StatusOK)
	}

	var completed Todo
	json.NewDecoder(rec.Body).Decode(&completed)
	if !completed.Completed {
		t.Fatal("complete: todo not marked as completed")
	}
}

func TestAPICompleteNotFound(t *testing.T) {
	mux, _ := setupServer()

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("PATCH", "/api/todos/nonexistent", nil)
	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("complete 404: got %d, want %d", rec.Code, http.StatusNotFound)
	}
}

func TestAPIDelete(t *testing.T) {
	mux, store := setupServer()
	todo := store.Create("to delete", "", "")

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("DELETE", "/api/todos/"+todo.ID, nil)
	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusNoContent {
		t.Fatalf("delete: got %d, want %d", rec.Code, http.StatusNoContent)
	}

	if len(store.List()) != 0 {
		t.Fatal("delete: todo still exists")
	}
}

func TestAPIDeleteNotFound(t *testing.T) {
	mux, _ := setupServer()

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("DELETE", "/api/todos/nonexistent", nil)
	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("delete 404: got %d, want %d", rec.Code, http.StatusNotFound)
	}
}

func TestAPICreateMissingTitle(t *testing.T) {
	mux, _ := setupServer()

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/api/todos", strings.NewReader(`{"priority":"high"}`))
	req.Header.Set("Content-Type", "application/json")
	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("create no title: got %d, want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestAPIMethodNotAllowed(t *testing.T) {
	mux, _ := setupServer()

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("PUT", "/api/todos", nil)
	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("method not allowed: got %d, want %d", rec.Code, http.StatusMethodNotAllowed)
	}
}

func TestAPICreateOptionalFields(t *testing.T) {
	mux, _ := setupServer()

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/api/todos", strings.NewReader(`{"title":"plain task"}`))
	req.Header.Set("Content-Type", "application/json")
	mux.ServeHTTP(rec, req)

	var todo Todo
	json.NewDecoder(rec.Body).Decode(&todo)
	if todo.Priority != "" || todo.Time != "" {
		t.Fatalf("optional fields: priority=%q time=%q, want empty", todo.Priority, todo.Time)
	}
}

// --- Inference: natural language → parse (no create) ---

func TestInferNaturalLanguageParse(t *testing.T) {
	requireSidecar(t)
	mux, store := setupServer()

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/infer", strings.NewReader("add buy milk"))
	mux.ServeHTTP(rec, req)

	var resp InferResponse
	json.NewDecoder(rec.Body).Decode(&resp)
	if resp.Action != "confirm" {
		t.Fatalf("parse: got action=%q, want 'confirm'", resp.Action)
	}
	if len(store.List()) != 0 {
		t.Fatal("parse: should not create a todo")
	}
}

// --- Inference: JSON → direct create ---

func TestInferJSONCreate(t *testing.T) {
	mux, store := setupServer()

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/infer", strings.NewReader(`{"title":"buy milk","priority":"urgent"}`))
	req.Header.Set("Content-Type", "application/json")
	mux.ServeHTTP(rec, req)

	var resp InferResponse
	json.NewDecoder(rec.Body).Decode(&resp)
	if !resp.OK {
		t.Fatalf("json create: expected ok=true, got error=%q", resp.Error)
	}

	todos := store.List()
	if len(todos) != 1 {
		t.Fatalf("json create: got %d todos, want 1", len(todos))
	}
	if todos[0].Title != "buy milk" || todos[0].Priority != "urgent" {
		t.Fatalf("json create: unexpected todo: %+v", todos[0])
	}
}

// --- Inference: parse then correct via JSON ---

func TestInferParseAndCorrect(t *testing.T) {
	requireSidecar(t)
	mux, store := setupServer()

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/infer", strings.NewReader("add buy milk"))
	mux.ServeHTTP(rec, req)

	var resp InferResponse
	json.NewDecoder(rec.Body).Decode(&resp)
	if resp.Action != "confirm" {
		t.Fatalf("parse: got action=%q, want 'confirm'", resp.Action)
	}

	rec = httptest.NewRecorder()
	req = httptest.NewRequest("POST", "/infer", strings.NewReader(`{"title":"buy almond milk","priority":"low priority"}`))
	req.Header.Set("Content-Type", "application/json")
	mux.ServeHTTP(rec, req)

	json.NewDecoder(rec.Body).Decode(&resp)
	if !resp.OK {
		t.Fatalf("correct: expected ok=true, got error=%q", resp.Error)
	}

	todos := store.List()
	if len(todos) != 1 {
		t.Fatalf("correct: got %d todos, want 1", len(todos))
	}
	if todos[0].Title != "buy almond milk" {
		t.Fatalf("correct: title=%q, want 'buy almond milk'", todos[0].Title)
	}
}

// --- Inference: JSON missing title ---

func TestInferJSONMissingTitle(t *testing.T) {
	mux, _ := setupServer()

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/infer", strings.NewReader(`{"priority":"urgent"}`))
	req.Header.Set("Content-Type", "application/json")
	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("json no title: got %d, want %d", rec.Code, http.StatusBadRequest)
	}
}

// --- Both APIs share the same store ---

func TestBothAPIsShareStore(t *testing.T) {
	mux, _ := setupServer()

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/api/todos", strings.NewReader(`{"title":"from classic api"}`))
	req.Header.Set("Content-Type", "application/json")
	mux.ServeHTTP(rec, req)

	rec = httptest.NewRecorder()
	req = httptest.NewRequest("POST", "/infer", strings.NewReader(`{"title":"from inference api"}`))
	req.Header.Set("Content-Type", "application/json")
	mux.ServeHTTP(rec, req)

	rec = httptest.NewRecorder()
	req = httptest.NewRequest("GET", "/api/todos", nil)
	mux.ServeHTTP(rec, req)

	var todos []*Todo
	json.NewDecoder(rec.Body).Decode(&todos)
	if len(todos) != 2 {
		t.Fatalf("shared store: got %d todos, want 2", len(todos))
	}
}
