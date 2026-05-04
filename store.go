package main

import (
	"crypto/rand"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

type Todo struct {
	ID        string `json:"id"`
	Title     string `json:"title"`
	Priority  string `json:"priority,omitempty"`
	Time      string `json:"time,omitempty"`
	Completed bool   `json:"completed"`
	CreatedAt string `json:"created_at"`
}

type TodoStore struct {
	mu    sync.RWMutex
	todos map[string]*Todo
}

func NewTodoStore() *TodoStore {
	return &TodoStore{todos: make(map[string]*Todo)}
}

func (s *TodoStore) Create(title, priority, timeExpr string) *Todo {
	s.mu.Lock()
	defer s.mu.Unlock()

	id := newID()
	todo := &Todo{
		ID:        id,
		Title:     title,
		Priority:  priority,
		Time:      timeExpr,
		Completed: false,
		CreatedAt: time.Now().UTC().Format(time.RFC3339),
	}
	s.todos[id] = todo
	return todo
}

func (s *TodoStore) Complete(id string) (*Todo, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	todo, ok := s.todos[id]
	if !ok {
		return nil, fmt.Errorf("todo not found: %s", id)
	}
	todo.Completed = true
	return todo, nil
}

func (s *TodoStore) Delete(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.todos[id]; !ok {
		return fmt.Errorf("todo not found: %s", id)
	}
	delete(s.todos, id)
	return nil
}

func (s *TodoStore) List() []*Todo {
	s.mu.RLock()
	defer s.mu.RUnlock()

	todos := make([]*Todo, 0, len(s.todos))
	for _, t := range s.todos {
		todos = append(todos, t)
	}
	sort.Slice(todos, func(i, j int) bool {
		return todos[i].CreatedAt < todos[j].CreatedAt
	})
	return todos
}

func (s *TodoStore) FindByTitle(query string) ([]*Todo, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	query = strings.ToLower(strings.TrimSpace(query))
	if query == "" {
		return nil, fmt.Errorf("empty search query")
	}

	var matches []*Todo
	for _, t := range s.todos {
		if strings.Contains(strings.ToLower(t.Title), query) {
			matches = append(matches, t)
		}
	}

	if len(matches) == 0 {
		return nil, fmt.Errorf("no todo matching '%s'", query)
	}
	return matches, nil
}

func newID() string {
	b := make([]byte, 8)
	rand.Read(b)
	return fmt.Sprintf("%x", b)
}
