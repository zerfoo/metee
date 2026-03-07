package registry

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/zerfoo/metee/model"
)

// stubModel is a minimal model.Model implementation for testing.
type stubModel struct{ name string }

func (s *stubModel) Train(_ context.Context, _ [][]float64, _ []float64) error { return nil }
func (s *stubModel) Predict(_ context.Context, _ [][]float64) ([]float64, error) {
	return nil, nil
}
func (s *stubModel) Save(_ context.Context, _ string) error  { return nil }
func (s *stubModel) Load(_ context.Context, _ string) error  { return nil }
func (s *stubModel) Importance() (map[string]float64, error) { return nil, nil }
func (s *stubModel) Name() string                            { return s.name }

func resetRegistry() {
	mu.Lock()
	defer mu.Unlock()
	backends = make(map[string]func() model.Model)
}

func TestRegisterAndGet(t *testing.T) {
	tests := []struct {
		name    string
		backend string
	}{
		{name: "simple backend", backend: "xgboost"},
		{name: "another backend", backend: "lightgbm"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resetRegistry()
			RegisterBackend(tt.backend, func() model.Model {
				return &stubModel{name: tt.backend}
			})

			m, err := GetBackend(tt.backend)
			if err != nil {
				t.Fatalf("GetBackend(%q) returned error: %v", tt.backend, err)
			}
			if m.Name() != tt.backend {
				t.Errorf("got name %q, want %q", m.Name(), tt.backend)
			}
		})
	}
}

func TestGetBackendNotFound(t *testing.T) {
	tests := []struct {
		name    string
		backend string
	}{
		{name: "empty registry", backend: "nonexistent"},
		{name: "wrong name", backend: "catboost"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resetRegistry()
			_, err := GetBackend(tt.backend)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !errors.Is(err, ErrBackendNotFound) {
				t.Errorf("expected ErrBackendNotFound, got %v", err)
			}
		})
	}
}

func TestListBackends(t *testing.T) {
	resetRegistry()
	RegisterBackend("beta", func() model.Model { return &stubModel{name: "beta"} })
	RegisterBackend("alpha", func() model.Model { return &stubModel{name: "alpha"} })

	got := ListBackends()
	if len(got) != 2 {
		t.Fatalf("expected 2 backends, got %d", len(got))
	}
	if got[0] != "alpha" || got[1] != "beta" {
		t.Errorf("expected [alpha beta], got %v", got)
	}
}

func TestConcurrentAccess(t *testing.T) {
	resetRegistry()
	var wg sync.WaitGroup
	const n = 100

	// Concurrent registrations.
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			name := "backend"
			RegisterBackend(name, func() model.Model {
				return &stubModel{name: name}
			})
		}(i)
	}

	// Concurrent gets (some may fail, that's fine).
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = GetBackend("backend")
		}()
	}

	// Concurrent lists.
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = ListBackends()
		}()
	}

	wg.Wait()
}
