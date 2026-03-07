// Package registry provides a thread-safe registry for model backends.
package registry

import (
	"errors"
	"fmt"
	"sort"
	"sync"

	"github.com/zerfoo/metee/model"
)

// ErrBackendNotFound is returned when a requested backend is not registered.
var ErrBackendNotFound = errors.New("registry: backend not found")

var (
	mu       sync.RWMutex
	backends = make(map[string]func() model.Model)
)

// RegisterBackend registers a backend factory under the given name.
func RegisterBackend(name string, factory func() model.Model) {
	mu.Lock()
	defer mu.Unlock()
	backends[name] = factory
}

// GetBackend creates a new model instance from the named backend's factory.
func GetBackend(name string) (model.Model, error) {
	mu.RLock()
	defer mu.RUnlock()
	factory, ok := backends[name]
	if !ok {
		return nil, fmt.Errorf("registry: backend %q not found: %w", name, ErrBackendNotFound)
	}
	return factory(), nil
}

// ListBackends returns the names of all registered backends in sorted order.
func ListBackends() []string {
	mu.RLock()
	defer mu.RUnlock()
	names := make([]string, 0, len(backends))
	for name := range backends {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}
