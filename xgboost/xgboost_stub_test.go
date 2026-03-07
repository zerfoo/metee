//go:build !xgboost

package xgboost

import (
	"context"
	"errors"
	"testing"

	"github.com/zerfoo/metee/model"
)

// Compile-time check that Booster satisfies model.Model.
var _ model.Model = (*Booster)(nil)

func TestNewBooster(t *testing.T) {
	b := NewBooster("test", DefaultParams(), []string{"f1", "f2"})
	if b.Name() != "test" {
		t.Errorf("Name() = %q, want test", b.Name())
	}
}

func TestStubMethodsReturnError(t *testing.T) {
	b := NewBooster("test", DefaultParams(), nil)
	ctx := context.Background()

	if err := b.Train(ctx, nil, nil); !errors.Is(err, errNoBuild) {
		t.Errorf("Train() error = %v, want errNoBuild", err)
	}

	if _, err := b.Predict(ctx, nil); !errors.Is(err, errNoBuild) {
		t.Errorf("Predict() error = %v, want errNoBuild", err)
	}

	if err := b.Save(ctx, "test.model"); !errors.Is(err, errNoBuild) {
		t.Errorf("Save() error = %v, want errNoBuild", err)
	}

	if err := b.Load(ctx, "test.model"); !errors.Is(err, errNoBuild) {
		t.Errorf("Load() error = %v, want errNoBuild", err)
	}

	if _, err := b.Importance(); !errors.Is(err, errNoBuild) {
		t.Errorf("Importance() error = %v, want errNoBuild", err)
	}
}
