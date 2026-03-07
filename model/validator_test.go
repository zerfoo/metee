package model_test

import (
	"context"

	"github.com/zerfoo/metee/model"
)

// Compile-time check that a concrete type can satisfy Validator.
var _ model.Validator = (*mockValidator)(nil)

type mockValidator struct{}

func (m *mockValidator) Validate(_ context.Context, _ [][]float64, _ []float64) (map[string]float64, error) {
	return nil, nil
}
