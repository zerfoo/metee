package model

import "context"

// Validator is an optional interface for models that support validation.
type Validator interface {
	Validate(ctx context.Context, features [][]float64, targets []float64) (map[string]float64, error)
}
