package model

import "context"

// Model is the core interface for all metee model backends (LightGBM, XGBoost, etc).
type Model interface {
	// Name returns a human-readable identifier for this model instance.
	Name() string

	// Train fits the model on the given feature matrix and target vector.
	Train(ctx context.Context, features [][]float64, targets []float64) error

	// Predict produces predictions for the given feature matrix.
	Predict(ctx context.Context, features [][]float64) ([]float64, error)

	// Save serializes the trained model to the given path.
	Save(ctx context.Context, path string) error

	// Load deserializes a model from the given path.
	Load(ctx context.Context, path string) error
}
