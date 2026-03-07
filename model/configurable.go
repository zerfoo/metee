package model

// Configurable is an optional interface for models whose parameters
// can be updated at runtime, enabling hyperparameter tuning integration.
type Configurable interface {
	SetParams(params map[string]any) error
}
