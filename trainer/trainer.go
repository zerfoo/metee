// Package trainer provides a training orchestrator for tree-based models.
package trainer

import (
	"context"
	"fmt"
	"path/filepath"

	"github.com/zerfoo/metee/data"
	"github.com/zerfoo/metee/model"
)

// TrainerConfig configures the training orchestrator.
type TrainerConfig struct {
	// MaxRounds is the max boosting iterations (reserved for future incremental training).
	MaxRounds int

	// EarlyStoppingRounds stops training if no improvement for N rounds
	// (reserved for future incremental training).
	EarlyStoppingRounds int

	// EarlyStoppingMetric computes a validation score from predictions and targets.
	// Higher is better.
	EarlyStoppingMetric func(preds, targets []float64) float64

	// CheckpointDir is the directory to save model checkpoints.
	// If empty, no checkpoints are saved.
	CheckpointDir string

	// CheckpointEvery saves a checkpoint every N rounds (reserved for future use).
	CheckpointEvery int
}

// TrainResult holds the result of a training run.
type TrainResult struct {
	BestScore        float64
	BestRound        int
	ValidationScores []float64
	CheckpointPath   string
}

// Trainer orchestrates model training, validation, and checkpointing.
type Trainer struct{}

// Run trains the model, validates it, optionally saves a checkpoint, and returns the result.
func (t *Trainer) Run(ctx context.Context, cfg TrainerConfig, m model.Model, train, valid *data.Dataset) (TrainResult, error) {
	select {
	case <-ctx.Done():
		return TrainResult{}, fmt.Errorf("trainer: %w", ctx.Err())
	default:
	}

	if err := m.Train(ctx, train.Features, train.Targets); err != nil {
		return TrainResult{}, fmt.Errorf("trainer: train: %w", err)
	}

	preds, err := m.Predict(ctx, valid.Features)
	if err != nil {
		return TrainResult{}, fmt.Errorf("trainer: predict: %w", err)
	}

	var result TrainResult
	if cfg.EarlyStoppingMetric != nil {
		score := cfg.EarlyStoppingMetric(preds, valid.Targets)
		result.BestScore = score
		result.BestRound = 1
		result.ValidationScores = []float64{score}
	}

	if cfg.CheckpointDir != "" {
		path := filepath.Join(cfg.CheckpointDir, m.Name()+".model")
		if err := m.Save(ctx, path); err != nil {
			return TrainResult{}, fmt.Errorf("trainer: checkpoint: %w", err)
		}
		result.CheckpointPath = path
	}

	return result, nil
}
