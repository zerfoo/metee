package transform

import (
	"context"
)

// Transform applies a post-prediction transformation.
type Transform interface {
	Apply(ctx context.Context, preds []float64, features [][]float64) ([]float64, error)
}

// Pipeline applies a sequence of transforms in order.
type Pipeline struct {
	Transforms []Transform
}

// Apply runs each transform in sequence, passing the output of each as input to the next.
// An empty pipeline returns a copy of the input.
func (p Pipeline) Apply(ctx context.Context, preds []float64, features [][]float64) ([]float64, error) {
	out := make([]float64, len(preds))
	copy(out, preds)

	for _, t := range p.Transforms {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
		var err error
		out, err = t.Apply(ctx, out, features)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

// RankNormalizeTransform applies rank normalization to predictions.
type RankNormalizeTransform struct{}

func (RankNormalizeTransform) Apply(_ context.Context, preds []float64, _ [][]float64) ([]float64, error) {
	return RankNormalize(preds), nil
}

// GaussianizeTransform applies gaussianization to predictions.
type GaussianizeTransform struct{}

func (GaussianizeTransform) Apply(_ context.Context, preds []float64, _ [][]float64) ([]float64, error) {
	return Gaussianize(preds), nil
}

// NeutralizeTransform applies feature neutralization to predictions.
type NeutralizeTransform struct {
	Proportion float64
}

func (t NeutralizeTransform) Apply(_ context.Context, preds []float64, features [][]float64) ([]float64, error) {
	return Neutralize(preds, features, t.Proportion), nil
}
