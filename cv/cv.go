package cv

import (
	"context"
	"math"

	"github.com/zerfoo/metee/data"
	"github.com/zerfoo/metee/model"
)

// CVResult holds the results of cross-validation.
type CVResult struct {
	Scores []float64
	Mean   float64
	Std    float64
}

// CrossValidate trains and evaluates a model on each fold, returning per-fold
// scores and aggregate statistics. Context cancellation stops remaining folds.
func CrossValidate(ctx context.Context, ds *data.Dataset, m model.Model, folds []Fold, metric func([]float64, []float64) float64) (CVResult, error) {
	scores := make([]float64, 0, len(folds))
	for _, fold := range folds {
		select {
		case <-ctx.Done():
			return CVResult{}, ctx.Err()
		default:
		}

		train, valid := ds.Split(fold.TrainEras, fold.ValidEras)

		if err := m.Train(ctx, train.Features, train.Targets); err != nil {
			return CVResult{}, err
		}

		preds, err := m.Predict(ctx, valid.Features)
		if err != nil {
			return CVResult{}, err
		}

		score := metric(preds, valid.Targets)
		scores = append(scores, score)
	}

	result := CVResult{Scores: scores}
	if len(scores) > 0 {
		var sum float64
		for _, s := range scores {
			sum += s
		}
		result.Mean = sum / float64(len(scores))

		var sumSq float64
		for _, s := range scores {
			d := s - result.Mean
			sumSq += d * d
		}
		result.Std = math.Sqrt(sumSq / float64(len(scores)))
	}
	return result, nil
}
