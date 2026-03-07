package tuning

import (
	"context"
	"math/rand"
	"time"

	"github.com/zerfoo/metee/cv"
	"github.com/zerfoo/metee/data"
	"github.com/zerfoo/metee/model"
)

// Trial records a single hyperparameter evaluation.
type Trial struct {
	Params   map[string]any
	Score    float64
	Duration time.Duration
}

// SearchResult holds the outcome of a hyperparameter search.
type SearchResult struct {
	BestParams map[string]any
	BestScore  float64
	AllTrials  []Trial
}

// GridSearch evaluates every combination from the parameter space's Grid()
// using cross-validation. Context cancellation stops remaining trials.
func GridSearch(ctx context.Context, ds *data.Dataset, modelFactory func() model.Model, space ParamSpace, folds []cv.Fold, metric func([]float64, []float64) float64) (SearchResult, error) {
	combos := space.Grid()
	if len(combos) == 0 {
		return SearchResult{}, nil
	}
	return runTrials(ctx, ds, modelFactory, combos, folds, metric)
}

// RandomSearch samples nTrials combinations from the parameter space
// using cross-validation. Context cancellation stops remaining trials.
func RandomSearch(ctx context.Context, ds *data.Dataset, modelFactory func() model.Model, space ParamSpace, folds []cv.Fold, metric func([]float64, []float64) float64, nTrials int, rng *rand.Rand) (SearchResult, error) {
	if nTrials <= 0 {
		return SearchResult{}, nil
	}
	combos := make([]map[string]any, nTrials)
	for i := range combos {
		combos[i] = space.Sample(rng)
	}
	return runTrials(ctx, ds, modelFactory, combos, folds, metric)
}

func runTrials(ctx context.Context, ds *data.Dataset, modelFactory func() model.Model, combos []map[string]any, folds []cv.Fold, metric func([]float64, []float64) float64) (SearchResult, error) {
	var result SearchResult
	first := true

	for _, params := range combos {
		select {
		case <-ctx.Done():
			return result, ctx.Err()
		default:
		}

		m := modelFactory()

		if c, ok := m.(model.Configurable); ok {
			if err := c.SetParams(params); err != nil {
				return result, err
			}
		}

		start := time.Now()
		cvResult, err := cv.CrossValidate(ctx, ds, m, folds, metric)
		if err != nil {
			return result, err
		}
		dur := time.Since(start)

		trial := Trial{
			Params:   params,
			Score:    cvResult.Mean,
			Duration: dur,
		}
		result.AllTrials = append(result.AllTrials, trial)

		if first || cvResult.Mean > result.BestScore {
			result.BestScore = cvResult.Mean
			result.BestParams = params
			first = false
		}
	}

	return result, nil
}
