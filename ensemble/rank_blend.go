package ensemble

import (
	"context"
	"errors"
	"math"

	"github.com/zerfoo/metee/transform"
)

// RankBlend combines multiple prediction sets by rank-normalizing each to [0,1]
// via transform.RankNormalize before computing a weighted average. Weights are
// normalized to sum to 1.
func RankBlend(_ context.Context, preds [][]float64, weights []float64) ([]float64, error) {
	if len(preds) == 0 {
		return nil, errors.New("no predictions to blend")
	}
	if len(weights) != len(preds) {
		return nil, errors.New("weights length must match number of prediction sets")
	}

	n := len(preds[0])
	for _, p := range preds[1:] {
		if len(p) != n {
			return nil, errors.New("all prediction vectors must have the same length")
		}
	}

	var wSum float64
	for _, w := range weights {
		wSum += w
	}
	if math.Abs(wSum) < 1e-12 {
		return nil, errors.New("weights sum to zero")
	}

	result := make([]float64, n)
	for i, p := range preds {
		ranked := transform.RankNormalize(p)
		w := weights[i] / wSum
		for j, v := range ranked {
			result[j] += w * v
		}
	}

	return result, nil
}
