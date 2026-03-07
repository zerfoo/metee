// Package transform provides post-prediction transformations for rank normalization,
// gaussianization, neutralization, and exposure computation.
package transform

import (
	"math"
	"sort"
)

// RankNormalize maps values to uniform [0, 1] via rank.
// Ties are handled with average rank.
// Empty or single-element inputs return a copy of the input.
func RankNormalize(values []float64) []float64 {
	n := len(values)
	if n <= 1 {
		out := make([]float64, n)
		copy(out, values)
		return out
	}

	ranks := averageRanks(values)
	out := make([]float64, n)
	for i, r := range ranks {
		out[i] = (r - 1) / float64(n-1)
	}
	return out
}

// Gaussianize maps values to Gaussian quantiles via rank normalization
// followed by the inverse error function.
// Empty or single-element inputs return a copy of the input.
func Gaussianize(values []float64) []float64 {
	n := len(values)
	if n <= 1 {
		out := make([]float64, n)
		copy(out, values)
		return out
	}

	ranks := averageRanks(values)
	out := make([]float64, n)
	for i, r := range ranks {
		// Map rank to (0, 1) avoiding 0 and 1 for erfinv stability.
		p := (r - 0.5) / float64(n)
		// Convert uniform quantile to Gaussian: x = sqrt(2) * erfinv(2*p - 1)
		out[i] = math.Sqrt2 * math.Erfinv(2*p-1)
	}
	return out
}

// averageRanks returns 1-based average ranks for the given values.
// Ties receive the mean of the ranks they would span.
func averageRanks(values []float64) []float64 {
	n := len(values)

	type indexValue struct {
		index int
		value float64
	}

	sorted := make([]indexValue, n)
	for i, v := range values {
		sorted[i] = indexValue{index: i, value: v}
	}

	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].value < sorted[j].value
	})

	ranks := make([]float64, n)
	i := 0
	for i < n {
		j := i
		for j < n && sorted[j].value == sorted[i].value {
			j++
		}
		avgRank := float64(i+j-1)/2.0 + 1.0
		for k := i; k < j; k++ {
			ranks[sorted[k].index] = avgRank
		}
		i = j
	}

	return ranks
}
