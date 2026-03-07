// Package metrics provides evaluation metrics for model performance.
package metrics

import (
	"math"
	"sort"

	"gonum.org/v1/gonum/mat"
)

// PearsonCorrelation calculates the Pearson correlation coefficient between two slices.
func PearsonCorrelation(x, y []float64) float64 {
	if len(x) != len(y) || len(x) == 0 {
		return math.NaN()
	}

	n := float64(len(x))

	var sumX, sumY float64
	for i := 0; i < len(x); i++ {
		sumX += x[i]
		sumY += y[i]
	}
	meanX := sumX / n
	meanY := sumY / n

	var numerator, sumXX, sumYY float64
	for i := 0; i < len(x); i++ {
		dx := x[i] - meanX
		dy := y[i] - meanY
		numerator += dx * dy
		sumXX += dx * dx
		sumYY += dy * dy
	}

	denominator := math.Sqrt(sumXX * sumYY)
	if denominator == 0 {
		return math.NaN()
	}

	return numerator / denominator
}

// SpearmanCorrelation calculates the Spearman rank correlation coefficient.
func SpearmanCorrelation(x, y []float64) float64 {
	if len(x) != len(y) || len(x) == 0 {
		return math.NaN()
	}

	ranksX := calculateRanks(x)
	ranksY := calculateRanks(y)

	return PearsonCorrelation(ranksX, ranksY)
}

// Sharpe calculates the Sharpe ratio of a return series.
func Sharpe(returns []float64) float64 {
	if len(returns) == 0 {
		return math.NaN()
	}

	n := float64(len(returns))
	var sum float64
	for _, r := range returns {
		sum += r
	}
	mean := sum / n

	var sumSq float64
	for _, r := range returns {
		d := r - mean
		sumSq += d * d
	}
	std := math.Sqrt(sumSq / n)

	if std == 0 {
		return math.NaN()
	}

	return mean / std
}

// MaxDrawdown calculates the maximum drawdown from a cumulative return series.
func MaxDrawdown(returns []float64) float64 {
	if len(returns) == 0 {
		return 0
	}

	cumulative := 0.0
	peak := 0.0
	maxDD := 0.0

	for _, r := range returns {
		cumulative += r
		if cumulative > peak {
			peak = cumulative
		}
		dd := peak - cumulative
		if dd > maxDD {
			maxDD = dd
		}
	}

	return maxDD
}

// PerEraMeanCorr calculates the mean per-era Pearson correlation.
// The eras slice assigns each sample to an era (integer group).
func PerEraMeanCorr(preds, targets []float64, eras []int) float64 {
	if len(preds) != len(targets) || len(preds) != len(eras) || len(preds) == 0 {
		return math.NaN()
	}

	groups := make(map[int][]int)
	for i, era := range eras {
		groups[era] = append(groups[era], i)
	}

	var sum float64
	var count int
	for _, indices := range groups {
		p := make([]float64, len(indices))
		t := make([]float64, len(indices))
		for i, idx := range indices {
			p[i] = preds[idx]
			t[i] = targets[idx]
		}
		corr := PearsonCorrelation(p, t)
		if !math.IsNaN(corr) {
			sum += corr
			count++
		}
	}

	if count == 0 {
		return math.NaN()
	}
	return sum / float64(count)
}

// EraReport holds aggregated per-era correlation statistics.
type EraReport struct {
	MeanCorr      float64
	Sharpe        float64
	MaxDrawdown   float64
	PositiveRatio float64
	PerEra        map[int]float64
}

// PerEraReport computes per-era Pearson correlations and derives aggregate statistics.
func PerEraReport(preds, targets []float64, eras []int) EraReport {
	if len(preds) != len(targets) || len(preds) != len(eras) || len(preds) == 0 {
		return EraReport{}
	}

	groups := make(map[int][]int)
	for i, era := range eras {
		groups[era] = append(groups[era], i)
	}

	perEra := make(map[int]float64, len(groups))
	var corrs []float64
	for era, indices := range groups {
		p := make([]float64, len(indices))
		t := make([]float64, len(indices))
		for i, idx := range indices {
			p[i] = preds[idx]
			t[i] = targets[idx]
		}
		corr := PearsonCorrelation(p, t)
		if math.IsNaN(corr) {
			continue
		}
		perEra[era] = corr
		corrs = append(corrs, corr)
	}

	if len(corrs) == 0 {
		return EraReport{PerEra: perEra}
	}

	var sum float64
	var positive int
	for _, c := range corrs {
		sum += c
		if c > 0 {
			positive++
		}
	}

	return EraReport{
		MeanCorr:      sum / float64(len(corrs)),
		Sharpe:        Sharpe(corrs),
		MaxDrawdown:   MaxDrawdown(corrs),
		PositiveRatio: float64(positive) / float64(len(corrs)),
		PerEra:        perEra,
	}
}

// SpearmanPerEra calculates the mean per-era Spearman correlation.
func SpearmanPerEra(preds, targets []float64, eras []int) float64 {
	if len(preds) != len(targets) || len(preds) != len(eras) || len(preds) == 0 {
		return math.NaN()
	}

	groups := make(map[int][]int)
	for i, era := range eras {
		groups[era] = append(groups[era], i)
	}

	var sum float64
	var count int
	for _, indices := range groups {
		p := make([]float64, len(indices))
		t := make([]float64, len(indices))
		for i, idx := range indices {
			p[i] = preds[idx]
			t[i] = targets[idx]
		}
		corr := SpearmanCorrelation(p, t)
		if !math.IsNaN(corr) {
			sum += corr
			count++
		}
	}

	if count == 0 {
		return math.NaN()
	}
	return sum / float64(count)
}

// FeatureNeutralCorrelation neutralizes predictions against features (proportion=1.0)
// and returns the Pearson correlation of the neutralized predictions with targets.
func FeatureNeutralCorrelation(preds, targets []float64, features [][]float64) float64 {
	neutralized := neutralize(preds, features)
	return PearsonCorrelation(neutralized, targets)
}

// neutralize removes linear exposure to features from predictions using SVD projection.
// This is inlined to avoid an import cycle with the transform package.
func neutralize(predictions []float64, features [][]float64) []float64 {
	n := len(predictions)
	result := make([]float64, n)
	copy(result, predictions)

	if n == 0 || len(features) == 0 {
		return result
	}

	nFeatures := len(features)
	cols := nFeatures + 1

	fData := make([]float64, n*cols)
	for i := 0; i < n; i++ {
		fData[i*cols] = 1.0
		for j, feat := range features {
			fData[i*cols+j+1] = feat[i]
		}
	}
	f := mat.NewDense(n, cols, fData)

	var svd mat.SVD
	if !svd.Factorize(f, mat.SVDThin) {
		return result
	}

	var u mat.Dense
	svd.UTo(&u)
	sv := svd.Values(nil)

	maxSV := 0.0
	for _, s := range sv {
		if s > maxSV {
			maxSV = s
		}
	}
	tol := maxSV * float64(max(n, cols)) * 1e-14

	for k := range sv {
		if sv[k] < tol {
			continue
		}
		var dot float64
		for i := 0; i < n; i++ {
			dot += u.At(i, k) * predictions[i]
		}
		for i := 0; i < n; i++ {
			result[i] -= u.At(i, k) * dot
		}
	}

	return result
}

func calculateRanks(values []float64) []float64 {
	n := len(values)
	ranks := make([]float64, n)

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

	i := 0
	for i < n {
		j := i
		currentValue := sorted[i].value
		for j < n && sorted[j].value == currentValue {
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
