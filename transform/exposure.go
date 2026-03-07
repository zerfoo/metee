package transform

import (
	"math"

	"github.com/zerfoo/metee/metrics"
)

// ComputeExposures returns the Pearson correlation of predictions with each feature column.
func ComputeExposures(predictions []float64, features [][]float64, featureNames []string) map[string]float64 {
	exposures := make(map[string]float64, len(featureNames))
	for i, name := range featureNames {
		if i < len(features) {
			exposures[name] = metrics.PearsonCorrelation(predictions, features[i])
		}
	}
	return exposures
}

// MaxExposure returns the feature name and value with the highest absolute exposure.
// If exposures is empty, it returns an empty string and NaN.
func MaxExposure(exposures map[string]float64) (string, float64) {
	if len(exposures) == 0 {
		return "", math.NaN()
	}

	var maxName string
	maxAbs := -1.0
	for name, val := range exposures {
		abs := math.Abs(val)
		if abs > maxAbs {
			maxAbs = abs
			maxName = name
		}
	}
	return maxName, exposures[maxName]
}
