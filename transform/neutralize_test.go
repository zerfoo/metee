package transform

import (
	"math"
	"testing"

	"github.com/zerfoo/metee/metrics"
)

func TestNeutralizeProportionZero(t *testing.T) {
	predictions := []float64{1, 2, 3, 4, 5}
	features := [][]float64{{1, 2, 3, 4, 5}}

	got := Neutralize(predictions, features, 0)

	for i := range got {
		if got[i] != predictions[i] {
			t.Errorf("index %d: got %f, want %f", i, got[i], predictions[i])
		}
	}
}

func TestNeutralizeReducesCorrelation(t *testing.T) {
	// Use predictions that are correlated but NOT perfectly collinear with
	// the feature so that residuals are non-trivial and Pearson is well-defined.
	predictions := []float64{3, 1, 4, 1, 5, 9, 2, 6, 5, 3}
	feature := []float64{2, 4, 6, 8, 10, 12, 14, 16, 18, 20}
	features := [][]float64{feature}

	neutralized := Neutralize(predictions, features, 1.0)
	neutCorr := metrics.PearsonCorrelation(neutralized, feature)

	if math.Abs(neutCorr) > 1e-10 {
		t.Errorf("neutralized correlation = %f, want ~0", neutCorr)
	}
}

func TestNeutralizePartialProportion(t *testing.T) {
	// Use predictions not perfectly correlated with the feature so that
	// partial neutralization changes the correlation (correlation is
	// scale-invariant, so scaling pred by 0.5 does not change it).
	predictions := []float64{3, 1, 4, 1, 5, 9, 2, 6, 5, 3}
	feature := []float64{2, 4, 6, 8, 10, 12, 14, 16, 18, 20}
	features := [][]float64{feature}

	rawCorr := math.Abs(metrics.PearsonCorrelation(predictions, feature))
	neutralized := Neutralize(predictions, features, 0.5)
	neutCorr := math.Abs(metrics.PearsonCorrelation(neutralized, feature))

	if neutCorr >= rawCorr {
		t.Errorf("partial neutralization did not reduce correlation: raw=%f, neutralized=%f", rawCorr, neutCorr)
	}
	if neutCorr < 1e-10 {
		t.Errorf("proportion=0.5 should not fully neutralize: got correlation %f", neutCorr)
	}
}

func TestNeutralizeMultipleFeatures(t *testing.T) {
	// Use predictions that are NOT a linear combination of the features
	// so that neutralization leaves non-zero residuals (otherwise
	// Pearson on all-zero data is undefined).
	predictions := []float64{3, 1, 4, 1, 5, 9, 2, 6, 5, 3}
	features := [][]float64{
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
		{10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
	}

	neutralized := Neutralize(predictions, features, 1.0)

	for i, feat := range features {
		corr := metrics.PearsonCorrelation(neutralized, feat)
		if math.Abs(corr) > 1e-10 {
			t.Errorf("feature %d: neutralized correlation = %f, want ~0", i, corr)
		}
	}
}

func TestNeutralizeEmptyInput(t *testing.T) {
	got := Neutralize(nil, nil, 1.0)
	if len(got) != 0 {
		t.Errorf("expected empty result, got %v", got)
	}
}

func TestNeutralizeEmptyFeatures(t *testing.T) {
	predictions := []float64{1, 2, 3}
	got := Neutralize(predictions, nil, 1.0)
	for i := range got {
		if got[i] != predictions[i] {
			t.Errorf("index %d: got %f, want %f", i, got[i], predictions[i])
		}
	}
}

func TestNeutralizeSingleFeature(t *testing.T) {
	predictions := []float64{3, 1, 4, 1, 5, 9, 2, 6}
	feature := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	features := [][]float64{feature}

	neutralized := Neutralize(predictions, features, 1.0)
	corr := metrics.PearsonCorrelation(neutralized, feature)
	if math.Abs(corr) > 1e-10 {
		t.Errorf("single feature neutralized correlation = %f, want ~0", corr)
	}
}

func TestNeutralizeDoesNotMutateInput(t *testing.T) {
	predictions := []float64{1, 2, 3, 4, 5}
	original := make([]float64, len(predictions))
	copy(original, predictions)

	features := [][]float64{{5, 4, 3, 2, 1}}
	Neutralize(predictions, features, 1.0)

	for i := range predictions {
		if predictions[i] != original[i] {
			t.Errorf("input mutated at index %d: got %f, want %f", i, predictions[i], original[i])
		}
	}
}
