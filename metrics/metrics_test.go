package metrics

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestPearsonCorrelation(t *testing.T) {
	tests := []struct {
		name string
		x, y []float64
		want float64
	}{
		{"perfect positive", []float64{1, 2, 3, 4, 5}, []float64{2, 4, 6, 8, 10}, 1.0},
		{"perfect negative", []float64{1, 2, 3, 4, 5}, []float64{10, 8, 6, 4, 2}, -1.0},
		{"uncorrelated", []float64{1, 2, 3, 4, 5}, []float64{2, 4, 1, 5, 3}, 0.3},
		{"empty", nil, nil, math.NaN()},
		{"length mismatch", []float64{1, 2}, []float64{1}, math.NaN()},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := PearsonCorrelation(tt.x, tt.y)
			if math.IsNaN(tt.want) {
				if !math.IsNaN(got) {
					t.Errorf("PearsonCorrelation() = %v, want NaN", got)
				}
				return
			}
			if math.Abs(got-tt.want) > 0.1 {
				t.Errorf("PearsonCorrelation() = %v, want ~%v", got, tt.want)
			}
		})
	}
}

func TestSpearmanCorrelation(t *testing.T) {
	t.Run("positive", func(t *testing.T) {
		x := []float64{1, 2, 3, 4, 5}
		y := []float64{5, 6, 7, 8, 7}
		got := SpearmanCorrelation(x, y)
		if got < 0.5 {
			t.Errorf("SpearmanCorrelation() = %v, want > 0.5", got)
		}
	})
	t.Run("empty", func(t *testing.T) {
		got := SpearmanCorrelation(nil, nil)
		if !math.IsNaN(got) {
			t.Errorf("SpearmanCorrelation(nil) = %v, want NaN", got)
		}
	})
	t.Run("length mismatch", func(t *testing.T) {
		got := SpearmanCorrelation([]float64{1}, []float64{1, 2})
		if !math.IsNaN(got) {
			t.Errorf("SpearmanCorrelation(mismatch) = %v, want NaN", got)
		}
	})
}

func TestSharpe(t *testing.T) {
	returns := []float64{0.01, 0.02, 0.01, 0.03, 0.02}
	got := Sharpe(returns)
	if got <= 0 {
		t.Errorf("Sharpe() = %v, want > 0", got)
	}
	if math.IsNaN(Sharpe(nil)) != true {
		t.Error("Sharpe(nil) should be NaN")
	}
}

func TestMaxDrawdown(t *testing.T) {
	returns := []float64{0.1, 0.05, -0.2, -0.1, 0.15}
	got := MaxDrawdown(returns)
	if got < 0.15 {
		t.Errorf("MaxDrawdown() = %v, want >= 0.15", got)
	}
	if MaxDrawdown(nil) != 0 {
		t.Error("MaxDrawdown(nil) should be 0")
	}
}

func TestPerEraMeanCorr(t *testing.T) {
	preds := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
	targets := []float64{0.15, 0.25, 0.35, 0.45, 0.55, 0.65}
	eras := []int{1, 1, 1, 2, 2, 2}

	got := PerEraMeanCorr(preds, targets, eras)
	if got < 0.9 {
		t.Errorf("PerEraMeanCorr() = %v, want > 0.9", got)
	}
}

func TestPerEraReport(t *testing.T) {
	t.Run("basic", func(t *testing.T) {
		preds := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
		targets := []float64{0.15, 0.25, 0.35, 0.45, 0.55, 0.65}
		eras := []int{1, 1, 1, 2, 2, 2}

		report := PerEraReport(preds, targets, eras)

		if report.MeanCorr < 0.9 {
			t.Errorf("MeanCorr = %v, want > 0.9", report.MeanCorr)
		}
		if report.PositiveRatio != 1.0 {
			t.Errorf("PositiveRatio = %v, want 1.0", report.PositiveRatio)
		}
		if len(report.PerEra) != 2 {
			t.Errorf("PerEra has %d entries, want 2", len(report.PerEra))
		}
		if report.Sharpe <= 0 {
			t.Errorf("Sharpe = %v, want > 0", report.Sharpe)
		}
		if report.MaxDrawdown < 0 {
			t.Errorf("MaxDrawdown = %v, want >= 0", report.MaxDrawdown)
		}
	})

	t.Run("mixed correlations", func(t *testing.T) {
		// Era 1: positive corr, Era 2: negative corr
		preds := []float64{1, 2, 3, 3, 2, 1}
		targets := []float64{1, 2, 3, 1, 2, 3}
		eras := []int{1, 1, 1, 2, 2, 2}

		report := PerEraReport(preds, targets, eras)

		if report.PositiveRatio != 0.5 {
			t.Errorf("PositiveRatio = %v, want 0.5", report.PositiveRatio)
		}
		if _, ok := report.PerEra[1]; !ok {
			t.Error("PerEra missing era 1")
		}
		if _, ok := report.PerEra[2]; !ok {
			t.Error("PerEra missing era 2")
		}
	})

	t.Run("empty input", func(t *testing.T) {
		report := PerEraReport(nil, nil, nil)
		if report.MeanCorr != 0 || report.Sharpe != 0 || report.MaxDrawdown != 0 || report.PositiveRatio != 0 {
			t.Errorf("empty report should be zero-value, got %+v", report)
		}
	})

	t.Run("length mismatch", func(t *testing.T) {
		report := PerEraReport([]float64{1}, []float64{1, 2}, []int{1})
		if report.MeanCorr != 0 {
			t.Errorf("mismatched input should return zero-value report")
		}
	})
}

func TestSpearmanPerEra(t *testing.T) {
	t.Run("basic", func(t *testing.T) {
		preds := []float64{1, 2, 3, 4, 5, 6}
		targets := []float64{1.1, 2.1, 3.1, 4.1, 5.1, 6.1}
		eras := []int{1, 1, 1, 2, 2, 2}

		got := SpearmanPerEra(preds, targets, eras)
		if got < 0.9 {
			t.Errorf("SpearmanPerEra() = %v, want > 0.9", got)
		}
	})

	t.Run("matches manual computation", func(t *testing.T) {
		preds := []float64{3, 1, 2, 6, 4, 5}
		targets := []float64{3, 2, 1, 6, 5, 4}
		eras := []int{1, 1, 1, 2, 2, 2}

		got := SpearmanPerEra(preds, targets, eras)

		// Each era has rank corr of [3,1,2] vs [3,2,1] = 0.5
		// Mean should be 0.5
		if math.Abs(got-0.5) > 0.01 {
			t.Errorf("SpearmanPerEra() = %v, want ~0.5", got)
		}
	})

	t.Run("empty input", func(t *testing.T) {
		got := SpearmanPerEra(nil, nil, nil)
		if !math.IsNaN(got) {
			t.Errorf("SpearmanPerEra(nil) = %v, want NaN", got)
		}
	})

	t.Run("length mismatch", func(t *testing.T) {
		got := SpearmanPerEra([]float64{1}, []float64{1, 2}, []int{1})
		if !math.IsNaN(got) {
			t.Errorf("SpearmanPerEra(mismatch) = %v, want NaN", got)
		}
	})
}

func TestFeatureNeutralCorrelation(t *testing.T) {
	t.Run("FNC less than raw when predictions derive from features", func(t *testing.T) {
		// Predictions come primarily from the feature (which also drives targets).
		// Neutralizing removes the feature signal, reducing correlation with targets.
		rng := rand.New(rand.NewPCG(42, 0))
		n := 200
		feature := make([]float64, n)
		targets := make([]float64, n)
		preds := make([]float64, n)
		for i := 0; i < n; i++ {
			feature[i] = rng.NormFloat64()
			// Both targets and predictions depend on the same feature
			targets[i] = feature[i] + 0.1*rng.NormFloat64()
			preds[i] = feature[i] + 0.1*rng.NormFloat64()
		}

		features := [][]float64{feature}
		rawCorr := PearsonCorrelation(preds, targets)
		fnc := FeatureNeutralCorrelation(preds, targets, features)

		if math.IsNaN(rawCorr) || math.IsNaN(fnc) {
			t.Fatalf("got NaN: rawCorr=%v, fnc=%v", rawCorr, fnc)
		}
		if fnc >= rawCorr {
			t.Errorf("FNC (%v) should be < raw Pearson (%v) when predictions derive from features", fnc, rawCorr)
		}
	})

	t.Run("FNC equals raw when uncorrelated with features", func(t *testing.T) {
		rng := rand.New(rand.NewPCG(99, 0))
		n := 200
		feature := make([]float64, n)
		targets := make([]float64, n)
		preds := make([]float64, n)
		for i := 0; i < n; i++ {
			feature[i] = float64(i)
			targets[i] = rng.NormFloat64()
			preds[i] = targets[i]
		}
		// Remove linear relationship between preds and feature
		var dotPF, dotFF float64
		meanP, meanF := 0.0, 0.0
		for i := 0; i < n; i++ {
			meanP += preds[i]
			meanF += feature[i]
		}
		meanP /= float64(n)
		meanF /= float64(n)
		for i := 0; i < n; i++ {
			dotPF += (preds[i] - meanP) * (feature[i] - meanF)
			dotFF += (feature[i] - meanF) * (feature[i] - meanF)
		}
		beta := dotPF / dotFF
		for i := 0; i < n; i++ {
			preds[i] -= beta * (feature[i] - meanF)
		}

		features := [][]float64{feature}
		rawCorr := PearsonCorrelation(preds, targets)
		fnc := FeatureNeutralCorrelation(preds, targets, features)

		if math.IsNaN(rawCorr) || math.IsNaN(fnc) {
			t.Fatalf("got NaN: rawCorr=%v, fnc=%v", rawCorr, fnc)
		}
		if math.Abs(fnc-rawCorr) > 0.05 {
			t.Errorf("FNC (%v) should approximately equal raw Pearson (%v) when preds are uncorrelated with features", fnc, rawCorr)
		}
	})

	t.Run("empty inputs", func(t *testing.T) {
		got := FeatureNeutralCorrelation(nil, nil, nil)
		if !math.IsNaN(got) {
			t.Errorf("FeatureNeutralCorrelation(nil) = %v, want NaN", got)
		}
	})
}
