package transform

import (
	"math"
	"testing"

	"github.com/zerfoo/metee/metrics"
)

func TestComputeExposures(t *testing.T) {
	predictions := []float64{1, 2, 3, 4, 5}
	features := [][]float64{
		{1, 2, 3, 4, 5}, // perfectly correlated
		{5, 4, 3, 2, 1}, // perfectly anti-correlated
		{1, 1, 1, 1, 1}, // no correlation (constant)
	}
	names := []string{"feat_pos", "feat_neg", "feat_const"}

	exposures := ComputeExposures(predictions, features, names)

	if len(exposures) != 3 {
		t.Fatalf("len = %d, want 3", len(exposures))
	}

	// Verify against manual Pearson computation.
	for i, name := range names {
		want := metrics.PearsonCorrelation(predictions, features[i])
		got := exposures[name]
		if math.IsNaN(want) && math.IsNaN(got) {
			continue
		}
		if math.Abs(got-want) > 1e-10 {
			t.Errorf("exposure[%s] = %f, want %f", name, got, want)
		}
	}

	if math.Abs(exposures["feat_pos"]-1.0) > 1e-10 {
		t.Errorf("feat_pos = %f, want 1.0", exposures["feat_pos"])
	}
	if math.Abs(exposures["feat_neg"]+1.0) > 1e-10 {
		t.Errorf("feat_neg = %f, want -1.0", exposures["feat_neg"])
	}
}

func TestComputeExposuresEmpty(t *testing.T) {
	exposures := ComputeExposures(nil, nil, nil)
	if len(exposures) != 0 {
		t.Errorf("expected empty map, got %v", exposures)
	}
}

func TestMaxExposure(t *testing.T) {
	tests := []struct {
		name      string
		exposures map[string]float64
		wantName  string
		wantVal   float64
		wantNaN   bool
	}{
		{
			name:      "empty",
			exposures: map[string]float64{},
			wantName:  "",
			wantNaN:   true,
		},
		{
			name:      "single",
			exposures: map[string]float64{"a": 0.5},
			wantName:  "a",
			wantVal:   0.5,
		},
		{
			name:      "negative has highest absolute",
			exposures: map[string]float64{"a": 0.3, "b": -0.8, "c": 0.5},
			wantName:  "b",
			wantVal:   -0.8,
		},
		{
			name:      "positive has highest absolute",
			exposures: map[string]float64{"a": 0.9, "b": -0.3, "c": 0.1},
			wantName:  "a",
			wantVal:   0.9,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotName, gotVal := MaxExposure(tt.exposures)
			if gotName != tt.wantName {
				t.Errorf("name = %q, want %q", gotName, tt.wantName)
			}
			if tt.wantNaN {
				if !math.IsNaN(gotVal) {
					t.Errorf("value = %f, want NaN", gotVal)
				}
			} else if math.Abs(gotVal-tt.wantVal) > 1e-10 {
				t.Errorf("value = %f, want %f", gotVal, tt.wantVal)
			}
		})
	}
}

func TestMaxExposureSingleFeature(t *testing.T) {
	predictions := []float64{1, 2, 3, 4, 5}
	features := [][]float64{{2, 4, 6, 8, 10}}
	names := []string{"feat_x"}

	exposures := ComputeExposures(predictions, features, names)
	name, val := MaxExposure(exposures)

	if name != "feat_x" {
		t.Errorf("name = %q, want feat_x", name)
	}
	if math.Abs(val-1.0) > 1e-10 {
		t.Errorf("value = %f, want 1.0", val)
	}
}
