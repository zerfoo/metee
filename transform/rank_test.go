package transform

import (
	"math"
	"testing"
)

func TestRankNormalize(t *testing.T) {
	tests := []struct {
		name  string
		input []float64
		want  []float64
	}{
		{
			name:  "empty",
			input: nil,
			want:  []float64{},
		},
		{
			name:  "single element",
			input: []float64{42},
			want:  []float64{42},
		},
		{
			name:  "sorted ascending",
			input: []float64{1, 2, 3, 4, 5},
			want:  []float64{0, 0.25, 0.5, 0.75, 1},
		},
		{
			name:  "sorted descending",
			input: []float64{5, 4, 3, 2, 1},
			want:  []float64{1, 0.75, 0.5, 0.25, 0},
		},
		{
			name:  "ties",
			input: []float64{1, 2, 2, 3},
			// ranks: 1, 2.5, 2.5, 4 -> normalized: 0, 0.5, 0.5, 1
			want: []float64{0, 0.5, 0.5, 1},
		},
		{
			name:  "all equal",
			input: []float64{5, 5, 5, 5},
			// ranks: 2.5, 2.5, 2.5, 2.5 -> normalized: 0.5, 0.5, 0.5, 0.5
			want: []float64{0.5, 0.5, 0.5, 0.5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := RankNormalize(tt.input)
			if len(got) != len(tt.want) {
				t.Fatalf("len = %d, want %d", len(got), len(tt.want))
			}
			for i := range got {
				if math.Abs(got[i]-tt.want[i]) > 1e-10 {
					t.Errorf("index %d: got %f, want %f", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestRankNormalizeUniform(t *testing.T) {
	// Verify output is in [0, 1].
	input := []float64{10, 5, 8, 3, 7, 1, 9, 2, 6, 4}
	got := RankNormalize(input)
	for i, v := range got {
		if v < 0 || v > 1 {
			t.Errorf("index %d: value %f out of [0, 1]", i, v)
		}
	}

	// Min should be 0, max should be 1.
	minVal, maxVal := got[0], got[0]
	for _, v := range got[1:] {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}
	if minVal != 0 {
		t.Errorf("min = %f, want 0", minVal)
	}
	if maxVal != 1 {
		t.Errorf("max = %f, want 1", maxVal)
	}
}

func TestRankNormalizeDoesNotMutateInput(t *testing.T) {
	input := []float64{3, 1, 2}
	original := make([]float64, len(input))
	copy(original, input)
	RankNormalize(input)
	for i := range input {
		if input[i] != original[i] {
			t.Errorf("input mutated at index %d: got %f, want %f", i, input[i], original[i])
		}
	}
}

func TestGaussianize(t *testing.T) {
	tests := []struct {
		name  string
		input []float64
	}{
		{name: "empty", input: nil},
		{name: "single", input: []float64{42}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Gaussianize(tt.input)
			if len(got) != len(tt.input) {
				t.Fatalf("len = %d, want %d", len(got), len(tt.input))
			}
			for i := range got {
				if got[i] != tt.input[i] {
					t.Errorf("index %d: got %f, want %f", i, got[i], tt.input[i])
				}
			}
		})
	}
}

func TestGaussianizeDistribution(t *testing.T) {
	// Large uniform sample -> gaussianized should have mean~0, std~1.
	n := 1000
	input := make([]float64, n)
	for i := range input {
		input[i] = float64(i)
	}

	got := Gaussianize(input)

	var sum float64
	for _, v := range got {
		sum += v
	}
	mean := sum / float64(n)
	if math.Abs(mean) > 0.05 {
		t.Errorf("mean = %f, want ~0", mean)
	}

	var sumSq float64
	for _, v := range got {
		d := v - mean
		sumSq += d * d
	}
	std := math.Sqrt(sumSq / float64(n))
	if math.Abs(std-1) > 0.1 {
		t.Errorf("std = %f, want ~1", std)
	}
}

func TestGaussianizeTies(t *testing.T) {
	input := []float64{1, 2, 2, 3}
	got := Gaussianize(input)
	// Tied values should get the same Gaussian quantile.
	if got[1] != got[2] {
		t.Errorf("tied values got different gaussianized scores: %f vs %f", got[1], got[2])
	}
	// Should be monotonically ordered for non-ties.
	if got[0] >= got[1] {
		t.Errorf("expected got[0] < got[1]: %f >= %f", got[0], got[1])
	}
	if got[2] >= got[3] {
		t.Errorf("expected got[2] < got[3]: %f >= %f", got[2], got[3])
	}
}

func TestGaussianizeSymmetry(t *testing.T) {
	input := []float64{1, 2, 3, 4, 5}
	got := Gaussianize(input)
	// Middle element should be ~0.
	if math.Abs(got[2]) > 1e-10 {
		t.Errorf("middle element = %f, want ~0", got[2])
	}
	// Symmetric: got[0] ≈ -got[4], got[1] ≈ -got[3].
	if math.Abs(got[0]+got[4]) > 1e-10 {
		t.Errorf("not symmetric: got[0]=%f, got[4]=%f", got[0], got[4])
	}
	if math.Abs(got[1]+got[3]) > 1e-10 {
		t.Errorf("not symmetric: got[1]=%f, got[3]=%f", got[1], got[3])
	}
}
