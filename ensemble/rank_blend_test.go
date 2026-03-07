package ensemble

import (
	"context"
	"math"
	"testing"
)

func TestRankBlendEqualWeightsSameScale(t *testing.T) {
	// With identical, uniformly-spaced inputs and equal weights,
	// rank blend produces rank-normalized values {0, 0.25, 0.5, 0.75, 1}.
	preds := [][]float64{
		{0.1, 0.2, 0.3, 0.4, 0.5},
		{0.1, 0.2, 0.3, 0.4, 0.5},
	}
	weights := []float64{1.0, 1.0}

	result, err := RankBlend(context.Background(), preds, weights)
	if err != nil {
		t.Fatalf("RankBlend() error: %v", err)
	}

	expected := []float64{0.0, 0.25, 0.5, 0.75, 1.0}
	for i, v := range result {
		if math.Abs(v-expected[i]) > 1e-9 {
			t.Errorf("result[%d] = %v, want %v", i, v, expected[i])
		}
	}
}

func TestRankBlendUnequalScales(t *testing.T) {
	// Predictions at very different scales. Rank blend should normalize them.
	preds := [][]float64{
		{0.1, 0.2, 0.3},       // small scale
		{100.0, 200.0, 300.0},  // large scale
	}
	weights := []float64{1.0, 1.0}

	rb, err := RankBlend(context.Background(), preds, weights)
	if err != nil {
		t.Fatalf("RankBlend() error: %v", err)
	}

	b := &Blender{Weights: weights}
	bb, err := b.Blend(context.Background(), preds)
	if err != nil {
		t.Fatalf("Blend() error: %v", err)
	}

	// Raw blend is dominated by the large-scale model; rank blend is not.
	// Rank blend output should differ from raw blend.
	same := true
	for i := range rb {
		if math.Abs(rb[i]-bb[i]) > 1e-9 {
			same = false
			break
		}
	}
	if same {
		t.Error("rank blend should differ from raw blend when scales differ")
	}

	// After rank normalization, both prediction sets map to {0, 0.5, 1},
	// so the blended result should also be {0, 0.5, 1}.
	expected := []float64{0.0, 0.5, 1.0}
	for i, v := range rb {
		if math.Abs(v-expected[i]) > 1e-9 {
			t.Errorf("result[%d] = %v, want %v", i, v, expected[i])
		}
	}
}

func TestRankBlendUnequalWeights(t *testing.T) {
	preds := [][]float64{
		{0.3, 0.1, 0.2}, // ranks: 1.0, 0.0, 0.5
		{0.1, 0.3, 0.2}, // ranks: 0.0, 1.0, 0.5
	}
	weights := []float64{3.0, 1.0}

	result, err := RankBlend(context.Background(), preds, weights)
	if err != nil {
		t.Fatalf("RankBlend() error: %v", err)
	}

	// Normalized weights: 0.75, 0.25
	// result[0] = 0.75*1.0 + 0.25*0.0 = 0.75
	// result[1] = 0.75*0.0 + 0.25*1.0 = 0.25
	// result[2] = 0.75*0.5 + 0.25*0.5 = 0.5
	expected := []float64{0.75, 0.25, 0.5}
	for i, v := range result {
		if math.Abs(v-expected[i]) > 1e-9 {
			t.Errorf("result[%d] = %v, want %v", i, v, expected[i])
		}
	}
}

func TestRankBlendOutputRange(t *testing.T) {
	preds := [][]float64{
		{-100, 0, 50, 200, 1000},
		{5, 3, 1, 4, 2},
	}
	weights := []float64{0.6, 0.4}

	result, err := RankBlend(context.Background(), preds, weights)
	if err != nil {
		t.Fatalf("RankBlend() error: %v", err)
	}

	for i, v := range result {
		if v < -1e-9 || v > 1.0+1e-9 {
			t.Errorf("result[%d] = %v, outside [0, 1]", i, v)
		}
	}
}

func TestRankBlendErrors(t *testing.T) {
	tests := []struct {
		name    string
		preds   [][]float64
		weights []float64
		wantErr string
	}{
		{
			name:    "empty preds",
			preds:   nil,
			weights: []float64{1.0},
			wantErr: "no predictions to blend",
		},
		{
			name:    "weight count mismatch",
			preds:   [][]float64{{1, 2}, {3, 4}},
			weights: []float64{1.0},
			wantErr: "weights length must match number of prediction sets",
		},
		{
			name:    "length mismatch",
			preds:   [][]float64{{1, 2}, {3}},
			weights: []float64{1.0, 1.0},
			wantErr: "all prediction vectors must have the same length",
		},
		{
			name:    "zero weights",
			preds:   [][]float64{{1, 2}},
			weights: []float64{0.0},
			wantErr: "weights sum to zero",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := RankBlend(context.Background(), tt.preds, tt.weights)
			if err == nil {
				t.Fatal("expected error")
			}
			if err.Error() != tt.wantErr {
				t.Errorf("error = %q, want %q", err.Error(), tt.wantErr)
			}
		})
	}
}

func TestRankBlendSinglePrediction(t *testing.T) {
	preds := [][]float64{{5.0, 3.0, 1.0, 4.0, 2.0}}
	weights := []float64{1.0}

	result, err := RankBlend(context.Background(), preds, weights)
	if err != nil {
		t.Fatalf("RankBlend() error: %v", err)
	}

	// Single set: rank normalized to [0,1]. Values 1,2,3,4,5 -> ranks 0,0.25,0.5,0.75,1
	expected := []float64{1.0, 0.5, 0.0, 0.75, 0.25}
	for i, v := range result {
		if math.Abs(v-expected[i]) > 1e-9 {
			t.Errorf("result[%d] = %v, want %v", i, v, expected[i])
		}
	}
}
