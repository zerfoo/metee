package transform

import (
	"context"
	"errors"
	"math"
	"testing"
)

func TestPipelineEmptyReturnsInputCopy(t *testing.T) {
	p := Pipeline{}
	input := []float64{3.0, 1.0, 2.0}
	got, err := p.Apply(context.Background(), input, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got) != len(input) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(input))
	}
	for i := range input {
		if got[i] != input[i] {
			t.Errorf("index %d: got %f, want %f", i, got[i], input[i])
		}
	}
	// Verify it's a copy, not the same slice.
	input[0] = 999.0
	if got[0] == 999.0 {
		t.Error("returned slice is not a copy of input")
	}
}

func TestPipelineSingleTransform(t *testing.T) {
	p := Pipeline{Transforms: []Transform{RankNormalizeTransform{}}}
	input := []float64{30.0, 10.0, 20.0}
	got, err := p.Apply(context.Background(), input, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := RankNormalize(input)
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("index %d: got %f, want %f", i, got[i], want[i])
		}
	}
}

func TestPipelineMultipleTransforms(t *testing.T) {
	p := Pipeline{
		Transforms: []Transform{
			RankNormalizeTransform{},
			GaussianizeTransform{},
		},
	}
	input := []float64{30.0, 10.0, 20.0}
	got, err := p.Apply(context.Background(), input, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Manually chain: rank normalize, then gaussianize.
	step1 := RankNormalize(input)
	want := Gaussianize(step1)
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-12 {
			t.Errorf("index %d: got %f, want %f", i, got[i], want[i])
		}
	}
}

func TestPipelineContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately.

	p := Pipeline{Transforms: []Transform{RankNormalizeTransform{}}}
	_, err := p.Apply(ctx, []float64{1, 2, 3}, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}

type errorTransform struct{}

func (errorTransform) Apply(_ context.Context, _ []float64, _ [][]float64) ([]float64, error) {
	return nil, errors.New("transform: test error")
}

func TestPipelineTransformError(t *testing.T) {
	p := Pipeline{
		Transforms: []Transform{
			RankNormalizeTransform{},
			errorTransform{},
		},
	}
	_, err := p.Apply(context.Background(), []float64{1, 2, 3}, nil)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if err.Error() != "transform: test error" {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRankNormalizeTransform(t *testing.T) {
	tr := RankNormalizeTransform{}
	input := []float64{3.0, 1.0, 2.0}
	got, err := tr.Apply(context.Background(), input, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := RankNormalize(input)
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("index %d: got %f, want %f", i, got[i], want[i])
		}
	}
}

func TestGaussianizeTransform(t *testing.T) {
	tr := GaussianizeTransform{}
	input := []float64{3.0, 1.0, 2.0}
	got, err := tr.Apply(context.Background(), input, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := Gaussianize(input)
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("index %d: got %f, want %f", i, got[i], want[i])
		}
	}
}

func TestNeutralizeTransform(t *testing.T) {
	preds := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	features := [][]float64{{1.0, 2.0, 3.0, 4.0, 5.0}}

	tr := NeutralizeTransform{Proportion: 1.0}
	got, err := tr.Apply(context.Background(), preds, features)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := Neutralize(preds, features, 1.0)
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-12 {
			t.Errorf("index %d: got %f, want %f", i, got[i], want[i])
		}
	}
}

func TestNeutralizeTransformZeroProportion(t *testing.T) {
	preds := []float64{1.0, 2.0, 3.0}
	features := [][]float64{{1.0, 2.0, 3.0}}

	tr := NeutralizeTransform{Proportion: 0.0}
	got, err := tr.Apply(context.Background(), preds, features)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for i := range preds {
		if got[i] != preds[i] {
			t.Errorf("index %d: got %f, want %f", i, got[i], preds[i])
		}
	}
}

func TestPipelineNeutralizeViaTransform(t *testing.T) {
	preds := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	features := [][]float64{{5.0, 4.0, 3.0, 2.0, 1.0}}

	p := Pipeline{
		Transforms: []Transform{
			RankNormalizeTransform{},
			NeutralizeTransform{Proportion: 0.5},
		},
	}
	got, err := p.Apply(context.Background(), preds, features)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	step1 := RankNormalize(preds)
	want := Neutralize(step1, features, 0.5)
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-12 {
			t.Errorf("index %d: got %f, want %f", i, got[i], want[i])
		}
	}
}

// cancelAfterTransform cancels the context after being called once.
type cancelAfterTransform struct {
	cancel context.CancelFunc
}

func (c *cancelAfterTransform) Apply(_ context.Context, preds []float64, _ [][]float64) ([]float64, error) {
	out := make([]float64, len(preds))
	copy(out, preds)
	c.cancel()
	return out, nil
}

func TestPipelineCancelBetweenTransforms(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	p := Pipeline{
		Transforms: []Transform{
			&cancelAfterTransform{cancel: cancel},
			RankNormalizeTransform{}, // Should not run.
		},
	}
	_, err := p.Apply(ctx, []float64{1, 2, 3}, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}
