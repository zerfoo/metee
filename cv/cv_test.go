package cv

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/zerfoo/metee/data"
)

// mockModel implements model.Model for testing.
type mockModel struct {
	trainErr   error
	predictErr error
	lastTrain  [][]float64
}

func (m *mockModel) Train(_ context.Context, features [][]float64, _ []float64) error {
	if m.trainErr != nil {
		return m.trainErr
	}
	m.lastTrain = features
	return nil
}

func (m *mockModel) Predict(_ context.Context, features [][]float64) ([]float64, error) {
	if m.predictErr != nil {
		return nil, m.predictErr
	}
	preds := make([]float64, len(features))
	for i, row := range features {
		var sum float64
		for _, v := range row {
			sum += v
		}
		preds[i] = sum / float64(len(row))
	}
	return preds, nil
}

func (m *mockModel) Save(_ context.Context, _ string) error  { return nil }
func (m *mockModel) Load(_ context.Context, _ string) error  { return nil }
func (m *mockModel) Importance() (map[string]float64, error) { return nil, nil }
func (m *mockModel) Name() string                            { return "mock" }

// simpleMetric returns the mean absolute difference (inverted for testing).
func simpleMetric(preds, targets []float64) float64 {
	if len(preds) == 0 {
		return 0
	}
	var sum float64
	for i := range preds {
		sum += preds[i] - targets[i]
	}
	return sum / float64(len(preds))
}

func TestCrossValidate(t *testing.T) {
	ds := makeTestDataset()
	folds := KFold(ds.Eras, 3)
	m := &mockModel{}

	result, err := CrossValidate(context.Background(), ds, m, folds, simpleMetric)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Scores) != 3 {
		t.Fatalf("expected 3 scores, got %d", len(result.Scores))
	}

	// Verify mean manually.
	var sum float64
	for _, s := range result.Scores {
		sum += s
	}
	wantMean := sum / float64(len(result.Scores))
	if math.Abs(result.Mean-wantMean) > 1e-12 {
		t.Errorf("Mean = %v, want %v", result.Mean, wantMean)
	}

	// Verify std manually.
	var sumSq float64
	for _, s := range result.Scores {
		d := s - wantMean
		sumSq += d * d
	}
	wantStd := math.Sqrt(sumSq / float64(len(result.Scores)))
	if math.Abs(result.Std-wantStd) > 1e-12 {
		t.Errorf("Std = %v, want %v", result.Std, wantStd)
	}
}

func TestCrossValidateContextCancellation(t *testing.T) {
	ds := makeTestDataset()
	folds := KFold(ds.Eras, 3)
	m := &mockModel{}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately.

	_, err := CrossValidate(ctx, ds, m, folds, simpleMetric)
	if err == nil {
		t.Fatal("expected error from cancelled context")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", err)
	}
}

func TestCrossValidateTrainError(t *testing.T) {
	ds := makeTestDataset()
	folds := KFold(ds.Eras, 3)
	trainErr := errors.New("train failed")
	m := &mockModel{trainErr: trainErr}

	_, err := CrossValidate(context.Background(), ds, m, folds, simpleMetric)
	if !errors.Is(err, trainErr) {
		t.Errorf("expected train error, got %v", err)
	}
}

func TestCrossValidatePredictError(t *testing.T) {
	ds := makeTestDataset()
	folds := KFold(ds.Eras, 3)
	predictErr := errors.New("predict failed")
	m := &mockModel{predictErr: predictErr}

	_, err := CrossValidate(context.Background(), ds, m, folds, simpleMetric)
	if !errors.Is(err, predictErr) {
		t.Errorf("expected predict error, got %v", err)
	}
}

func TestCrossValidateEmptyFolds(t *testing.T) {
	ds := makeTestDataset()
	m := &mockModel{}

	result, err := CrossValidate(context.Background(), ds, m, nil, simpleMetric)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Scores) != 0 {
		t.Errorf("expected 0 scores, got %d", len(result.Scores))
	}
	if result.Mean != 0 || result.Std != 0 {
		t.Errorf("expected zero Mean/Std for empty folds, got Mean=%v Std=%v", result.Mean, result.Std)
	}
}

func TestCrossValidateWalkForward(t *testing.T) {
	ds := makeTestDataset()
	folds := WalkForward(ds.Eras, 2, 1, 1)
	m := &mockModel{}

	result, err := CrossValidate(context.Background(), ds, m, folds, simpleMetric)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Scores) != len(folds) {
		t.Errorf("expected %d scores, got %d", len(folds), len(result.Scores))
	}
}

// makeTestDataset creates a small dataset with 4 eras, 3 samples per era.
func makeTestDataset() *data.Dataset {
	nEras := 4
	samplesPerEra := 3
	n := nEras * samplesPerEra

	ds := &data.Dataset{
		Features:     make([][]float64, n),
		Targets:      make([]float64, n),
		Eras:         make([]int, n),
		FeatureNames: []string{"f1", "f2"},
	}

	for i := 0; i < n; i++ {
		era := i/samplesPerEra + 1
		ds.Eras[i] = era
		ds.Features[i] = []float64{float64(i), float64(era)}
		ds.Targets[i] = float64(i) * 0.1
	}
	return ds
}
