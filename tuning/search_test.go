package tuning

import (
	"context"
	"math/rand"
	"testing"

	"github.com/zerfoo/metee/cv"
	"github.com/zerfoo/metee/data"
	"github.com/zerfoo/metee/model"
)

// mockModel implements model.Model and model.Configurable for testing.
type mockModel struct {
	params map[string]any
}

func (m *mockModel) Train(_ context.Context, _ [][]float64, _ []float64) error { return nil }

func (m *mockModel) Predict(_ context.Context, features [][]float64) ([]float64, error) {
	// Return predictions that vary based on params so different configs produce different scores.
	base := 0.5
	if m.params != nil {
		if v, ok := m.params["depth"]; ok {
			switch v.(int) {
			case 3:
				base = 0.3
			case 6:
				base = 0.9
			}
		}
	}
	preds := make([]float64, len(features))
	for i := range preds {
		preds[i] = base
	}
	return preds, nil
}

func (m *mockModel) Save(_ context.Context, _ string) error  { return nil }
func (m *mockModel) Load(_ context.Context, _ string) error  { return nil }
func (m *mockModel) Importance() (map[string]float64, error) { return nil, nil }
func (m *mockModel) Name() string                            { return "mock" }
func (m *mockModel) SetParams(params map[string]any) error {
	m.params = params
	return nil
}

var _ model.Model = (*mockModel)(nil)
var _ model.Configurable = (*mockModel)(nil)

func testDataset() *data.Dataset {
	n := 20
	features := make([][]float64, n)
	targets := make([]float64, n)
	eras := make([]int, n)
	for i := 0; i < n; i++ {
		features[i] = []float64{float64(i)}
		targets[i] = float64(i) * 0.1
		eras[i] = i / 5 // eras 0,1,2,3
	}
	return &data.Dataset{
		Features:     features,
		Targets:      targets,
		Eras:         eras,
		FeatureNames: []string{"f1"},
	}
}

func sumMetric(preds, targets []float64) float64 {
	var s float64
	for _, p := range preds {
		s += p
	}
	return s
}

func TestGridSearchBasic(t *testing.T) {
	ds := testDataset()
	space := ParamSpace{
		"depth": Discrete(3, 6),
		"lr":    Discrete(0.1, 0.2),
	}
	folds := cv.KFold([]int{0, 1, 2, 3}, 2)
	factory := func() model.Model { return &mockModel{} }

	result, err := GridSearch(context.Background(), ds, factory, space, folds, sumMetric)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// 2 * 2 = 4 trials
	if len(result.AllTrials) != 4 {
		t.Fatalf("AllTrials len = %d, want 4", len(result.AllTrials))
	}

	// depth=6 produces higher predictions (0.9 vs 0.3), so best should have depth=6
	if result.BestParams["depth"] != 6 {
		t.Errorf("BestParams[depth] = %v, want 6", result.BestParams["depth"])
	}

	if result.BestScore <= 0 {
		t.Errorf("BestScore = %v, want > 0", result.BestScore)
	}
}

func TestGridSearchEmpty(t *testing.T) {
	ds := testDataset()
	space := ParamSpace{
		"lr": Uniform(0.01, 0.1), // continuous only, Grid() returns nil
	}
	folds := cv.KFold([]int{0, 1, 2, 3}, 2)
	factory := func() model.Model { return &mockModel{} }

	result, err := GridSearch(context.Background(), ds, factory, space, folds, sumMetric)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.AllTrials) != 0 {
		t.Errorf("AllTrials len = %d, want 0", len(result.AllTrials))
	}
}

func TestRandomSearchBasic(t *testing.T) {
	ds := testDataset()
	space := ParamSpace{
		"depth": Discrete(3, 6),
	}
	folds := cv.KFold([]int{0, 1, 2, 3}, 2)
	factory := func() model.Model { return &mockModel{} }
	rng := rand.New(rand.NewSource(42))

	result, err := RandomSearch(context.Background(), ds, factory, space, folds, sumMetric, 3, rng)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.AllTrials) != 3 {
		t.Fatalf("AllTrials len = %d, want 3", len(result.AllTrials))
	}
	if result.BestScore <= 0 {
		t.Errorf("BestScore = %v, want > 0", result.BestScore)
	}
}

func TestRandomSearchZeroTrials(t *testing.T) {
	ds := testDataset()
	space := ParamSpace{"depth": Discrete(3)}
	folds := cv.KFold([]int{0, 1}, 2)
	factory := func() model.Model { return &mockModel{} }
	rng := rand.New(rand.NewSource(42))

	result, err := RandomSearch(context.Background(), ds, factory, space, folds, sumMetric, 0, rng)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.AllTrials) != 0 {
		t.Errorf("AllTrials len = %d, want 0", len(result.AllTrials))
	}
}

func TestGridSearchContextCancellation(t *testing.T) {
	ds := testDataset()
	space := ParamSpace{
		"depth": Discrete(3, 6, 9),
		"lr":    Discrete(0.1, 0.2, 0.3),
	}
	folds := cv.KFold([]int{0, 1, 2, 3}, 2)

	ctx, cancel := context.WithCancel(context.Background())
	trialCount := 0
	factory := func() model.Model {
		trialCount++
		if trialCount >= 2 {
			cancel()
		}
		return &mockModel{}
	}

	result, err := GridSearch(ctx, ds, factory, space, folds, sumMetric)
	if err == nil {
		t.Fatal("expected context cancellation error")
	}
	if len(result.AllTrials) >= 9 {
		t.Errorf("expected fewer than 9 trials due to cancellation, got %d", len(result.AllTrials))
	}
}

func TestRandomSearchContextCancellation(t *testing.T) {
	ds := testDataset()
	space := ParamSpace{
		"depth": Discrete(3, 6, 9),
	}
	folds := cv.KFold([]int{0, 1, 2, 3}, 2)
	rng := rand.New(rand.NewSource(42))

	ctx, cancel := context.WithCancel(context.Background())
	trialCount := 0
	factory := func() model.Model {
		trialCount++
		if trialCount >= 2 {
			cancel()
		}
		return &mockModel{}
	}

	result, err := RandomSearch(ctx, ds, factory, space, folds, sumMetric, 10, rng)
	if err == nil {
		t.Fatal("expected context cancellation error")
	}
	if len(result.AllTrials) >= 10 {
		t.Errorf("expected fewer than 10 trials due to cancellation, got %d", len(result.AllTrials))
	}
}

func TestGridSearchBestParamsMatchesHighestScore(t *testing.T) {
	ds := testDataset()
	space := ParamSpace{
		"depth": Discrete(3, 6),
	}
	folds := cv.KFold([]int{0, 1, 2, 3}, 2)
	factory := func() model.Model { return &mockModel{} }

	result, err := GridSearch(context.Background(), ds, factory, space, folds, sumMetric)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Find the actual best trial
	var bestTrial Trial
	for i, trial := range result.AllTrials {
		if i == 0 || trial.Score > bestTrial.Score {
			bestTrial = trial
		}
	}

	if result.BestScore != bestTrial.Score {
		t.Errorf("BestScore = %v, want %v (actual best trial)", result.BestScore, bestTrial.Score)
	}

	if result.BestParams["depth"] != bestTrial.Params["depth"] {
		t.Errorf("BestParams[depth] = %v, want %v", result.BestParams["depth"], bestTrial.Params["depth"])
	}
}

func TestTrialDurationPositive(t *testing.T) {
	ds := testDataset()
	space := ParamSpace{"depth": Discrete(3)}
	folds := cv.KFold([]int{0, 1, 2, 3}, 2)
	factory := func() model.Model { return &mockModel{} }

	result, err := GridSearch(context.Background(), ds, factory, space, folds, sumMetric)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for i, trial := range result.AllTrials {
		if trial.Duration <= 0 {
			t.Errorf("AllTrials[%d].Duration = %v, want > 0", i, trial.Duration)
		}
	}
}
