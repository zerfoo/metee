package ensemble

import (
	"context"
	"errors"
	"testing"

	"github.com/zerfoo/metee/cv"
	"github.com/zerfoo/metee/data"
	"github.com/zerfoo/metee/model"
)

// mockModel implements model.Model for testing.
type mockModel struct {
	name       string
	trained    bool
	trainErr   error
	predictErr error
	trainCalls int
	predictions []float64
}

func (m *mockModel) Train(_ context.Context, features [][]float64, targets []float64) error {
	m.trainCalls++
	if m.trainErr != nil {
		return m.trainErr
	}
	m.trained = true
	return nil
}

func (m *mockModel) Predict(_ context.Context, features [][]float64) ([]float64, error) {
	if m.predictErr != nil {
		return nil, m.predictErr
	}
	if m.predictions != nil {
		return m.predictions, nil
	}
	preds := make([]float64, len(features))
	for i := range preds {
		preds[i] = float64(i) * 0.1
	}
	return preds, nil
}

func (m *mockModel) Save(_ context.Context, _ string) error          { return nil }
func (m *mockModel) Load(_ context.Context, _ string) error          { return nil }
func (m *mockModel) Importance() (map[string]float64, error)         { return nil, nil }
func (m *mockModel) Name() string                                    { return m.name }

func testDataset() *data.Dataset {
	return &data.Dataset{
		IDs:          []string{"a", "b", "c", "d", "e", "f"},
		Features:     [][]float64{{1}, {2}, {3}, {4}, {5}, {6}},
		Targets:      []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
		Eras:         []int{1, 1, 2, 2, 3, 3},
		FeatureNames: []string{"f1"},
	}
}

func TestStacker_Fit(t *testing.T) {
	ds := testDataset()
	folds := cv.KFold(ds.Eras, 3)

	base1 := &mockModel{name: "base1"}
	base2 := &mockModel{name: "base2"}
	meta := &mockModel{name: "meta"}

	s := &Stacker{
		BaseModels: []model.Model{base1, base2},
		MetaModel:  meta,
		Folds:      folds,
	}

	err := s.Fit(context.Background(), ds)
	if err != nil {
		t.Fatalf("Fit() error = %v", err)
	}

	// Each base model: trained once per fold (3) + once on full dataset (1) = 4
	if base1.trainCalls != 4 {
		t.Errorf("base1 trainCalls = %d, want 4", base1.trainCalls)
	}
	if base2.trainCalls != 4 {
		t.Errorf("base2 trainCalls = %d, want 4", base2.trainCalls)
	}

	// Meta model should be trained once.
	if meta.trainCalls != 1 {
		t.Errorf("meta trainCalls = %d, want 1", meta.trainCalls)
	}

	// All models should be trained.
	if !base1.trained || !base2.trained || !meta.trained {
		t.Error("not all models were trained")
	}
}

func TestStacker_Predict(t *testing.T) {
	base1 := &mockModel{name: "base1", trained: true, predictions: []float64{0.1, 0.2}}
	base2 := &mockModel{name: "base2", trained: true, predictions: []float64{0.3, 0.4}}
	meta := &mockModel{name: "meta", trained: true, predictions: []float64{0.5, 0.6}}

	s := &Stacker{
		BaseModels: []model.Model{base1, base2},
		MetaModel:  meta,
	}

	features := [][]float64{{1}, {2}}
	preds, err := s.Predict(context.Background(), features)
	if err != nil {
		t.Fatalf("Predict() error = %v", err)
	}

	if len(preds) != 2 {
		t.Fatalf("Predict() returned %d predictions, want 2", len(preds))
	}
	if preds[0] != 0.5 || preds[1] != 0.6 {
		t.Errorf("Predict() = %v, want [0.5 0.6]", preds)
	}
}

func TestStacker_Fit_NoBaseModels(t *testing.T) {
	s := &Stacker{
		MetaModel: &mockModel{name: "meta"},
		Folds:     []cv.Fold{{TrainEras: []int{1}, ValidEras: []int{2}}},
	}
	err := s.Fit(context.Background(), testDataset())
	if err == nil {
		t.Fatal("expected error for no base models")
	}
}

func TestStacker_Fit_NoMetaModel(t *testing.T) {
	s := &Stacker{
		BaseModels: []model.Model{&mockModel{name: "b"}},
		Folds:      []cv.Fold{{TrainEras: []int{1}, ValidEras: []int{2}}},
	}
	err := s.Fit(context.Background(), testDataset())
	if err == nil {
		t.Fatal("expected error for no meta model")
	}
}

func TestStacker_Fit_NoFolds(t *testing.T) {
	s := &Stacker{
		BaseModels: []model.Model{&mockModel{name: "b"}},
		MetaModel:  &mockModel{name: "meta"},
	}
	err := s.Fit(context.Background(), testDataset())
	if err == nil {
		t.Fatal("expected error for no folds")
	}
}

func TestStacker_Fit_TrainError(t *testing.T) {
	ds := testDataset()
	folds := cv.KFold(ds.Eras, 2)

	s := &Stacker{
		BaseModels: []model.Model{&mockModel{name: "bad", trainErr: errors.New("train failed")}},
		MetaModel:  &mockModel{name: "meta"},
		Folds:      folds,
	}
	err := s.Fit(context.Background(), ds)
	if err == nil {
		t.Fatal("expected error from base model train failure")
	}
}

func TestStacker_Fit_PredictError(t *testing.T) {
	ds := testDataset()
	folds := cv.KFold(ds.Eras, 2)

	s := &Stacker{
		BaseModels: []model.Model{&mockModel{name: "bad", predictErr: errors.New("predict failed")}},
		MetaModel:  &mockModel{name: "meta"},
		Folds:      folds,
	}
	err := s.Fit(context.Background(), ds)
	if err == nil {
		t.Fatal("expected error from base model predict failure")
	}
}

func TestStacker_Fit_ContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	ds := testDataset()
	folds := cv.KFold(ds.Eras, 2)

	s := &Stacker{
		BaseModels: []model.Model{&mockModel{name: "b"}},
		MetaModel:  &mockModel{name: "meta"},
		Folds:      folds,
	}
	err := s.Fit(ctx, ds)
	if err == nil {
		t.Fatal("expected error from cancelled context")
	}
}

func TestStacker_Predict_NoBaseModels(t *testing.T) {
	s := &Stacker{MetaModel: &mockModel{name: "meta"}}
	_, err := s.Predict(context.Background(), [][]float64{{1}})
	if err == nil {
		t.Fatal("expected error for no base models")
	}
}

func TestStacker_Predict_NoMetaModel(t *testing.T) {
	s := &Stacker{BaseModels: []model.Model{&mockModel{name: "b"}}}
	_, err := s.Predict(context.Background(), [][]float64{{1}})
	if err == nil {
		t.Fatal("expected error for no meta model")
	}
}

func TestStacker_Predict_ContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	s := &Stacker{
		BaseModels: []model.Model{&mockModel{name: "b"}},
		MetaModel:  &mockModel{name: "meta"},
	}
	_, err := s.Predict(ctx, [][]float64{{1}})
	if err == nil {
		t.Fatal("expected error from cancelled context")
	}
}

func TestStacker_Predict_BasePredictError(t *testing.T) {
	s := &Stacker{
		BaseModels: []model.Model{&mockModel{name: "bad", predictErr: errors.New("fail")}},
		MetaModel:  &mockModel{name: "meta"},
	}
	_, err := s.Predict(context.Background(), [][]float64{{1}})
	if err == nil {
		t.Fatal("expected error from base predict failure")
	}
}

func TestStacker_Predict_MetaPredictError(t *testing.T) {
	s := &Stacker{
		BaseModels: []model.Model{&mockModel{name: "b", predictions: []float64{0.1}}},
		MetaModel:  &mockModel{name: "meta", predictErr: errors.New("fail")},
	}
	_, err := s.Predict(context.Background(), [][]float64{{1}})
	if err == nil {
		t.Fatal("expected error from meta predict failure")
	}
}
