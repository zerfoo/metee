package trainer

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/zerfoo/metee/data"
)

// mockModel implements model.Model for testing.
type mockModel struct {
	name        string
	trainErr    error
	predictErr  error
	saveErr     error
	predictions []float64
	trained     bool
	savedPath   string
}

func (m *mockModel) Train(_ context.Context, _ [][]float64, _ []float64) error {
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
		preds[i] = 0.5
	}
	return preds, nil
}

func (m *mockModel) Save(_ context.Context, path string) error {
	if m.saveErr != nil {
		return m.saveErr
	}
	m.savedPath = path
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, []byte("mock"), 0o644)
}

func (m *mockModel) Load(_ context.Context, _ string) error { return nil }

func (m *mockModel) Importance() (map[string]float64, error) {
	return map[string]float64{"f1": 1.0}, nil
}

func (m *mockModel) Name() string {
	if m.name != "" {
		return m.name
	}
	return "mock"
}

func makeDataset(n int) *data.Dataset {
	features := make([][]float64, n)
	targets := make([]float64, n)
	for i := 0; i < n; i++ {
		features[i] = []float64{float64(i), float64(i) * 2}
		targets[i] = float64(i) * 0.1
	}
	return &data.Dataset{
		Features:     features,
		Targets:      targets,
		FeatureNames: []string{"f1", "f2"},
	}
}

func TestRun(t *testing.T) {
	tests := []struct {
		name       string
		cfg        TrainerConfig
		model      *mockModel
		train      *data.Dataset
		valid      *data.Dataset
		wantErr    bool
		errContain string
		check      func(t *testing.T, result TrainResult, m *mockModel)
	}{
		{
			name: "basic train and validate",
			cfg: TrainerConfig{
				EarlyStoppingMetric: func(preds, targets []float64) float64 {
					return 0.42
				},
			},
			model: &mockModel{},
			train: makeDataset(10),
			valid: makeDataset(5),
			check: func(t *testing.T, result TrainResult, m *mockModel) {
				t.Helper()
				if !m.trained {
					t.Error("model was not trained")
				}
				if result.BestScore != 0.42 {
					t.Errorf("BestScore = %v, want 0.42", result.BestScore)
				}
				if result.BestRound != 1 {
					t.Errorf("BestRound = %v, want 1", result.BestRound)
				}
				if len(result.ValidationScores) != 1 || result.ValidationScores[0] != 0.42 {
					t.Errorf("ValidationScores = %v, want [0.42]", result.ValidationScores)
				}
				if result.CheckpointPath != "" {
					t.Errorf("CheckpointPath = %q, want empty", result.CheckpointPath)
				}
			},
		},
		{
			name: "with checkpoint",
			cfg: TrainerConfig{
				CheckpointDir: t.TempDir(),
				EarlyStoppingMetric: func(preds, targets []float64) float64 {
					return 0.55
				},
			},
			model: &mockModel{name: "mymodel"},
			train: makeDataset(10),
			valid: makeDataset(5),
			check: func(t *testing.T, result TrainResult, m *mockModel) {
				t.Helper()
				if result.CheckpointPath == "" {
					t.Fatal("CheckpointPath is empty")
				}
				if _, err := os.Stat(result.CheckpointPath); err != nil {
					t.Errorf("checkpoint file not found: %v", err)
				}
				wantPath := filepath.Join(filepath.Dir(result.CheckpointPath), "mymodel.model")
				if result.CheckpointPath != wantPath {
					t.Errorf("CheckpointPath = %q, want %q", result.CheckpointPath, wantPath)
				}
			},
		},
		{
			name:  "no checkpoint dir",
			cfg:   TrainerConfig{},
			model: &mockModel{},
			train: makeDataset(10),
			valid: makeDataset(5),
			check: func(t *testing.T, result TrainResult, m *mockModel) {
				t.Helper()
				if result.CheckpointPath != "" {
					t.Errorf("CheckpointPath = %q, want empty", result.CheckpointPath)
				}
			},
		},
		{
			name:  "no metric function",
			cfg:   TrainerConfig{},
			model: &mockModel{},
			train: makeDataset(10),
			valid: makeDataset(5),
			check: func(t *testing.T, result TrainResult, m *mockModel) {
				t.Helper()
				if result.BestScore != 0 {
					t.Errorf("BestScore = %v, want 0", result.BestScore)
				}
				if result.ValidationScores != nil {
					t.Errorf("ValidationScores = %v, want nil", result.ValidationScores)
				}
			},
		},
		{
			name:       "train error",
			cfg:        TrainerConfig{},
			model:      &mockModel{trainErr: errors.New("train failed")},
			train:      makeDataset(10),
			valid:      makeDataset(5),
			wantErr:    true,
			errContain: "trainer: train:",
		},
		{
			name:       "predict error",
			cfg:        TrainerConfig{},
			model:      &mockModel{predictErr: errors.New("predict failed")},
			train:      makeDataset(10),
			valid:      makeDataset(5),
			wantErr:    true,
			errContain: "trainer: predict:",
		},
		{
			name: "save error",
			cfg: TrainerConfig{
				CheckpointDir: t.TempDir(),
			},
			model:      &mockModel{saveErr: errors.New("save failed")},
			train:      makeDataset(10),
			valid:      makeDataset(5),
			wantErr:    true,
			errContain: "trainer: checkpoint:",
		},
		{
			name:  "context cancelled before train",
			cfg:   TrainerConfig{},
			model: &mockModel{},
			train: makeDataset(10),
			valid: makeDataset(5),
			check: func(t *testing.T, result TrainResult, m *mockModel) {
				// handled separately
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()

			trainer := &Trainer{}
			result, err := trainer.Run(ctx, tt.cfg, tt.model, tt.train, tt.valid)

			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if tt.errContain != "" {
					if got := err.Error(); !contains(got, tt.errContain) {
						t.Errorf("error = %q, want to contain %q", got, tt.errContain)
					}
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if tt.check != nil {
				tt.check(t, result, tt.model)
			}
		})
	}
}

func TestRunContextCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	trainer := &Trainer{}
	_, err := trainer.Run(ctx, TrainerConfig{}, &mockModel{}, makeDataset(5), makeDataset(3))
	if err == nil {
		t.Fatal("expected error for cancelled context")
	}
	if got := err.Error(); !contains(got, "trainer:") {
		t.Errorf("error = %q, want to contain 'trainer:'", got)
	}
}

func TestRunCustomPredictions(t *testing.T) {
	preds := []float64{0.1, 0.2, 0.3}
	m := &mockModel{predictions: preds}
	valid := makeDataset(3)

	var gotPreds []float64
	cfg := TrainerConfig{
		EarlyStoppingMetric: func(p, targets []float64) float64 {
			gotPreds = p
			return 0.99
		},
	}

	trainer := &Trainer{}
	result, err := trainer.Run(context.Background(), cfg, m, makeDataset(10), valid)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.BestScore != 0.99 {
		t.Errorf("BestScore = %v, want 0.99", result.BestScore)
	}
	if len(gotPreds) != 3 {
		t.Fatalf("metric received %d predictions, want 3", len(gotPreds))
	}
	for i, p := range gotPreds {
		if p != preds[i] {
			t.Errorf("preds[%d] = %v, want %v", i, p, preds[i])
		}
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && searchString(s, substr)
}

func searchString(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
