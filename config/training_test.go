package config

import (
	"reflect"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestTrainingConfigRoundTrip(t *testing.T) {
	original := TrainingConfig{
		Backend:       "lightgbm",
		ModelParams:   map[string]any{"num_leaves": 31, "learning_rate": 0.05},
		DataPath:      "/data/train.parquet",
		TargetColumn:  "target",
		EraColumn:     "era",
		IDColumn:      "id",
		FeaturePrefix: "feature_",
		TrainEras:     []int{1, 2, 3, 4, 5},
		ValidEras:     []int{6, 7},
		Seed:          42,
	}

	data, err := yaml.Marshal(original)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var decoded TrainingConfig
	if err := yaml.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if decoded.Backend != original.Backend {
		t.Errorf("Backend = %q, want %q", decoded.Backend, original.Backend)
	}
	if decoded.DataPath != original.DataPath {
		t.Errorf("DataPath = %q, want %q", decoded.DataPath, original.DataPath)
	}
	if decoded.TargetColumn != original.TargetColumn {
		t.Errorf("TargetColumn = %q, want %q", decoded.TargetColumn, original.TargetColumn)
	}
	if decoded.EraColumn != original.EraColumn {
		t.Errorf("EraColumn = %q, want %q", decoded.EraColumn, original.EraColumn)
	}
	if decoded.IDColumn != original.IDColumn {
		t.Errorf("IDColumn = %q, want %q", decoded.IDColumn, original.IDColumn)
	}
	if decoded.FeaturePrefix != original.FeaturePrefix {
		t.Errorf("FeaturePrefix = %q, want %q", decoded.FeaturePrefix, original.FeaturePrefix)
	}
	if !reflect.DeepEqual(decoded.TrainEras, original.TrainEras) {
		t.Errorf("TrainEras = %v, want %v", decoded.TrainEras, original.TrainEras)
	}
	if !reflect.DeepEqual(decoded.ValidEras, original.ValidEras) {
		t.Errorf("ValidEras = %v, want %v", decoded.ValidEras, original.ValidEras)
	}
	if decoded.Seed != original.Seed {
		t.Errorf("Seed = %d, want %d", decoded.Seed, original.Seed)
	}
	if decoded.ModelParams["num_leaves"] == nil {
		t.Error("ModelParams missing num_leaves")
	}
	if decoded.ModelParams["learning_rate"] == nil {
		t.Error("ModelParams missing learning_rate")
	}
}

func TestEnsembleConfigRoundTrip(t *testing.T) {
	original := EnsembleConfig{
		Models: []TrainingConfig{
			{
				Backend:      "lightgbm",
				ModelParams:  map[string]any{"num_leaves": 31},
				DataPath:     "/data/train.parquet",
				TargetColumn: "target",
				EraColumn:    "era",
				TrainEras:    []int{1, 2, 3},
				ValidEras:    []int{4, 5},
				Seed:         42,
			},
			{
				Backend:      "xgboost",
				ModelParams:  map[string]any{"max_depth": 6},
				DataPath:     "/data/train.parquet",
				TargetColumn: "target",
				EraColumn:    "era",
				TrainEras:    []int{1, 2, 3},
				ValidEras:    []int{4, 5},
				Seed:         99,
			},
		},
		Weights:     []float64{0.6, 0.4},
		BlendMethod: "rank",
	}

	data, err := yaml.Marshal(original)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var decoded EnsembleConfig
	if err := yaml.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if len(decoded.Models) != 2 {
		t.Fatalf("Models count = %d, want 2", len(decoded.Models))
	}
	if decoded.Models[0].Backend != "lightgbm" {
		t.Errorf("Models[0].Backend = %q, want %q", decoded.Models[0].Backend, "lightgbm")
	}
	if decoded.Models[1].Backend != "xgboost" {
		t.Errorf("Models[1].Backend = %q, want %q", decoded.Models[1].Backend, "xgboost")
	}
	if !reflect.DeepEqual(decoded.Weights, original.Weights) {
		t.Errorf("Weights = %v, want %v", decoded.Weights, original.Weights)
	}
	if decoded.BlendMethod != original.BlendMethod {
		t.Errorf("BlendMethod = %q, want %q", decoded.BlendMethod, original.BlendMethod)
	}
	if decoded.Models[0].Seed != 42 {
		t.Errorf("Models[0].Seed = %d, want 42", decoded.Models[0].Seed)
	}
	if decoded.Models[1].Seed != 99 {
		t.Errorf("Models[1].Seed = %d, want 99", decoded.Models[1].Seed)
	}
}

func TestTrainingConfigZeroValue(t *testing.T) {
	var cfg TrainingConfig
	data, err := yaml.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal zero value: %v", err)
	}
	var decoded TrainingConfig
	if err := yaml.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal zero value: %v", err)
	}
	if decoded.Backend != "" {
		t.Errorf("zero value Backend = %q, want empty", decoded.Backend)
	}
	if decoded.Seed != 0 {
		t.Errorf("zero value Seed = %d, want 0", decoded.Seed)
	}
}
