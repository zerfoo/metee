//go:build xgboost

package xgboost

import (
	"context"
	"os"
	"testing"

	"github.com/zerfoo/metee/metrics"
)

func TestBoosterRoundTrip(t *testing.T) {
	nSamples := 200
	nFeatures := 5

	features := make([][]float64, nSamples)
	targets := make([]float64, nSamples)
	featureNames := make([]string, nFeatures)

	for j := range nFeatures {
		featureNames[j] = "f" + string(rune('a'+j))
	}

	// Deterministic data: target = 0.5*f0 + 0.3*f1
	for i := range nSamples {
		features[i] = make([]float64, nFeatures)
		for j := range nFeatures {
			features[i][j] = float64(i*nFeatures+j) / float64(nSamples*nFeatures)
		}
		targets[i] = features[i][0]*0.5 + features[i][1]*0.3
	}

	params := DefaultParams()
	params.NumRounds = 50
	params.MaxDepth = 4
	params.Verbose = 0

	ctx := context.Background()
	booster := NewBooster("test", params, featureNames)

	// Train
	if err := booster.Train(ctx, features, targets); err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Predict
	preds, err := booster.Predict(ctx, features)
	if err != nil {
		t.Fatalf("Predict() error: %v", err)
	}
	if len(preds) != nSamples {
		t.Fatalf("Predict() returned %d values, want %d", len(preds), nSamples)
	}

	corr := metrics.PearsonCorrelation(preds, targets)
	if corr < 0.5 {
		t.Errorf("correlation = %v, want > 0.5", corr)
	}

	// Save and Load
	tmpFile, err := os.CreateTemp("", "metee-xgb-test-*.model")
	if err != nil {
		t.Fatalf("CreateTemp error: %v", err)
	}
	defer os.Remove(tmpFile.Name())
	tmpFile.Close()

	if err := booster.Save(ctx, tmpFile.Name()); err != nil {
		t.Fatalf("Save() error: %v", err)
	}

	booster2 := NewBooster("test-loaded", params, featureNames)
	if err := booster2.Load(ctx, tmpFile.Name()); err != nil {
		t.Fatalf("Load() error: %v", err)
	}

	preds2, err := booster2.Predict(ctx, features)
	if err != nil {
		t.Fatalf("Predict() after Load() error: %v", err)
	}

	corr2 := metrics.PearsonCorrelation(preds, preds2)
	if corr2 < 0.999 {
		t.Errorf("predictions diverged after save/load, correlation = %v", corr2)
	}

	// Feature importance
	imp, err := booster.Importance()
	if err != nil {
		t.Fatalf("Importance() error: %v", err)
	}
	if len(imp) == 0 {
		t.Error("Importance() returned empty map")
	}
}

func TestBoosterErrors(t *testing.T) {
	ctx := context.Background()
	b := NewBooster("test", DefaultParams(), nil)

	// Predict before training
	_, err := b.Predict(ctx, [][]float64{{1, 2}})
	if err == nil {
		t.Error("Predict() on untrained booster should return error")
	}

	// Save before training
	if err := b.Save(ctx, "/tmp/nope.model"); err == nil {
		t.Error("Save() on untrained booster should return error")
	}

	// Train with empty data
	if err := b.Train(ctx, nil, nil); err == nil {
		t.Error("Train() with empty data should return error")
	}
}

func TestBoosterName(t *testing.T) {
	b := NewBooster("mymodel", DefaultParams(), nil)
	if b.Name() != "mymodel" {
		t.Errorf("Name() = %q, want mymodel", b.Name())
	}
}
