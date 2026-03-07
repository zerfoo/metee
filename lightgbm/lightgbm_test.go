//go:build lightgbm

package lightgbm

import (
	"context"
	"math/rand"
	"os"
	"testing"

	"github.com/zerfoo/metee/metrics"
)

func TestBoosterRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	nSamples := 100
	nFeatures := 10

	features := make([][]float64, nSamples)
	targets := make([]float64, nSamples)
	featureNames := make([]string, nFeatures)

	for j := 0; j < nFeatures; j++ {
		featureNames[j] = "f" + string(rune('0'+j))
	}

	for i := 0; i < nSamples; i++ {
		features[i] = make([]float64, nFeatures)
		for j := 0; j < nFeatures; j++ {
			features[i][j] = rng.Float64()
		}
		// Target is a simple linear combination + noise
		targets[i] = features[i][0]*0.5 + features[i][1]*0.3 + rng.Float64()*0.1
	}

	params := DefaultParams()
	params.NumIterations = 50
	params.NumLeaves = 8

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

	corr := metrics.PearsonCorrelation(preds, targets)
	if corr < 0.5 {
		t.Errorf("correlation = %v, want > 0.5", corr)
	}

	// Save and Load
	tmpFile, err := os.CreateTemp("", "metee-test-*.model")
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
