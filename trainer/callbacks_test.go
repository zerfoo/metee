package trainer

import (
	"bytes"
	"context"
	"strings"
	"testing"
)

// recordingCallback records the order of callback invocations.
type recordingCallback struct {
	calls  []string
	result *TrainResult
}

func (r *recordingCallback) OnTrainStart(_ context.Context) {
	r.calls = append(r.calls, "start")
}

func (r *recordingCallback) OnTrainEnd(_ context.Context, result TrainResult) {
	r.calls = append(r.calls, "end")
	r.result = &result
}

func TestCallbacksInvocationOrder(t *testing.T) {
	rec := &recordingCallback{}
	trainer := &Trainer{Callbacks: []Callback{rec}}

	cfg := TrainerConfig{
		EarlyStoppingMetric: func(preds, targets []float64) float64 { return 0.42 },
	}

	result, err := trainer.Run(context.Background(), cfg, &mockModel{}, makeDataset(10), makeDataset(5))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(rec.calls) != 2 {
		t.Fatalf("calls = %v, want [start end]", rec.calls)
	}
	if rec.calls[0] != "start" || rec.calls[1] != "end" {
		t.Errorf("calls = %v, want [start end]", rec.calls)
	}
	if rec.result == nil {
		t.Fatal("OnTrainEnd result is nil")
	}
	if rec.result.BestScore != result.BestScore {
		t.Errorf("callback result.BestScore = %v, want %v", rec.result.BestScore, result.BestScore)
	}
}

func TestMultipleCallbacks(t *testing.T) {
	rec1 := &recordingCallback{}
	rec2 := &recordingCallback{}
	trainer := &Trainer{Callbacks: []Callback{rec1, rec2}}

	cfg := TrainerConfig{}
	_, err := trainer.Run(context.Background(), cfg, &mockModel{}, makeDataset(5), makeDataset(3))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for i, rec := range []*recordingCallback{rec1, rec2} {
		if len(rec.calls) != 2 {
			t.Errorf("callback %d: calls = %v, want [start end]", i, rec.calls)
		}
	}
}

func TestNoCallbacks(t *testing.T) {
	trainer := &Trainer{}
	_, err := trainer.Run(context.Background(), TrainerConfig{}, &mockModel{}, makeDataset(5), makeDataset(3))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestLogCallbackOutput(t *testing.T) {
	var buf bytes.Buffer
	lc := &LogCallback{Writer: &buf}
	trainer := &Trainer{Callbacks: []Callback{lc}}

	cfg := TrainerConfig{
		EarlyStoppingMetric: func(preds, targets []float64) float64 { return 0.75 },
	}

	_, err := trainer.Run(context.Background(), cfg, &mockModel{}, makeDataset(10), makeDataset(5))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	output := buf.String()
	if !strings.Contains(output, "training started") {
		t.Errorf("output missing 'training started': %q", output)
	}
	if !strings.Contains(output, "training ended") {
		t.Errorf("output missing 'training ended': %q", output)
	}
	if !strings.Contains(output, "0.7500") {
		t.Errorf("output missing best score: %q", output)
	}
}

func TestCallbacksNotCalledOnCancelledContext(t *testing.T) {
	rec := &recordingCallback{}
	trainer := &Trainer{Callbacks: []Callback{rec}}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := trainer.Run(ctx, TrainerConfig{}, &mockModel{}, makeDataset(5), makeDataset(3))
	if err == nil {
		t.Fatal("expected error for cancelled context")
	}
	if len(rec.calls) != 0 {
		t.Errorf("callbacks should not be called on cancelled context, got %v", rec.calls)
	}
}
