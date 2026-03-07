package trainer

import (
	"context"
	"fmt"
	"io"
)

// Callback defines hooks invoked during training.
type Callback interface {
	OnTrainStart(ctx context.Context)
	OnTrainEnd(ctx context.Context, result TrainResult)
}

// LogCallback logs training results to a writer.
type LogCallback struct {
	Writer io.Writer
}

func (l *LogCallback) OnTrainStart(_ context.Context) {
	fmt.Fprintln(l.Writer, "training started")
}

func (l *LogCallback) OnTrainEnd(_ context.Context, result TrainResult) {
	fmt.Fprintf(l.Writer, "training ended: best_score=%.4f best_round=%d\n", result.BestScore, result.BestRound)
}
