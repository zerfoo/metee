# Metee

Tree-based machine learning library for Go. "Metee" means "trees" in Swahili/Kikuyu.

Metee provides LightGBM and XGBoost bindings, ensemble methods, cross-validation, hyperparameter tuning, and a full feature-engineering pipeline — all behind clean Go interfaces. It is a standalone complement to the [Zerfoo](https://github.com/zerfoo/zerfoo) neural network framework with no dependency on it.

## Features

- **Model backends** — LightGBM and XGBoost via CGO bindings (optional build tags), with stubs for CGO-free builds
- **Ensemble methods** — Rank-normalized blending and out-of-fold stacking
- **Cross-validation** — Era-aware KFold and WalkForward splits
- **Hyperparameter tuning** — Grid search and random search over typed parameter spaces
- **Feature transforms** — Rank normalization, Gaussianization, SVD-based neutralization, exposure computation
- **Metrics** — Pearson, Spearman, Sharpe ratio, max drawdown, feature-neutral correlation (FNC), per-era reports
- **Data loading** — CSV and Parquet with streaming support
- **Config** — Generic YAML loader with environment variable overrides and struct validation
- **Training orchestrator** — Checkpointing, early stopping, and callback hooks

## Install

```bash
go get github.com/zerfoo/metee
```

Requires Go 1.25+.

## Quick Start

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/zerfoo/metee/cv"
	"github.com/zerfoo/metee/data"
	"github.com/zerfoo/metee/metrics"
	"github.com/zerfoo/metee/registry"
	_ "github.com/zerfoo/metee/lightgbm" // register backend (requires -tags lightgbm)
)

func main() {
	ctx := context.Background()

	// Load data
	ds, err := data.LoadCSV("train.csv", data.CSVOptions{
		TargetColumn: "target",
		IDColumn:     "id",
		EraColumn:    "era",
	})
	if err != nil {
		log.Fatal(err)
	}

	// Get a model from the registry
	m, err := registry.GetBackend("lightgbm")
	if err != nil {
		log.Fatal(err)
	}

	// Cross-validate with walk-forward splits
	folds := cv.WalkForward(ds, 4, 2)
	results, err := cv.CrossValidate(ctx, m, ds, folds, metrics.Spearman)
	if err != nil {
		log.Fatal(err)
	}

	for i, r := range results {
		fmt.Printf("Fold %d: %.4f\n", i, r.Score)
	}
}
```

## Build Tags

CGO backends are optional. Without build tags, stub implementations return descriptive errors and `go build ./...` always succeeds.

| Tag | Backend | Requirement |
|-----|---------|-------------|
| `lightgbm` | LightGBM | `libLightGBM` headers and shared library |
| `xgboost` | XGBoost | `libxgboost` headers and shared library |

```bash
# CPU-only, no CGO:
go test ./...

# With LightGBM:
CGO_CFLAGS="-I/usr/local/include" CGO_LDFLAGS="-L/usr/local/lib -lLightGBM" \
  go test -tags lightgbm ./...

# With both backends:
go test -tags "lightgbm,xgboost" ./...
```

## Package Overview

| Package | Purpose |
|---------|---------|
| `model/` | `Model`, `Validator`, and `Configurable` interfaces |
| `registry/` | Thread-safe backend registry (`RegisterBackend` / `GetBackend`) |
| `lightgbm/` | LightGBM CGO bindings + stub |
| `xgboost/` | XGBoost CGO bindings + stub |
| `data/` | `Dataset` type, CSV and Parquet loaders |
| `transform/` | Rank normalization, Gaussianization, neutralization, exposure, pipeline |
| `metrics/` | Pearson, Spearman, Sharpe, max drawdown, FNC, per-era reports |
| `cv/` | KFold, WalkForward splits, `CrossValidate` |
| `tuning/` | Parameter spaces (`Discrete`, `Uniform`, `LogUniform`, `IntRange`), grid/random search |
| `trainer/` | Training orchestrator with checkpointing and callbacks |
| `ensemble/` | Rank blending and out-of-fold stacking |
| `config/` | Generic YAML loader with env overrides and validation |

## Interfaces

### model.Model

All backends implement the core model interface:

```go
type Model interface {
    Train(ctx context.Context, features [][]float64, targets []float64) error
    Predict(ctx context.Context, features [][]float64) ([]float64, error)
    Save(ctx context.Context, path string) error
    Load(ctx context.Context, path string) error
    Importance() (map[string]float64, error)
    Name() string
}
```

### model.Validator (optional)

```go
type Validator interface {
    Validate(ctx context.Context, features [][]float64, targets []float64) (map[string]float64, error)
}
```

### model.Configurable (optional)

For runtime parameter updates during hyperparameter tuning:

```go
type Configurable interface {
    SetParams(params map[string]any) error
}
```

## Dependencies

| Dependency | Purpose |
|-----------|---------|
| `gonum.org/v1/gonum` | Matrix operations (neutralization, FNC) |
| `gopkg.in/yaml.v3` | YAML config parsing |
| `github.com/parquet-go/parquet-go` | Parquet data loading |

## A Product of [Feza, Inc](https://feza.ai)
