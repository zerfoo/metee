# Metee Design Document

## Overview

Metee ("trees" in Swahili/Kikuyu) is a Go library for tree-based machine learning models and ensemble methods. It is designed as a standalone complement to the zerfoo neural network framework, with no dependencies on zerfoo.

## Package Layout

```
metee/
├── metee.go                    # Package doc
├── model/
│   ├── model.go                # Model interface
│   ├── validator.go            # Validator interface (optional)
│   └── configurable.go         # Configurable interface (optional)
├── registry/
│   └── registry.go             # Thread-safe backend registry
├── config/
│   ├── config.go               # Generic YAML loader with env overrides + validation
│   └── training.go             # TrainingConfig, EnsembleConfig structs
├── data/
│   ├── dataset.go              # Dataset type with Split/EraGroups
│   ├── csv.go                  # CSV loader
│   └── parquet.go              # Parquet loader
├── lightgbm/
│   ├── params.go               # Hyperparameter config (pure Go)
│   ├── lightgbm.go             # CGO bindings (build tag: lightgbm)
│   └── lightgbm_stub.go        # Stub when LightGBM unavailable
├── xgboost/
│   ├── params.go               # Hyperparameter config (pure Go)
│   ├── xgboost.go              # CGO bindings (build tag: xgboost)
│   └── xgboost_stub.go         # Stub when XGBoost unavailable
├── transform/
│   ├── rank.go                 # RankNormalize, Gaussianize
│   ├── neutralize.go           # Neutralize (SVD-based feature neutralization)
│   ├── exposure.go             # ComputeExposures, MaxExposure
│   └── pipeline.go             # Transform interface, Pipeline, built-in transforms
├── metrics/
│   └── metrics.go              # Pearson, Spearman, Sharpe, MaxDrawdown, FNC, PerEraReport
├── cv/
│   ├── split.go                # Fold, KFold, WalkForward
│   └── cv.go                   # CrossValidate
├── tuning/
│   └── space.go                # ParamRange, ParamSpace, Grid, Sample
├── trainer/
│   └── trainer.go              # Trainer orchestrator with checkpointing
├── ensemble/
│   ├── blend.go                # Blender (weighted averaging)
│   ├── rank_blend.go           # RankBlend (rank-normalize then blend)
│   └── stacking.go             # Stacker (out-of-fold stacking ensemble)
└── docs/
    ├── design.md               # This document
    └── plan.md                 # Task tracker
```

## Interfaces

### model.Model

All tree-based models implement the `Model` interface:

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

### model.Validator

Optional interface for models that support validation:

```go
type Validator interface {
    Validate(ctx context.Context, features [][]float64, targets []float64) (map[string]float64, error)
}
```

### model.Configurable

Optional interface for runtime parameter updates (tuning integration):

```go
type Configurable interface {
    SetParams(params map[string]any) error
}
```

### transform.Transform

Post-prediction transformation interface:

```go
type Transform interface {
    Apply(ctx context.Context, preds []float64, features [][]float64) ([]float64, error)
}
```

Built-in implementations: `RankNormalizeTransform`, `GaussianizeTransform`, `NeutralizeTransform`.

### tuning.ParamRange

Hyperparameter range interface:

```go
type ParamRange interface {
    Values() []any
    Sample(rng *rand.Rand) any
}
```

Constructors: `Discrete`, `Uniform`, `LogUniform`, `IntRange`.

## Registry Pattern

The `registry` package provides a global, thread-safe registry for model backends. Backends register via `init()` in their respective packages (guarded by build tags):

```go
// In lightgbm/lightgbm.go (//go:build lightgbm)
func init() {
    registry.RegisterBackend("lightgbm", func() model.Model { return NewBooster(...) })
}
```

Consumers retrieve backends without import-time coupling:

```go
m, err := registry.GetBackend("lightgbm")
```

`ErrBackendNotFound` is returned for unregistered backends. `ListBackends()` returns all registered names.

## Config System

The `config` package provides generic YAML loading:

- `Load[T](path) (T, error)` -- unmarshal YAML into any struct type
- `LoadWithEnv[T](path, prefix) (T, error)` -- YAML + environment variable overrides (`PREFIX_FIELDNAME`)
- `Validate(v) []string` -- check `validate:"required"` struct tags

Supported env override types: `string`, `int`, `int64`, `float64`, `bool`.

Pre-defined config structs in `config/training.go`:
- `TrainingConfig` -- single model training parameters
- `EnsembleConfig` -- multi-model ensemble parameters

## Build Tag Strategy

CGO-dependent backends use build tags to enable optional compilation:

| Tag | Package | Effect |
|-----|---------|--------|
| `lightgbm` | `lightgbm/` | Enables CGO bindings to libLightGBM |
| `xgboost` | `xgboost/` | Enables CGO bindings to libxgboost |

Without tags, stub files (`*_stub.go`) provide the same types but return descriptive errors from all methods. This ensures `go build ./...` always succeeds.

```bash
# Stub mode (no CGO):
go test ./...

# With LightGBM:
CGO_CFLAGS="-I/usr/local/include" CGO_LDFLAGS="-L/usr/local/lib -lLightGBM" \
go test -tags lightgbm ./...

# With both:
go test -tags "lightgbm,xgboost" ./...
```

## Dependency Diagram

```
model ◄──────── registry
  ▲                ▲
  │                │
  ├── lightgbm ────┤
  ├── xgboost ─────┘
  │
  ├── data
  │     ▲
  │     │
  ├── cv ◄──────── ensemble/stacking
  │     ▲
  │     │
  │   tuning
  │
  ├── trainer ──── config
  │
  ├── transform ── metrics
  │     ▲
  │     │
  └── ensemble/rank_blend
```

External dependencies:
- `gopkg.in/yaml.v3` -- YAML parsing (config)
- `gonum.org/v1/gonum` -- matrix ops (transform/neutralize, metrics/FNC)
- `github.com/parquet-go/parquet-go` -- Parquet I/O (data)

## Error Handling Conventions

- Sentinel errors for expected cases (e.g., `registry.ErrBackendNotFound`)
- Wrap with context: `fmt.Errorf("pkg: context: %w", err)`
- Error prefixes by package: `registry:`, `config:`, `ensemble:`, `trainer:`, `data:`
- Stub methods return package-level sentinel errors (e.g., `errNoBuild`)

## Testing Standards

- Table-driven tests with `t.Helper()` in helpers
- All tests run with `-race`
- Target >= 95% coverage per package
- Test fixtures in `testdata/` directories
- Mock models implement `model.Model` interface directly in `_test.go` files
