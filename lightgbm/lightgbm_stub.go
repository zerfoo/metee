//go:build !lightgbm

package lightgbm

import (
	"context"
	"errors"
)

var errNoBuild = errors.New("metee: lightgbm support not compiled; rebuild with -tags lightgbm")

// Booster is a LightGBM model. This is a stub that returns errors when
// LightGBM is not available at build time.
type Booster struct {
	params       Params
	name         string
	featureNames []string
}

// NewBooster creates a new Booster with the given params.
func NewBooster(name string, params Params, featureNames []string) *Booster {
	return &Booster{params: params, name: name, featureNames: featureNames}
}

func (b *Booster) Train(_ context.Context, _ [][]float64, _ []float64) error {
	return errNoBuild
}

func (b *Booster) Predict(_ context.Context, _ [][]float64) ([]float64, error) {
	return nil, errNoBuild
}

func (b *Booster) Save(_ context.Context, _ string) error { return errNoBuild }
func (b *Booster) Load(_ context.Context, _ string) error { return errNoBuild }

func (b *Booster) Importance() (map[string]float64, error) { return nil, errNoBuild }
func (b *Booster) Name() string                            { return b.name }
