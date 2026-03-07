package ensemble

import (
	"context"
	"errors"
	"fmt"

	"github.com/zerfoo/metee/cv"
	"github.com/zerfoo/metee/data"
	"github.com/zerfoo/metee/model"
)

// Stacker implements a stacking ensemble. Base models generate out-of-fold
// predictions that become the feature matrix for a meta-model.
type Stacker struct {
	BaseModels []model.Model
	MetaModel  model.Model
	Folds      []cv.Fold
}

// Fit trains all base models using out-of-fold predictions, then trains the
// meta-model on the stacked prediction matrix.
func (s *Stacker) Fit(ctx context.Context, ds *data.Dataset) error {
	if len(s.BaseModels) == 0 {
		return errors.New("ensemble: stacker: no base models")
	}
	if s.MetaModel == nil {
		return errors.New("ensemble: stacker: no meta model")
	}
	if len(s.Folds) == 0 {
		return errors.New("ensemble: stacker: no folds")
	}

	nSamples := ds.NSamples()
	nBase := len(s.BaseModels)

	// Build OOF prediction matrix [nSamples x nBase].
	oof := make([][]float64, nSamples)
	for i := range oof {
		oof[i] = make([]float64, nBase)
	}

	// Build a sample index lookup: for each sample, which row index is it?
	// We need to map fold valid eras back to dataset row indices.
	eraToIndices := make(map[int][]int)
	for i, era := range ds.Eras {
		eraToIndices[era] = append(eraToIndices[era], i)
	}

	for baseIdx, baseModel := range s.BaseModels {
		select {
		case <-ctx.Done():
			return fmt.Errorf("ensemble: stacker: %w", ctx.Err())
		default:
		}

		for _, fold := range s.Folds {
			select {
			case <-ctx.Done():
				return fmt.Errorf("ensemble: stacker: %w", ctx.Err())
			default:
			}

			train, valid := ds.Split(fold.TrainEras, fold.ValidEras)

			if err := baseModel.Train(ctx, train.Features, train.Targets); err != nil {
				return fmt.Errorf("ensemble: stacker: train base %q: %w", baseModel.Name(), err)
			}

			preds, err := baseModel.Predict(ctx, valid.Features)
			if err != nil {
				return fmt.Errorf("ensemble: stacker: predict base %q: %w", baseModel.Name(), err)
			}

			// Map predictions back to original dataset indices.
			validIdx := 0
			for _, era := range fold.ValidEras {
				for _, origIdx := range eraToIndices[era] {
					if validIdx < len(preds) {
						oof[origIdx][baseIdx] = preds[validIdx]
						validIdx++
					}
				}
			}
		}

		// Retrain on full dataset so base model is ready for prediction.
		if err := baseModel.Train(ctx, ds.Features, ds.Targets); err != nil {
			return fmt.Errorf("ensemble: stacker: retrain base %q: %w", baseModel.Name(), err)
		}
	}

	// Train meta-model on OOF predictions.
	if err := s.MetaModel.Train(ctx, oof, ds.Targets); err != nil {
		return fmt.Errorf("ensemble: stacker: train meta: %w", err)
	}

	return nil
}

// Predict generates predictions by feeding base model outputs to the meta-model.
func (s *Stacker) Predict(ctx context.Context, features [][]float64) ([]float64, error) {
	if len(s.BaseModels) == 0 {
		return nil, errors.New("ensemble: stacker: no base models")
	}
	if s.MetaModel == nil {
		return nil, errors.New("ensemble: stacker: no meta model")
	}

	nSamples := len(features)
	nBase := len(s.BaseModels)

	// Get predictions from each base model.
	stacked := make([][]float64, nSamples)
	for i := range stacked {
		stacked[i] = make([]float64, nBase)
	}

	for baseIdx, baseModel := range s.BaseModels {
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("ensemble: stacker: %w", ctx.Err())
		default:
		}

		preds, err := baseModel.Predict(ctx, features)
		if err != nil {
			return nil, fmt.Errorf("ensemble: stacker: predict base %q: %w", baseModel.Name(), err)
		}

		for i, p := range preds {
			stacked[i][baseIdx] = p
		}
	}

	// Feed stacked predictions to meta-model.
	result, err := s.MetaModel.Predict(ctx, stacked)
	if err != nil {
		return nil, fmt.Errorf("ensemble: stacker: predict meta: %w", err)
	}

	return result, nil
}
