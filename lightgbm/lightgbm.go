//go:build lightgbm

package lightgbm

/*
#cgo CFLAGS: -I/usr/local/include
#cgo LDFLAGS: -L/usr/local/lib -l_lightgbm
#include <LightGBM/c_api.h>
#include <stdlib.h>
*/
import "C"
import (
	"context"
	"errors"
	"fmt"
	"os"
	"unsafe"

	"github.com/zerfoo/metee/model"
	"github.com/zerfoo/metee/registry"
)

func init() {
	registry.RegisterBackend("lightgbm", func() model.Model { return NewBooster("", DefaultParams(), nil) })
}

// Booster wraps a LightGBM booster handle.
type Booster struct {
	handle       C.BoosterHandle
	params       Params
	name         string
	featureNames []string
}

// NewBooster creates a new Booster with the given params.
func NewBooster(name string, params Params, featureNames []string) *Booster {
	return &Booster{params: params, name: name, featureNames: featureNames}
}

func (b *Booster) Train(ctx context.Context, features [][]float64, targets []float64) error {
	if len(features) == 0 {
		return errors.New("no training data")
	}

	nRows := len(features)
	nCols := len(features[0])

	// Flatten features into row-major float64 array
	flat := make([]float64, nRows*nCols)
	for i, row := range features {
		copy(flat[i*nCols:], row)
	}

	// Create dataset
	var dataHandle C.DatasetHandle
	ret := C.LGBM_DatasetCreateFromMat(
		unsafe.Pointer(&flat[0]),
		C.C_API_DTYPE_FLOAT64,
		C.int32_t(nRows),
		C.int32_t(nCols),
		C.int(1), // is_row_major
		C.CString(b.params.ToConfigString()),
		nil, // reference
		&dataHandle,
	)
	if ret != 0 {
		return fmt.Errorf("LGBM_DatasetCreateFromMat failed: %d", ret)
	}
	defer C.LGBM_DatasetFree(dataHandle)

	// Set labels
	ret = C.LGBM_DatasetSetField(
		dataHandle,
		C.CString("label"),
		unsafe.Pointer(&targets[0]),
		C.int(nRows),
		C.C_API_DTYPE_FLOAT64,
	)
	if ret != 0 {
		return fmt.Errorf("LGBM_DatasetSetField(label) failed: %d", ret)
	}

	// Create booster
	ret = C.LGBM_BoosterCreate(
		dataHandle,
		C.CString(b.params.ToConfigString()),
		&b.handle,
	)
	if ret != 0 {
		return fmt.Errorf("LGBM_BoosterCreate failed: %d", ret)
	}

	// Train
	var isFinished C.int
	for i := 0; i < b.params.NumIterations; i++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		ret = C.LGBM_BoosterUpdateOneIter(b.handle, &isFinished)
		if ret != 0 {
			return fmt.Errorf("LGBM_BoosterUpdateOneIter failed at iter %d: %d", i, ret)
		}
		if isFinished != 0 {
			break
		}
	}

	return nil
}

func (b *Booster) Predict(_ context.Context, features [][]float64) ([]float64, error) {
	if b.handle == nil {
		return nil, errors.New("booster not trained or loaded")
	}
	if len(features) == 0 {
		return nil, nil
	}

	nRows := len(features)
	nCols := len(features[0])

	flat := make([]float64, nRows*nCols)
	for i, row := range features {
		copy(flat[i*nCols:], row)
	}

	var outLen C.int64_t
	result := make([]float64, nRows)

	ret := C.LGBM_BoosterPredictForMat(
		b.handle,
		unsafe.Pointer(&flat[0]),
		C.C_API_DTYPE_FLOAT64,
		C.int32_t(nRows),
		C.int32_t(nCols),
		C.int(1), // is_row_major
		C.int(C.C_API_PREDICT_NORMAL),
		C.int(0),  // start_iteration
		C.int(-1), // num_iteration (-1 = all)
		C.CString(""),
		&outLen,
		(*C.double)(unsafe.Pointer(&result[0])),
	)
	if ret != 0 {
		return nil, fmt.Errorf("LGBM_BoosterPredictForMat failed: %d", ret)
	}

	return result[:outLen], nil
}

func (b *Booster) Save(_ context.Context, path string) error {
	if b.handle == nil {
		return errors.New("booster not trained or loaded")
	}

	ret := C.LGBM_BoosterSaveModel(
		b.handle,
		C.int(0),  // start_iteration
		C.int(-1), // num_iteration
		C.int(0),  // feature_importance_type
		C.CString(path),
	)
	if ret != 0 {
		return fmt.Errorf("LGBM_BoosterSaveModel failed: %d", ret)
	}
	return nil
}

func (b *Booster) Load(_ context.Context, path string) error {
	modelBytes, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("reading model file: %w", err)
	}

	var numIterations C.int
	cModel := C.CString(string(modelBytes))
	defer C.free(unsafe.Pointer(cModel))

	ret := C.LGBM_BoosterLoadModelFromString(
		cModel,
		&numIterations,
		&b.handle,
	)
	if ret != 0 {
		return fmt.Errorf("LGBM_BoosterLoadModelFromString failed: %d", ret)
	}
	return nil
}

func (b *Booster) Importance() (map[string]float64, error) {
	if b.handle == nil {
		return nil, errors.New("booster not trained or loaded")
	}

	var nFeatures C.int
	C.LGBM_BoosterGetNumFeature(b.handle, &nFeatures)

	importance := make([]float64, int(nFeatures))
	ret := C.LGBM_BoosterFeatureImportance(
		b.handle,
		C.int(0), // num_iteration
		C.int(0), // importance_type (split)
		(*C.double)(unsafe.Pointer(&importance[0])),
	)
	if ret != 0 {
		return nil, fmt.Errorf("LGBM_BoosterFeatureImportance failed: %d", ret)
	}

	result := make(map[string]float64, int(nFeatures))
	for i, v := range importance {
		name := fmt.Sprintf("feature_%d", i)
		if i < len(b.featureNames) {
			name = b.featureNames[i]
		}
		result[name] = v
	}
	return result, nil
}

func (b *Booster) Name() string { return b.name }
