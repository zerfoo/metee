//go:build lightgbm

package lightgbm

/*
#cgo CFLAGS: -I/usr/local/include
#cgo LDFLAGS: -L/usr/local/lib -l_lightgbm
#include <stdint.h>
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

	// Free previous booster to avoid leaking C memory across repeated Train() calls
	// (e.g., during cross-validation or hyperparameter tuning).
	if b.handle != nil {
		C.LGBM_BoosterFree(b.handle)
		b.handle = nil
	}

	nRows := len(features)
	nCols := len(features[0])

	// Flatten features into row-major float64 array
	flat := make([]float64, nRows*nCols)
	for i, row := range features {
		copy(flat[i*nCols:], row)
	}

	// Create dataset
	configStr := C.CString(b.params.ToConfigString())
	defer C.free(unsafe.Pointer(configStr))

	var dataHandle C.DatasetHandle
	ret := C.LGBM_DatasetCreateFromMat(
		unsafe.Pointer(&flat[0]),
		C.C_API_DTYPE_FLOAT64,
		C.int32_t(nRows),
		C.int32_t(nCols),
		C.int(1), // is_row_major
		configStr,
		nil, // reference
		&dataHandle,
	)
	if ret != 0 {
		return fmt.Errorf("LGBM_DatasetCreateFromMat failed: %d", ret)
	}
	defer C.LGBM_DatasetFree(dataHandle)

	// Set labels (LightGBM expects float32 labels)
	labels32 := make([]float32, nRows)
	for i, t := range targets {
		labels32[i] = float32(t)
	}
	labelField := C.CString("label")
	defer C.free(unsafe.Pointer(labelField))
	ret = C.LGBM_DatasetSetField(
		dataHandle,
		labelField,
		unsafe.Pointer(&labels32[0]),
		C.int(nRows),
		C.C_API_DTYPE_FLOAT32,
	)
	if ret != 0 {
		return fmt.Errorf("LGBM_DatasetSetField(label) failed: %d", ret)
	}

	// Create booster
	boosterConfig := C.CString(b.params.ToConfigString())
	defer C.free(unsafe.Pointer(boosterConfig))
	ret = C.LGBM_BoosterCreate(
		dataHandle,
		boosterConfig,
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

	emptyStr := C.CString("")
	defer C.free(unsafe.Pointer(emptyStr))
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
		emptyStr,
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

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	ret := C.LGBM_BoosterSaveModel(
		b.handle,
		C.int(0),  // start_iteration
		C.int(-1), // num_iteration
		C.int(0),  // feature_importance_type
		cPath,
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

// SetParams updates booster hyperparameters from a generic map.
// This implements model.Configurable for use with tuning.RandomSearch.
func (b *Booster) SetParams(params map[string]any) error {
	for k, v := range params {
		switch k {
		case "num_iterations":
			b.params.NumIterations = toInt(v)
		case "num_leaves":
			b.params.NumLeaves = toInt(v)
		case "max_depth":
			b.params.MaxDepth = toInt(v)
		case "min_data_in_leaf":
			b.params.MinDataInLeaf = toInt(v)
		case "bagging_freq":
			b.params.BaggingFreq = toInt(v)
		case "num_threads":
			b.params.NumThreads = toInt(v)
		case "learning_rate":
			b.params.LearningRate = toFloat(v)
		case "feature_fraction":
			b.params.FeatureFraction = toFloat(v)
		case "bagging_fraction":
			b.params.BaggingFraction = toFloat(v)
		case "lambda_l1":
			b.params.LambdaL1 = toFloat(v)
		case "lambda_l2":
			b.params.LambdaL2 = toFloat(v)
		case "objective":
			if s, ok := v.(string); ok {
				b.params.Objective = s
			}
		case "metric":
			if s, ok := v.(string); ok {
				b.params.Metric = s
			}
		}
	}
	return nil
}

func toInt(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	default:
		return 0
	}
}

func toFloat(v any) float64 {
	switch n := v.(type) {
	case float64:
		return n
	case int:
		return float64(n)
	case int64:
		return float64(n)
	default:
		return 0
	}
}
