//go:build xgboost

package xgboost

/*
#cgo CFLAGS: -I/usr/include
#cgo LDFLAGS: -lxgboost
#include <xgboost/c_api.h>
#include <stdlib.h>
*/
import "C"
import (
	"context"
	"errors"
	"fmt"
	"unsafe"

	"github.com/zerfoo/metee/model"
	"github.com/zerfoo/metee/registry"
)

func init() {
	registry.RegisterBackend("xgboost", func() model.Model { return NewBooster("", DefaultParams(), nil) })
}

// Booster wraps an XGBoost booster handle.
type Booster struct {
	handle       C.BoosterHandle
	dmat         C.DMatrixHandle
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
		return errors.New("xgboost: no training data")
	}

	nRows := len(features)
	nCols := len(features[0])

	// XGBoost C API uses float32
	flat := make([]float32, nRows*nCols)
	for i, row := range features {
		for j, v := range row {
			flat[i*nCols+j] = float32(v)
		}
	}

	// Create DMatrix
	var dmat C.DMatrixHandle
	ret := C.XGDMatrixCreateFromMat(
		(*C.float)(unsafe.Pointer(&flat[0])),
		C.bst_ulong(nRows),
		C.bst_ulong(nCols),
		C.float(-1), // missing value sentinel
		&dmat,
	)
	if ret != 0 {
		return fmt.Errorf("xgboost: XGDMatrixCreateFromMat failed: %d", ret)
	}
	defer C.XGDMatrixFree(dmat)

	// Set labels (float32)
	labels := make([]float32, len(targets))
	for i, v := range targets {
		labels[i] = float32(v)
	}
	ret = C.XGDMatrixSetFloatInfo(
		dmat,
		C.CString("label"),
		(*C.float)(unsafe.Pointer(&labels[0])),
		C.bst_ulong(len(labels)),
	)
	if ret != 0 {
		return fmt.Errorf("xgboost: XGDMatrixSetFloatInfo(label) failed: %d", ret)
	}

	// Create booster
	dmats := []C.DMatrixHandle{dmat}
	ret = C.XGBoosterCreate(
		&dmats[0],
		C.bst_ulong(1),
		&b.handle,
	)
	if ret != 0 {
		return fmt.Errorf("xgboost: XGBoosterCreate failed: %d", ret)
	}

	// Set params
	for k, v := range b.params.ToConfigMap() {
		ck := C.CString(k)
		cv := C.CString(v)
		C.XGBoosterSetParam(b.handle, ck, cv)
		C.free(unsafe.Pointer(ck))
		C.free(unsafe.Pointer(cv))
	}

	// Train iteratively
	for i := 0; i < b.params.NumRounds; i++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		ret = C.XGBoosterUpdateOneIter(b.handle, C.int(i), dmat)
		if ret != 0 {
			return fmt.Errorf("xgboost: XGBoosterUpdateOneIter failed at iter %d: %d", i, ret)
		}
	}

	return nil
}

func (b *Booster) Predict(_ context.Context, features [][]float64) ([]float64, error) {
	if b.handle == nil {
		return nil, errors.New("xgboost: booster not trained or loaded")
	}
	if len(features) == 0 {
		return nil, nil
	}

	nRows := len(features)
	nCols := len(features[0])

	flat := make([]float32, nRows*nCols)
	for i, row := range features {
		for j, v := range row {
			flat[i*nCols+j] = float32(v)
		}
	}

	// Create DMatrix for prediction
	var dmat C.DMatrixHandle
	ret := C.XGDMatrixCreateFromMat(
		(*C.float)(unsafe.Pointer(&flat[0])),
		C.bst_ulong(nRows),
		C.bst_ulong(nCols),
		C.float(-1),
		&dmat,
	)
	if ret != 0 {
		return nil, fmt.Errorf("xgboost: XGDMatrixCreateFromMat failed: %d", ret)
	}
	defer C.XGDMatrixFree(dmat)

	var outLen C.bst_ulong
	var outResult *C.float

	ret = C.XGBoosterPredict(
		b.handle,
		dmat,
		C.int(0), // option_mask: normal prediction
		C.uint(0), // ntree_limit: 0 = all trees
		C.int(0), // training: false
		&outLen,
		&outResult,
	)
	if ret != 0 {
		return nil, fmt.Errorf("xgboost: XGBoosterPredict failed: %d", ret)
	}

	// Convert float32 results to float64
	result := make([]float64, int(outLen))
	cSlice := unsafe.Slice(outResult, int(outLen))
	for i, v := range cSlice {
		result[i] = float64(v)
	}

	return result, nil
}

func (b *Booster) Save(_ context.Context, path string) error {
	if b.handle == nil {
		return errors.New("xgboost: booster not trained or loaded")
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	ret := C.XGBoosterSaveModel(b.handle, cPath)
	if ret != 0 {
		return fmt.Errorf("xgboost: XGBoosterSaveModel failed: %d", ret)
	}
	return nil
}

func (b *Booster) Load(_ context.Context, path string) error {
	if b.handle == nil {
		// Create an empty booster first
		ret := C.XGBoosterCreate(nil, C.bst_ulong(0), &b.handle)
		if ret != 0 {
			return fmt.Errorf("xgboost: XGBoosterCreate failed: %d", ret)
		}
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	ret := C.XGBoosterLoadModel(b.handle, cPath)
	if ret != 0 {
		return fmt.Errorf("xgboost: XGBoosterLoadModel failed: %d", ret)
	}
	return nil
}

func (b *Booster) Importance() (map[string]float64, error) {
	if b.handle == nil {
		return nil, errors.New("xgboost: booster not trained or loaded")
	}

	// Use FeatureScore API
	cConfig := C.CString(`{"importance_type": "weight"}`)
	defer C.free(unsafe.Pointer(cConfig))

	var outNFeatures C.bst_ulong
	var outFeatures **C.char
	var outDim C.bst_ulong
	var outShape *C.bst_ulong
	var outScores *C.float

	ret := C.XGBoosterFeatureScore(
		b.handle,
		cConfig,
		&outNFeatures,
		&outFeatures,
		&outDim,
		&outShape,
		&outScores,
	)
	if ret != 0 {
		return nil, fmt.Errorf("xgboost: XGBoosterFeatureScore failed: %d", ret)
	}

	n := int(outNFeatures)
	result := make(map[string]float64, n)

	features := unsafe.Slice(outFeatures, n)
	scores := unsafe.Slice(outScores, n)

	for i := 0; i < n; i++ {
		fname := C.GoString(features[i])
		// Map XGBoost default feature names (f0, f1, ...) to our names
		if i < len(b.featureNames) {
			fname = b.featureNames[i]
		}
		result[fname] = float64(scores[i])
	}

	return result, nil
}

func (b *Booster) Name() string { return b.name }
