package transform

import (
	"gonum.org/v1/gonum/mat"
)

// Neutralize removes linear exposure to the given features from predictions.
// pred_neutralized = pred - proportion * proj, where proj is the orthogonal
// projection of pred onto the column space of [1 | features].
// Uses SVD to handle rank-deficient feature matrices.
// proportion in [0, 1]: 0 returns original, 1 = full neutralization.
func Neutralize(predictions []float64, features [][]float64, proportion float64) []float64 {
	n := len(predictions)
	result := make([]float64, n)
	copy(result, predictions)

	if n == 0 || len(features) == 0 || proportion == 0 {
		return result
	}

	nFeatures := len(features)
	cols := nFeatures + 1

	// Build feature matrix F (n x cols) with intercept column.
	fData := make([]float64, n*cols)
	for i := 0; i < n; i++ {
		fData[i*cols] = 1.0
		for j, feat := range features {
			fData[i*cols+j+1] = feat[i]
		}
	}
	f := mat.NewDense(n, cols, fData)

	// SVD of F. Thin SVD gives U (n x cols).
	// Projection onto col(F) = U_r U_r^T pred, where U_r are columns
	// corresponding to non-negligible singular values.
	var svd mat.SVD
	if !svd.Factorize(f, mat.SVDThin) {
		return result
	}

	var u mat.Dense
	svd.UTo(&u)
	sv := svd.Values(nil)

	maxSV := 0.0
	for _, s := range sv {
		if s > maxSV {
			maxSV = s
		}
	}
	tol := maxSV * float64(max(n, cols)) * 1e-14

	for k := range sv {
		if sv[k] < tol {
			continue
		}
		var dot float64
		for i := 0; i < n; i++ {
			dot += u.At(i, k) * predictions[i]
		}
		for i := 0; i < n; i++ {
			result[i] -= proportion * u.At(i, k) * dot
		}
	}

	return result
}
