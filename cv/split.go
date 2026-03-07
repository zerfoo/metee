// Package cv provides cross-validation utilities for era-based datasets.
package cv

import "sort"

// Fold represents a train/validation split of eras.
type Fold struct {
	TrainEras []int
	ValidEras []int
}

// KFold splits eras into k non-overlapping folds. Each fold uses one group
// as validation and the remaining groups as training.
func KFold(eras []int, k int) []Fold {
	unique := uniqueSorted(eras)
	if len(unique) == 0 || k <= 0 {
		return nil
	}
	if k > len(unique) {
		k = len(unique)
	}

	folds := make([]Fold, k)
	for i := 0; i < k; i++ {
		var train, valid []int
		for j, era := range unique {
			if j%k == i {
				valid = append(valid, era)
			} else {
				train = append(train, era)
			}
		}
		folds[i] = Fold{TrainEras: train, ValidEras: valid}
	}
	return folds
}

// WalkForward generates sliding-window folds. Each fold trains on trainSize
// consecutive eras and validates on the next validSize eras, advancing by step.
func WalkForward(eras []int, trainSize, validSize, step int) []Fold {
	unique := uniqueSorted(eras)
	if len(unique) == 0 || trainSize <= 0 || validSize <= 0 || step <= 0 {
		return nil
	}

	var folds []Fold
	for i := 0; i+trainSize+validSize <= len(unique); i += step {
		fold := Fold{
			TrainEras: append([]int(nil), unique[i:i+trainSize]...),
			ValidEras: append([]int(nil), unique[i+trainSize:i+trainSize+validSize]...),
		}
		folds = append(folds, fold)
	}
	return folds
}

func uniqueSorted(eras []int) []int {
	if len(eras) == 0 {
		return nil
	}
	seen := make(map[int]struct{}, len(eras))
	for _, e := range eras {
		seen[e] = struct{}{}
	}
	result := make([]int, 0, len(seen))
	for e := range seen {
		result = append(result, e)
	}
	sort.Ints(result)
	return result
}
