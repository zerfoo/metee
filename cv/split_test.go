package cv

import (
	"sort"
	"testing"
)

func TestKFold(t *testing.T) {
	tests := []struct {
		name    string
		eras    []int
		k       int
		wantK   int
		wantAll bool // all eras covered
	}{
		{"5 eras 3 folds", []int{1, 1, 2, 2, 3, 3, 4, 4, 5, 5}, 3, 3, true},
		{"3 eras 3 folds", []int{1, 2, 3}, 3, 3, true},
		{"k larger than eras", []int{1, 2}, 5, 2, true},
		{"single era", []int{1, 1, 1}, 3, 1, true},
		{"empty input", nil, 3, 0, false},
		{"k zero", []int{1, 2, 3}, 0, 0, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			folds := KFold(tt.eras, tt.k)
			if len(folds) != tt.wantK {
				t.Fatalf("KFold() returned %d folds, want %d", len(folds), tt.wantK)
			}
			if !tt.wantAll {
				return
			}

			// All eras should be covered across validation folds.
			allValid := make(map[int]bool)
			for _, f := range folds {
				for _, e := range f.ValidEras {
					allValid[e] = true
				}
				// No overlap between train and valid in each fold.
				trainSet := make(map[int]bool)
				for _, e := range f.TrainEras {
					trainSet[e] = true
				}
				for _, e := range f.ValidEras {
					if trainSet[e] {
						t.Errorf("era %d appears in both train and valid", e)
					}
				}
			}

			unique := uniqueSorted(tt.eras)
			for _, e := range unique {
				if !allValid[e] {
					t.Errorf("era %d not covered by any validation fold", e)
				}
			}
		})
	}
}

func TestKFoldNonOverlapping(t *testing.T) {
	eras := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	folds := KFold(eras, 5)
	if len(folds) != 5 {
		t.Fatalf("expected 5 folds, got %d", len(folds))
	}

	// Validation sets across folds should not overlap.
	seen := make(map[int]int)
	for i, f := range folds {
		for _, e := range f.ValidEras {
			if prev, ok := seen[e]; ok {
				t.Errorf("era %d in valid of fold %d and fold %d", e, prev, i)
			}
			seen[e] = i
		}
	}
}

func TestWalkForward(t *testing.T) {
	tests := []struct {
		name      string
		eras      []int
		trainSize int
		validSize int
		step      int
		wantFolds int
	}{
		{"basic", []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 5, 2, 2, 2},
		{"step 1", []int{1, 2, 3, 4, 5}, 2, 1, 1, 3},
		{"not enough eras", []int{1, 2}, 3, 1, 1, 0},
		{"empty", nil, 3, 1, 1, 0},
		{"zero trainSize", []int{1, 2, 3}, 0, 1, 1, 0},
		{"zero validSize", []int{1, 2, 3}, 2, 0, 1, 0},
		{"zero step", []int{1, 2, 3}, 2, 1, 0, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			folds := WalkForward(tt.eras, tt.trainSize, tt.validSize, tt.step)
			if len(folds) != tt.wantFolds {
				t.Fatalf("WalkForward() returned %d folds, want %d", len(folds), tt.wantFolds)
			}

			for i, f := range folds {
				if len(f.TrainEras) != tt.trainSize {
					t.Errorf("fold %d: train size %d, want %d", i, len(f.TrainEras), tt.trainSize)
				}
				if len(f.ValidEras) != tt.validSize {
					t.Errorf("fold %d: valid size %d, want %d", i, len(f.ValidEras), tt.validSize)
				}
				// Valid eras should come after train eras.
				if len(f.TrainEras) > 0 && len(f.ValidEras) > 0 {
					maxTrain := f.TrainEras[len(f.TrainEras)-1]
					minValid := f.ValidEras[0]
					if minValid <= maxTrain {
						t.Errorf("fold %d: valid era %d not after train era %d", i, minValid, maxTrain)
					}
				}
			}
		})
	}
}

func TestWalkForwardWindows(t *testing.T) {
	eras := []int{1, 2, 3, 4, 5}
	folds := WalkForward(eras, 2, 1, 1)
	if len(folds) != 3 {
		t.Fatalf("expected 3 folds, got %d", len(folds))
	}

	expected := []Fold{
		{TrainEras: []int{1, 2}, ValidEras: []int{3}},
		{TrainEras: []int{2, 3}, ValidEras: []int{4}},
		{TrainEras: []int{3, 4}, ValidEras: []int{5}},
	}
	for i, f := range folds {
		if !intSliceEqual(f.TrainEras, expected[i].TrainEras) {
			t.Errorf("fold %d: train %v, want %v", i, f.TrainEras, expected[i].TrainEras)
		}
		if !intSliceEqual(f.ValidEras, expected[i].ValidEras) {
			t.Errorf("fold %d: valid %v, want %v", i, f.ValidEras, expected[i].ValidEras)
		}
	}
}

func TestUniqueSorted(t *testing.T) {
	got := uniqueSorted([]int{3, 1, 2, 1, 3, 2})
	want := []int{1, 2, 3}
	if !intSliceEqual(got, want) {
		t.Errorf("uniqueSorted() = %v, want %v", got, want)
	}

	if uniqueSorted(nil) != nil {
		t.Error("uniqueSorted(nil) should be nil")
	}
}

func intSliceEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	sort.Ints(a)
	sort.Ints(b)
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
