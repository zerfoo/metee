package data

import "testing"

func makeTestDataset() *Dataset {
	return &Dataset{
		IDs:          []string{"a", "b", "c", "d", "e", "f"},
		Features:     [][]float64{{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}},
		Targets:      []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
		Eras:         []int{1, 1, 2, 2, 3, 3},
		FeatureNames: []string{"f1", "f2"},
	}
}

func TestDatasetDimensions(t *testing.T) {
	ds := makeTestDataset()
	if ds.NSamples() != 6 {
		t.Errorf("NSamples() = %d, want 6", ds.NSamples())
	}
	if ds.NFeatures() != 2 {
		t.Errorf("NFeatures() = %d, want 2", ds.NFeatures())
	}
}

func TestSplit(t *testing.T) {
	ds := makeTestDataset()
	train, valid := ds.Split([]int{1, 2}, []int{3})

	if train.NSamples() != 4 {
		t.Errorf("train samples = %d, want 4", train.NSamples())
	}
	if valid.NSamples() != 2 {
		t.Errorf("valid samples = %d, want 2", valid.NSamples())
	}
	if len(train.FeatureNames) != 2 {
		t.Errorf("train feature names = %d, want 2", len(train.FeatureNames))
	}
}

func TestEraGroups(t *testing.T) {
	ds := makeTestDataset()
	groups := ds.EraGroups()

	if len(groups) != 3 {
		t.Errorf("EraGroups() returned %d groups, want 3", len(groups))
	}
	for era, g := range groups {
		if g.NSamples() != 2 {
			t.Errorf("era %d has %d samples, want 2", era, g.NSamples())
		}
	}
}

func TestEmptyDataset(t *testing.T) {
	ds := &Dataset{}
	if ds.NSamples() != 0 {
		t.Errorf("NSamples() = %d, want 0", ds.NSamples())
	}
	if ds.NFeatures() != 0 {
		t.Errorf("NFeatures() = %d, want 0", ds.NFeatures())
	}
}
