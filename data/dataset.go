// Package data provides dataset types and loading utilities for tree-based models.
package data

// Dataset holds tabular data for training and evaluation.
type Dataset struct {
	IDs          []string    // Row identifiers
	Features     [][]float64 // [nSamples][nFeatures]
	Targets      []float64   // [nSamples]
	Eras         []int       // [nSamples] era/group assignment
	FeatureNames []string    // Feature column names
}

// NSamples returns the number of samples in the dataset.
func (d *Dataset) NSamples() int {
	return len(d.Features)
}

// NFeatures returns the number of features per sample.
func (d *Dataset) NFeatures() int {
	if len(d.Features) == 0 {
		return 0
	}
	return len(d.Features[0])
}

// Split partitions the dataset by era into train and validation sets.
func (d *Dataset) Split(trainEras, validEras []int) (*Dataset, *Dataset) {
	trainSet := make(map[int]bool, len(trainEras))
	for _, e := range trainEras {
		trainSet[e] = true
	}
	validSet := make(map[int]bool, len(validEras))
	for _, e := range validEras {
		validSet[e] = true
	}

	train := &Dataset{FeatureNames: d.FeatureNames}
	valid := &Dataset{FeatureNames: d.FeatureNames}

	for i, era := range d.Eras {
		if trainSet[era] {
			appendSample(train, d, i)
		} else if validSet[era] {
			appendSample(valid, d, i)
		}
	}

	return train, valid
}

// EraGroups returns a map from era to a Dataset containing only that era's samples.
func (d *Dataset) EraGroups() map[int]*Dataset {
	groups := make(map[int]*Dataset)
	for i, era := range d.Eras {
		ds, ok := groups[era]
		if !ok {
			ds = &Dataset{FeatureNames: d.FeatureNames}
			groups[era] = ds
		}
		appendSample(ds, d, i)
	}
	return groups
}

func appendSample(dst *Dataset, src *Dataset, i int) {
	if len(src.IDs) > i {
		dst.IDs = append(dst.IDs, src.IDs[i])
	}
	dst.Features = append(dst.Features, src.Features[i])
	if len(src.Targets) > i {
		dst.Targets = append(dst.Targets, src.Targets[i])
	}
	if len(src.Eras) > i {
		dst.Eras = append(dst.Eras, src.Eras[i])
	}
}
