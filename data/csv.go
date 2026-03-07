package data

import (
	"encoding/csv"
	"fmt"
	"io"
	"strconv"
)

// LoadCSV reads a CSV dataset from an io.Reader.
// The CSV must have a header row. The targetCol and eraCol specify
// which columns to use as target and era respectively. The idCol
// specifies the row identifier column. All other columns are treated
// as features.
func LoadCSV(r io.Reader, idCol, targetCol, eraCol string) (*Dataset, error) {
	reader := csv.NewReader(r)

	header, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("reading header: %w", err)
	}

	colIndex := make(map[string]int, len(header))
	for i, name := range header {
		colIndex[name] = i
	}

	idIdx, hasID := colIndex[idCol]
	targetIdx, hasTarget := colIndex[targetCol]
	eraIdx, hasEra := colIndex[eraCol]

	// Determine feature columns (everything except id, target, era)
	skip := make(map[int]bool)
	if hasID {
		skip[idIdx] = true
	}
	if hasTarget {
		skip[targetIdx] = true
	}
	if hasEra {
		skip[eraIdx] = true
	}

	var featureNames []string
	var featureIndices []int
	for i, name := range header {
		if !skip[i] {
			featureNames = append(featureNames, name)
			featureIndices = append(featureIndices, i)
		}
	}

	ds := &Dataset{FeatureNames: featureNames}

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("reading row: %w", err)
		}

		if hasID {
			ds.IDs = append(ds.IDs, record[idIdx])
		}

		if hasTarget {
			v, err := strconv.ParseFloat(record[targetIdx], 64)
			if err != nil {
				return nil, fmt.Errorf("parsing target %q: %w", record[targetIdx], err)
			}
			ds.Targets = append(ds.Targets, v)
		}

		if hasEra {
			v, err := strconv.Atoi(record[eraIdx])
			if err != nil {
				return nil, fmt.Errorf("parsing era %q: %w", record[eraIdx], err)
			}
			ds.Eras = append(ds.Eras, v)
		}

		row := make([]float64, len(featureIndices))
		for j, fi := range featureIndices {
			v, err := strconv.ParseFloat(record[fi], 64)
			if err != nil {
				return nil, fmt.Errorf("parsing feature %q: %w", record[fi], err)
			}
			row[j] = v
		}
		ds.Features = append(ds.Features, row)
	}

	return ds, nil
}
