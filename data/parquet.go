package data

import (
	"context"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/parquet-go/parquet-go"
)

// LoadOption configures how LoadParquet loads data.
type LoadOption func(*loadConfig)

type loadConfig struct {
	featurePrefix string
	targetColumn  string
	eraColumn     string
	idColumn      string
}

func defaultLoadConfig() *loadConfig {
	return &loadConfig{
		featurePrefix: "feature_",
		targetColumn:  "target",
		eraColumn:     "era",
		idColumn:      "id",
	}
}

// WithFeaturePrefix sets the prefix used to identify feature columns.
func WithFeaturePrefix(prefix string) LoadOption {
	return func(c *loadConfig) { c.featurePrefix = prefix }
}

// WithTargetColumn sets the name of the target column.
func WithTargetColumn(name string) LoadOption {
	return func(c *loadConfig) { c.targetColumn = name }
}

// WithEraColumn sets the name of the era column.
func WithEraColumn(name string) LoadOption {
	return func(c *loadConfig) { c.eraColumn = name }
}

// WithIDColumn sets the name of the ID column.
func WithIDColumn(name string) LoadOption {
	return func(c *loadConfig) { c.idColumn = name }
}

// LoadParquet reads a Parquet file into a Dataset, streaming rows to limit
// memory usage. Feature columns are identified by the configured prefix.
func LoadParquet(ctx context.Context, path string, opts ...LoadOption) (*Dataset, error) {
	cfg := defaultLoadConfig()
	for _, o := range opts {
		o(cfg)
	}

	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("data: parquet: %w", err)
	}
	defer f.Close()

	stat, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("data: parquet: %w", err)
	}

	pf, err := parquet.OpenFile(f, stat.Size())
	if err != nil {
		return nil, fmt.Errorf("data: parquet: %w", err)
	}

	schema := pf.Schema()
	fields := schema.Fields()

	colByName := make(map[string]int, len(fields))
	for i, field := range fields {
		colByName[field.Name()] = i
	}

	var featureNames []string
	var featureIndices []int
	for i, field := range fields {
		name := field.Name()
		if strings.HasPrefix(name, cfg.featurePrefix) {
			featureNames = append(featureNames, name)
			featureIndices = append(featureIndices, i)
		}
	}

	targetIdx, hasTarget := colByName[cfg.targetColumn]
	eraIdx, hasEra := colByName[cfg.eraColumn]
	idIdx, hasID := colByName[cfg.idColumn]

	ds := &Dataset{FeatureNames: featureNames}

	for _, rg := range pf.RowGroups() {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("data: parquet: %w", err)
		}

		if err := readRowGroup(ctx, rg, ds, idIdx, hasID, targetIdx, hasTarget, eraIdx, hasEra, featureIndices, featureNames); err != nil {
			return nil, err
		}
	}

	return ds, nil
}

func readRowGroup(ctx context.Context, rg parquet.RowGroup, ds *Dataset, idIdx int, hasID bool, targetIdx int, hasTarget bool, eraIdx int, hasEra bool, featureIndices []int, featureNames []string) error {
	rows := rg.Rows()
	defer rows.Close()

	rowBuf := make([]parquet.Row, 128)
	for {
		if err := ctx.Err(); err != nil {
			return fmt.Errorf("data: parquet: %w", err)
		}

		n, err := rows.ReadRows(rowBuf)
		for i := 0; i < n; i++ {
			if procErr := processRow(rowBuf[i], ds, idIdx, hasID, targetIdx, hasTarget, eraIdx, hasEra, featureIndices, featureNames); procErr != nil {
				return procErr
			}
		}

		if err != nil {
			if err == io.EOF {
				return nil
			}
			return fmt.Errorf("data: parquet: %w", err)
		}
	}
}

func processRow(row parquet.Row, ds *Dataset, idIdx int, hasID bool, targetIdx int, hasTarget bool, eraIdx int, hasEra bool, featureIndices []int, featureNames []string) error {
	if hasID {
		ds.IDs = append(ds.IDs, valueToString(row[idIdx]))
	}

	if hasTarget {
		v, err := valueToFloat64(row[targetIdx])
		if err != nil {
			return fmt.Errorf("data: parquet: target: %w", err)
		}
		ds.Targets = append(ds.Targets, v)
	}

	if hasEra {
		v, err := valueToInt(row[eraIdx])
		if err != nil {
			return fmt.Errorf("data: parquet: era: %w", err)
		}
		ds.Eras = append(ds.Eras, v)
	}

	featureRow := make([]float64, len(featureIndices))
	for j, fi := range featureIndices {
		v, err := valueToFloat64(row[fi])
		if err != nil {
			return fmt.Errorf("data: parquet: feature %q: %w", featureNames[j], err)
		}
		featureRow[j] = v
	}
	ds.Features = append(ds.Features, featureRow)
	return nil
}

func valueToString(v parquet.Value) string {
	if v.IsNull() {
		return ""
	}
	return v.String()
}

func valueToFloat64(v parquet.Value) (float64, error) {
	if v.IsNull() {
		return 0, nil
	}
	switch v.Kind() {
	case parquet.Double:
		return v.Double(), nil
	case parquet.Float:
		return float64(v.Float()), nil
	case parquet.Int32:
		return float64(v.Int32()), nil
	case parquet.Int64:
		return float64(v.Int64()), nil
	default:
		return strconv.ParseFloat(v.String(), 64)
	}
}

func valueToInt(v parquet.Value) (int, error) {
	if v.IsNull() {
		return 0, nil
	}
	switch v.Kind() {
	case parquet.Int32:
		return int(v.Int32()), nil
	case parquet.Int64:
		return int(v.Int64()), nil
	default:
		return strconv.Atoi(strings.TrimSpace(v.String()))
	}
}
