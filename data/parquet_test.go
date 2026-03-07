package data

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/parquet-go/parquet-go"
)

type testRow struct {
	ID       string  `parquet:"id"`
	Era      int32   `parquet:"era"`
	Feature1 float64 `parquet:"feature_a"`
	Feature2 float64 `parquet:"feature_b"`
	Target   float64 `parquet:"target"`
}

type testRowCustomPrefix struct {
	ID     string  `parquet:"id"`
	Era    int32   `parquet:"era"`
	ColA   float64 `parquet:"col_x"`
	ColB   float64 `parquet:"col_y"`
	Target float64 `parquet:"target"`
}

type testRowNoEraNoID struct {
	Feature1 float64 `parquet:"feature_a"`
	Feature2 float64 `parquet:"feature_b"`
	Target   float64 `parquet:"target"`
}

func writeTestParquet[T any](t *testing.T, dir string, name string, rows []T) string {
	t.Helper()
	path := filepath.Join(dir, name)
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	w := parquet.NewGenericWriter[T](f)
	_, err = w.Write(rows)
	if err != nil {
		f.Close()
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		f.Close()
		t.Fatal(err)
	}
	f.Close()
	return path
}

func TestLoadParquet_AllOptions(t *testing.T) {
	dir := t.TempDir()
	rows := []testRow{
		{ID: "a", Era: 1, Feature1: 0.1, Feature2: 0.2, Target: 0.5},
		{ID: "b", Era: 1, Feature1: 0.3, Feature2: 0.4, Target: 0.6},
		{ID: "c", Era: 2, Feature1: 0.5, Feature2: 0.6, Target: 0.7},
	}
	path := writeTestParquet(t, dir, "all.parquet", rows)

	ds, err := LoadParquet(context.Background(), path,
		WithFeaturePrefix("feature_"),
		WithTargetColumn("target"),
		WithEraColumn("era"),
		WithIDColumn("id"),
	)
	if err != nil {
		t.Fatal(err)
	}

	if ds.NSamples() != 3 {
		t.Fatalf("expected 3 samples, got %d", ds.NSamples())
	}
	if ds.NFeatures() != 2 {
		t.Fatalf("expected 2 features, got %d", ds.NFeatures())
	}
	if len(ds.IDs) != 3 {
		t.Fatalf("expected 3 IDs, got %d", len(ds.IDs))
	}
	if len(ds.Eras) != 3 {
		t.Fatalf("expected 3 eras, got %d", len(ds.Eras))
	}
	if len(ds.Targets) != 3 {
		t.Fatalf("expected 3 targets, got %d", len(ds.Targets))
	}

	// Check values.
	if ds.IDs[0] != "a" {
		t.Errorf("expected ID 'a', got %q", ds.IDs[0])
	}
	if ds.Eras[2] != 2 {
		t.Errorf("expected era 2, got %d", ds.Eras[2])
	}
	if ds.Targets[0] != 0.5 {
		t.Errorf("expected target 0.5, got %f", ds.Targets[0])
	}
	if ds.Features[1][0] != 0.3 {
		t.Errorf("expected feature 0.3, got %f", ds.Features[1][0])
	}
}

func TestLoadParquet_Defaults(t *testing.T) {
	dir := t.TempDir()
	rows := []testRow{
		{ID: "x", Era: 5, Feature1: 1.0, Feature2: 2.0, Target: 0.9},
	}
	path := writeTestParquet(t, dir, "defaults.parquet", rows)

	ds, err := LoadParquet(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}

	if ds.NSamples() != 1 {
		t.Fatalf("expected 1 sample, got %d", ds.NSamples())
	}
	if ds.IDs[0] != "x" {
		t.Errorf("expected ID 'x', got %q", ds.IDs[0])
	}
	if ds.Eras[0] != 5 {
		t.Errorf("expected era 5, got %d", ds.Eras[0])
	}
}

func TestLoadParquet_CustomPrefix(t *testing.T) {
	dir := t.TempDir()
	rows := []testRowCustomPrefix{
		{ID: "r1", Era: 1, ColA: 10.0, ColB: 20.0, Target: 0.1},
	}
	path := writeTestParquet(t, dir, "prefix.parquet", rows)

	ds, err := LoadParquet(context.Background(), path,
		WithFeaturePrefix("col_"),
	)
	if err != nil {
		t.Fatal(err)
	}

	if ds.NFeatures() != 2 {
		t.Fatalf("expected 2 features, got %d", ds.NFeatures())
	}
	if ds.FeatureNames[0] != "col_x" && ds.FeatureNames[1] != "col_x" {
		t.Errorf("expected feature col_x in names: %v", ds.FeatureNames)
	}
}

func TestLoadParquet_MissingOptionalColumns(t *testing.T) {
	dir := t.TempDir()
	rows := []testRowNoEraNoID{
		{Feature1: 0.1, Feature2: 0.2, Target: 0.5},
		{Feature1: 0.3, Feature2: 0.4, Target: 0.6},
	}
	path := writeTestParquet(t, dir, "noeranoid.parquet", rows)

	ds, err := LoadParquet(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}

	if ds.NSamples() != 2 {
		t.Fatalf("expected 2 samples, got %d", ds.NSamples())
	}
	if len(ds.IDs) != 0 {
		t.Errorf("expected 0 IDs, got %d", len(ds.IDs))
	}
	if len(ds.Eras) != 0 {
		t.Errorf("expected 0 eras, got %d", len(ds.Eras))
	}
	if len(ds.Targets) != 2 {
		t.Errorf("expected 2 targets, got %d", len(ds.Targets))
	}
}

func TestLoadParquet_NonexistentFile(t *testing.T) {
	_, err := LoadParquet(context.Background(), "/tmp/does-not-exist-12345.parquet")
	if err == nil {
		t.Fatal("expected error for nonexistent file")
	}
}

func TestLoadParquet_ContextCancellation(t *testing.T) {
	dir := t.TempDir()
	rows := []testRow{
		{ID: "a", Era: 1, Feature1: 0.1, Feature2: 0.2, Target: 0.5},
	}
	path := writeTestParquet(t, dir, "cancel.parquet", rows)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately.

	_, err := LoadParquet(ctx, path)
	if err == nil {
		t.Fatal("expected error from cancelled context")
	}
}

func TestLoadParquet_InvalidFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.parquet")
	if err := os.WriteFile(path, []byte("not a parquet file"), 0644); err != nil {
		t.Fatal(err)
	}

	_, err := LoadParquet(context.Background(), path)
	if err == nil {
		t.Fatal("expected error for invalid parquet file")
	}
}

type testRowFloat32Target struct {
	ID       string  `parquet:"id"`
	Era      int64   `parquet:"era"`
	Feature1 float32 `parquet:"feature_f"`
	Target   float32 `parquet:"target"`
}

func TestLoadParquet_Float32AndInt64(t *testing.T) {
	dir := t.TempDir()
	rows := []testRowFloat32Target{
		{ID: "a", Era: 10, Feature1: 1.5, Target: 0.25},
	}
	path := writeTestParquet(t, dir, "f32.parquet", rows)

	ds, err := LoadParquet(context.Background(), path,
		WithFeaturePrefix("feature_f"),
	)
	if err != nil {
		t.Fatal(err)
	}

	if ds.NSamples() != 1 {
		t.Fatalf("expected 1 sample, got %d", ds.NSamples())
	}
	if ds.Eras[0] != 10 {
		t.Errorf("expected era 10, got %d", ds.Eras[0])
	}
	if ds.Features[0][0] < 1.49 || ds.Features[0][0] > 1.51 {
		t.Errorf("expected feature ~1.5, got %f", ds.Features[0][0])
	}
	if ds.Targets[0] < 0.24 || ds.Targets[0] > 0.26 {
		t.Errorf("expected target ~0.25, got %f", ds.Targets[0])
	}
}

type testRowInt32Target struct {
	Feature1 float64 `parquet:"feature_x"`
	Target   int32   `parquet:"target"`
	Era      int32   `parquet:"era"`
}

func TestLoadParquet_Int32Target(t *testing.T) {
	dir := t.TempDir()
	rows := []testRowInt32Target{
		{Feature1: 0.5, Target: 3, Era: 1},
	}
	path := writeTestParquet(t, dir, "int32target.parquet", rows)

	ds, err := LoadParquet(context.Background(), path,
		WithFeaturePrefix("feature_"),
	)
	if err != nil {
		t.Fatal(err)
	}

	if ds.Targets[0] != 3.0 {
		t.Errorf("expected target 3.0, got %f", ds.Targets[0])
	}
}

type testRowInt64Target struct {
	Feature1 float64 `parquet:"feature_x"`
	Target   int64   `parquet:"target"`
}

func TestLoadParquet_Int64Target(t *testing.T) {
	dir := t.TempDir()
	rows := []testRowInt64Target{
		{Feature1: 0.5, Target: 42},
	}
	path := writeTestParquet(t, dir, "int64target.parquet", rows)

	ds, err := LoadParquet(context.Background(), path,
		WithFeaturePrefix("feature_"),
	)
	if err != nil {
		t.Fatal(err)
	}

	if ds.Targets[0] != 42.0 {
		t.Errorf("expected target 42.0, got %f", ds.Targets[0])
	}
}

func TestLoadParquet_NoFeatures(t *testing.T) {
	type rowNoFeatures struct {
		ID     string  `parquet:"id"`
		Target float64 `parquet:"target"`
	}
	dir := t.TempDir()
	rows := []rowNoFeatures{{ID: "a", Target: 0.5}}
	path := writeTestParquet(t, dir, "nofeatures.parquet", rows)

	ds, err := LoadParquet(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}

	if ds.NFeatures() != 0 {
		t.Errorf("expected 0 features, got %d", ds.NFeatures())
	}
	if ds.NSamples() != 1 {
		t.Errorf("expected 1 sample, got %d", ds.NSamples())
	}
}

func TestValueToString_Null(t *testing.T) {
	v := parquet.Value{}
	s := valueToString(v)
	if s != "" {
		t.Errorf("expected empty string for null, got %q", s)
	}
}

func TestValueToFloat64_Null(t *testing.T) {
	v := parquet.Value{}
	f, err := valueToFloat64(v)
	if err != nil {
		t.Fatal(err)
	}
	if f != 0 {
		t.Errorf("expected 0 for null, got %f", f)
	}
}

func TestValueToInt_Null(t *testing.T) {
	v := parquet.Value{}
	i, err := valueToInt(v)
	if err != nil {
		t.Fatal(err)
	}
	if i != 0 {
		t.Errorf("expected 0 for null, got %d", i)
	}
}

func TestValueToFloat64_StringFallback(t *testing.T) {
	v := parquet.ValueOf("3.14").Level(0, 1, 0)
	f, err := valueToFloat64(v)
	if err != nil {
		t.Fatal(err)
	}
	if f < 3.13 || f > 3.15 {
		t.Errorf("expected ~3.14, got %f", f)
	}
}

func TestValueToInt_StringFallback(t *testing.T) {
	v := parquet.ValueOf("42").Level(0, 1, 0)
	i, err := valueToInt(v)
	if err != nil {
		t.Fatal(err)
	}
	if i != 42 {
		t.Errorf("expected 42, got %d", i)
	}
}

type testRowStringTarget struct {
	Feature1 float64 `parquet:"feature_a"`
	Target   string  `parquet:"target"`
}

func TestLoadParquet_BadTarget(t *testing.T) {
	dir := t.TempDir()
	rows := []testRowStringTarget{
		{Feature1: 0.1, Target: "notanumber"},
	}
	path := writeTestParquet(t, dir, "badtarget.parquet", rows)

	_, err := LoadParquet(context.Background(), path)
	if err == nil {
		t.Fatal("expected error for non-numeric target")
	}
	if !strings.Contains(err.Error(), "target") {
		t.Errorf("error should mention target: %v", err)
	}
}

type testRowStringEra struct {
	Feature1 float64 `parquet:"feature_a"`
	Target   float64 `parquet:"target"`
	Era      string  `parquet:"era"`
}

func TestLoadParquet_BadEra(t *testing.T) {
	dir := t.TempDir()
	rows := []testRowStringEra{
		{Feature1: 0.1, Target: 0.5, Era: "notanint"},
	}
	path := writeTestParquet(t, dir, "badera.parquet", rows)

	_, err := LoadParquet(context.Background(), path)
	if err == nil {
		t.Fatal("expected error for non-integer era")
	}
	if !strings.Contains(err.Error(), "era") {
		t.Errorf("error should mention era: %v", err)
	}
}

type testRowStringFeature struct {
	Feature1 string  `parquet:"feature_a"`
	Target   float64 `parquet:"target"`
}

func TestLoadParquet_BadFeature(t *testing.T) {
	dir := t.TempDir()
	rows := []testRowStringFeature{
		{Feature1: "notanumber", Target: 0.5},
	}
	path := writeTestParquet(t, dir, "badfeature.parquet", rows)

	_, err := LoadParquet(context.Background(), path)
	if err == nil {
		t.Fatal("expected error for non-numeric feature")
	}
	if !strings.Contains(err.Error(), "feature") {
		t.Errorf("error should mention feature: %v", err)
	}
}
