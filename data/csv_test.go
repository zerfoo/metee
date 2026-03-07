package data

import (
	"strings"
	"testing"
)

func TestLoadCSV(t *testing.T) {
	input := `id,f1,f2,target,era
a,1.0,2.0,0.5,1
b,3.0,4.0,0.6,1
c,5.0,6.0,0.7,2
`
	ds, err := LoadCSV(strings.NewReader(input), "id", "target", "era")
	if err != nil {
		t.Fatalf("LoadCSV() error: %v", err)
	}

	if ds.NSamples() != 3 {
		t.Errorf("NSamples() = %d, want 3", ds.NSamples())
	}
	if ds.NFeatures() != 2 {
		t.Errorf("NFeatures() = %d, want 2", ds.NFeatures())
	}
	if len(ds.IDs) != 3 || ds.IDs[0] != "a" {
		t.Errorf("IDs = %v, want [a b c]", ds.IDs)
	}
	if len(ds.Targets) != 3 || ds.Targets[0] != 0.5 {
		t.Errorf("Targets = %v, want [0.5 0.6 0.7]", ds.Targets)
	}
	if len(ds.Eras) != 3 || ds.Eras[2] != 2 {
		t.Errorf("Eras = %v, want [1 1 2]", ds.Eras)
	}
	if ds.FeatureNames[0] != "f1" || ds.FeatureNames[1] != "f2" {
		t.Errorf("FeatureNames = %v, want [f1 f2]", ds.FeatureNames)
	}
}

func TestLoadCSVEmpty(t *testing.T) {
	input := `id,f1,target,era
`
	ds, err := LoadCSV(strings.NewReader(input), "id", "target", "era")
	if err != nil {
		t.Fatalf("LoadCSV() error: %v", err)
	}
	if ds.NSamples() != 0 {
		t.Errorf("NSamples() = %d, want 0", ds.NSamples())
	}
}
