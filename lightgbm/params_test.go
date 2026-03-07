package lightgbm

import (
	"strings"
	"testing"
)

func TestDefaultParams(t *testing.T) {
	p := DefaultParams()
	if p.Objective != "regression" {
		t.Errorf("Objective = %q, want regression", p.Objective)
	}
	if p.NumLeaves != 31 {
		t.Errorf("NumLeaves = %d, want 31", p.NumLeaves)
	}
	if p.LearningRate != 0.05 {
		t.Errorf("LearningRate = %v, want 0.05", p.LearningRate)
	}
}

func TestToConfigString(t *testing.T) {
	p := DefaultParams()
	s := p.ToConfigString()

	expected := []string{
		"objective=regression",
		"num_leaves=31",
		"learning_rate=0.05",
		"verbose=-1",
	}
	for _, e := range expected {
		if !strings.Contains(s, e) {
			t.Errorf("ToConfigString() missing %q in: %s", e, s)
		}
	}
}
