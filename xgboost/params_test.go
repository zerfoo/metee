package xgboost

import "testing"

func TestDefaultParams(t *testing.T) {
	p := DefaultParams()

	tests := []struct {
		name string
		got  any
		want any
	}{
		{"MaxDepth", p.MaxDepth, 6},
		{"LearningRate", p.LearningRate, 0.3},
		{"NumRounds", p.NumRounds, 100},
		{"Objective", p.Objective, "reg:squarederror"},
		{"Subsample", p.Subsample, 1.0},
		{"ColsampleBytree", p.ColsampleBytree, 1.0},
		{"MinChildWeight", p.MinChildWeight, 1.0},
		{"RegAlpha", p.RegAlpha, 0.0},
		{"RegLambda", p.RegLambda, 1.0},
		{"NumThreads", p.NumThreads, 0},
		{"Verbose", p.Verbose, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.got != tt.want {
				t.Errorf("%s = %v, want %v", tt.name, tt.got, tt.want)
			}
		})
	}
}

func TestToConfigMap(t *testing.T) {
	p := DefaultParams()
	m := p.ToConfigMap()

	expected := map[string]string{
		"max_depth":        "6",
		"learning_rate":    "0.3",
		"n_estimators":     "100",
		"objective":        "reg:squarederror",
		"subsample":        "1",
		"colsample_bytree": "1",
		"min_child_weight": "1",
		"reg_alpha":        "0",
		"reg_lambda":       "1",
		"nthread":          "0",
		"verbosity":        "0",
	}

	for key, want := range expected {
		got, ok := m[key]
		if !ok {
			t.Errorf("ToConfigMap() missing key %q", key)
			continue
		}
		if got != want {
			t.Errorf("ToConfigMap()[%q] = %q, want %q", key, got, want)
		}
	}

	if len(m) != len(expected) {
		t.Errorf("ToConfigMap() has %d keys, want %d", len(m), len(expected))
	}
}

func TestToConfigMapCustomValues(t *testing.T) {
	p := Params{
		MaxDepth:        10,
		LearningRate:    0.01,
		NumRounds:       500,
		Objective:       "reg:pseudohubererror",
		Subsample:       0.8,
		ColsampleBytree: 0.7,
		MinChildWeight:  5.0,
		RegAlpha:        0.1,
		RegLambda:       0.5,
		NumThreads:      4,
		Verbose:         1,
	}
	m := p.ToConfigMap()

	if m["max_depth"] != "10" {
		t.Errorf("max_depth = %q, want 10", m["max_depth"])
	}
	if m["learning_rate"] != "0.01" {
		t.Errorf("learning_rate = %q, want 0.01", m["learning_rate"])
	}
	if m["nthread"] != "4" {
		t.Errorf("nthread = %q, want 4", m["nthread"])
	}
}
