package tuning

import (
	"math"
	"math/rand"
	"testing"
)

func TestDiscreteValues(t *testing.T) {
	d := Discrete("a", "b", "c")
	vals := d.Values()
	if len(vals) != 3 {
		t.Fatalf("Values() len = %d, want 3", len(vals))
	}
	if vals[0] != "a" || vals[1] != "b" || vals[2] != "c" {
		t.Errorf("Values() = %v, want [a b c]", vals)
	}
}

func TestDiscreteSample(t *testing.T) {
	d := Discrete(1, 2, 3)
	rng := rand.New(rand.NewSource(42))
	seen := map[any]bool{}
	for i := 0; i < 100; i++ {
		v := d.Sample(rng)
		seen[v] = true
	}
	for _, want := range []any{1, 2, 3} {
		if !seen[want] {
			t.Errorf("Sample() never produced %v", want)
		}
	}
}

func TestUniformValues(t *testing.T) {
	u := Uniform(0, 1)
	if u.Values() != nil {
		t.Error("Uniform.Values() should be nil")
	}
}

func TestUniformSample(t *testing.T) {
	u := Uniform(2.0, 5.0)
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 100; i++ {
		v := u.Sample(rng).(float64)
		if v < 2.0 || v >= 5.0 {
			t.Errorf("Sample() = %v, want in [2.0, 5.0)", v)
		}
	}
}

func TestLogUniformValues(t *testing.T) {
	l := LogUniform(0.001, 1.0)
	if l.Values() != nil {
		t.Error("LogUniform.Values() should be nil")
	}
}

func TestLogUniformSample(t *testing.T) {
	l := LogUniform(0.001, 1.0)
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 100; i++ {
		v := l.Sample(rng).(float64)
		if v < 0.001 || v > 1.0 {
			t.Errorf("Sample() = %v, want in [0.001, 1.0]", v)
		}
	}
}

func TestLogUniformDistribution(t *testing.T) {
	l := LogUniform(0.001, 1.0)
	rng := rand.New(rand.NewSource(42))
	belowMedian := 0
	// Geometric mean of 0.001 and 1.0 is sqrt(0.001) ~ 0.0316
	median := math.Sqrt(0.001 * 1.0)
	n := 10000
	for i := 0; i < n; i++ {
		v := l.Sample(rng).(float64)
		if v < median {
			belowMedian++
		}
	}
	ratio := float64(belowMedian) / float64(n)
	if ratio < 0.4 || ratio > 0.6 {
		t.Errorf("log-uniform distribution skewed: %.2f below median", ratio)
	}
}

func TestIntRangeValues(t *testing.T) {
	r := IntRange(3, 6)
	vals := r.Values()
	if len(vals) != 4 {
		t.Fatalf("Values() len = %d, want 4", len(vals))
	}
	for i, want := range []int{3, 4, 5, 6} {
		if vals[i] != want {
			t.Errorf("Values()[%d] = %v, want %d", i, vals[i], want)
		}
	}
}

func TestIntRangeSample(t *testing.T) {
	r := IntRange(1, 3)
	rng := rand.New(rand.NewSource(42))
	seen := map[int]bool{}
	for i := 0; i < 100; i++ {
		v := r.Sample(rng).(int)
		if v < 1 || v > 3 {
			t.Errorf("Sample() = %d, want in [1, 3]", v)
		}
		seen[v] = true
	}
	for _, want := range []int{1, 2, 3} {
		if !seen[want] {
			t.Errorf("Sample() never produced %d", want)
		}
	}
}

func TestGridSingleParam(t *testing.T) {
	ps := ParamSpace{
		"x": Discrete(1, 2, 3),
	}
	grid := ps.Grid()
	if len(grid) != 3 {
		t.Fatalf("Grid() len = %d, want 3", len(grid))
	}
}

func TestGridCartesianProduct(t *testing.T) {
	ps := ParamSpace{
		"a": Discrete("x", "y"),
		"b": IntRange(1, 2),
	}
	grid := ps.Grid()
	// 2 * 2 = 4 combinations
	if len(grid) != 4 {
		t.Fatalf("Grid() len = %d, want 4", len(grid))
	}

	// Verify all combinations exist.
	type combo struct {
		a string
		b int
	}
	seen := map[combo]bool{}
	for _, m := range grid {
		seen[combo{a: m["a"].(string), b: m["b"].(int)}] = true
	}
	for _, a := range []string{"x", "y"} {
		for _, b := range []int{1, 2} {
			if !seen[combo{a, b}] {
				t.Errorf("Grid() missing combination a=%s, b=%d", a, b)
			}
		}
	}
}

func TestGridSkipsContinuous(t *testing.T) {
	ps := ParamSpace{
		"lr": Uniform(0.01, 0.1),
	}
	grid := ps.Grid()
	if grid != nil {
		t.Errorf("Grid() with only continuous params should be nil, got %v", grid)
	}
}

func TestGridMixedParams(t *testing.T) {
	ps := ParamSpace{
		"depth": Discrete(3, 6),
		"lr":    Uniform(0.01, 0.1), // skipped in grid
	}
	grid := ps.Grid()
	if len(grid) != 2 {
		t.Fatalf("Grid() len = %d, want 2", len(grid))
	}
	for _, m := range grid {
		if _, ok := m["lr"]; ok {
			t.Error("Grid() should not include continuous params")
		}
	}
}

func TestSample(t *testing.T) {
	ps := ParamSpace{
		"depth": Discrete(3, 6, 9),
		"lr":    Uniform(0.01, 0.1),
		"alpha": LogUniform(0.001, 1.0),
		"trees": IntRange(50, 200),
	}
	rng := rand.New(rand.NewSource(42))
	sample := ps.Sample(rng)

	if len(sample) != 4 {
		t.Fatalf("Sample() len = %d, want 4", len(sample))
	}

	if _, ok := sample["depth"]; !ok {
		t.Error("Sample() missing depth")
	}

	lr := sample["lr"].(float64)
	if lr < 0.01 || lr >= 0.1 {
		t.Errorf("lr = %v, want in [0.01, 0.1)", lr)
	}

	alpha := sample["alpha"].(float64)
	if alpha < 0.001 || alpha > 1.0 {
		t.Errorf("alpha = %v, want in [0.001, 1.0]", alpha)
	}

	trees := sample["trees"].(int)
	if trees < 50 || trees > 200 {
		t.Errorf("trees = %d, want in [50, 200]", trees)
	}
}

func TestGridEmpty(t *testing.T) {
	ps := ParamSpace{}
	grid := ps.Grid()
	if grid != nil {
		t.Errorf("Grid() on empty space should be nil, got %v", grid)
	}
}

func TestSampleEmpty(t *testing.T) {
	ps := ParamSpace{}
	rng := rand.New(rand.NewSource(42))
	sample := ps.Sample(rng)
	if len(sample) != 0 {
		t.Errorf("Sample() on empty space should be empty, got %v", sample)
	}
}
