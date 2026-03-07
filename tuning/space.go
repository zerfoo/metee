// Package tuning provides hyperparameter search spaces and search strategies.
package tuning

import (
	"math"
	"math/rand"
)

// ParamRange defines a range of values for a hyperparameter.
type ParamRange interface {
	// Values returns all discrete values in the range.
	// For continuous ranges, this returns nil.
	Values() []any

	// Sample returns a random value from the range.
	Sample(rng *rand.Rand) any
}

// Discrete returns a ParamRange that enumerates the given values.
func Discrete(values ...any) ParamRange {
	return discreteRange{values: values}
}

type discreteRange struct {
	values []any
}

func (d discreteRange) Values() []any {
	out := make([]any, len(d.values))
	copy(out, d.values)
	return out
}

func (d discreteRange) Sample(rng *rand.Rand) any {
	return d.values[rng.Intn(len(d.values))]
}

// Uniform returns a ParamRange that samples uniformly from [low, high).
func Uniform(low, high float64) ParamRange {
	return uniformRange{low: low, high: high}
}

type uniformRange struct {
	low, high float64
}

func (u uniformRange) Values() []any { return nil }

func (u uniformRange) Sample(rng *rand.Rand) any {
	return u.low + rng.Float64()*(u.high-u.low)
}

// LogUniform returns a ParamRange that samples in log space from [low, high).
// Both low and high must be positive.
func LogUniform(low, high float64) ParamRange {
	return logUniformRange{low: low, high: high}
}

type logUniformRange struct {
	low, high float64
}

func (l logUniformRange) Values() []any { return nil }

func (l logUniformRange) Sample(rng *rand.Rand) any {
	logLow := math.Log(l.low)
	logHigh := math.Log(l.high)
	return math.Exp(logLow + rng.Float64()*(logHigh-logLow))
}

// IntRange returns a ParamRange that enumerates integers from low to high (inclusive).
func IntRange(low, high int) ParamRange {
	return intRange{low: low, high: high}
}

type intRange struct {
	low, high int
}

func (r intRange) Values() []any {
	out := make([]any, 0, r.high-r.low+1)
	for i := r.low; i <= r.high; i++ {
		out = append(out, i)
	}
	return out
}

func (r intRange) Sample(rng *rand.Rand) any {
	return r.low + rng.Intn(r.high-r.low+1)
}

// ParamSpace maps parameter names to their ranges.
type ParamSpace map[string]ParamRange

// Grid returns the cartesian product of all discrete/enumerable ranges.
// Parameters with continuous ranges (Values() == nil) are skipped.
func (ps ParamSpace) Grid() []map[string]any {
	names := make([]string, 0, len(ps))
	valueSets := make([][]any, 0, len(ps))

	for name, pr := range ps {
		vals := pr.Values()
		if vals == nil {
			continue
		}
		names = append(names, name)
		valueSets = append(valueSets, vals)
	}

	if len(names) == 0 {
		return nil
	}

	// Compute cartesian product.
	result := []map[string]any{{}}
	for i, name := range names {
		var next []map[string]any
		for _, combo := range result {
			for _, val := range valueSets[i] {
				newCombo := make(map[string]any, len(combo)+1)
				for k, v := range combo {
					newCombo[k] = v
				}
				newCombo[name] = val
				next = append(next, newCombo)
			}
		}
		result = next
	}
	return result
}

// Sample returns a random sample from the parameter space.
func (ps ParamSpace) Sample(rng *rand.Rand) map[string]any {
	out := make(map[string]any, len(ps))
	for name, pr := range ps {
		out[name] = pr.Sample(rng)
	}
	return out
}
