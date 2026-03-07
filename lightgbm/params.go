// Package lightgbm provides LightGBM model bindings for tree-based learning.
package lightgbm

import "fmt"

// Params holds LightGBM hyperparameters.
type Params struct {
	Objective       string  // e.g. "regression", "binary"
	Metric          string  // e.g. "mse", "binary_logloss"
	NumLeaves       int     // max number of leaves per tree
	LearningRate    float64 // shrinkage rate
	NumIterations   int     // number of boosting rounds
	MinDataInLeaf   int     // minimum samples per leaf
	MaxDepth        int     // max tree depth (-1 for no limit)
	FeatureFraction float64 // subsample ratio of features
	BaggingFraction float64 // subsample ratio of rows
	BaggingFreq     int     // bagging frequency (0 = disabled)
	LambdaL1        float64 // L1 regularization
	LambdaL2        float64 // L2 regularization
	Verbose         int     // verbosity level (-1 = silent)
	NumThreads      int     // number of threads (0 = auto)
}

// DefaultParams returns sensible defaults for regression.
func DefaultParams() Params {
	return Params{
		Objective:       "regression",
		Metric:          "mse",
		NumLeaves:       31,
		LearningRate:    0.05,
		NumIterations:   1000,
		MinDataInLeaf:   20,
		MaxDepth:        -1,
		FeatureFraction: 0.8,
		BaggingFraction: 0.8,
		BaggingFreq:     5,
		LambdaL1:        0.0,
		LambdaL2:        0.0,
		Verbose:         -1,
		NumThreads:      0,
	}
}

// ToConfigString converts params to LightGBM's key=value config format.
func (p Params) ToConfigString() string {
	return fmt.Sprintf(
		"objective=%s metric=%s num_leaves=%d learning_rate=%g "+
			"num_iterations=%d min_data_in_leaf=%d max_depth=%d "+
			"feature_fraction=%g bagging_fraction=%g bagging_freq=%d "+
			"lambda_l1=%g lambda_l2=%g verbose=%d num_threads=%d",
		p.Objective, p.Metric, p.NumLeaves, p.LearningRate,
		p.NumIterations, p.MinDataInLeaf, p.MaxDepth,
		p.FeatureFraction, p.BaggingFraction, p.BaggingFreq,
		p.LambdaL1, p.LambdaL2, p.Verbose, p.NumThreads,
	)
}
