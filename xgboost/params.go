// Package xgboost provides XGBoost model bindings for tree-based learning.
package xgboost

import "fmt"

// Params holds XGBoost hyperparameters.
type Params struct {
	MaxDepth        int     // max tree depth
	LearningRate    float64 // shrinkage rate (eta)
	NumRounds       int     // number of boosting rounds
	Objective       string  // e.g. "reg:squarederror"
	Subsample       float64 // subsample ratio of rows
	ColsampleBytree float64 // subsample ratio of columns per tree
	MinChildWeight  float64 // minimum sum of instance weight in a child
	RegAlpha        float64 // L1 regularization
	RegLambda       float64 // L2 regularization
	NumThreads      int     // number of threads (0 = auto)
	Verbose         int     // verbosity level (0 = silent)
}

// DefaultParams returns sensible defaults for regression.
func DefaultParams() Params {
	return Params{
		MaxDepth:        6,
		LearningRate:    0.3,
		NumRounds:       100,
		Objective:       "reg:squarederror",
		Subsample:       1.0,
		ColsampleBytree: 1.0,
		MinChildWeight:  1.0,
		RegAlpha:        0.0,
		RegLambda:       1.0,
		NumThreads:      0,
		Verbose:         0,
	}
}

// ToConfigMap converts params to a key=value map for the XGBoost C API.
func (p Params) ToConfigMap() map[string]string {
	return map[string]string{
		"max_depth":        fmt.Sprintf("%d", p.MaxDepth),
		"learning_rate":    fmt.Sprintf("%g", p.LearningRate),
		"n_estimators":     fmt.Sprintf("%d", p.NumRounds),
		"objective":        p.Objective,
		"subsample":        fmt.Sprintf("%g", p.Subsample),
		"colsample_bytree": fmt.Sprintf("%g", p.ColsampleBytree),
		"min_child_weight": fmt.Sprintf("%g", p.MinChildWeight),
		"reg_alpha":        fmt.Sprintf("%g", p.RegAlpha),
		"reg_lambda":       fmt.Sprintf("%g", p.RegLambda),
		"nthread":          fmt.Sprintf("%d", p.NumThreads),
		"verbosity":        fmt.Sprintf("%d", p.Verbose),
	}
}
