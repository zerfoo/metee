package config

// TrainingConfig holds configuration for training a single model.
type TrainingConfig struct {
	Backend       string         `yaml:"backend"`
	ModelParams   map[string]any `yaml:"model_params"`
	DataPath      string         `yaml:"data_path"`
	TargetColumn  string         `yaml:"target_column"`
	EraColumn     string         `yaml:"era_column"`
	IDColumn      string         `yaml:"id_column"`
	FeaturePrefix string         `yaml:"feature_prefix"`
	TrainEras     []int          `yaml:"train_eras"`
	ValidEras     []int          `yaml:"valid_eras"`
	Seed          int64          `yaml:"seed"`
}

// EnsembleConfig holds configuration for an ensemble of models.
type EnsembleConfig struct {
	Models      []TrainingConfig `yaml:"models"`
	Weights     []float64        `yaml:"weights"`
	BlendMethod string           `yaml:"blend_method"`
}
