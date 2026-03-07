package config

import (
	"testing"
)

type testConfig struct {
	Name         string  `yaml:"name" validate:"required"`
	LearningRate float64 `yaml:"learning_rate"`
	NumRounds    int     `yaml:"num_rounds"`
	Enabled      bool    `yaml:"enabled"`
}

type requiredConfig struct {
	Backend string `yaml:"backend" validate:"required"`
	Seed    int64  `yaml:"seed" validate:"required"`
}

func TestLoad(t *testing.T) {
	tests := []struct {
		name    string
		path    string
		wantErr bool
		check   func(t *testing.T, cfg testConfig)
	}{
		{
			name: "valid YAML",
			path: "testdata/valid.yaml",
			check: func(t *testing.T, cfg testConfig) {
				t.Helper()
				if cfg.Name != "test-model" {
					t.Errorf("Name = %q, want %q", cfg.Name, "test-model")
				}
				if cfg.LearningRate != 0.01 {
					t.Errorf("LearningRate = %v, want %v", cfg.LearningRate, 0.01)
				}
				if cfg.NumRounds != 100 {
					t.Errorf("NumRounds = %d, want %d", cfg.NumRounds, 100)
				}
				if !cfg.Enabled {
					t.Error("Enabled = false, want true")
				}
			},
		},
		{
			name:    "nonexistent file",
			path:    "testdata/nonexistent.yaml",
			wantErr: true,
		},
		{
			name:    "invalid YAML",
			path:    "testdata/invalid.yaml",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg, err := Load[testConfig](tt.path)
			if (err != nil) != tt.wantErr {
				t.Fatalf("Load() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.check != nil {
				tt.check(t, cfg)
			}
		})
	}
}

func TestLoadWithEnv(t *testing.T) {
	tests := []struct {
		name    string
		path    string
		prefix  string
		envVars map[string]string
		wantErr bool
		check   func(t *testing.T, cfg testConfig)
	}{
		{
			name:   "env overrides string and float",
			path:   "testdata/valid.yaml",
			prefix: "TEST",
			envVars: map[string]string{
				"TEST_NAME":         "overridden",
				"TEST_LEARNINGRATE": "0.05",
			},
			check: func(t *testing.T, cfg testConfig) {
				t.Helper()
				if cfg.Name != "overridden" {
					t.Errorf("Name = %q, want %q", cfg.Name, "overridden")
				}
				if cfg.LearningRate != 0.05 {
					t.Errorf("LearningRate = %v, want %v", cfg.LearningRate, 0.05)
				}
				// Unchanged fields retain YAML values.
				if cfg.NumRounds != 100 {
					t.Errorf("NumRounds = %d, want %d", cfg.NumRounds, 100)
				}
			},
		},
		{
			name:   "env overrides int and bool",
			path:   "testdata/valid.yaml",
			prefix: "APP",
			envVars: map[string]string{
				"APP_NUMROUNDS": "200",
				"APP_ENABLED":   "false",
			},
			check: func(t *testing.T, cfg testConfig) {
				t.Helper()
				if cfg.NumRounds != 200 {
					t.Errorf("NumRounds = %d, want %d", cfg.NumRounds, 200)
				}
				if cfg.Enabled {
					t.Error("Enabled = true, want false")
				}
			},
		},
		{
			name:   "invalid int env var",
			path:   "testdata/valid.yaml",
			prefix: "BAD",
			envVars: map[string]string{
				"BAD_NUMROUNDS": "notanumber",
			},
			wantErr: true,
		},
		{
			name:   "invalid float env var",
			path:   "testdata/valid.yaml",
			prefix: "BAD",
			envVars: map[string]string{
				"BAD_LEARNINGRATE": "notafloat",
			},
			wantErr: true,
		},
		{
			name:   "invalid bool env var",
			path:   "testdata/valid.yaml",
			prefix: "BAD",
			envVars: map[string]string{
				"BAD_ENABLED": "notabool",
			},
			wantErr: true,
		},
		{
			name:    "nonexistent file",
			path:    "testdata/nonexistent.yaml",
			prefix:  "TEST",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for k, v := range tt.envVars {
				t.Setenv(k, v)
			}

			cfg, err := LoadWithEnv[testConfig](tt.path, tt.prefix)
			if (err != nil) != tt.wantErr {
				t.Fatalf("LoadWithEnv() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.check != nil {
				tt.check(t, cfg)
			}
		})
	}
}

type int64Config struct {
	Seed int64 `yaml:"seed"`
}

func TestLoadWithEnv_Int64(t *testing.T) {
	t.Setenv("I64_SEED", "9999999999")

	cfg, err := LoadWithEnv[int64Config]("testdata/valid.yaml", "I64")
	if err != nil {
		t.Fatalf("LoadWithEnv() error = %v", err)
	}
	if cfg.Seed != 9999999999 {
		t.Errorf("Seed = %d, want %d", cfg.Seed, 9999999999)
	}
}

func TestLoadWithEnv_InvalidInt64(t *testing.T) {
	t.Setenv("I64_SEED", "notint64")

	_, err := LoadWithEnv[int64Config]("testdata/valid.yaml", "I64")
	if err == nil {
		t.Fatal("expected error for invalid int64 env var")
	}
}

func TestValidate(t *testing.T) {
	tests := []struct {
		name     string
		input    any
		wantErrs int
	}{
		{
			name:     "all required fields set",
			input:    requiredConfig{Backend: "lightgbm", Seed: 42},
			wantErrs: 0,
		},
		{
			name:     "missing all required fields",
			input:    requiredConfig{},
			wantErrs: 2,
		},
		{
			name:     "missing one required field",
			input:    requiredConfig{Backend: "lightgbm"},
			wantErrs: 1,
		},
		{
			name:     "pointer to struct",
			input:    &requiredConfig{},
			wantErrs: 2,
		},
		{
			name:     "non-struct returns nil",
			input:    "a string",
			wantErrs: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			errs := Validate(tt.input)
			if len(errs) != tt.wantErrs {
				t.Errorf("Validate() returned %d errors, want %d: %v", len(errs), tt.wantErrs, errs)
			}
		})
	}
}
