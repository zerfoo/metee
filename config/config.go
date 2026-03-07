// Package config provides YAML configuration loading with environment variable
// overrides and struct validation.
package config

import (
	"fmt"
	"os"
	"reflect"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// Load reads a YAML file at path and unmarshals it into a value of type T.
func Load[T any](path string) (T, error) {
	var cfg T
	data, err := os.ReadFile(path)
	if err != nil {
		return cfg, fmt.Errorf("config: %w", err)
	}
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return cfg, fmt.Errorf("config: %w", err)
	}
	return cfg, nil
}

// LoadWithEnv reads a YAML file and then applies environment variable overrides.
// For each exported struct field, it checks for an env var named PREFIX_FIELDNAME
// (uppercase). Supported types: string, int, int64, float64, bool.
func LoadWithEnv[T any](path string, prefix string) (T, error) {
	cfg, err := Load[T](path)
	if err != nil {
		return cfg, err
	}
	if err := applyEnvOverrides(&cfg, prefix); err != nil {
		return cfg, fmt.Errorf("config: %w", err)
	}
	return cfg, nil
}

// Validate checks struct fields tagged with `validate:"required"` and returns
// a list of error messages for fields that have their zero value.
func Validate(v any) []string {
	rv := reflect.ValueOf(v)
	if rv.Kind() == reflect.Ptr {
		rv = rv.Elem()
	}
	if rv.Kind() != reflect.Struct {
		return nil
	}
	rt := rv.Type()
	var errs []string
	for i := range rt.NumField() {
		field := rt.Field(i)
		tag := field.Tag.Get("validate")
		if !strings.Contains(tag, "required") {
			continue
		}
		if rv.Field(i).IsZero() {
			errs = append(errs, fmt.Sprintf("field %q is required", field.Name))
		}
	}
	return errs
}

func applyEnvOverrides(cfg any, prefix string) error {
	rv := reflect.ValueOf(cfg)
	if rv.Kind() == reflect.Ptr {
		rv = rv.Elem()
	}
	if rv.Kind() != reflect.Struct {
		return nil
	}
	rt := rv.Type()
	for i := range rt.NumField() {
		field := rt.Field(i)
		if !field.IsExported() {
			continue
		}
		envKey := prefix + "_" + strings.ToUpper(field.Name)
		envVal, ok := os.LookupEnv(envKey)
		if !ok {
			continue
		}
		fv := rv.Field(i)
		if err := setField(fv, envVal, envKey); err != nil {
			return err
		}
	}
	return nil
}

func setField(fv reflect.Value, envVal, envKey string) error {
	switch fv.Kind() {
	case reflect.String:
		fv.SetString(envVal)
	case reflect.Int:
		v, err := strconv.ParseInt(envVal, 10, 64)
		if err != nil {
			return fmt.Errorf("env %s: %w", envKey, err)
		}
		fv.SetInt(v)
	case reflect.Int64:
		v, err := strconv.ParseInt(envVal, 10, 64)
		if err != nil {
			return fmt.Errorf("env %s: %w", envKey, err)
		}
		fv.SetInt(v)
	case reflect.Float64:
		v, err := strconv.ParseFloat(envVal, 64)
		if err != nil {
			return fmt.Errorf("env %s: %w", envKey, err)
		}
		fv.SetFloat(v)
	case reflect.Bool:
		v, err := strconv.ParseBool(envVal)
		if err != nil {
			return fmt.Errorf("env %s: %w", envKey, err)
		}
		fv.SetBool(v)
	}
	return nil
}
