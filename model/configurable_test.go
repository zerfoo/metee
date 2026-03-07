package model_test

import "github.com/zerfoo/metee/model"

// Compile-time check that a concrete type can satisfy Configurable.
var _ model.Configurable = (*mockConfigurable)(nil)

type mockConfigurable struct{}

func (m *mockConfigurable) SetParams(_ map[string]any) error {
	return nil
}
