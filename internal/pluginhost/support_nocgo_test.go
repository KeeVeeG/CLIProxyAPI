//go:build !cgo && !windows

package pluginhost

import "testing"

func TestSupportPluginHeaderValueNoCgo(t *testing.T) {
	if got := SupportPluginHeaderValue(); got != "0" {
		t.Fatalf("SupportPluginHeaderValue() = %q, want %q", got, "0")
	}
}
