//go:build cgo && !windows

package pluginhost

import "testing"

func TestSupportPluginHeaderValueCgo(t *testing.T) {
	if got := SupportPluginHeaderValue(); got != "1" {
		t.Fatalf("SupportPluginHeaderValue() = %q, want %q", got, "1")
	}
}
