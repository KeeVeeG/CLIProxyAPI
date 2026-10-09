//go:build windows

package pluginhost

import "testing"

func TestSupportPluginHeaderValueWindows(t *testing.T) {
	if got := SupportPluginHeaderValue(); got != "1" {
		t.Fatalf("SupportPluginHeaderValue() = %q, want %q", got, "1")
	}
}
