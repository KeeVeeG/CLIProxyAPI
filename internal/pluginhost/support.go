package pluginhost

// SupportPluginHeaderValue reports whether the current binary can load standard
// dynamic-library plugins on the current platform.
//
// Windows uses the native syscall DLL loader, so it supports plugins even when
// the binary was built with CGO_ENABLED=0. On Unix-like platforms the loader
// uses cgo and the value reflects whether cgo was enabled at build time.
func SupportPluginHeaderValue() string {
	return supportPluginValue
}
