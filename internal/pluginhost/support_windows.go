//go:build windows

package pluginhost

// Windows loads plugin DLLs through the native syscall loader. This remains
// available for CGO_ENABLED=0 builds, so Windows binaries support plugins
// independently of the cgo build tag.
const supportPluginValue = "1"
