package auth

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	internalconfig "github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/registry"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/executor"
	"github.com/tidwall/gjson"
)

// parallelClaudeHandoffCompactorExecutor deliberately blocks every chunk after
// the probe. This makes the worker pool observable without relying on timing.
type parallelClaudeHandoffCompactorExecutor struct {
	expectedWorkers      int
	forceContextTooLarge bool

	calls      atomic.Int32
	active     atomic.Int32
	maxActive  atomic.Int32
	started    atomic.Int32
	callback   atomic.Int32
	allStarted chan struct{}
	release    chan struct{}
	closeOnce  sync.Once

	mu      sync.Mutex
	authIDs []string
	markers []string
}

func (*parallelClaudeHandoffCompactorExecutor) Identifier() string { return "codex" }

func (e *parallelClaudeHandoffCompactorExecutor) Refresh(_ context.Context, auth *Auth) (*Auth, error) {
	return auth, nil
}

func (*parallelClaudeHandoffCompactorExecutor) CountTokens(context.Context, *Auth, cliproxyexecutor.Request, cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	return cliproxyexecutor.Response{}, &Error{HTTPStatus: http.StatusNotImplemented, Message: "not used"}
}

func (*parallelClaudeHandoffCompactorExecutor) ExecuteStream(context.Context, *Auth, cliproxyexecutor.Request, cliproxyexecutor.Options) (*cliproxyexecutor.StreamResult, error) {
	return nil, &Error{HTTPStatus: http.StatusNotImplemented, Message: "not used"}
}

func (*parallelClaudeHandoffCompactorExecutor) HttpRequest(context.Context, *Auth, *http.Request) (*http.Response, error) {
	return nil, &Error{HTTPStatus: http.StatusNotImplemented, Message: "not used"}
}

func (e *parallelClaudeHandoffCompactorExecutor) Execute(ctx context.Context, auth *Auth, req cliproxyexecutor.Request, _ cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	call := int(e.calls.Add(1))
	active := e.active.Add(1)
	for {
		max := e.maxActive.Load()
		if active <= max || e.maxActive.CompareAndSwap(max, active) {
			break
		}
	}
	defer e.active.Add(-1)

	text := gjson.GetBytes(req.Payload, "input.0.content.0.text").String()
	marker := ""
	for _, field := range strings.Fields(text) {
		if strings.HasPrefix(field, "chunk-") {
			marker = strings.TrimRight(field, ",.:;)")
			break
		}
	}
	e.mu.Lock()
	e.authIDs = append(e.authIDs, auth.ID)
	e.markers = append(e.markers, marker)
	e.mu.Unlock()

	if call > 1 {
		if e.started.Add(1) == int32(e.expectedWorkers) {
			e.closeOnce.Do(func() { close(e.allStarted) })
		}
		select {
		case <-e.release:
		case <-ctx.Done():
			return cliproxyexecutor.Response{}, ctx.Err()
		}
		if e.forceContextTooLarge && !strings.Contains(text, "Original Claude message fragment") {
			return cliproxyexecutor.Response{}, &Error{HTTPStatus: http.StatusBadRequest, Code: "context_too_large", Message: "input exceeds context window"}
		}
	}
	response, err := json.Marshal(map[string]any{
		"output": []any{map[string]any{
			"content": []any{map[string]any{"text": "summary:" + marker}},
		}},
	})
	if err != nil {
		return cliproxyexecutor.Response{}, err
	}
	return cliproxyexecutor.Response{Payload: response}, nil
}

func newParallelClaudeHandoffTestManager(t *testing.T, executor *parallelClaudeHandoffCompactorExecutor) (*Manager, []byte, int) {
	t.Helper()
	manager := NewManager(nil, nil, nil)
	manager.RegisterExecutor(executor)
	manager.SetRetryConfig(0, 0, 0)
	const model = "gpt-6-luna"
	for index, id := range []string{"claude-handoff-parallel-a", "claude-handoff-parallel-b"} {
		registry.GetGlobalRegistry().RegisterClient(id, "codex", []*registry.ModelInfo{{ID: model}})
		entryID := id
		t.Cleanup(func() { registry.GetGlobalRegistry().UnregisterClient(entryID) })
		if _, err := manager.Register(context.Background(), &Auth{
			ID: id, Provider: "codex", Status: StatusActive,
			Metadata: map[string]any{"disable_cooling": true},
		}); err != nil {
			t.Fatalf("Register(%s) error = %v", id, err)
		}
		_ = index
	}
	manager.SetConfig(&internalconfig.Config{Routing: internalconfig.RoutingConfig{ClaudeHandoff: internalconfig.ClaudeHandoffConfig{
		Enabled: true, CompactorProvider: "codex", CompactorModel: model,
		CompactorRetryMaxInterval: 1,
	}}})

	messages := make([]map[string]any, 0, 12)
	for index := 0; index < 12; index++ {
		messages = append(messages, map[string]any{
			"role":    "user",
			"content": strings.Repeat(fmt.Sprintf("chunk-%02d ", index), 2000),
		})
	}
	payload, err := json.Marshal(map[string]any{"messages": messages})
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	return manager, payload, len(messages)
}

func TestClaudeHandoffCompactorParallelChunksPreserveOrderAndBoundConcurrency(t *testing.T) {
	executor := &parallelClaudeHandoffCompactorExecutor{}
	manager, payload, _ := newParallelClaudeHandoffTestManager(t, executor)
	compactorPayload, err := claudeHandoffCompactorPayload(payload)
	if err != nil {
		t.Fatalf("claudeHandoffCompactorPayload() error = %v", err)
	}
	const maxBytes = 32 * 1024
	chunks, err := splitClaudeHandoffPayload(compactorPayload, maxBytes)
	if err != nil {
		t.Fatalf("splitClaudeHandoffPayload() error = %v", err)
	}
	if len(chunks) < 3 {
		t.Fatalf("chunks = %d, want at least 3", len(chunks))
	}
	executor.expectedWorkers = minInt(claudeHandoffCompactorParallelism, len(chunks)-1)
	executor.allStarted = make(chan struct{})
	executor.release = make(chan struct{})

	var callbackCalls atomic.Int32
	opts := claudeHandoffOptions()
	opts.Metadata[cliproxyexecutor.SelectedAuthCallbackMetadataKey] = func(string) {
		callbackCalls.Add(1)
	}
	cfg := claudeHandoffConfig(manager.runtimeConfigSnapshot())
	result := make(chan struct {
		summary string
		err     error
	}, 1)
	go func() {
		summary, compactErr := manager.compactClaudeHandoffContextWithLimit(context.Background(), payload, opts, cfg, maxBytes, 0)
		result <- struct {
			summary string
			err     error
		}{summary, compactErr}
	}()

	select {
	case <-executor.allStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for parallel compactor workers")
	}
	if got := executor.maxActive.Load(); got < 2 {
		t.Fatalf("max concurrent compactor calls = %d, want at least 2", got)
	}
	if got := executor.maxActive.Load(); got > int32(claudeHandoffCompactorParallelism) {
		t.Fatalf("max concurrent compactor calls = %d, want <= %d", got, claudeHandoffCompactorParallelism)
	}
	if got := callbackCalls.Load(); got != 1 {
		t.Fatalf("selected-auth callback calls = %d, want exactly one probe callback", got)
	}
	close(executor.release)

	select {
	case outcome := <-result:
		if outcome.err != nil {
			t.Fatalf("compactClaudeHandoffContextWithLimit() error = %v", outcome.err)
		}
		for index, chunk := range chunks {
			marker := strings.Fields(gjson.GetBytes(chunk, "messages.0.content").String())[0]
			want := "summary:" + marker
			if index > 0 {
				want = "\n\n" + want
			}
			if !strings.Contains(outcome.summary, want) {
				t.Fatalf("summary missing ordered chunk %d marker %q: %q", index, marker, outcome.summary)
			}
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for compactor result")
	}
	if got := int(executor.calls.Load()); got != len(chunks) {
		t.Fatalf("compactor calls = %d, want exactly one per chunk (%d)", got, len(chunks))
	}
	executor.mu.Lock()
	defer executor.mu.Unlock()
	if len(executor.authIDs) != len(chunks) {
		t.Fatalf("recorded auth selections = %d, want %d", len(executor.authIDs), len(chunks))
	}
	for index, authID := range executor.authIDs[1:] {
		if authID != executor.authIDs[0] {
			t.Fatalf("chunk %d used auth %q, want probe auth %q", index+1, authID, executor.authIDs[0])
		}
	}
}

func TestClaudeHandoffCompactorParallelChunksCancelSiblings(t *testing.T) {
	executor := &parallelClaudeHandoffCompactorExecutor{}
	manager, payload, _ := newParallelClaudeHandoffTestManager(t, executor)
	compactorPayload, err := claudeHandoffCompactorPayload(payload)
	if err != nil {
		t.Fatalf("claudeHandoffCompactorPayload() error = %v", err)
	}
	const maxBytes = 32 * 1024
	chunks, err := splitClaudeHandoffPayload(compactorPayload, maxBytes)
	if err != nil {
		t.Fatalf("splitClaudeHandoffPayload() error = %v", err)
	}
	executor.expectedWorkers = minInt(claudeHandoffCompactorParallelism, len(chunks)-1)
	executor.allStarted = make(chan struct{})
	executor.release = make(chan struct{})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	result := make(chan error, 1)
	go func() {
		_, compactErr := manager.compactClaudeHandoffContextWithLimit(ctx, payload, claudeHandoffOptions(), claudeHandoffConfig(manager.runtimeConfigSnapshot()), maxBytes, 0)
		result <- compactErr
	}()
	select {
	case <-executor.allStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for parallel compactor workers")
	}
	cancel()
	select {
	case compactErr := <-result:
		if !errors.Is(compactErr, context.Canceled) {
			t.Fatalf("compaction error = %v, want context.Canceled", compactErr)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("parallel compactor did not stop after cancellation")
	}
	if got := executor.active.Load(); got != 0 {
		t.Fatalf("active compactor calls after cancellation = %d, want 0", got)
	}
}

func TestClaudeHandoffCompactorParallelNestedProbeDoesNotCallOuterCallbackConcurrently(t *testing.T) {
	executor := &parallelClaudeHandoffCompactorExecutor{forceContextTooLarge: true}
	manager, payload, _ := newParallelClaudeHandoffTestManager(t, executor)
	compactorPayload, err := claudeHandoffCompactorPayload(payload)
	if err != nil {
		t.Fatalf("claudeHandoffCompactorPayload() error = %v", err)
	}
	const maxBytes = 32 * 1024
	chunks, err := splitClaudeHandoffPayload(compactorPayload, maxBytes)
	if err != nil {
		t.Fatalf("splitClaudeHandoffPayload() error = %v", err)
	}
	executor.expectedWorkers = minInt(claudeHandoffCompactorParallelism, len(chunks)-1)
	executor.allStarted = make(chan struct{})
	executor.release = make(chan struct{})
	var callbackActive atomic.Int32
	var callbackMax atomic.Int32
	opts := claudeHandoffOptions()
	opts.Metadata[cliproxyexecutor.SelectedAuthCallbackMetadataKey] = func(string) {
		active := callbackActive.Add(1)
		for {
			max := callbackMax.Load()
			if active <= max || callbackMax.CompareAndSwap(max, active) {
				break
			}
		}
		time.Sleep(20 * time.Millisecond)
		callbackActive.Add(-1)
	}
	result := make(chan error, 1)
	go func() {
		_, compactErr := manager.compactClaudeHandoffContextWithLimit(context.Background(), payload, opts, claudeHandoffConfig(manager.runtimeConfigSnapshot()), maxBytes, 0)
		result <- compactErr
	}()
	select {
	case <-executor.allStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for parallel compactor workers")
	}
	close(executor.release)
	select {
	case compactErr := <-result:
		if compactErr != nil {
			t.Fatalf("compactClaudeHandoffContextWithLimit() error = %v", compactErr)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for recursively split compactor chunks")
	}
	if got := callbackMax.Load(); got > 1 {
		t.Fatalf("outer selected-auth callback ran concurrently: max active = %d", got)
	}
	if got := executor.maxActive.Load(); got > int32(claudeHandoffCompactorParallelism) {
		t.Fatalf("nested compactor calls exceeded global worker limit: max active = %d, limit = %d", got, claudeHandoffCompactorParallelism)
	}
}

func minInt(left, right int) int {
	if left < right {
		return left
	}
	return right
}
