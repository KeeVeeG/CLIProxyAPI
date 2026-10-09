package auth

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	internalconfig "github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/registry"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v8/sdk/translator"
	"github.com/tidwall/gjson"
)

func claudeHandoffOptions() cliproxyexecutor.Options {
	return cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FormatClaude,
		Metadata: map[string]any{
			cliproxyexecutor.CanonicalSessionIDMetadataKey: "session-test",
		},
	}
}

func claudeQuotaError(message string, status int) error {
	return &Error{
		HTTPStatus: status,
		Message:    `{"type":"error","error":{"type":"rate_limit_error","message":"` + message + `"}}`,
	}
}

func TestSplitClaudeHandoffPayloadKeepsChunksBelowCompactorLimit(t *testing.T) {
	var messages []map[string]any
	for index := 0; index < 5; index++ {
		messages = append(messages, map[string]any{
			"role":    "user",
			"content": strings.Repeat(fmt.Sprintf("message-%d ", index), 18000),
		})
	}
	payload, err := json.Marshal(map[string]any{"model": "claude-opus-5-5", "messages": messages})
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	chunks, err := splitClaudeHandoffPayload(payload, 128*1024)
	if err != nil {
		t.Fatalf("splitClaudeHandoffPayload() error = %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("chunks = %d, want at least 2", len(chunks))
	}
	for index, chunk := range chunks {
		if len(chunk) > 128*1024 {
			t.Fatalf("chunk %d has %d bytes, want <= %d", index, len(chunk), 128*1024)
		}
		if !gjson.GetBytes(chunk, "messages").IsArray() {
			t.Fatalf("chunk %d has invalid messages payload: %s", index, chunk)
		}
	}
}

func TestSplitClaudeHandoffPayloadDoesNotMultiplyGiantMessageCalls(t *testing.T) {
	payload, err := json.Marshal(map[string]any{
		"model": "claude-opus-5-5",
		"messages": []map[string]any{{
			"role":    "user",
			"content": strings.Repeat("large tool output ", 300000),
		}, {
			"role":    "user",
			"content": "latest request",
		}},
	})
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	chunks, err := splitClaudeHandoffPayload(payload, claudeHandoffCompactorChunkBytes)
	if err != nil {
		t.Fatalf("splitClaudeHandoffPayload() error = %v", err)
	}
	// A giant result should consume the safe budget per fragment. A prior
	// implementation used maxBytes/4 and turned this payload into ~40 calls.
	if len(chunks) > 15 {
		t.Fatalf("chunks = %d, want at most 15 for %d-byte giant message", len(chunks), len(payload))
	}
	for index, chunk := range chunks {
		if len(chunk) > claudeHandoffCompactorChunkBytes {
			t.Fatalf("chunk %d has %d bytes, want <= %d", index, len(chunk), claudeHandoffCompactorChunkBytes)
		}
	}
}

func TestSplitClaudeHandoffPayloadEscapedGiantMessageStaysBelowLimit(t *testing.T) {
	payload, err := json.Marshal(map[string]any{
		"model": "claude-opus-5-5",
		"messages": []map[string]any{{
			"role":    "user",
			"content": strings.Repeat(`quote " slash \\ tool_result Ж 🧪 `, 12000),
		}, {
			"role":    "user",
			"content": "latest request",
		}},
	})
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	chunks, err := splitClaudeHandoffPayload(payload, 32*1024)
	if err != nil {
		t.Fatalf("splitClaudeHandoffPayload() error = %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("chunks = %d, want at least 2", len(chunks))
	}
	for index, chunk := range chunks {
		if len(chunk) > 32*1024 {
			t.Fatalf("chunk %d has %d bytes, want <= %d", index, len(chunk), 32*1024)
		}
		if strings.Contains(string(chunk), "\uFFFD") {
			t.Fatalf("chunk %d contains a replacement character after UTF-8 splitting", index)
		}
	}
}

func TestClaudeHandoffCompactorPayloadDropsRequestToolSchemas(t *testing.T) {
	payload, err := json.Marshal(map[string]any{
		"model":               "claude-opus-5-5",
		"system":              []map[string]any{{"type": "text", "text": "keep system context"}},
		"tools":               []map[string]any{{"name": "large_tool", "description": strings.Repeat("schema ", 1000)}},
		"tool_choice":         map[string]any{"type": "auto"},
		"parallel_tool_calls": true,
		"messages":            []map[string]any{{"role": "user", "content": "keep conversation"}},
	})
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	compactorPayload, err := claudeHandoffCompactorPayload(payload)
	if err != nil {
		t.Fatalf("claudeHandoffCompactorPayload() error = %v", err)
	}
	if gjson.GetBytes(compactorPayload, "tools").Exists() || gjson.GetBytes(compactorPayload, "tool_choice").Exists() || gjson.GetBytes(compactorPayload, "parallel_tool_calls").Exists() {
		t.Fatalf("request-only tool fields remain in compactor payload: %s", compactorPayload)
	}
	if !gjson.GetBytes(compactorPayload, "system").Exists() || !gjson.GetBytes(compactorPayload, "messages").IsArray() {
		t.Fatalf("compactor payload dropped conversation fields: %s", compactorPayload)
	}
	if gjson.GetBytes(payload, "tools").IsArray() == false {
		t.Fatalf("test payload unexpectedly lost original tools: %s", payload)
	}
}

func TestClaudeHandoffCompactorChunksDoNotRepeatToolSchemas(t *testing.T) {
	payload, err := json.Marshal(map[string]any{
		"model": "claude-opus-5-5",
		"tools": []map[string]any{{"name": "large_tool", "description": strings.Repeat("schema ", 20000)}},
		"messages": []map[string]any{{
			"role":    "user",
			"content": strings.Repeat("large tool output ", 40000),
		}, {"role": "user", "content": "latest request"}},
	})
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	compactorPayload, err := claudeHandoffCompactorPayload(payload)
	if err != nil {
		t.Fatalf("claudeHandoffCompactorPayload() error = %v", err)
	}
	chunks, err := splitClaudeHandoffPayload(compactorPayload, claudeHandoffCompactorChunkBytes)
	if err != nil {
		t.Fatalf("splitClaudeHandoffPayload() error = %v", err)
	}
	if len(chunks) > 4 {
		t.Fatalf("chunks = %d, want at most 4 after removing repeated tool schemas", len(chunks))
	}
	for index, chunk := range chunks {
		if gjson.GetBytes(chunk, "tools").Exists() {
			t.Fatalf("chunk %d still contains tools", index)
		}
		if len(chunk) > claudeHandoffCompactorChunkBytes {
			t.Fatalf("chunk %d has %d bytes, want <= %d", index, len(chunk), claudeHandoffCompactorChunkBytes)
		}
	}
}

type claudeHandoffContextRotationExecutor struct {
	calls atomic.Int32
}

func (*claudeHandoffContextRotationExecutor) Identifier() string { return "codex" }
func (*claudeHandoffContextRotationExecutor) Refresh(_ context.Context, auth *Auth) (*Auth, error) {
	return auth, nil
}
func (*claudeHandoffContextRotationExecutor) CountTokens(context.Context, *Auth, cliproxyexecutor.Request, cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	return cliproxyexecutor.Response{}, &Error{HTTPStatus: http.StatusNotImplemented, Message: "not used"}
}
func (*claudeHandoffContextRotationExecutor) ExecuteStream(context.Context, *Auth, cliproxyexecutor.Request, cliproxyexecutor.Options) (*cliproxyexecutor.StreamResult, error) {
	return nil, &Error{HTTPStatus: http.StatusNotImplemented, Message: "not used"}
}
func (*claudeHandoffContextRotationExecutor) HttpRequest(context.Context, *Auth, *http.Request) (*http.Response, error) {
	return nil, &Error{HTTPStatus: http.StatusNotImplemented, Message: "not used"}
}
func (e *claudeHandoffContextRotationExecutor) Execute(_ context.Context, auth *Auth, _ cliproxyexecutor.Request, _ cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	e.calls.Add(1)
	if auth.ID == "codex-a" {
		return cliproxyexecutor.Response{}, &Error{HTTPStatus: http.StatusBadRequest, Code: "context_too_large", Message: "input exceeds context window"}
	}
	return cliproxyexecutor.Response{Payload: []byte(`{"output":[{"content":[{"text":"{\"goal\":\"continue\"}"}]}]}`)}, nil
}

func TestClaudeHandoffCompactorRotatesPastContextTooLargeAccount(t *testing.T) {
	manager := NewManager(nil, claudeHandoffFlowSelector{}, nil)
	executor := &claudeHandoffContextRotationExecutor{}
	manager.RegisterExecutor(executor)
	manager.SetRetryConfig(0, 1, 0)
	const model = "gpt-6-luna"
	for _, id := range []string{"codex-a", "codex-b"} {
		registry.GetGlobalRegistry().RegisterClient(id, "codex", []*registry.ModelInfo{{ID: model}})
		entryID := id
		t.Cleanup(func() { registry.GetGlobalRegistry().UnregisterClient(entryID) })
		if _, err := manager.Register(context.Background(), &Auth{ID: id, Provider: "codex", Status: StatusActive, Metadata: map[string]any{"disable_cooling": true}}); err != nil {
			t.Fatalf("Register(%s) error = %v", id, err)
		}
	}
	manager.SetConfig(&internalconfig.Config{Routing: internalconfig.RoutingConfig{ClaudeHandoff: internalconfig.ClaudeHandoffConfig{
		Enabled: true, CompactorProvider: "codex", CompactorModel: model, CompactorRetryMaxInterval: 1,
	}}})
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	summary, err := manager.compactClaudeHandoffContext(ctx, []byte(`{"messages":[{"role":"user","content":"small"}]}`), claudeHandoffOptions(), claudeHandoffConfig(manager.runtimeConfigSnapshot()))
	if err != nil {
		t.Fatalf("compactClaudeHandoffContext() error = %v", err)
	}
	if summary != `{"goal":"continue"}` {
		t.Fatalf("summary = %q", summary)
	}
	if got := executor.calls.Load(); got != 2 {
		t.Fatalf("compactor calls = %d, want both codex credentials", got)
	}
}

func TestClaudeHandoffCompactorSplitsLargeTranscript(t *testing.T) {
	manager := NewManager(nil, nil, nil)
	recorder := &claudeHandoffPayloadRecordingCompactorExecutor{}
	manager.RegisterExecutor(recorder)
	const model = "gpt-6-luna"
	const authID = "claude-handoff-chunk-compactor-auth"
	registry.GetGlobalRegistry().RegisterClient(authID, "codex", []*registry.ModelInfo{{ID: model}})
	t.Cleanup(func() { registry.GetGlobalRegistry().UnregisterClient(authID) })
	if _, err := manager.Register(context.Background(), &Auth{ID: authID, Provider: "codex", Status: StatusActive, Metadata: map[string]any{"disable_cooling": true}}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}
	manager.SetConfig(&internalconfig.Config{Routing: internalconfig.RoutingConfig{ClaudeHandoff: internalconfig.ClaudeHandoffConfig{
		Enabled: true, CompactorProvider: "codex", CompactorModel: model,
	}}})
	messages := make([]map[string]any, 0, 5)
	for index := 0; index < 5; index++ {
		messages = append(messages, map[string]any{"role": "user", "content": strings.Repeat(fmt.Sprintf("chunk-%d ", index), 30000)})
	}
	payload, err := json.Marshal(map[string]any{"messages": messages})
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	summary, err := manager.compactClaudeHandoffContext(ctx, payload, claudeHandoffOptions(), claudeHandoffConfig(manager.runtimeConfigSnapshot()))
	if err != nil {
		t.Fatalf("compactClaudeHandoffContext() error = %v", err)
	}
	if summary == "" {
		t.Fatal("compactor summary is empty")
	}
	if got := recorder.calls.Load(); got < 2 {
		t.Fatalf("compactor calls = %d, want at least two chunks", got)
	}
}

func TestIsClaudeHandoffCandidateRequiresExactQuotaSignal(t *testing.T) {
	const quotaMessage = "This request would exceed your account's rate limit. Please try again later."
	tests := []struct {
		name     string
		provider string
		format   sdktranslator.Format
		status   int
		message  string
		want     bool
	}{
		{name: "claude account rate limit", provider: "claude", format: sdktranslator.FormatClaude, status: http.StatusTooManyRequests, message: quotaMessage, want: true},
		{name: "monthly spend limit", provider: "claude", format: sdktranslator.FormatClaude, status: http.StatusTooManyRequests, message: "This request would exceed your account's monthly spend limit.", want: true},
		{name: "five hour limit", provider: "claude", format: sdktranslator.FormatClaude, status: http.StatusTooManyRequests, message: "5-hour limit exceeded", want: true},
		{name: "different provider", provider: "openai", format: sdktranslator.FormatClaude, status: http.StatusTooManyRequests, message: quotaMessage, want: false},
		{name: "translated request", provider: "claude", format: sdktranslator.FormatOpenAI, status: http.StatusTooManyRequests, message: quotaMessage, want: false},
		{name: "bad request", provider: "claude", format: sdktranslator.FormatClaude, status: http.StatusBadRequest, message: quotaMessage, want: false},
		{name: "ordinary rate limit", provider: "claude", format: sdktranslator.FormatClaude, status: http.StatusTooManyRequests, message: "rate limit exceeded", want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts := claudeHandoffOptions()
			opts.SourceFormat = tt.format
			got := isClaudeHandoffCandidate(tt.provider, opts, claudeQuotaError(tt.message, tt.status))
			if got != tt.want {
				t.Fatalf("isClaudeHandoffCandidate() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsClaudeHandoffCandidateSkipsInternalCompactor(t *testing.T) {
	opts := claudeHandoffOptions()
	opts.Metadata[claudeHandoffInternalMetadataKey] = true
	if isClaudeHandoffCandidate("claude", opts, claudeQuotaError("This request would exceed your account's rate limit", http.StatusTooManyRequests)) {
		t.Fatal("compactor request must not recursively trigger a Claude handoff")
	}
}

func TestIsClaudeHandoffCandidateRejectsUnstructuredOrWrongTypeBody(t *testing.T) {
	for _, body := range []string{
		`rate_limit_error: would exceed your account's rate limit`,
		`{"error":{"type":"api_error","message":"rate_limit_error: would exceed your account's rate limit"}}`,
		`{"error":{"type":"rate_limit_error","message":"Please try again later"},"debug":"would exceed your account's rate limit"}`,
	} {
		t.Run(body, func(t *testing.T) {
			err := &Error{HTTPStatus: http.StatusTooManyRequests, Message: body}
			if isClaudeHandoffCandidate("claude", claudeHandoffOptions(), err) {
				t.Fatalf("unstructured or unrelated error must not trigger compaction: %s", body)
			}
		})
	}
}

func TestIsClaudeHandoffCandidateReadsWrappedStructuredError(t *testing.T) {
	err := fmt.Errorf("execution failed: %w", claudeQuotaError("This request would exceed your account's rate limit. Please try again later.", http.StatusTooManyRequests))
	if !isClaudeHandoffCandidate("claude", claudeHandoffOptions(), err) {
		t.Fatal("wrapped upstream quota signal was lost")
	}
}

func TestIsClaudeHandoffCandidateReadsStreamBootstrapQuotaError(t *testing.T) {
	err := newStreamBootstrapError(claudeQuotaError("This request would exceed your account's rate limit. Please try again later.", http.StatusTooManyRequests), nil)
	if !isClaudeHandoffCandidate("claude", claudeHandoffOptions(), err) {
		t.Fatal("stream bootstrap quota signal was lost")
	}
}

func TestBuildCompactorPayloadUsesXHighAndPreservesTranscript(t *testing.T) {
	original := []byte(`{"model":"claude-opus-5-5","messages":[{"role":"user","content":"keep this fact"}]}`)
	payload, err := buildClaudeHandoffCompactorPayload(original, "gpt-6-luna", 4096)
	if err != nil {
		t.Fatalf("buildClaudeHandoffCompactorPayload() error = %v", err)
	}
	if got := gjson.GetBytes(payload, "model").String(); got != "gpt-6-luna" {
		t.Fatalf("model = %q, want gpt-6-luna", got)
	}
	if got := gjson.GetBytes(payload, "reasoning.effort").String(); got != "xhigh" {
		t.Fatalf("reasoning.effort = %q, want xhigh", got)
	}
	if got := gjson.GetBytes(payload, "max_output_tokens").Int(); got != 4096 {
		t.Fatalf("max_output_tokens = %d, want 4096", got)
	}
	if !strings.Contains(gjson.GetBytes(payload, "input.0.content.0.text").String(), string(original)) {
		t.Fatalf("compactor payload lost original Claude transcript: %s", payload)
	}
}

func TestRewriteClaudeHandoffPayloadPreservesEnvelopeAndDoesNotMutateInput(t *testing.T) {
	original := []byte(`{"model":"claude-opus-5-5","system":"system fact","tools":[{"name":"read_file"}],"stream":true,"metadata":{"request_id":"r1"},"messages":[{"role":"user","content":[{"type":"text","text":"old"}]},{"role":"assistant","content":[{"type":"text","text":"old answer"}]},{"role":"user","content":[{"type":"text","text":"latest"}]}]}`)
	before := append([]byte(nil), original...)
	rewritten, err := rewriteClaudeHandoffPayload(original, `{"goal":"continue","pending_tasks":["run tests"]}`, 2)
	if err != nil {
		t.Fatalf("rewriteClaudeHandoffPayload() error = %v", err)
	}
	if !reflect.DeepEqual(original, before) {
		t.Fatal("rewriteClaudeHandoffPayload mutated the caller's payload")
	}
	for _, path := range []string{"model", "system", "tools.0.name", "stream", "metadata.request_id"} {
		if got := gjson.GetBytes(rewritten, path).String(); got != gjson.GetBytes(original, path).String() {
			t.Fatalf("envelope field %s changed: got %q, want %q", path, got, gjson.GetBytes(original, path).String())
		}
	}
	messages := gjson.GetBytes(rewritten, "messages").Array()
	if len(messages) != 2 {
		t.Fatalf("rewritten messages = %d, want summary plus one recent message", len(messages))
	}
	if !strings.Contains(messages[0].Get("content.0.text").String(), claudeHandoffSummaryTag) || !strings.Contains(messages[0].Get("content.0.text").String(), "run tests") {
		t.Fatalf("summary marker/content missing from first message: %s", messages[0].Raw)
	}
	if messages[len(messages)-1].Get("role").String() != "user" {
		t.Fatalf("rewritten request ended with role %q, want user", messages[len(messages)-1].Get("role").String())
	}
	if messages[1].Get("content.0.text").String() != "latest" {
		t.Fatalf("recent tail message was not retained: %s", messages[1].Raw)
	}
}

func TestRewriteClaudeHandoffPayloadAppendsUserAfterAssistantTail(t *testing.T) {
	payload := []byte(`{"messages":[{"role":"user","content":"old"}]}`)
	rewritten, err := rewriteClaudeHandoffPayloadWithTail(payload, `{"goal":"continue"}`, []string{`{"role":"assistant","content":"latest answer"}`})
	if err != nil {
		t.Fatalf("rewriteClaudeHandoffPayload() error = %v", err)
	}
	messages := gjson.GetBytes(rewritten, "messages").Array()
	if len(messages) != 3 {
		t.Fatalf("rewritten messages = %d, want summary, assistant tail, continuation user", len(messages))
	}
	if messages[0].Get("role").String() != "user" || messages[1].Get("role").String() != "assistant" || messages[2].Get("role").String() != "user" {
		t.Fatalf("rewritten roles = %q,%q,%q, want user,assistant,user", messages[0].Get("role").String(), messages[1].Get("role").String(), messages[2].Get("role").String())
	}
	if !strings.Contains(messages[2].Get("content.0.text").String(), "Continue from the Claude handoff context") {
		t.Fatalf("continuation user message missing: %s", messages[2].Raw)
	}
}

func TestSanitizeClaudeHandoffMessagesDropsThinkingAndOrphanToolBlocks(t *testing.T) {
	payload := []byte(`{"messages":[
{"role":"user","content":[{"type":"text","text":"request"}]},
{"role":"assistant","content":[{"type":"thinking","thinking":"secret"},{"type":"tool_use","id":"orphan","name":"read_file","input":{}}]},
{"role":"user","content":[{"type":"tool_result","tool_use_id":"orphan","content":"file"},{"type":"tool_result","tool_use_id":"missing","content":"drop"}]},
{"role":"assistant","content":[{"type":"tool_use","id":"paired","name":"read_file","input":{}},{"type":"text","text":"done"}]},
{"role":"user","content":[{"type":"tool_result","tool_use_id":"paired","content":"file"}]}
]}`)
	result, err := rewriteClaudeHandoffPayload(payload, "summary", 3)
	if err != nil {
		t.Fatalf("rewriteClaudeHandoffPayload() error = %v", err)
	}
	text := string(result)
	if strings.Contains(text, "secret") || strings.Contains(text, "orphan") || strings.Contains(text, "missing") {
		t.Fatalf("sanitizer retained thinking/orphan tool block: %s", text)
	}
	if !strings.Contains(text, "paired") || !strings.Contains(text, "done") {
		t.Fatalf("sanitizer dropped paired tool cycle: %s", text)
	}
}

func TestSanitizeClaudeHandoffMessagesDropsUnfinishedToolUseAndThinking(t *testing.T) {
	payload := []byte(`{"messages":[
{"role":"user","content":[{"type":"text","text":"request"}]},
{"role":"assistant","content":[{"type":"thinking","thinking":"hidden"},{"type":"redacted_thinking","data":"opaque"},{"type":"tool_use","id":"pending","name":"read_file","input":{}},{"type":"text","text":"visible"}]},
{"role":"user","content":[{"type":"text","text":"latest"}]}
]}`)
	result, err := rewriteClaudeHandoffPayload(payload, "summary", 8)
	if err != nil {
		t.Fatalf("rewriteClaudeHandoffPayload() error = %v", err)
	}
	text := string(result)
	for _, unwanted := range []string{"hidden", "opaque", "pending"} {
		if strings.Contains(text, unwanted) {
			t.Fatalf("sanitizer retained %q: %s", unwanted, text)
		}
	}
	if !strings.Contains(text, "visible") || !strings.Contains(text, "latest") {
		t.Fatalf("sanitizer lost visible text: %s", text)
	}
}

func TestSanitizeClaudeHandoffMessageSlicePreservesPostBoundaryAssistantTurn(t *testing.T) {
	// A subsequent Claude request can begin with the assistant turn that
	// followed the compaction boundary. The post-boundary sanitizer must not
	// apply the initial "start at user" rule to this slice.
	payload := []byte(`{"messages":[
{"role":"assistant","content":[{"type":"text","text":"post-boundary answer"}]},
{"role":"user","content":[{"type":"text","text":"next request"}]}
]}`)
	messages := gjson.GetBytes(payload, "messages").Array()
	got := sanitizeClaudeHandoffMessageSlice(messages)
	if len(got) != 2 {
		t.Fatalf("sanitized post-boundary messages = %d, want 2", len(got))
	}
	if !strings.Contains(got[0], "post-boundary answer") || !strings.Contains(got[1], "next request") {
		t.Fatalf("post-boundary assistant/user turns were not preserved: %v", got)
	}
}

func TestApplyClaudeHandoffStateIsIdempotent(t *testing.T) {
	manager := NewManager(nil, nil, nil)
	manager.SetConfig(&internalconfig.Config{Routing: internalconfig.RoutingConfig{ClaudeHandoff: internalconfig.ClaudeHandoffConfig{Enabled: true, KeepRecentMessages: 1}}})
	manager.rememberClaudeHandoffSummary("session-test", `{"goal":"continue"}`)
	req := cliproxyexecutor.Request{Model: "claude-opus-5-5", Payload: []byte(`{"model":"claude-opus-5-5","messages":[{"role":"user","content":"latest"}]}`)}
	opts := claudeHandoffOptions()
	firstReq, firstOpts := manager.applyClaudeHandoffState(req, opts, []string{"claude"})
	if firstOpts.OriginalRequest == nil || !strings.Contains(gjson.GetBytes(firstReq.Payload, "messages.0.content.0.text").String(), claudeHandoffSummaryTag) {
		t.Fatalf("first application did not rewrite request: %s", firstReq.Payload)
	}
	secondReq, _ := manager.applyClaudeHandoffState(firstReq, firstOpts, []string{"claude"})
	if !reflect.DeepEqual(firstReq.Payload, secondReq.Payload) {
		t.Fatalf("Claude handoff rewrite is not idempotent:\nfirst=%s\nsecond=%s", firstReq.Payload, secondReq.Payload)
	}
}

func TestApplyClaudeHandoffStateKeepsTurnsAddedAfterBoundary(t *testing.T) {
	manager := NewManager(nil, nil, nil)
	manager.SetConfig(&internalconfig.Config{Routing: internalconfig.RoutingConfig{ClaudeHandoff: internalconfig.ClaudeHandoffConfig{Enabled: true, KeepRecentMessages: 1}}})
	basePayload := []byte(`{"model":"claude-opus-5-5","messages":[
{"role":"user","content":"before"},
{"role":"assistant","content":"answer"},
{"role":"user","content":"boundary"}
]}`)
	manager.rememberClaudeHandoffSummaryForPayload("session-test", `{"goal":"continue"}`, basePayload)
	nextPayload := []byte(`{"model":"claude-opus-5-5","messages":[
{"role":"user","content":"before"},
{"role":"assistant","content":"answer"},
{"role":"user","content":"boundary"},
{"role":"assistant","content":"post-boundary answer"},
{"role":"user","content":"new request"}
]}`)
	req, _ := manager.applyClaudeHandoffState(cliproxyexecutor.Request{Model: "claude-opus-5-5", Payload: nextPayload}, claudeHandoffOptions(), []string{"claude"})
	messages := gjson.GetBytes(req.Payload, "messages").Array()
	if len(messages) != 4 {
		t.Fatalf("rewritten message count = %d, want summary + boundary and 2 post-boundary turns", len(messages))
	}
	text := string(req.Payload)
	for _, want := range []string{"boundary", "post-boundary answer", "new request"} {
		if !strings.Contains(text, want) {
			t.Fatalf("post-boundary turn %q was dropped: %s", want, text)
		}
	}
	if messages[len(messages)-1].Get("role").String() != "user" {
		t.Fatalf("rewritten message tail role = %q, want user", messages[len(messages)-1].Get("role").String())
	}
	if strings.Contains(text, `"before"`) || strings.Contains(text, `"answer"`) {
		t.Fatalf("pre-boundary transcript was replayed after summary: %s", text)
	}
}

func TestClaudeHandoffStateKeepsTurnsAddedAfterBoundary(t *testing.T) {
	manager := NewManager(nil, nil, nil)
	manager.SetConfig(&internalconfig.Config{Routing: internalconfig.RoutingConfig{ClaudeHandoff: internalconfig.ClaudeHandoffConfig{Enabled: true, KeepRecentMessages: 1}}})
	original := []byte(`{"model":"claude-opus-5-5","messages":[{"role":"user","content":"old request"},{"role":"assistant","content":"old answer"},{"role":"user","content":"current request"}]}`)
	manager.rememberClaudeHandoffSummaryForPayload("session-test", `{"goal":"continue"}`, original)
	updated := []byte(`{"model":"claude-opus-5-5","messages":[{"role":"user","content":"old request"},{"role":"assistant","content":"old answer"},{"role":"user","content":"current request"},{"role":"assistant","content":"current answer"},{"role":"user","content":"new request"}]}`)
	rewritten, opts := manager.applyClaudeHandoffState(cliproxyexecutor.Request{Model: "claude-opus-5-5", Payload: updated}, claudeHandoffOptions(), []string{"claude"})
	if opts.OriginalRequest == nil {
		t.Fatal("applyClaudeHandoffState did not update OriginalRequest")
	}
	text := string(rewritten.Payload)
	for _, want := range []string{"current request", "current answer", "new request"} {
		if !strings.Contains(text, want) {
			t.Fatalf("post-boundary turn %q was lost: %s", want, text)
		}
	}
	if strings.Contains(text, "old request") || strings.Contains(text, "old answer") {
		t.Fatalf("pre-boundary transcript leaked into rewritten payload: %s", text)
	}
}

func TestRewriteClaudeHandoffPayloadRejectsMissingMessages(t *testing.T) {
	if _, err := rewriteClaudeHandoffPayload([]byte(`{"model":"claude-opus-5-5"}`), "summary", 1); err == nil {
		t.Fatal("rewriteClaudeHandoffPayload accepted request without messages array")
	}
}

func TestClaudeHandoffSummaryTagIsValidJSONContent(t *testing.T) {
	payload, err := rewriteClaudeHandoffPayload([]byte(`{"messages":[{"role":"user","content":"latest"}]}`), `{"goal":"x"}`, 1)
	if err != nil {
		t.Fatalf("rewriteClaudeHandoffPayload() error = %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("rewritten payload is invalid JSON: %v", err)
	}
}

func TestClaudeHandoffConfigDefaultsDoNotMutateInput(t *testing.T) {
	cfg := &internalconfig.Config{Routing: internalconfig.RoutingConfig{ClaudeHandoff: internalconfig.ClaudeHandoffConfig{Enabled: true}}}
	before := cfg.Routing.ClaudeHandoff
	got := claudeHandoffConfig(cfg)
	if !reflect.DeepEqual(before, cfg.Routing.ClaudeHandoff) {
		t.Fatal("claudeHandoffConfig mutated the runtime configuration")
	}
	if got.CompactorProvider != "codex" || got.CompactorModel != "gpt-6-luna" || got.CompactorMaxOutputTokens != 8192 || got.KeepRecentMessages != 8 || got.MaxSummaryBytes <= 0 {
		t.Fatalf("claudeHandoffConfig defaults = %+v", got)
	}
}

func TestIsTemporaryClaudeHandoffError(t *testing.T) {
	for _, tt := range []struct {
		name string
		err  error
		want bool
	}{
		{name: "nil", want: false},
		{name: "canceled", err: context.Canceled, want: false},
		{name: "deadline", err: context.DeadlineExceeded, want: false},
		{name: "transport", err: fmt.Errorf("connection reset"), want: true},
		{name: "rate limit", err: &Error{HTTPStatus: http.StatusTooManyRequests}, want: true},
		{name: "bad gateway", err: &Error{HTTPStatus: http.StatusBadGateway}, want: true},
		{name: "service unavailable", err: &Error{HTTPStatus: http.StatusServiceUnavailable}, want: true},
		{name: "authentication", err: &Error{HTTPStatus: http.StatusUnauthorized}, want: false},
		{name: "invalid model", err: &Error{HTTPStatus: http.StatusNotFound}, want: false},
		{name: "no compactor auth", err: &Error{Code: "auth_not_found", Message: "no auth available"}, want: false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := isTemporaryClaudeHandoffError(tt.err); got != tt.want {
				t.Fatalf("isTemporaryClaudeHandoffError() = %v, want %v", got, tt.want)
			}
		})
	}
}

type claudeHandoffRetryExecutor struct {
	calls atomic.Int32
}

func (*claudeHandoffRetryExecutor) Identifier() string { return "codex" }
func (*claudeHandoffRetryExecutor) Refresh(_ context.Context, auth *Auth) (*Auth, error) {
	return auth, nil
}
func (*claudeHandoffRetryExecutor) CountTokens(context.Context, *Auth, cliproxyexecutor.Request, cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	return cliproxyexecutor.Response{}, &Error{HTTPStatus: http.StatusNotImplemented, Message: "not used"}
}
func (*claudeHandoffRetryExecutor) ExecuteStream(context.Context, *Auth, cliproxyexecutor.Request, cliproxyexecutor.Options) (*cliproxyexecutor.StreamResult, error) {
	return nil, &Error{HTTPStatus: http.StatusNotImplemented, Message: "not used"}
}
func (*claudeHandoffRetryExecutor) HttpRequest(context.Context, *Auth, *http.Request) (*http.Response, error) {
	return nil, &Error{HTTPStatus: http.StatusNotImplemented, Message: "not used"}
}
func (e *claudeHandoffRetryExecutor) Execute(context.Context, *Auth, cliproxyexecutor.Request, cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	if e.calls.Add(1) == 1 {
		return cliproxyexecutor.Response{}, &Error{HTTPStatus: http.StatusServiceUnavailable, Message: "temporary compactor outage"}
	}
	return cliproxyexecutor.Response{Payload: []byte(`{"output":[{"content":[{"text":"{\"goal\":\"continue\"}"}]}]}`)}, nil
}

func TestCompactClaudeContextRetriesTemporaryCompactorFailure(t *testing.T) {
	manager := NewManager(nil, nil, nil)
	executor := &claudeHandoffRetryExecutor{}
	manager.RegisterExecutor(executor)
	model := "gpt-6-luna"
	authID := "claude-handoff-compactor-auth"
	registry.GetGlobalRegistry().RegisterClient(authID, "codex", []*registry.ModelInfo{{ID: model}})
	t.Cleanup(func() { registry.GetGlobalRegistry().UnregisterClient(authID) })
	if _, err := manager.Register(context.Background(), &Auth{ID: authID, Provider: "codex", Status: StatusActive, Metadata: map[string]any{"disable_cooling": true}}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}
	manager.SetConfig(&internalconfig.Config{Routing: internalconfig.RoutingConfig{ClaudeHandoff: internalconfig.ClaudeHandoffConfig{
		Enabled: true, CompactorProvider: "codex", CompactorModel: model, CompactorRetryMaxInterval: 1,
	}}})
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	summary, err := manager.compactClaudeHandoffContext(ctx, []byte(`{"messages":[{"role":"user","content":"keep"}]}`), claudeHandoffOptions(), claudeHandoffConfig(manager.runtimeConfigSnapshot()))
	if err != nil {
		t.Fatalf("compactClaudeHandoffContext() error = %v", err)
	}
	if summary != `{"goal":"continue"}` {
		t.Fatalf("summary = %q", summary)
	}
	if got := executor.calls.Load(); got != 2 {
		t.Fatalf("compactor calls = %d, want temporary failure plus success", got)
	}
}

type claudeHandoffFlowExecutor struct{}

func (e *claudeHandoffFlowExecutor) Identifier() string { return "claude" }
func (e *claudeHandoffFlowExecutor) Refresh(_ context.Context, auth *Auth) (*Auth, error) {
	return auth, nil
}
func (e *claudeHandoffFlowExecutor) CountTokens(context.Context, *Auth, cliproxyexecutor.Request, cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	return cliproxyexecutor.Response{}, &Error{HTTPStatus: http.StatusNotImplemented, Message: "not used"}
}
func (e *claudeHandoffFlowExecutor) ExecuteStream(_ context.Context, auth *Auth, req cliproxyexecutor.Request, _ cliproxyexecutor.Options) (*cliproxyexecutor.StreamResult, error) {
	chunks := make(chan cliproxyexecutor.StreamChunk, 1)
	switch auth.ID {
	case "claude-a":
		chunks <- cliproxyexecutor.StreamChunk{Err: claudeQuotaError("This request would exceed your account's rate limit. Please try again later.", http.StatusTooManyRequests)}
	case "claude-b":
		if !isAlreadyClaudeHandoffCompacted(req.Payload) {
			chunks <- cliproxyexecutor.StreamChunk{Err: &Error{HTTPStatus: http.StatusBadRequest, Message: "stream Claude handoff envelope missing"}}
		} else {
			chunks <- cliproxyexecutor.StreamChunk{Payload: []byte(`{"ok":true}`)}
		}
	default:
		return nil, &Error{HTTPStatus: http.StatusNotFound, Message: "unexpected Claude auth"}
	}
	close(chunks)
	return &cliproxyexecutor.StreamResult{Chunks: chunks}, nil
}
func (e *claudeHandoffFlowExecutor) HttpRequest(context.Context, *Auth, *http.Request) (*http.Response, error) {
	return nil, &Error{HTTPStatus: http.StatusNotImplemented, Message: "not used"}
}
func (e *claudeHandoffFlowExecutor) Execute(_ context.Context, auth *Auth, req cliproxyexecutor.Request, _ cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	if auth.ID == "claude-a" {
		return cliproxyexecutor.Response{}, claudeQuotaError("This request would exceed your account's rate limit. Please try again later.", http.StatusTooManyRequests)
	}
	if auth.ID == "claude-b" {
		if !isAlreadyClaudeHandoffCompacted(req.Payload) {
			return cliproxyexecutor.Response{}, &Error{HTTPStatus: http.StatusBadRequest, Message: "Claude handoff envelope missing: " + string(req.Payload)}
		}
		return cliproxyexecutor.Response{Payload: []byte(`{"ok":true}`)}, nil
	}
	return cliproxyexecutor.Response{}, &Error{HTTPStatus: http.StatusNotFound, Message: "unexpected Claude auth"}
}

type claudeHandoffFlowCompactorExecutor struct{}

func (e *claudeHandoffFlowCompactorExecutor) Identifier() string { return "codex" }
func (e *claudeHandoffFlowCompactorExecutor) Refresh(_ context.Context, auth *Auth) (*Auth, error) {
	return auth, nil
}
func (e *claudeHandoffFlowCompactorExecutor) CountTokens(context.Context, *Auth, cliproxyexecutor.Request, cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	return cliproxyexecutor.Response{}, &Error{HTTPStatus: http.StatusNotImplemented, Message: "not used"}
}
func (e *claudeHandoffFlowCompactorExecutor) ExecuteStream(context.Context, *Auth, cliproxyexecutor.Request, cliproxyexecutor.Options) (*cliproxyexecutor.StreamResult, error) {
	return nil, &Error{HTTPStatus: http.StatusNotImplemented, Message: "not used"}
}
func (e *claudeHandoffFlowCompactorExecutor) HttpRequest(context.Context, *Auth, *http.Request) (*http.Response, error) {
	return nil, &Error{HTTPStatus: http.StatusNotImplemented, Message: "not used"}
}
func (e *claudeHandoffFlowCompactorExecutor) Execute(_ context.Context, _ *Auth, req cliproxyexecutor.Request, _ cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	if gjson.GetBytes(req.Payload, "reasoning.effort").String() != "xhigh" {
		return cliproxyexecutor.Response{}, &Error{HTTPStatus: http.StatusBadRequest, Message: "compactor reasoning was not xhigh"}
	}
	return cliproxyexecutor.Response{Payload: []byte(`{"output":[{"content":[{"text":"{\"goal\":\"continue\",\"pending_tasks\":[\"finish request\"]}"}]}]}`)}, nil
}

type claudeHandoffPayloadRecordingCompactorExecutor struct {
	payload  []byte
	metadata map[string]any
	calls    atomic.Int32
}

func (e *claudeHandoffPayloadRecordingCompactorExecutor) Identifier() string { return "codex" }
func (e *claudeHandoffPayloadRecordingCompactorExecutor) Refresh(_ context.Context, auth *Auth) (*Auth, error) {
	return auth, nil
}
func (*claudeHandoffPayloadRecordingCompactorExecutor) CountTokens(context.Context, *Auth, cliproxyexecutor.Request, cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	return cliproxyexecutor.Response{}, &Error{HTTPStatus: http.StatusNotImplemented, Message: "not used"}
}
func (*claudeHandoffPayloadRecordingCompactorExecutor) ExecuteStream(context.Context, *Auth, cliproxyexecutor.Request, cliproxyexecutor.Options) (*cliproxyexecutor.StreamResult, error) {
	return nil, &Error{HTTPStatus: http.StatusNotImplemented, Message: "not used"}
}
func (*claudeHandoffPayloadRecordingCompactorExecutor) HttpRequest(context.Context, *Auth, *http.Request) (*http.Response, error) {
	return nil, &Error{HTTPStatus: http.StatusNotImplemented, Message: "not used"}
}
func (e *claudeHandoffPayloadRecordingCompactorExecutor) Execute(_ context.Context, _ *Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	e.calls.Add(1)
	e.payload = append(e.payload[:0], req.Payload...)
	e.metadata = opts.Metadata
	return cliproxyexecutor.Response{Payload: []byte(`{"output":[{"content":[{"text":"{\"goal\":\"continue\"}"}]}]}`)}, nil
}

func TestPerformClaudeHandoffCompactsCurrentPayloadInsteadOfOriginalRequest(t *testing.T) {
	manager := NewManager(nil, nil, nil)
	recorder := &claudeHandoffPayloadRecordingCompactorExecutor{}
	manager.RegisterExecutor(recorder)
	manager.SetConfig(&internalconfig.Config{Routing: internalconfig.RoutingConfig{ClaudeHandoff: internalconfig.ClaudeHandoffConfig{
		Enabled: true, CompactorProvider: "codex", CompactorModel: "gpt-6-luna",
	}}})
	const authID = "claude-handoff-current-payload-compactor-auth"
	registry.GetGlobalRegistry().RegisterClient(authID, "codex", []*registry.ModelInfo{{ID: "gpt-6-luna"}})
	t.Cleanup(func() { registry.GetGlobalRegistry().UnregisterClient(authID) })
	if _, err := manager.Register(context.Background(), &Auth{ID: authID, Provider: "codex", Status: StatusActive, Metadata: map[string]any{"disable_cooling": true}}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	oversizedOriginal := []byte(`{"messages":[{"role":"user","content":"oversized-` + strings.Repeat("x", 512*1024) + `"}]}`)
	currentPayload := []byte(`{"model":"claude-opus-5-5","messages":[{"role":"user","content":"current request"}]}`)
	opts := claudeHandoffOptions()
	opts.OriginalRequest = oversizedOriginal
	opts.Metadata[cliproxyexecutor.CanonicalSessionIDMetadataKey] = "claude-current-payload-session"
	opts.Metadata[cliproxyexecutor.RequestedModelMetadataKey] = "claude-opus-5-5-xhigh"
	opts.Metadata[cliproxyexecutor.AuthSelectionModelMetadataKey] = "claude-opus-5-5-xhigh"
	if _, err := manager.performClaudeHandoff(context.Background(), cliproxyexecutor.Request{Model: "claude-opus-5-5", Payload: currentPayload}, opts, claudeQuotaError("quota", http.StatusTooManyRequests)); err != nil {
		t.Fatalf("performClaudeHandoff() error = %v", err)
	}

	compactorText := gjson.GetBytes(recorder.payload, "input.0.content.0.text").String()
	if !strings.Contains(compactorText, "current request") {
		t.Fatalf("compactor did not receive current payload: %s", compactorText)
	}
	if strings.Contains(compactorText, "oversized-") || len(recorder.payload) >= len(oversizedOriginal) {
		t.Fatalf("compactor received oversized OriginalRequest: payload bytes=%d original bytes=%d", len(recorder.payload), len(oversizedOriginal))
	}
	if got := recorder.metadata[cliproxyexecutor.RequestedModelMetadataKey]; got != "gpt-6-luna" {
		t.Fatalf("compactor requested model metadata = %#v, want gpt-6-luna", got)
	}
	if got := recorder.metadata[cliproxyexecutor.AuthSelectionModelMetadataKey]; got != "gpt-6-luna" {
		t.Fatalf("compactor auth selection metadata = %#v, want gpt-6-luna", got)
	}
	if got := recorder.metadata[claudeHandoffInternalMetadataKey]; got != true {
		t.Fatalf("compactor internal metadata = %#v, want true", got)
	}
}

type claudeHandoffFlowSelector struct{}

func (claudeHandoffFlowSelector) Pick(_ context.Context, _ string, _ string, _ cliproxyexecutor.Options, auths []*Auth) (*Auth, error) {
	var selected *Auth
	for _, auth := range auths {
		if auth != nil && (selected == nil || auth.ID < selected.ID) {
			selected = auth
		}
	}
	if selected == nil {
		return nil, &Error{Code: "auth_not_found", Message: "no auth"}
	}
	return selected, nil
}

func TestExecuteClaudeQuotaCompactsBeforeSelectingNextOAuthAccount(t *testing.T) {
	manager := NewManager(nil, claudeHandoffFlowSelector{}, nil)
	manager.SetRetryConfig(0, 0, 0)
	manager.RegisterExecutor(&claudeHandoffFlowExecutor{})
	manager.RegisterExecutor(&claudeHandoffFlowCompactorExecutor{})
	manager.SetConfig(&internalconfig.Config{Routing: internalconfig.RoutingConfig{ClaudeHandoff: internalconfig.ClaudeHandoffConfig{
		Enabled: true, CompactorProvider: "codex", CompactorModel: "gpt-6-luna", KeepRecentMessages: 2,
	}}})
	for _, entry := range []struct {
		id       string
		provider string
		model    string
	}{
		{id: "claude-a", provider: "claude", model: "claude-opus-5-5"},
		{id: "claude-b", provider: "claude", model: "claude-opus-5-5"},
		{id: "codex-a", provider: "codex", model: "gpt-6-luna"},
	} {
		registry.GetGlobalRegistry().RegisterClient(entry.id, entry.provider, []*registry.ModelInfo{{ID: entry.model}})
		entryID := entry.id
		t.Cleanup(func() { registry.GetGlobalRegistry().UnregisterClient(entryID) })
		attributes := map[string]string{}
		if entry.provider == "claude" {
			attributes["auth_kind"] = "oauth"
		}
		if _, err := manager.Register(context.Background(), &Auth{ID: entry.id, Provider: entry.provider, Status: StatusActive, Attributes: attributes, Metadata: map[string]any{"disable_cooling": true}}); err != nil {
			t.Fatalf("Register(%s) error = %v", entry.id, err)
		}
	}
	payload := []byte(`{"model":"claude-opus-5-5","messages":[{"role":"user","content":"work on this"}]}`)
	resp, err := manager.Execute(context.Background(), []string{"claude"}, cliproxyexecutor.Request{Model: "claude-opus-5-5", Payload: payload}, cliproxyexecutor.Options{
		SourceFormat:    sdktranslator.FormatClaude,
		OriginalRequest: payload,
		Metadata: map[string]any{
			cliproxyexecutor.CanonicalSessionIDMetadataKey: "claude-handoff-flow-session",
		},
	})
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if string(resp.Payload) != `{"ok":true}` {
		t.Fatalf("final response = %s", resp.Payload)
	}
}

type claudeHandoffThreeAccountExecutor struct {
	nonStreamPayloads map[string][]byte
	streamPayloads    map[string][]byte
}

func (e *claudeHandoffThreeAccountExecutor) Identifier() string { return "claude" }
func (e *claudeHandoffThreeAccountExecutor) Refresh(_ context.Context, auth *Auth) (*Auth, error) {
	return auth, nil
}
func (*claudeHandoffThreeAccountExecutor) CountTokens(context.Context, *Auth, cliproxyexecutor.Request, cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	return cliproxyexecutor.Response{}, &Error{HTTPStatus: http.StatusNotImplemented, Message: "not used"}
}
func (e *claudeHandoffThreeAccountExecutor) ExecuteStream(_ context.Context, auth *Auth, req cliproxyexecutor.Request, _ cliproxyexecutor.Options) (*cliproxyexecutor.StreamResult, error) {
	if e.streamPayloads == nil {
		e.streamPayloads = make(map[string][]byte)
	}
	e.streamPayloads[auth.ID] = append([]byte(nil), req.Payload...)
	chunks := make(chan cliproxyexecutor.StreamChunk, 1)
	switch auth.ID {
	case "claude-a", "claude-b":
		chunks <- cliproxyexecutor.StreamChunk{Err: claudeQuotaError("This request would exceed your account's rate limit. Please try again later.", http.StatusTooManyRequests)}
	case "claude-c":
		if !isAlreadyClaudeHandoffCompacted(req.Payload) {
			chunks <- cliproxyexecutor.StreamChunk{Err: &Error{HTTPStatus: http.StatusBadRequest, Message: "stream handoff payload missing"}}
		} else {
			chunks <- cliproxyexecutor.StreamChunk{Payload: []byte(`{"ok":true}`)}
		}
	default:
		chunks <- cliproxyexecutor.StreamChunk{Err: &Error{HTTPStatus: http.StatusNotFound, Message: "unexpected Claude auth"}}
	}
	close(chunks)
	return &cliproxyexecutor.StreamResult{Chunks: chunks}, nil
}
func (*claudeHandoffThreeAccountExecutor) HttpRequest(context.Context, *Auth, *http.Request) (*http.Response, error) {
	return nil, &Error{HTTPStatus: http.StatusNotImplemented, Message: "not used"}
}
func (e *claudeHandoffThreeAccountExecutor) Execute(_ context.Context, auth *Auth, req cliproxyexecutor.Request, _ cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	if e.nonStreamPayloads == nil {
		e.nonStreamPayloads = make(map[string][]byte)
	}
	e.nonStreamPayloads[auth.ID] = append([]byte(nil), req.Payload...)
	switch auth.ID {
	case "claude-a", "claude-b":
		return cliproxyexecutor.Response{}, claudeQuotaError("This request would exceed your account's rate limit. Please try again later.", http.StatusTooManyRequests)
	case "claude-c":
		if !isAlreadyClaudeHandoffCompacted(req.Payload) {
			return cliproxyexecutor.Response{}, &Error{HTTPStatus: http.StatusBadRequest, Message: "handoff payload missing"}
		}
		return cliproxyexecutor.Response{Payload: []byte(`{"ok":true}`)}, nil
	default:
		return cliproxyexecutor.Response{}, &Error{HTTPStatus: http.StatusNotFound, Message: "unexpected Claude auth"}
	}
}

func registerThreeClaudeHandoffAccounts(t *testing.T, manager *Manager, compactor *claudeHandoffPayloadRecordingCompactorExecutor) *claudeHandoffThreeAccountExecutor {
	t.Helper()
	executor := &claudeHandoffThreeAccountExecutor{}
	manager.RegisterExecutor(executor)
	manager.RegisterExecutor(compactor)
	for _, entry := range []struct {
		id       string
		provider string
		model    string
	}{
		{id: "claude-a", provider: "claude", model: "claude-opus-5-5"},
		{id: "claude-b", provider: "claude", model: "claude-opus-5-5"},
		{id: "claude-c", provider: "claude", model: "claude-opus-5-5"},
		{id: "codex-a", provider: "codex", model: "gpt-6-luna"},
	} {
		registry.GetGlobalRegistry().RegisterClient(entry.id, entry.provider, []*registry.ModelInfo{{ID: entry.model}})
		entryID := entry.id
		t.Cleanup(func() { registry.GetGlobalRegistry().UnregisterClient(entryID) })
		attributes := map[string]string{}
		if entry.provider == "claude" {
			attributes["auth_kind"] = "oauth"
		}
		if _, err := manager.Register(context.Background(), &Auth{ID: entry.id, Provider: entry.provider, Status: StatusActive, Attributes: attributes, Metadata: map[string]any{"disable_cooling": true}}); err != nil {
			t.Fatalf("Register(%s) error = %v", entry.id, err)
		}
	}
	return executor
}

func claudeHandoffThreeAccountConfig() *internalconfig.Config {
	return &internalconfig.Config{Routing: internalconfig.RoutingConfig{ClaudeHandoff: internalconfig.ClaudeHandoffConfig{
		Enabled: true, CompactorProvider: "codex", CompactorModel: "gpt-6-luna", KeepRecentMessages: 2,
	}}}
}

func TestExecuteClaudeQuotaCompactsOnceAcrossTwoFailures(t *testing.T) {
	manager := NewManager(nil, claudeHandoffFlowSelector{}, nil)
	manager.SetRetryConfig(0, 0, 0)
	compactor := &claudeHandoffPayloadRecordingCompactorExecutor{}
	executor := registerThreeClaudeHandoffAccounts(t, manager, compactor)
	manager.SetConfig(claudeHandoffThreeAccountConfig())
	payload := []byte(`{"model":"claude-opus-5-5","messages":[{"role":"user","content":"work"}]}`)
	resp, err := manager.Execute(context.Background(), []string{"claude"}, cliproxyexecutor.Request{Model: "claude-opus-5-5", Payload: payload}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FormatClaude, OriginalRequest: payload,
		Metadata: map[string]any{cliproxyexecutor.CanonicalSessionIDMetadataKey: "claude-three-account-nonstream"},
	})
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if string(resp.Payload) != `{"ok":true}` {
		t.Fatalf("response = %s", resp.Payload)
	}
	if got := compactor.calls.Load(); got != 1 {
		t.Fatalf("compactor calls = %d, want exactly one", got)
	}
	for _, authID := range []string{"claude-b", "claude-c"} {
		if !isAlreadyClaudeHandoffCompacted(executor.nonStreamPayloads[authID]) {
			t.Fatalf("%s did not receive compacted payload", authID)
		}
	}
}

func TestExecuteStreamClaudeQuotaCompactsOnceAcrossTwoFailures(t *testing.T) {
	manager := NewManager(nil, claudeHandoffFlowSelector{}, nil)
	manager.SetRetryConfig(0, 0, 0)
	compactor := &claudeHandoffPayloadRecordingCompactorExecutor{}
	executor := &claudeHandoffThreeAccountExecutor{}
	manager.RegisterExecutor(executor)
	manager.RegisterExecutor(compactor)
	for _, entry := range []struct{ id, provider, model string }{
		{id: "claude-a", provider: "claude", model: "claude-opus-5-5"},
		{id: "claude-b", provider: "claude", model: "claude-opus-5-5"},
		{id: "claude-c", provider: "claude", model: "claude-opus-5-5"},
		{id: "codex-a", provider: "codex", model: "gpt-6-luna"},
	} {
		registry.GetGlobalRegistry().RegisterClient(entry.id, entry.provider, []*registry.ModelInfo{{ID: entry.model}})
		entryID := entry.id
		t.Cleanup(func() { registry.GetGlobalRegistry().UnregisterClient(entryID) })
		attributes := map[string]string{}
		if entry.provider == "claude" {
			attributes["auth_kind"] = "oauth"
		}
		if _, err := manager.Register(context.Background(), &Auth{ID: entry.id, Provider: entry.provider, Status: StatusActive, Attributes: attributes, Metadata: map[string]any{"disable_cooling": true}}); err != nil {
			t.Fatalf("Register(%s) error = %v", entry.id, err)
		}
	}
	manager.SetConfig(claudeHandoffThreeAccountConfig())
	payload := []byte(`{"model":"claude-opus-5-5","stream":true,"messages":[{"role":"user","content":"work"}]}`)
	stream, err := manager.ExecuteStream(context.Background(), []string{"claude"}, cliproxyexecutor.Request{Model: "claude-opus-5-5", Payload: payload}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FormatClaude, OriginalRequest: payload, Stream: true,
		Metadata: map[string]any{cliproxyexecutor.CanonicalSessionIDMetadataKey: "claude-three-account-stream"},
	})
	if err != nil {
		t.Fatalf("ExecuteStream() error = %v", err)
	}
	var received []byte
	for chunk := range stream.Chunks {
		if chunk.Err != nil {
			t.Fatalf("stream chunk error = %v", chunk.Err)
		}
		received = append(received, chunk.Payload...)
	}
	if string(received) != `{"ok":true}` {
		t.Fatalf("stream response = %s", received)
	}
	if got := compactor.calls.Load(); got != 1 {
		t.Fatalf("compactor calls = %d, want exactly one", got)
	}
	for _, authID := range []string{"claude-b", "claude-c"} {
		if !isAlreadyClaudeHandoffCompacted(executor.streamPayloads[authID]) {
			t.Fatalf("%s did not receive compacted stream payload", authID)
		}
	}
}

func TestExecuteOpenAIRouteToClaudeCompactsOnQuota(t *testing.T) {
	manager := NewManager(nil, claudeHandoffFlowSelector{}, nil)
	manager.SetRetryConfig(0, 0, 0)
	manager.RegisterExecutor(&claudeHandoffFlowExecutor{})
	manager.RegisterExecutor(&claudeHandoffFlowCompactorExecutor{})
	for _, entry := range []struct {
		id       string
		provider string
		model    string
	}{
		{id: "claude-a", provider: "claude", model: "claude-opus-5-5"},
		{id: "claude-b", provider: "claude", model: "claude-opus-5-5"},
		{id: "codex-a", provider: "codex", model: "gpt-6-luna"},
	} {
		registry.GetGlobalRegistry().RegisterClient(entry.id, entry.provider, []*registry.ModelInfo{{ID: entry.model}})
		entryID := entry.id
		t.Cleanup(func() { registry.GetGlobalRegistry().UnregisterClient(entryID) })
		attributes := map[string]string{}
		if entry.provider == "claude" {
			attributes["auth_kind"] = "oauth"
		}
		if _, err := manager.Register(context.Background(), &Auth{ID: entry.id, Provider: entry.provider, Status: StatusActive, Attributes: attributes, Metadata: map[string]any{"disable_cooling": true}}); err != nil {
			t.Fatalf("Register(%s) error = %v", entry.id, err)
		}
	}
	manager.SetConfig(&internalconfig.Config{Routing: internalconfig.RoutingConfig{ClaudeHandoff: internalconfig.ClaudeHandoffConfig{
		Enabled: true, CompactorProvider: "codex", CompactorModel: "gpt-6-luna", KeepRecentMessages: 2,
	}}})
	payload := []byte(`{"model":"claude-opus-5-5","messages":[{"role":"user","content":"work"}]}`)
	resp, err := manager.Execute(context.Background(), []string{"claude"}, cliproxyexecutor.Request{Model: "claude-opus-5-5", Payload: payload}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FormatOpenAI, OriginalRequest: payload,
		Metadata: map[string]any{cliproxyexecutor.CanonicalSessionIDMetadataKey: "openai-to-claude-session"},
	})
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if string(resp.Payload) != `{"ok":true}` {
		t.Fatalf("response = %s", resp.Payload)
	}
}

// A model-fallback plugin can enter the auth manager with a non-Claude source
// format while its selected fallback model resolves to the Claude provider.
// A historical transcript must be compacted before the first Claude attempt,
// and that single envelope must be reused after further Claude quota failures.
func TestExecuteOpenAIRouteToClaudeCompactsHistoricalPayloadOnce(t *testing.T) {
	manager := NewManager(nil, claudeHandoffFlowSelector{}, nil)
	manager.SetRetryConfig(0, 0, 0)
	compactor := &claudeHandoffPayloadRecordingCompactorExecutor{}
	executor := registerThreeClaudeHandoffAccounts(t, manager, compactor)
	manager.SetConfig(claudeHandoffThreeAccountConfig())
	payload := []byte(`{"model":"claude-opus-5-5","messages":[{"role":"user","content":"old request"},{"role":"assistant","content":"old answer"},{"role":"user","content":"latest request"}]}`)
	resp, err := manager.Execute(context.Background(), []string{"claude"}, cliproxyexecutor.Request{Model: "claude-opus-5-5", Payload: payload}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FormatOpenAI, OriginalRequest: payload,
		Metadata: map[string]any{cliproxyexecutor.CanonicalSessionIDMetadataKey: "openai-to-claude-history-session"},
	})
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if string(resp.Payload) != `{"ok":true}` {
		t.Fatalf("response = %s", resp.Payload)
	}
	if got := compactor.calls.Load(); got != 1 {
		t.Fatalf("compactor calls = %d, want exactly one", got)
	}
	for _, authID := range []string{"claude-a", "claude-b", "claude-c"} {
		if !isAlreadyClaudeHandoffCompacted(executor.nonStreamPayloads[authID]) {
			t.Fatalf("%s did not receive the compacted Claude envelope", authID)
		}
	}
}

func TestExecuteStreamOpenAIRouteToClaudeCompactsHistoricalPayloadOnce(t *testing.T) {
	manager := NewManager(nil, claudeHandoffFlowSelector{}, nil)
	manager.SetRetryConfig(0, 0, 0)
	compactor := &claudeHandoffPayloadRecordingCompactorExecutor{}
	executor := registerThreeClaudeHandoffAccounts(t, manager, compactor)
	manager.SetConfig(claudeHandoffThreeAccountConfig())
	payload := []byte(`{"model":"claude-opus-5-5","stream":true,"messages":[{"role":"user","content":"old request"},{"role":"assistant","content":"old answer"},{"role":"user","content":"latest request"}]}`)
	stream, err := manager.ExecuteStream(context.Background(), []string{"claude"}, cliproxyexecutor.Request{Model: "claude-opus-5-5", Payload: payload}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FormatOpenAI, OriginalRequest: payload, Stream: true,
		Metadata: map[string]any{cliproxyexecutor.CanonicalSessionIDMetadataKey: "openai-to-claude-history-stream-session"},
	})
	if err != nil {
		t.Fatalf("ExecuteStream() error = %v", err)
	}
	var received []byte
	for chunk := range stream.Chunks {
		if chunk.Err != nil {
			t.Fatalf("stream chunk error = %v", chunk.Err)
		}
		received = append(received, chunk.Payload...)
	}
	if string(received) != `{"ok":true}` {
		t.Fatalf("stream response = %s", received)
	}
	if got := compactor.calls.Load(); got != 1 {
		t.Fatalf("compactor calls = %d, want exactly one", got)
	}
	for _, authID := range []string{"claude-a", "claude-b", "claude-c"} {
		if !isAlreadyClaudeHandoffCompacted(executor.streamPayloads[authID]) {
			t.Fatalf("%s did not receive the compacted Claude stream envelope", authID)
		}
	}
}

func TestClaudeHandoffRouteEligibilityRequiresMessagesPayloadForNonClaudeSource(t *testing.T) {
	providers := []string{"claude"}
	if !claudeHandoffRouteEligible(cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatOpenAI}, providers, []byte(`{"messages":[{"role":"user","content":"x"}]}`)) {
		t.Fatal("OpenAI chat payload should be eligible for a Claude-only route")
	}
	if claudeHandoffRouteEligible(cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatOpenAIResponse}, providers, []byte(`{"input":[{"role":"user","content":"x"}]}`)) {
		t.Fatal("OpenAI Responses payload without messages must not enter Claude handoff")
	}
	if claudeHandoffRouteEligible(cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatOpenAI}, []string{"claude", "codex"}, []byte(`{"messages":[{"role":"user","content":"x"}]}`)) {
		t.Fatal("mixed routes must defer non-Claude-source handoff until Claude is selected")
	}
}

func TestMaybeCompactClaudeHandoffHistoryOnFirstRequest(t *testing.T) {
	manager := NewManager(nil, nil, nil)
	manager.RegisterExecutor(&claudeHandoffFlowCompactorExecutor{})
	manager.SetConfig(&internalconfig.Config{Routing: internalconfig.RoutingConfig{ClaudeHandoff: internalconfig.ClaudeHandoffConfig{
		Enabled: true, CompactorProvider: "codex", CompactorModel: "gpt-6-luna", KeepRecentMessages: 2,
	}}})
	const authID = "claude-startup-compactor-auth"
	const model = "gpt-6-luna"
	registry.GetGlobalRegistry().RegisterClient(authID, "codex", []*registry.ModelInfo{{ID: model}})
	t.Cleanup(func() { registry.GetGlobalRegistry().UnregisterClient(authID) })
	if _, err := manager.Register(context.Background(), &Auth{ID: authID, Provider: "codex", Status: StatusActive, Metadata: map[string]any{"disable_cooling": true}}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}
	payload := []byte(`{"model":"claude-opus-5-5","messages":[{"role":"user","content":"old request"},{"role":"assistant","content":"old answer"},{"role":"user","content":"latest request"}]}`)
	req, opts, err := manager.maybeCompactClaudeHandoffHistory(context.Background(), cliproxyexecutor.Request{Model: "claude-opus-5-5", Payload: payload}, cliproxyexecutor.Options{
		SourceFormat:    sdktranslator.FormatClaude,
		OriginalRequest: payload,
		Metadata: map[string]any{
			cliproxyexecutor.CanonicalSessionIDMetadataKey: "claude-startup-history-session",
		},
	}, []string{"claude"})
	if err != nil {
		t.Fatalf("maybeCompactClaudeHandoffHistory() error = %v", err)
	}
	if !isAlreadyClaudeHandoffCompacted(req.Payload) || !isAlreadyClaudeHandoffCompacted(opts.OriginalRequest) {
		t.Fatalf("historical request was not compacted: payload=%s original=%s", req.Payload, opts.OriginalRequest)
	}
}

func TestMaybeCompactClaudeHandoffHistoryAfterThirtyMinutesIdle(t *testing.T) {
	manager := NewManager(nil, nil, nil)
	compactor := &claudeHandoffPayloadRecordingCompactorExecutor{}
	manager.RegisterExecutor(compactor)
	manager.SetConfig(&internalconfig.Config{Routing: internalconfig.RoutingConfig{ClaudeHandoff: internalconfig.ClaudeHandoffConfig{
		Enabled: true, CompactorProvider: "codex", CompactorModel: "gpt-6-luna", KeepRecentMessages: 2,
	}}})
	const authID = "claude-idle-compactor-auth"
	registry.GetGlobalRegistry().RegisterClient(authID, "codex", []*registry.ModelInfo{{ID: "gpt-6-luna"}})
	t.Cleanup(func() { registry.GetGlobalRegistry().UnregisterClient(authID) })
	if _, err := manager.Register(context.Background(), &Auth{ID: authID, Provider: "codex", Status: StatusActive, Metadata: map[string]any{"disable_cooling": true}}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}
	const sessionID = "claude-idle-history-session"
	manager.rememberClaudeHandoffSummary(sessionID, `{"goal":"old summary"}`)
	manager.rememberClaudeHandoffRequest(sessionID, time.Now().Add(-31*time.Minute))
	payload := []byte(`{"model":"claude-opus-5-5","messages":[{"role":"user","content":"old request"},{"role":"assistant","content":"old answer"},{"role":"user","content":"latest request"}]}`)
	req, _, err := manager.maybeCompactClaudeHandoffHistory(context.Background(), cliproxyexecutor.Request{Model: "claude-opus-5-5", Payload: payload}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FormatClaude,
		Metadata: map[string]any{
			cliproxyexecutor.CanonicalSessionIDMetadataKey: sessionID,
		},
	}, []string{"claude"})
	if err != nil {
		t.Fatalf("maybeCompactClaudeHandoffHistory() error = %v", err)
	}
	if !isAlreadyClaudeHandoffCompacted(req.Payload) {
		t.Fatalf("idle Claude session was not compacted: %s", req.Payload)
	}
	if got := compactor.calls.Load(); got != 1 {
		t.Fatalf("idle compactor calls = %d, want exactly one fresh compact", got)
	}
}

func TestExecuteStreamOpenAIRouteToClaudeCompactsBootstrapErrorBeforeNextOAuthAccount(t *testing.T) {
	manager := NewManager(nil, claudeHandoffFlowSelector{}, nil)
	manager.SetRetryConfig(0, 0, 0)
	manager.RegisterExecutor(&claudeHandoffFlowExecutor{})
	manager.RegisterExecutor(&claudeHandoffFlowCompactorExecutor{})
	manager.SetConfig(&internalconfig.Config{Routing: internalconfig.RoutingConfig{ClaudeHandoff: internalconfig.ClaudeHandoffConfig{
		Enabled: true, CompactorProvider: "codex", CompactorModel: "gpt-6-luna", KeepRecentMessages: 2,
	}}})
	for _, entry := range []struct {
		id       string
		provider string
		model    string
	}{
		{id: "claude-a", provider: "claude", model: "claude-opus-5-5"},
		{id: "claude-b", provider: "claude", model: "claude-opus-5-5"},
		{id: "codex-a", provider: "codex", model: "gpt-6-luna"},
	} {
		registry.GetGlobalRegistry().RegisterClient(entry.id, entry.provider, []*registry.ModelInfo{{ID: entry.model}})
		entryID := entry.id
		t.Cleanup(func() { registry.GetGlobalRegistry().UnregisterClient(entryID) })
		attributes := map[string]string{}
		if entry.provider == "claude" {
			attributes["auth_kind"] = "oauth"
		}
		if _, err := manager.Register(context.Background(), &Auth{ID: entry.id, Provider: entry.provider, Status: StatusActive, Attributes: attributes, Metadata: map[string]any{"disable_cooling": true}}); err != nil {
			t.Fatalf("Register(%s) error = %v", entry.id, err)
		}
	}
	payload := []byte(`{"model":"claude-opus-5-5","stream":true,"messages":[{"role":"user","content":"work on this"}]}`)
	stream, err := manager.ExecuteStream(context.Background(), []string{"claude"}, cliproxyexecutor.Request{Model: "claude-opus-5-5", Payload: payload}, cliproxyexecutor.Options{
		SourceFormat:    sdktranslator.FormatOpenAI,
		OriginalRequest: payload,
		Stream:          true,
		Metadata: map[string]any{
			cliproxyexecutor.CanonicalSessionIDMetadataKey: "claude-handoff-stream-session",
		},
	})
	if err != nil {
		t.Fatalf("ExecuteStream() error = %v", err)
	}
	var received []byte
	for chunk := range stream.Chunks {
		if chunk.Err != nil {
			t.Fatalf("stream chunk error = %v", chunk.Err)
		}
		received = append(received, chunk.Payload...)
	}
	if string(received) != `{"ok":true}` {
		t.Fatalf("stream response = %s", received)
	}
}

func TestClaudeHandoffRetryDelayIsBoundedAndUsesRetryAfter(t *testing.T) {
	for _, tt := range []struct {
		attempt int
		want    time.Duration
	}{
		{attempt: 0, want: time.Second},
		{attempt: 1, want: 2 * time.Second},
		{attempt: 4, want: 16 * time.Second},
		{attempt: 10, want: 30 * time.Second},
	} {
		if got := claudeHandoffRetryDelay(nil, tt.attempt, 30*time.Second); got != tt.want {
			t.Fatalf("claudeHandoffRetryDelay(attempt=%d) = %v, want %v", tt.attempt, got, tt.want)
		}
	}
	retryAfter := 10 * time.Second
	err := customStatusError{code: http.StatusTooManyRequests, retryAfter: &retryAfter}
	if got := claudeHandoffRetryDelay(err, 0, 30*time.Second); got != retryAfter {
		t.Fatalf("RetryAfter delay = %v, want %v", got, retryAfter)
	}
	retryAfter = time.Hour
	if got := claudeHandoffRetryDelay(err, 0, 30*time.Second); got != 30*time.Second {
		t.Fatalf("capped RetryAfter delay = %v, want 30s", got)
	}
}
