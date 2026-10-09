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
	payload, err := buildClaudeHandoffCompactorPayload(original, "gpt-6.1-sol", 4096)
	if err != nil {
		t.Fatalf("buildClaudeHandoffCompactorPayload() error = %v", err)
	}
	if got := gjson.GetBytes(payload, "model").String(); got != "gpt-6.1-sol" {
		t.Fatalf("model = %q, want gpt-6.1-sol", got)
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
	if len(messages) != 3 {
		t.Fatalf("rewritten messages = %d, want summary pair plus one recent message", len(messages))
	}
	if !strings.Contains(messages[0].Get("content.0.text").String(), claudeHandoffSummaryTag) || !strings.Contains(messages[0].Get("content.0.text").String(), "run tests") {
		t.Fatalf("summary marker/content missing from first message: %s", messages[0].Raw)
	}
	if messages[1].Get("role").String() != "assistant" {
		t.Fatalf("summary acknowledgement role = %q, want assistant", messages[1].Get("role").String())
	}
	if messages[2].Get("content.0.text").String() != "latest" {
		t.Fatalf("recent tail message was not retained: %s", messages[2].Raw)
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
	if len(messages) != 5 {
		t.Fatalf("rewritten message count = %d, want summary pair + boundary and 2 post-boundary turns", len(messages))
	}
	text := string(req.Payload)
	for _, want := range []string{"boundary", "post-boundary answer", "new request"} {
		if !strings.Contains(text, want) {
			t.Fatalf("post-boundary turn %q was dropped: %s", want, text)
		}
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
	if got.CompactorProvider != "codex" || got.CompactorModel != "gpt-6.1-sol" || got.CompactorMaxOutputTokens != 8192 || got.KeepRecentMessages != 8 || got.MaxSummaryBytes <= 0 {
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
	model := "gpt-6.1-sol"
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
		Enabled: true, CompactorProvider: "codex", CompactorModel: "gpt-6.1-sol", KeepRecentMessages: 2,
	}}})
	for _, entry := range []struct {
		id       string
		provider string
		model    string
	}{
		{id: "claude-a", provider: "claude", model: "claude-opus-5-5"},
		{id: "claude-b", provider: "claude", model: "claude-opus-5-5"},
		{id: "codex-a", provider: "codex", model: "gpt-6.1-sol"},
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

func TestExecuteStreamClaudeQuotaCompactsBootstrapErrorBeforeNextOAuthAccount(t *testing.T) {
	manager := NewManager(nil, claudeHandoffFlowSelector{}, nil)
	manager.SetRetryConfig(0, 0, 0)
	manager.RegisterExecutor(&claudeHandoffFlowExecutor{})
	manager.RegisterExecutor(&claudeHandoffFlowCompactorExecutor{})
	manager.SetConfig(&internalconfig.Config{Routing: internalconfig.RoutingConfig{ClaudeHandoff: internalconfig.ClaudeHandoffConfig{
		Enabled: true, CompactorProvider: "codex", CompactorModel: "gpt-6.1-sol", KeepRecentMessages: 2,
	}}})
	for _, entry := range []struct {
		id       string
		provider string
		model    string
	}{
		{id: "claude-a", provider: "claude", model: "claude-opus-5-5"},
		{id: "claude-b", provider: "claude", model: "claude-opus-5-5"},
		{id: "codex-a", provider: "codex", model: "gpt-6.1-sol"},
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
		SourceFormat:    sdktranslator.FormatClaude,
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
