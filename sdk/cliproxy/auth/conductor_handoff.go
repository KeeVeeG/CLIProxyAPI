package auth

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	internalconfig "github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v8/sdk/translator"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const (
	handoffInternalMetadataKey = "cliproxy.handoff.internal"
	handoffSummaryTag          = "<cli-proxy-context-handoff>"
	handoffSummaryEndTag       = "</cli-proxy-context-handoff>"
)

type handoffState struct {
	Summary          string
	UpdatedAt        time.Time
	BoundaryMessages int
	BoundaryHash     string
}

type handoffPermanentError struct{ error }

func (handoffPermanentError) HandoffPermanent() bool { return true }

// handoffConfig returns normalized defaults without mutating the live config.
func handoffConfig(cfg *internalconfig.Config) internalconfig.HandoffConfig {
	if cfg == nil {
		return internalconfig.HandoffConfig{}
	}
	out := cfg.Routing.Handoff
	if strings.TrimSpace(out.CompactorProvider) == "" {
		out.CompactorProvider = "codex"
	}
	if strings.TrimSpace(out.CompactorModel) == "" {
		out.CompactorModel = "gpt-6.1-sol"
	}
	if out.CompactorMaxOutputTokens <= 0 {
		out.CompactorMaxOutputTokens = 8192
	}
	if out.CompactorRetryMaxInterval <= 0 {
		out.CompactorRetryMaxInterval = 30
	}
	if out.KeepRecentMessages <= 0 {
		out.KeepRecentMessages = 8
	}
	if out.MaxSummaryBytes <= 0 {
		out.MaxSummaryBytes = 256 * 1024
	}
	return out
}

func handoffInternal(opts cliproxyexecutor.Options) bool {
	if opts.Metadata == nil {
		return false
	}
	value, _ := opts.Metadata[handoffInternalMetadataKey].(bool)
	return value
}

func handoffSessionID(opts cliproxyexecutor.Options) string {
	if opts.Metadata == nil {
		return ""
	}
	if value, ok := opts.Metadata[cliproxyexecutor.CanonicalSessionIDMetadataKey].(string); ok {
		if value = strings.TrimSpace(value); value != "" {
			return value
		}
	}
	if value, ok := opts.Metadata[cliproxyexecutor.DerivedSessionIDMetadataKey].(string); ok {
		return strings.TrimSpace(value)
	}
	return ""
}

func handoffSummaryFromState(m *Manager, sessionID string) (handoffState, bool) {
	if m == nil || sessionID == "" {
		return handoffState{}, false
	}
	m.handoffMu.RLock()
	state, ok := m.handoffStates[sessionID]
	m.handoffMu.RUnlock()
	// Handoff state is deliberately bounded in time. A Claude session can be
	// reused for days, but carrying a stale summary into a new session would be
	// worse than paying for one fresh compaction.
	if ok && !state.UpdatedAt.IsZero() && time.Since(state.UpdatedAt) > 24*time.Hour {
		m.handoffMu.Lock()
		delete(m.handoffStates, sessionID)
		m.handoffMu.Unlock()
		return handoffState{}, false
	}
	return state, ok && strings.TrimSpace(state.Summary) != ""
}

func (m *Manager) handoffSessionLock(sessionID string) *sync.Mutex {
	if m == nil || sessionID == "" {
		return nil
	}
	value, _ := m.handoffLocks.LoadOrStore(sessionID, &sync.Mutex{})
	return value.(*sync.Mutex)
}

func (m *Manager) rememberHandoffSummary(sessionID, summary string) {
	if m == nil || sessionID == "" || strings.TrimSpace(summary) == "" {
		return
	}
	m.handoffMu.Lock()
	if m.handoffStates == nil {
		m.handoffStates = make(map[string]handoffState)
	}
	m.handoffStates[sessionID] = handoffState{Summary: summary, UpdatedAt: time.Now()}
	m.handoffMu.Unlock()
}

func (m *Manager) rememberHandoffSummaryForPayload(sessionID, summary string, payload []byte) {
	state := handoffState{Summary: summary, UpdatedAt: time.Now()}
	messages := gjson.GetBytes(payload, "messages")
	if messages.IsArray() && len(messages.Array()) > 0 {
		state.BoundaryMessages = len(messages.Array())
		if messages.Array()[len(messages.Array())-1].Get("role").String() == "user" {
			state.BoundaryMessages--
		}
		if state.BoundaryMessages > 0 {
			state.BoundaryHash = handoffMessagesHash(messages.Array()[:state.BoundaryMessages])
		}
	}
	m.handoffMu.Lock()
	if m.handoffStates == nil {
		m.handoffStates = make(map[string]handoffState)
	}
	m.handoffStates[sessionID] = state
	m.handoffMu.Unlock()
}

func handoffMessagesHash(messages []gjson.Result) string {
	hash := sha256.New()
	for _, message := range messages {
		_, _ = hash.Write([]byte(message.Raw))
		_, _ = hash.Write([]byte{'\n'})
	}
	return fmt.Sprintf("%x", hash.Sum(nil))
}

func isClaudeHandoffCandidate(provider string, opts cliproxyexecutor.Options, err error) bool {
	if !strings.EqualFold(strings.TrimSpace(provider), "claude") || opts.SourceFormat.String() != "claude" || handoffInternal(opts) || err == nil {
		return false
	}
	if statusCodeFromError(err) != http.StatusTooManyRequests {
		return false
	}
	body := []byte(extractErrorBody(err))
	if !gjson.ValidBytes(body) || gjson.GetBytes(body, "error.type").String() != "rate_limit_error" {
		return false
	}
	message := strings.ToLower(gjson.GetBytes(body, "error.message").String())
	if message == "" {
		return false
	}
	return strings.Contains(message, "would exceed your account's rate limit") ||
		strings.Contains(message, "would exceed your account's monthly spend limit") ||
		strings.Contains(message, "5-hour limit exceeded") ||
		strings.Contains(message, "7-day rate limit exceeded") ||
		strings.Contains(message, "both limits exceeded")
}

func isTemporaryHandoffError(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	var permanent handoffPermanentError
	if errors.As(err, &permanent) {
		return false
	}
	var authErr *Error
	if errors.As(err, &authErr) && authErr != nil {
		switch strings.ToLower(strings.TrimSpace(authErr.Code)) {
		case "auth_not_found", "provider_not_found", "executor_not_found", "model_not_found", "invalid_request", "request_invalid":
			return false
		}
	}
	status := statusCodeFromError(err)
	if status == 0 {
		return true
	}
	switch status {
	case http.StatusRequestTimeout, http.StatusTooManyRequests,
		http.StatusBadGateway, http.StatusServiceUnavailable, http.StatusGatewayTimeout:
		return true
	default:
		return status >= 520 && status <= 526
	}
}

func handoffRetryDelay(err error, attempt int, maxInterval time.Duration) time.Duration {
	if retryAfter := retryAfterFromError(err); retryAfter != nil && *retryAfter > 0 {
		if *retryAfter > maxInterval {
			return maxInterval
		}
		return *retryAfter
	}
	delay := time.Second
	for index := 0; index < attempt && delay < maxInterval; index++ {
		delay *= 2
	}
	if delay > maxInterval {
		return maxInterval
	}
	return delay
}

// contextForHandoff preserves cancellation and deadlines while clearing the
// round-tripper values installed for the failed Claude attempt. Manager.Execute
// will install the selected compactor credential's transport itself.
func contextForHandoff(ctx context.Context) context.Context {
	if ctx == nil {
		return context.Background()
	}
	ctx = context.WithValue(ctx, roundTripperContextKey{}, nil)
	return context.WithValue(ctx, "cliproxy.roundtripper", nil)
}

func (m *Manager) hasHandoffCompactorAuth(provider, pinnedID string) bool {
	if m == nil {
		return false
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	for _, auth := range m.auths {
		if auth == nil || !strings.EqualFold(strings.TrimSpace(auth.Provider), strings.TrimSpace(provider)) {
			continue
		}
		if pinnedID == "" || auth.ID == pinnedID {
			return true
		}
	}
	return false
}

func (m *Manager) applyHandoffState(req cliproxyexecutor.Request, opts cliproxyexecutor.Options, providers []string) (cliproxyexecutor.Request, cliproxyexecutor.Options) {
	cfg := handoffConfig(m.runtimeConfigSnapshot())
	if !cfg.Enabled || handoffInternal(opts) || opts.SourceFormat.String() != "claude" || !containsProvider(providers, "claude") {
		return req, opts
	}
	sessionID := handoffSessionID(opts)
	state, ok := handoffSummaryFromState(m, sessionID)
	if !ok || isAlreadyHandoffCompacted(req.Payload) {
		return req, opts
	}
	rewritten, errRewrite := rewriteClaudeHandoffPayloadWithState(req.Payload, state, cfg.KeepRecentMessages)
	if errRewrite != nil {
		return req, opts
	}
	req.Payload = rewritten
	opts.OriginalRequest = bytes.Clone(rewritten)
	return req, opts
}

func isAlreadyHandoffCompacted(payload []byte) bool {
	if bytes.Contains(payload, []byte(handoffSummaryTag)) {
		return true
	}
	// encoding/json escapes HTML-sensitive delimiters in the synthetic text
	// block, so detect the canonical escaped representation as well.
	return bytes.Contains(payload, []byte(`\u003ccli-proxy-context-handoff\u003e`))
}

func buildCompactorPayload(original []byte, model string, maxOutputTokens int) ([]byte, error) {
	instructions := "You are a context handoff compactor. Return only a concise JSON object with these keys: goal, constraints, decisions, current_state, files_changed, tests_and_results, pending_tasks, important_tool_results, latest_user_request, open_questions. Preserve concrete facts, file paths, commands, errors, tool outcomes, and unfinished work. Do not include hidden chain-of-thought, speculation, or markdown fences. Do not answer the task."
	text := instructions + "\n\n<claude_request>\n" + string(original) + "\n</claude_request>"
	body := map[string]any{
		"model":        model,
		"instructions": instructions,
		"input": []any{map[string]any{
			"role":    "user",
			"content": []any{map[string]any{"type": "input_text", "text": text}},
		}},
		"reasoning":         map[string]any{"effort": "xhigh"},
		"max_output_tokens": maxOutputTokens,
		"stream":            false,
	}
	return json.Marshal(body)
}

func extractCompactorText(payload []byte) string {
	var parts []string
	for _, item := range gjson.GetBytes(payload, "output").Array() {
		for _, content := range item.Get("content").Array() {
			if text := strings.TrimSpace(content.Get("text").String()); text != "" {
				parts = append(parts, text)
			}
		}
	}
	if len(parts) == 0 {
		if text := strings.TrimSpace(gjson.GetBytes(payload, "choices.0.message.content").String()); text != "" {
			parts = append(parts, text)
		}
	}
	return strings.TrimSpace(strings.Join(parts, "\n"))
}

func normalizeCompactorText(text string) string {
	text = strings.TrimSpace(text)
	if strings.HasPrefix(text, "```") && strings.HasSuffix(text, "```") {
		lines := strings.Split(text, "\n")
		if len(lines) >= 3 {
			text = strings.Join(lines[1:len(lines)-1], "\n")
		}
	}
	return strings.TrimSpace(text)
}

func (m *Manager) compactClaudeContext(ctx context.Context, original []byte, opts cliproxyexecutor.Options, cfg internalconfig.HandoffConfig) (string, error) {
	compactorBody, errBuild := buildCompactorPayload(original, cfg.CompactorModel, cfg.CompactorMaxOutputTokens)
	if errBuild != nil {
		return "", errBuild
	}
	metadata := cloneSchedulerAnyMap(opts.Metadata)
	if metadata == nil {
		metadata = make(map[string]any, 1)
	}
	metadata[handoffInternalMetadataKey] = true
	compactorOpts := cliproxyexecutor.Options{
		SourceFormat:    sdktranslator.FormatOpenAIResponse,
		ResponseFormat:  sdktranslator.FormatOpenAIResponse,
		OriginalRequest: bytes.Clone(compactorBody),
		Metadata:        metadata,
	}
	if cfg.CompactorAuthID != "" {
		metadata[cliproxyexecutor.PinnedAuthMetadataKey] = cfg.CompactorAuthID
	}
	compactorReq := cliproxyexecutor.Request{Model: cfg.CompactorModel, Payload: compactorBody, Metadata: metadata}
	maxInterval := time.Duration(cfg.CompactorRetryMaxInterval) * time.Second
	var lastErr error
	for attempt := 0; ; attempt++ {
		// The Claude attempt may have installed a provider-specific round-tripper
		// in the context. The compactor must resolve its own OpenAI transport.
		resp, errExecute := m.Execute(contextForHandoff(ctx), []string{cfg.CompactorProvider}, compactorReq, compactorOpts)
		if errExecute == nil {
			text := normalizeCompactorText(extractCompactorText(resp.Payload))
			if text != "" && len([]byte(text)) <= cfg.MaxSummaryBytes {
				return text, nil
			}
			if text == "" {
				errExecute = handoffPermanentError{error: fmt.Errorf("compactor returned no summary text")}
			} else {
				errExecute = handoffPermanentError{error: fmt.Errorf("compactor summary exceeds %d bytes", cfg.MaxSummaryBytes)}
			}
		}
		lastErr = errExecute
		temporary := isTemporaryHandoffError(errExecute)
		// A registered compactor credential can be temporarily unavailable
		// because the normal selector cooled it after a transient failure. Keep
		// retrying that case; an absent provider/auth remains a permanent setup
		// error and does not spin forever.
		if !temporary && m.hasHandoffCompactorAuth(cfg.CompactorProvider, cfg.CompactorAuthID) {
			var authErr *Error
			if errors.As(errExecute, &authErr) && authErr != nil && strings.EqualFold(authErr.Code, "auth_not_found") {
				temporary = true
			}
		}
		if !temporary {
			return "", fmt.Errorf("context compaction failed permanently: %w", errExecute)
		}
		delay := handoffRetryDelay(errExecute, attempt, maxInterval)
		timer := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			if !timer.Stop() {
				<-timer.C
			}
			return "", ctx.Err()
		case <-timer.C:
		}
		if lastErr == nil {
			return "", fmt.Errorf("context compaction stopped")
		}
	}
}

func (m *Manager) performClaudeHandoff(ctx context.Context, req cliproxyexecutor.Request, opts cliproxyexecutor.Options, cause error) ([]byte, error) {
	if m == nil {
		return nil, fmt.Errorf("context handoff manager is unavailable")
	}
	cfg := handoffConfig(m.runtimeConfigSnapshot())
	if !cfg.Enabled {
		return nil, fmt.Errorf("context handoff is disabled")
	}
	sessionID := handoffSessionID(opts)
	if sessionID == "" {
		return nil, fmt.Errorf("context handoff requires a stable Claude session id")
	}
	lock := m.handoffSessionLock(sessionID)
	if lock == nil {
		return nil, fmt.Errorf("context handoff session lock is unavailable")
	}
	lock.Lock()
	defer lock.Unlock()
	if state, ok := handoffSummaryFromState(m, sessionID); ok {
		payload := opts.OriginalRequest
		if len(payload) == 0 {
			payload = req.Payload
		}
		// Once a request already contains a handoff envelope, compact the
		// current envelope again. This prevents a long-running session from
		// accumulating an ever-growing post-handoff tail on the next quota.
		if isAlreadyHandoffCompacted(payload) {
			summary, errCompact := m.compactClaudeContext(ctx, payload, opts, cfg)
			if errCompact != nil {
				return nil, fmt.Errorf("refresh Claude context after %v: %w", cause, errCompact)
			}
			rewritten, errRewrite := rewriteClaudeHandoffPayload(payload, summary, cfg.KeepRecentMessages)
			if errRewrite != nil {
				return nil, fmt.Errorf("rewrite refreshed Claude request after compaction: %w", errRewrite)
			}
			m.rememberHandoffSummaryForPayload(sessionID, summary, payload)
			return rewritten, nil
		}
		return rewriteClaudeHandoffPayloadWithState(payload, state, cfg.KeepRecentMessages)
	}
	original := opts.OriginalRequest
	if len(original) == 0 {
		original = req.Payload
	}
	if len(original) == 0 {
		return nil, fmt.Errorf("context handoff has no original request body")
	}
	summary, errCompact := m.compactClaudeContext(ctx, original, opts, cfg)
	if errCompact != nil {
		return nil, fmt.Errorf("compact Claude context after %v: %w", cause, errCompact)
	}
	rewritten, errRewrite := rewriteClaudeHandoffPayload(original, summary, cfg.KeepRecentMessages)
	if errRewrite != nil {
		return nil, fmt.Errorf("rewrite Claude request after compaction: %w", errRewrite)
	}
	m.rememberHandoffSummaryForPayload(sessionID, summary, original)
	return rewritten, nil
}

func rewriteClaudeHandoffPayloadWithState(payload []byte, state handoffState, keepRecent int) ([]byte, error) {
	if state.BoundaryMessages <= 0 || state.BoundaryHash == "" {
		return rewriteClaudeHandoffPayload(payload, state.Summary, keepRecent)
	}
	messages := gjson.GetBytes(payload, "messages")
	if !messages.IsArray() || len(messages.Array()) < state.BoundaryMessages || handoffMessagesHash(messages.Array()[:state.BoundaryMessages]) != state.BoundaryHash {
		return rewriteClaudeHandoffPayload(payload, state.Summary, keepRecent)
	}
	tail := sanitizeHandoffMessageSlice(messages.Array()[state.BoundaryMessages:])
	return rewriteClaudeHandoffPayloadWithTail(payload, state.Summary, tail)
}

func rewriteClaudeHandoffPayload(payload []byte, summary string, keepRecent int) ([]byte, error) {
	messages := gjson.GetBytes(payload, "messages")
	if !messages.IsArray() {
		return nil, fmt.Errorf("Claude handoff requires messages array")
	}
	tail := sanitizeHandoffMessages(messages.Array(), keepRecent)
	return rewriteClaudeHandoffPayloadWithTail(payload, summary, tail)
}

func rewriteClaudeHandoffPayloadWithTail(payload []byte, summary string, tail []string) ([]byte, error) {
	handoffUser, _ := json.Marshal(map[string]any{
		"role": "user",
		"content": []map[string]any{{
			"type": "text",
			"text": handoffSummaryTag + "\n" + summary + "\n" + handoffSummaryEndTag,
		}},
	})
	handoffAssistant, _ := json.Marshal(map[string]any{
		"role": "assistant",
		"content": []map[string]any{{
			"type": "text",
			"text": "I will continue from the handoff context.",
		}},
	})
	items := make([]string, 0, len(tail)+2)
	items = append(items, string(handoffUser), string(handoffAssistant))
	items = append(items, tail...)
	updated, errSet := sjson.SetRawBytes(payload, "messages", []byte("["+strings.Join(items, ",")+"]"))
	if errSet != nil {
		return nil, errSet
	}
	return updated, nil
}

func sanitizeHandoffMessages(messages []gjson.Result, keepRecent int) []string {
	if keepRecent <= 0 {
		keepRecent = 8
	}
	start := len(messages) - keepRecent
	if start < 0 {
		start = 0
	}
	for start < len(messages) && messages[start].Get("role").String() != "user" {
		start++
	}
	if start >= len(messages) {
		return nil
	}
	return sanitizeHandoffMessageSlice(messages[start:])
}

// sanitizeHandoffMessageSlice removes provider-internal reasoning and broken
// tool blocks while preserving every remaining user/assistant message. It is
// used for the post-boundary portion of a session, where truncating to a fixed
// number of messages could silently discard turns created after the handoff.
func sanitizeHandoffMessageSlice(tail []gjson.Result) []string {
	toolUses := make(map[string]bool)
	toolResults := make(map[string]bool)
	for _, message := range tail {
		for _, block := range message.Get("content").Array() {
			switch block.Get("type").String() {
			case "tool_use":
				if id := block.Get("id").String(); id != "" {
					toolUses[id] = true
				}
			case "tool_result":
				if id := block.Get("tool_use_id").String(); id != "" {
					toolResults[id] = true
				}
			}
		}
	}
	out := make([]string, 0, len(tail))
	for _, message := range tail {
		role := message.Get("role").String()
		if role != "user" && role != "assistant" {
			continue
		}
		updated := message
		content := message.Get("content")
		if content.IsArray() {
			blocks := make([]string, 0, len(content.Array()))
			for _, block := range content.Array() {
				typeName := block.Get("type").String()
				if typeName == "thinking" || typeName == "redacted_thinking" {
					continue
				}
				if typeName == "tool_use" && !toolResults[block.Get("id").String()] {
					continue
				}
				if typeName == "tool_result" && !toolUses[block.Get("tool_use_id").String()] {
					continue
				}
				blocks = append(blocks, block.Raw)
			}
			if len(blocks) == 0 {
				continue
			}
			raw, errSet := sjson.SetRawBytes([]byte(message.Raw), "content", []byte("["+strings.Join(blocks, ",")+"]"))
			if errSet != nil {
				continue
			}
			updated = gjson.ParseBytes(raw)
		}
		out = append(out, updated.Raw)
	}
	return out
}
