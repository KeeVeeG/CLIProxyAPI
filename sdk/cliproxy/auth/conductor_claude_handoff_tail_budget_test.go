package auth

import (
	"fmt"
	"strings"
	"testing"

	internalconfig "github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	"github.com/tidwall/gjson"
)

func TestClaudeHandoffTailBudgetKeepsRecentCompleteMessages(t *testing.T) {
	tail := make([]gjson.Result, 0, 8)
	for i := 0; i < 8; i++ {
		raw := fmt.Sprintf(`{"role":"user","content":"turn-%d-%s"}`, i, strings.Repeat("x", 48))
		tail = append(tail, gjson.Parse(raw))
	}
	const maxBytes = 180
	got := sanitizeClaudeHandoffMessageSliceWithBudget(tail, maxBytes)
	var total int
	for _, raw := range got {
		total += len([]byte(raw))
	}
	if total > maxBytes {
		t.Fatalf("bounded tail size = %d bytes, want <= %d: %v", total, maxBytes, got)
	}
	if len(got) == 0 || !strings.Contains(got[len(got)-1], "turn-7") {
		t.Fatalf("bounded tail did not retain the newest complete message: %v", got)
	}
	if strings.Contains(strings.Join(got, ""), "turn-0") {
		t.Fatalf("bounded tail retained an old message after reaching its budget: %v", got)
	}
}

func TestClaudeHandoffTailBudgetDoesNotSplitToolExchange(t *testing.T) {
	tail := []gjson.Result{
		gjson.Parse(`{"role":"user","content":"old"}`),
		gjson.Parse(`{"role":"assistant","content":[{"type":"tool_use","id":"paired","name":"read_file","input":{}}]}`),
		gjson.Parse(`{"role":"user","content":[{"type":"tool_result","tool_use_id":"paired","content":"result"}]}`),
		gjson.Parse(`{"role":"user","content":"latest request"}`),
	}
	got := sanitizeClaudeHandoffMessageSliceWithBudget(tail, 220)
	text := strings.Join(got, "")
	if strings.Contains(text, `"tool_result"`) && !strings.Contains(text, `"tool_use"`) {
		t.Fatalf("bounded tail retained an orphan tool_result: %v", got)
	}
	if !strings.Contains(text, "latest request") {
		t.Fatalf("bounded tail lost the newest user request: %v", got)
	}
}

func TestClaudeHandoffRewriteAppliesConfiguredTailBudget(t *testing.T) {
	payload := []byte(`{"model":"claude-opus-5-5","messages":[
{"role":"user","content":"old-1"},
{"role":"assistant","content":"old-2"},
{"role":"user","content":"latest-"}` + strings.Repeat("z", 600) + `"}]}`)
	rewritten, err := rewriteClaudeHandoffPayloadBudget(payload, `{"goal":"continue"}`, 8, 128)
	if err != nil {
		t.Fatalf("rewriteClaudeHandoffPayloadBudget() error = %v", err)
	}
	messages := gjson.GetBytes(rewritten, "messages").Array()
	if len(messages) != 2 {
		t.Fatalf("rewritten messages = %d, want summary plus continuation after oversized tail", len(messages))
	}
	if messages[0].Get("role").String() != "user" || messages[1].Get("role").String() != "user" {
		t.Fatalf("rewritten roles = %q,%q, want user,user", messages[0].Get("role").String(), messages[1].Get("role").String())
	}
	if strings.Contains(string(rewritten), "latest-") {
		t.Fatalf("oversized post-summary message was retained: %s", rewritten)
	}
}

func TestClaudeHandoffTailBudgetDefaultIsBounded(t *testing.T) {
	cfg := claudeHandoffConfig(&internalconfig.Config{})
	if cfg.MaxTailBytes != claudeHandoffDefaultMaxTailBytes {
		t.Fatalf("default max tail bytes = %d, want %d", cfg.MaxTailBytes, claudeHandoffDefaultMaxTailBytes)
	}
}
