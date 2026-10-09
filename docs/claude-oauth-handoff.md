# Claude OAuth context handoff

The Claude routing handoff is an opt-in failover path for Claude Messages OAuth
credentials. It is useful with `fill-first` when one Claude account reaches its
request or subscription limit while the active conversation is large.

When Claude returns HTTP 429 with a structured `rate_limit_error` message that
indicates the account or subscription limit would be exceeded, the conductor
holds the request before selecting the next Claude credential. It sends the
original Claude request to the configured compactor route (by default the
`codex` OAuth provider and `gpt-6-luna`) using an OpenAI Responses request with
`reasoning.effort: xhigh`. The summary is inserted into a small Claude user
message, recent safe user/assistant turns are retained, and the next Claude
credential receives the rewritten request. The original request envelope
(`system`, `tools`, model, stream flag, and other fields) is preserved.

Enable it under `routing`:

```yaml
routing:
  strategy: fill-first
  claude-handoff:
    enabled: true
    compactor-provider: codex
    compactor-model: gpt-6-luna
    compactor-max-output-tokens: 8192
    compactor-retry-max-interval: 30
    keep-recent-messages: 8
    max-summary-bytes: 262144
    fail-open: false
```

The compactor request carries the same session metadata but is marked
`cliproxy.claude_handoff.internal`, so a Claude quota response from the compactor
cannot recursively trigger another Claude handoff. Set `compactor-auth-id` when one
specific OpenAI OAuth credential should be used for compaction.

Temporary compactor failures (network failures, 408, 429, 502, 503, 504, and
Cloudflare 520–526) retry with exponential backoff, capped by
`compactor-retry-max-interval`, until the request context is canceled or a
permanent error is returned. Missing compactor credentials, invalid requests,
an empty response, or an oversized summary fail immediately. With the default
`fail-open: false`, the full uncompressed transcript is never sent to the next
Claude account after a failed compaction. Set `fail-open: true` only if losing
the request is worse than paying for the full context.

The compactor keeps each OpenAI request below a conservative input budget. A
large Claude transcript is split at message boundaries, each part is compacted
through the configured OpenAI OAuth pool, and the resulting summaries are
combined. If one compactor credential returns `context_too_large`, the next
credential is still tried before the chunk is reduced. This is needed because
the effective `gpt-6-luna` request window can be smaller than the larger
client-advertised maximum.

The first chunk is executed serially to find a working OAuth credential. The
remaining chunks then run in a bounded batch of seven requests pinned to that
credential, preserving summary order. A temporary failure releases the pin for
that chunk so normal credential rotation can continue.

Claude handoff state is keyed by the normalized session/conversation ID already
extracted by CLIProxyAPI. After a proxy restart, the first request that already
contains conversation history is compacted before the first Claude account is
used. During a running session, a historical request received after at least
30 minutes of inactivity is compacted again. State is kept for 24 hours. A
later quota in the same session can compact the current Claude handoff envelope
again, so repeated account rotation does not grow the transcript without bound.
The request-scoped handoff marker prevents a startup or idle compact from being
run a second time when that same request immediately receives a Claude quota
response. Model-fallback-router attempts also pass through this hook: when a
fallback attempt resolves to Claude, a historical `messages` payload is compacted
before the Claude request. The hook does not rewrite OpenAI Responses `input`
payloads because they do not have the Claude Messages shape required for a safe
handoff envelope. The mechanism runs in the local conductor path; Home dispatch
mode does not currently run this hook.
