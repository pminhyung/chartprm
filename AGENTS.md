# AGENTS.md

## Harness Rules (auto-managed)

### Artifact Structure
- `docs/active/` — Always loaded at session start (~500 line budget). Keep compact.
- `docs/reference/` — Never auto-loaded. Search when historical context needed.

### Feedback Accumulation (always active)
- When user gives directives, corrections, approvals, or preferences → auto-save to docs/active/feedback.md
- Read docs/active/feedback.md before starting any task
- Apply relevant feedback to current work

### Session Management (hook-triggered)
- On SessionStart hook → read docs/active/ for context restoration
- On Stop hook with code changes → use session-historian agent to record

### Output Verification (auto)
- After generating visual artifacts (PPTX, PNG, HTML, PDF) → auto verify-output
- Never report "file generated" without visual inspection

### Code Evaluation (auto for 10+ line changes)
- After significant code changes → delegate to evaluator agent for independent review
- Fix Critical Issues before declaring completion
