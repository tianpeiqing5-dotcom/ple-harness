# plan-verify: planner, worker, independent verifier, fix rounds

A plan-verify harness built on the `minimal` starter. For each task:

1. **Planner** (read-only tools) inspects the task and its files and writes a
   plan: deliverables with exact paths/formats, a numbered hard-constraint
   checklist (C1, C2, ...), the computation method for every number, and how
   to check each deliverable.
2. **Worker** executes the plan. **Compute with code, never by hand**: every
   number in a deliverable must be produced by a script run in the task
   container (`calc_*.py`), and the scripts stay in the workdir so they can
   be re-run.
3. **Verifier** (read-only, fresh conversation) re-reads every deliverable
   against the task and the constraint checklist, re-runs the calc scripts
   and diffs their output against the deliverables, and replies `PASS` or a
   numbered list of gaps. Up to `MAX_FIX_ROUNDS = 2` fix rounds follow while
   time remains.

## Cost control

- Per-task token budget (`TOKEN_BUDGET`, default 1,500,000): when it is
  used up, the current role is forced to wrap up with `tool_choice="none"`.
- Hard cap on tool-call turns per role (`MAX_TURNS_PLANNER` 25,
  `MAX_TURNS_WORKER` 80, `MAX_TURNS_VERIFIER` 30, `MAX_TURNS_FIX` 40).
- Context compaction once the conversation passes `COMPACT_CHARS`
  (default 120,000): the middle is summarized into a progress note and only
  the most recent turns are kept. Old tool results are also cut on a 400
  context-length error, as in the minimal starter.
- Wrap-up `WRAP_UP_SEC = 240` s before the time limit.

## Evidence logging

Every plan, write (path, bytes, sha256), tool call, usage report, verdict
and final summary is appended as JSONL to
`$EVIDENCE_DIR/<session>.jsonl` on the harness host (default
`/tmp/ple-evidence`), never inside the task container, so deliverables stay
clean. Key events are also sent through `session.log()` into the trajectory.
This is the raw material for the Stage 1 writeup's failure analysis.

## Files

| File | Purpose |
|---|---|
| `serve.py` | Plan-verify agent loop, cost control, evidence logging |
| `taskbridge.py` | Serve protocol implementation (shared across Python samples) |
| `requirements.txt` | Pinned `openai` and `websockets` dependencies |
| `submission.yaml` | Model declaration, build command, and server startup command |

## Local runs

For local runs the entry uses `model_api: openai-compatible/deepseek-flash`
against a local OpenAI-compatible proxy (`OPENAI_BASE_URL`,
`OPENAI_API_KEY`). That entry runs locally only; submit it with a
`google-cloud/<id>` model and generate `artifacts/` with the same
`model_api` you submit.

## CI (GitHub Actions)

`.github/workflows/kit-check.yml` runs `uv run python -m kit check` on
every push to `main` and on manual dispatch. It needs one repository
secret: `DEEPSEEK_API_KEY` (Settings → Secrets and variables → Actions),
the DeepSeek API key used as `MODEL_API_KEY` for the check run.
