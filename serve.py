# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Plan-verify harness: planner, worker, independent verifier, fix rounds.

Roles:
1. Planner (read-only): inspects the task and its files, writes the plan:
   deliverables with exact paths/formats, a numbered hard-constraint
   checklist (C1, C2, ...), the computation method for every number, and how
   to check each deliverable.
2. Worker: executes the plan. Compute-with-code: every number in a
   deliverable must be produced by a script run in the task container, never
   hand-computed; calc scripts stay in the workdir so they can be re-run.
3. Verifier (read-only, fresh conversation): re-reads every deliverable
   against the task and the constraint checklist, re-runs the calc scripts
   and diffs their output against the deliverables, and reports PASS or a
   numbered list of gaps. Up to MAX_FIX_ROUNDS fix rounds follow.

MCP servers: when the task offers its own tools on MCP servers, they are
connected once in solve() (taskbridge.MCPClient) and exposed to the worker
as <server>__<tool> functions when the set is small, or through mcp_find /
mcp_call when it is large. The planner and verifier get the same MCP tools
for read-only use.

Cost control: per-task token budget (TOKEN_BUDGET), hard cap on tool turns
per role, context compaction when the conversation grows past COMPACT_CHARS,
and wrap-up WRAP_UP_SEC before the time limit.

Evidence: every plan, write, tool call, usage report, verdict and summary is
appended as JSONL to $EVIDENCE_DIR/<session>.jsonl on the harness host (never
inside the task container, so deliverables stay clean) and key events are
also sent through session.log() into the trajectory.
"""

import asyncio
import hashlib
import json
import os
import re
import time

import openai
import taskbridge

MODEL = os.environ["MODEL_NAME"]
# The model's highest reasoning level: max for Claude, GLM, GPT-6 and
# DeepSeek-V4, high for the others (Gemini refuses max). REASONING_EFFORT
# overrides it.
MAX_EFFORT_MODELS = ("claude", "glm", "gpt-6", "deepseek-v4")
EFFORT = os.environ.get("REASONING_EFFORT") or (
    "max" if any(m in MODEL.lower() for m in MAX_EFFORT_MODELS) else "high"
)
# a fix round wraps up this long before the time limit, and a wrap-up call may
# take as long
WRAP_UP_SEC = 240
WRAP_UP = {
    "role": "user",
    "content": "Time is up for this step. Stop and reply with your answer.",
}
BUDGET_MSG = {
    "role": "user",
    "content": (
        "You have used the token budget for this task. Stop calling tools "
        "and reply with your answer now."
    ),
}
TURN_CAP_MSG = {
    "role": "user",
    "content": (
        "You have used your tool-call budget for this step. Stop calling "
        "tools and reply with your answer now."
    ),
}
RESULT_CHARS = 20_000  # a longer tool result loses its middle
# Seconds between retries once the SDK's own quick retries are exhausted.
RETRY_WAITS = (5, 15, 30, 60, 120)
# How endpoints word a 400 saying the conversation no longer fits the context
# window: Claude, OpenAI and vLLM, Gemini, xAI.
CONTEXT_FULL = re.compile(
    r"context.?(length|window)|prompt is too long|maximum prompt length"
    r"|exceed[^.]{0,40}(token|context)",
    re.I,
)
CUT_KEEP = 1000  # characters a tool result keeps once the context is full
# times in a row an empty or cut-off reply gets a "continue" before the run
# stops
MAX_NUDGES = 3
# lower it for a small context window
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "32000"))
# Claude caches a prompt prefix only when marked with cache_control. Each call
# marks the system prompt and the latest message (using 2 of Claude's 4 cache
# breakpoints) so each turn reads the prior conversation from cache. Other
# providers cache automatically without markers.
CACHE_MARK = "claude" in MODEL.lower()
# DeepSeek's thinking models reject a follow-up call when an assistant
# message in the history has no reasoning_content: the field must be passed
# back verbatim. The per-turn append keeps it (model_dump carries extras),
# but a turn the model answered without reasoning, and the synthetic progress
# note compact() inserts, have none — so those get a placeholder instead.
NEED_REASONING = "deepseek" in MODEL.lower()
REASONING_PLACEHOLDER = "[no reasoning returned for this turn]"
# the planner wraps up once it has used this share of the task's time
PLAN_SHARE = 0.1
# the worker wraps up when this share is left, to keep time for checking and
# fixing
CHECK_SHARE = 0.25
# a check or a fix round starts only with this long to work before its own
# wrap-up
MIN_STEP_SEC = 120
MAX_FIX_ROUNDS = 2

# --- cost control ---
TOKEN_BUDGET = int(os.environ.get("TOKEN_BUDGET", "1500000"))
MAX_TOOL_TURNS = {
    "planner": int(os.environ.get("MAX_TURNS_PLANNER", "25")),
    "worker": int(os.environ.get("MAX_TURNS_WORKER", "80")),
    "verifier": int(os.environ.get("MAX_TURNS_VERIFIER", "30")),
    "fix": int(os.environ.get("MAX_TURNS_FIX", "40")),
}
COMPACT_CHARS = int(os.environ.get("COMPACT_CHARS", "120000"))
COMPACT_KEEP_TURNS = 8
EVIDENCE_DIR = os.environ.get("EVIDENCE_DIR", "/tmp/ple-evidence")

# The evaluator sets OPENAI_BASE_URL and OPENAI_API_KEY for every model.
# The SDK handles quick retries; recover() handles longer backoffs. Because
# calls are non-streaming, a max-effort response can take over 15 minutes
# before any bytes arrive, so the timeout matches the egress proxy's 30-minute
# idle limit.
client = openai.AsyncOpenAI(max_retries=5, timeout=1800)

PLANNER = """You plan an engineering task that another engineer will carry out in a Linux container with no internet \
access. Inspect only the files the task points to and a listing of the workdir; do not explore the harness, the \
grader, or anything outside the workdir. Do not create or change any file. \
Reply with the plan in exactly these sections:

## Deliverables
Each deliverable with its exact path and format, and what it must contain.

## Hard constraints
A numbered list C1, C2, ... of EVERY hard constraint in the task: numeric limits with units, exact formats, exact \
paths, required contents, required sections. A constraint you miss here is a constraint nobody will check, so be \
thorough. One breach of a hard constraint caps that item's score at 0.4.

## Method
For each number the task needs, the calculation to run and the script that will run it (e.g. calc_loads.py).

## Checks
How to verify each deliverable, including which script outputs to compare against which numbers.{mcp}

Use MCP tools only to read data the task points to; never call one that creates, changes or sends anything."""

WORKER = """You are an expert engineer working on a task in a Linux container with no internet access. Your tools \
act inside it, and bash starts in {workdir}. A plan for the task follows it; use it, and correct it where the files \
show it is wrong.

COMPUTE WITH CODE, never by hand: every number in a deliverable must be produced by a script you run in the \
container. Save each calculation as a script (calc_*.py) in the workdir, run it, and write the deliverables from the \
script's output. Never compute a number in your head and type it into a deliverable. If a deliverable is a \
spreadsheet, its numbers must come from live formulas or from a script that writes the file.

Write each deliverable exactly where and in the format the task asks, then open it again to check it. When you are \
done, reply without calling a tool: a short summary of each deliverable, where it is, and the key results, plus an \
evidence table with one row per key number: claim -> script that computed it -> file and line where it appears.{mcp}

When a deliverable's data lives on one of the task's own servers, get it through that server's tools rather than \
guessing it."""

VERIFIER = """You check an engineer's work on a task in a Linux container. You get the task, the plan, and the \
engineer's summary of what was delivered. Open every deliverable with your tools (bash starts in {workdir}) and \
check it against the task and the plan. Inspect only the deliverables in the plan, the calc scripts, and the files \
the task points to; do not explore unrelated files or the grading code.

1. Go through the plan's hard constraints C1, C2, ... one by one and give a verdict for each (C1 PASS, C2 FAIL, \
...). Check the path, the format, every part the task asks for.
2. Re-run every calc_*.py script the engineer left in the workdir and compare its output with the numbers in the \
deliverables. Any mismatch is a gap.
3. Do not create or change any file. Work efficiently: batch multiple checks into single bash commands where \
possible.

If everything holds, end your reply with a line containing only the word PASS. Otherwise reply with a numbered list \
of the gaps, each specific enough to fix, starting with the per-constraint verdicts.{mcp}

Use MCP tools only to read; never call one that creates, changes or sends anything."""


def tool(name, description, **params):
  """Builds an OpenAI function-tool definition dictionary.

  Args:
    name: Tool function name.
    description: Human-readable tool description.
    **params: Parameter names mapped to their JSON Schema type strings.

  Returns:
    An OpenAI function-tool specification dictionary.
  """
  return {
      "type": "function",
      "function": {
          "name": name,
          "description": description,
          "parameters": {
              "type": "object",
              "properties": {k: {"type": v} for k, v in params.items()},
              "required": list(params),
          },
      },
  }


TOOLS = [
    tool(
        "bash",
        "Run a bash command (120 s limit) and return its exit code and output.",
        command="string",
    ),
    tool("read_file", "Read a text file.", path="string"),
    tool(
        "write_file",
        "Create or overwrite a file; parent folders are created.",
        path="string",
        content="string",
    ),
]
READ_ONLY = TOOLS[:2]

# --- MCP servers (taskbridge.MCPClient) ---
# A task may offer its own tools on MCP servers (documents, spreadsheets,
# email, CAD, ...). Kit check's mcp stage warns when a harness cannot use
# them, so the worker gets them as tools and the planner/verifier get
# read-only access through the same set. Small tool sets are exposed as
# direct <server>__<tool> functions (same thresholds as the kit's
# custom-openai sample); larger ones go through mcp_find/mcp_call.
DIRECT_MCP_CHARS = 40_000
MAX_MCP_TOOLS = 128
MCP_START_SEC = 120  # give up on one server after this long

MCP_TOOLS = [
    tool(
        "mcp_find",
        "Show the description and parameters of the task's own tools that match a query.",
        query="string",
    ),
    tool(
        "mcp_call",
        "Run one of the task's own tools.",
        tool="string",
        arguments="object",
    ),
]


class MCPServer(taskbridge.MCPClient):
  """Client for one task MCP server and its discovered tools."""

  def __init__(self, session, spec):
    super().__init__(session, spec["url"], client="plan-verify")
    self.name = spec["name"]
    self.tools = []

  async def start(self):
    """Initializes the MCP session and fetches the server's tool list."""
    await self.initialize()
    self.tools = await self.list_tools()

  async def call(self, name, arguments):
    """Calls tool `name` and formats the result as text.

    Args:
      name: Unprefixed MCP tool name on this server.
      arguments: Tool input arguments dictionary.

    Returns:
      Formatted text output (prefixed with `error: ` if `isError` is set).
    """
    result = await self.call_tool(name, arguments)
    parts = []
    content = (
        result.get("content") if isinstance(result.get("content"), list) else []
    )
    for item in content:
      kind = item.get("type") if isinstance(item, dict) else None
      if kind == "text":
        parts.append(str(item.get("text", "")))
      elif kind in ("resource", "resource_link"):
        parts.append(json.dumps(item)[:RESULT_CHARS])
      else:
        parts.append(f"[{kind or 'unknown'} content not shown]")
    if not parts and result.get("structuredContent") is not None:
      parts.append(json.dumps(result["structuredContent"]))
    text = "\n".join(parts) or "(no content)"
    return f"error: {text}" if result.get("isError") else text


def _unique_name(name, taken):
  """Returns `name` unchanged, or suffixed until it is not in `taken`."""
  if name not in taken:
    return name
  n = 2
  while f"{name}_{n}" in taken:
    n += 1
  return f"{name}_{n}"


def find_mcp(query, mcp):
  """Finds registry tools matching `query` and formats their schemas.

  Args:
    query: Tool name, or words describing what is needed.
    mcp: Registry mapping tool name to `(server, tool definition)`.

  Returns:
    Formatted descriptions and JSON parameter schemas of matching tools.
  """
  words = [w for w in re.split(r"[^a-z0-9]+", query.lower()) if w]

  def score(item):
    name, (_, t) = item
    text = f"{name} {t.get('description') or ''}".lower()
    return (
        (query == name) * 100
        + sum(w in name.lower() for w in words) * 3
        + sum(w in text for w in words)
    )

  best = [
      item
      for item in sorted(mcp.items(), key=score, reverse=True)[:8]
      if score(item) > 0
  ]
  if not best:
    return (
        f"no tool matches {query!r}; the names are listed in the system"
        " prompt"
    )
  found = [
      f"{name}: {(t.get('description') or '')[:400]}\nparameters:"
      f" {json.dumps(t.get('inputSchema') or {})}"
      for name, (_, t) in best
  ]
  while len(found) > 1 and len("\n\n".join(found)) > RESULT_CHARS:
    found.pop()
  return "\n\n".join(found)


async def call_mcp_tool(mcp, tool_name, arguments):
  """Calls a registry tool by name and returns its text result."""
  entry = (mcp or {}).get(tool_name)
  if entry is None:
    return f"error: there is no MCP tool named {tool_name}"
  server, t = entry
  try:
    return clip(await server.call(t["name"], arguments))
  except (taskbridge.TaskError, taskbridge.MCPError, ValueError) as exc:
    return f"error: {type(exc).__name__}: {exc}"


async def connect_mcp(session, evidence=None):
  """Connects to the task's MCP servers and builds the tool lists.

  Args:
    session: Active `taskbridge.Session` with `mcp_servers`.
    evidence: Evidence log to record the connected servers and tools in.

  Returns:
    `(registry, read_tools, write_tools, mcp_section)`: the tool registry
    mapping `<server>__<tool>` names to `(server, tool definition)`, the tool
    lists for the read-only roles (planner, verifier) and the writing roles
    (worker, fix), and the system-prompt section describing the MCP tools
    (`""` when the task offers none).
  """
  registry = {}
  for spec in session.mcp_servers:
    if (
        not isinstance(spec, dict)
        or not spec.get("name")
        or not spec.get("url")
    ):
      continue
    server = MCPServer(session, spec)
    try:
      await asyncio.wait_for(server.start(), timeout=MCP_START_SEC)
    except (
        taskbridge.TaskError,
        taskbridge.MCPError,
        ValueError,
        asyncio.TimeoutError,
    ) as exc:
      await session.log(f"MCP server {server.name} unavailable: {exc}")
      continue
    for t in server.tools:
      name = re.sub(r"[^A-Za-z0-9_-]", "_", f"{server.name}__{t['name']}")
      registry[_unique_name(name, registry)] = (server, t)
    await session.log(f"MCP server {server.name}: {len(server.tools)} tools")
  if evidence is not None:
    evidence.record(
        "mcp",
        servers=[s.get("name") for s in session.mcp_servers],
        tools=sorted(registry),
    )
  if not registry:
    return registry, READ_ONLY, TOOLS, ""
  functions = [
      {
          "type": "function",
          "function": {
              "name": name,
              "description": (t.get("description") or t["name"])[:1024],
              "parameters": (
                  {
                      k: v
                      for k, v in (t.get("inputSchema") or {}).items()
                      if k != "$schema"
                  }
                  or {"type": "object", "properties": {}}
              ),
          },
      }
      for name, (_, t) in registry.items()
  ]
  if (
      len(json.dumps(functions)) <= DIRECT_MCP_CHARS
      and len(functions) <= MAX_MCP_TOOLS
  ):
    mcp_tools = functions
    section = (
        "\n\nThe task also offers its own tools (the functions named"
        " <server>__<tool>)."
    )
  else:
    mcp_tools = MCP_TOOLS
    servers = {}
    for name, (server, _) in registry.items():
      servers.setdefault(server.name, []).append(name)
    section = (
        "\n\nThe task also offers its own tools, listed below by server."
        " Call mcp_find to see a tool's parameters, then mcp_call to run"
        " it.\n"
        + "\n".join(
            f"{s}: {', '.join(names)}" for s, names in servers.items()
        )
    )
  return registry, READ_ONLY + mcp_tools, TOOLS + mcp_tools, section


def clip(text):
  """Truncates `text` in the middle if it exceeds `RESULT_CHARS` characters.

  Args:
    text: Tool output string to clip.

  Returns:
    The original string if within `RESULT_CHARS`, or a middle-truncated string.
  """
  if len(text) <= RESULT_CHARS:
    return text
  half = RESULT_CHARS // 2
  return (
      f"{text[:half]}\n[... {len(text) - RESULT_CHARS} characters cut"
      f" ...]\n{text[-half:]}"
  )


def marked(messages):
  """Returns the messages a model call sends.

  For Claude, this is a copy whose system prompt and newest message carry
  `cache_control`. The stored conversation stays as it is, so each turn marks
  its own newest message.

  Args:
    messages: The conversation so far, in OpenAI chat format.

  Returns:
    `messages` itself for other models, or the marked copy for Claude.
  """
  if not CACHE_MARK:
    return messages
  out = list(messages)
  for i in {0, len(out) - 1}:
    m, content = out[i], out[i].get("content")
    if isinstance(content, str) and content:
      out[i] = {
          **m,
          "content": [{
              "type": "text",
              "text": content,
              "cache_control": {"type": "ephemeral"},
          }],
      }
    else:
      out[i] = {**m, "cache_control": {"type": "ephemeral"}}
  return out


def _with_reasoning(messages):
  """Fills in reasoning_content on assistant messages that lack it.

  DeepSeek thinking models refuse a call whose history holds an assistant
  message without reasoning_content. Model turns keep theirs (see agent());
  this covers the synthetic progress note compact() inserts and any turn the
  model answered without reasoning. Mutates in place so the stored history
  stays valid too. A no-op for non-DeepSeek models.

  Args:
    messages: The conversation, modified in place.

  Returns:
    The same list.
  """
  if NEED_REASONING:
    for m in messages:
      if m.get("role") == "assistant" and not m.get("reasoning_content"):
        m["reasoning_content"] = REASONING_PLACEHOLDER
  return messages


class Evidence:
  """Structured per-task evidence log, written on the harness host.

  The file lives in $EVIDENCE_DIR (default /tmp/ple-evidence) and never
  inside the task container, so it cannot pollute deliverables or break a
  task that forbids extra files. Key events are also sent through
  session.log() so the trajectory carries them.
  """

  def __init__(self, session_id):
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id or "local")
    self.path = os.path.join(EVIDENCE_DIR, f"{safe}.jsonl")
    self._file = open(self.path, "a", encoding="utf-8")

  def record(self, event, **fields):
    rec = {"ts": round(time.time(), 3), "event": event, **fields}
    self._file.write(json.dumps(rec, ensure_ascii=False) + "\n")
    self._file.flush()

  def close(self):
    self._file.close()


class CostTracker:
  """Cumulative token usage for one task, against TOKEN_BUDGET."""

  def __init__(self, budget):
    self.budget = budget
    self.input_tokens = 0
    self.output_tokens = 0

  def add(self, input_tokens, output_tokens):
    self.input_tokens += input_tokens
    self.output_tokens += output_tokens

  @property
  def total(self):
    return self.input_tokens + self.output_tokens

  @property
  def exceeded(self):
    return self.total >= self.budget


async def run_tool(session, fn, role, evidence, mcp=None):
  """Executes a single tool call in the task container and returns its output.

  Rewrites `fn["arguments"]` as canonical JSON (or `"{}"` if invalid) so the
  conversation history remains valid for strict endpoints. Any tool failure is
  returned as an error string for the model. Every write is recorded in the
  evidence log with path, byte size and sha256.

  Args:
    session: Active `taskbridge.Session` connected to the task container.
    fn: Function-call dictionary containing `name` and `arguments`.
    role: Agent role making the call ("planner", "worker", "verifier", "fix").
    evidence: Evidence log to record the call in.
    mcp: Registry mapping `<server>__<tool>` names to `(server, tool
      definition)` for the task's MCP servers, or `None` when the task
      offers none.

  Returns:
    The clipped tool output or error string to send back to the model.
  """
  name = fn["name"]
  try:
    args = json.loads(fn["arguments"] or "{}")
    if not isinstance(args, dict):
      raise TypeError("not a JSON object")
    fn["arguments"] = json.dumps(args, allow_nan=False)
  except (TypeError, ValueError) as exc:
    fn["arguments"] = "{}"
    return (
        f"error: the arguments are not a JSON object ({exc}); send the call"
        " again"
    )
  if evidence is not None:
    evidence.record("tool", role=role, name=name,
                    args=str(fn["arguments"])[:200])
  try:
    if name == "bash":
      result = await session.exec(args["command"])
      return clip(f"exit code {result['exit_code']}\n{result['output']}")
    if name == "read_file":
      return clip(
          (await session.read_file(args["path"])).decode("utf-8", "replace")
      )
    if name == "write_file":
      content = args["content"]
      size = await session.write_file(args["path"], content)
      if evidence is not None:
        evidence.record(
            "write", role=role, path=args["path"], bytes=size,
            sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )
      return f"wrote {size} bytes to {args['path']}"
    if name == "mcp_find":
      return clip(find_mcp(args.get("query", ""), mcp or {}))
    if name == "mcp_call":
      call_args = args.get("arguments") or {}
      if not isinstance(call_args, dict):
        return "error: mcp_call arguments must be a JSON object"
      return await call_mcp_tool(mcp, args.get("tool", ""), call_args)
    if mcp is not None and name in mcp:
      server, t = mcp[name]
      return clip(await server.call(t["name"], args))
    return f"error: there is no tool named {name}"
  # ValueError: content with an unpaired surrogate
  except (taskbridge.TaskError, taskbridge.MCPError, KeyError, ValueError) as exc:
    return f"error: {type(exc).__name__}: {exc}"


async def recover(session, messages, exc, waits, until):
  """Handles a failed model call and returns whether to retry.

  On a 400 context-overflow error, shortens the oldest tool results. On a rate
  limit, server error, or connection drop, sleeps for the next interval in
  `RETRY_WAITS` (capped at the step's wrap-up deadline). Errors are logged
  rather than added to the conversation.

  Args:
    session: Active `taskbridge.Session` for the task.
    messages: The conversation, shortened in place when the context is full.
    exc: The `openai.APIError` the call raised.
    waits: Iterator over the remaining intervals in `RETRY_WAITS`.
    until: Seconds before the time limit at which the step wraps up.

  Returns:
    `True` to retry the call, `False` to re-raise.
  """
  status = getattr(exc, "status_code", None)  # None: no HTTP answer
  if status == 400 and CONTEXT_FULL.search(str(exc)):
    return await shorten(session, messages)
  wait = next(waits, None)
  if wait is None or not (
      status is None or status in (408, 409, 429) or status >= 500
  ):
    return False
  left = session.time_left()
  if left is not None and left > until:
    wait = min(wait, left - until)
  await session.log(
      f"model call failed ({str(exc)[:200]}); again in {wait:.0f} s"
  )
  await asyncio.sleep(wait)
  return True


async def shorten(session, messages):
  """Truncates the older half of long tool results to `CUT_KEEP` characters.

  Repeated calls truncate progressively newer results. Claude's thinking blocks
  are kept intact because each block's signature is self-contained and Claude
  rejects tool loops if the latest assistant turn loses its thinking block.

  Args:
    session: Active `taskbridge.Session` for the task.
    messages: The conversation, modified in place.

  Returns:
    Whether any tool result was truncated.
  """
  long = [
      n
      for n, m in enumerate(messages)
      if m["role"] == "tool" and len(m["content"]) > 2 * CUT_KEEP
  ]
  if not long:
    return False
  cut = long[: max(1, len(long) // 2)]
  for n in cut:
    messages[n]["content"] = (
        messages[n]["content"][:CUT_KEEP]
        + "\n[... cut to save context; run the tool again if needed]"
    )
  await session.log(f"context full: cut {len(cut)} old tool results")
  return True


def _message_text(message):
  """Renders one chat message as plain text for a compaction summary."""
  role = message.get("role", "?")
  content = message.get("content")
  if isinstance(content, list):
    content = " ".join(
        part.get("text", "") for part in content if isinstance(part, dict)
    )
  text = content or ""
  calls = message.get("tool_calls") or []
  if calls:
    called = "; ".join(
        f"{c.get('function', {}).get('name', '?')}"
        f" {str(c.get('function', {}).get('arguments', ''))[:200]}"
        for c in calls
    )
    text = (text + f" [tool calls: {called}]").strip()
  return f"{role}: {text}"


async def compact(session, messages, evidence):
  """Summarizes the middle of a long conversation into a progress note.

  Keeps the system prompt, the original task, a model-written summary of the
  dropped middle, and the most recent turns (starting at a user message so no
  tool result is left orphaned). Returns False when there is nothing safe to
  drop.

  Args:
    session: Active `taskbridge.Session` for the task.
    messages: The conversation, rebuilt in place.
    evidence: Evidence log to record the compaction in.

  Returns:
    Whether the conversation was compacted.
  """
  if len(messages) <= COMPACT_KEEP_TURNS + 3:
    return False
  tail = messages[-COMPACT_KEEP_TURNS:]
  start = next(
      (i for i, m in enumerate(tail) if m.get("role") == "user"), None
  )
  if start is None:
    tail = []
    middle = messages[2:]
  else:
    tail = tail[start:]
    middle = messages[2:len(messages) - len(tail)]
  if not middle:
    return False
  transcript = "\n".join(_message_text(m) for m in middle)[:60000]
  try:
    reply = await asyncio.wait_for(
        client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarize this engineering work transcript into a "
                        "compact progress note: what was planned, what "
                        "deliverables exist and where, the key numbers and "
                        "how they were computed, what is still open, and any "
                        "errors seen. Keep it under 1500 words."
                    ),
                },
                {"role": "user", "content": transcript},
            ],
            tool_choice="none",
            reasoning_effort=EFFORT,
            max_completion_tokens=4000,
        ),
        600,
    )
  except (openai.APIError, asyncio.TimeoutError) as exc:
    await session.log(f"compaction failed ({str(exc)[:150]}); keeping history")
    return False
  note = (reply.choices[0].message.content or "").strip()
  if not note:
    return False
  head = messages[:2]
  note_msg = {
      "role": "assistant",
      "content": "[Progress note after compaction]\n" + note,
  }
  if NEED_REASONING:
    # a fabricated assistant message still needs reasoning_content, or the
    # next DeepSeek call is refused
    note_msg["reasoning_content"] = (
        "[Compacted history: the dropped middle was summarized into the"
        " progress note above.]"
    )
  messages[:] = head + [note_msg] + tail
  if evidence is not None:
    evidence.record("compact", kept_turns=len(tail), note_chars=len(note))
  await session.log(
      f"context compacted: kept {len(tail)} recent turns, note {len(note)} chars"
  )
  return True


def conversation(system, session, prompt, mcp=""):
  """Creates the initial `[system, user]` message list for a role.

  Args:
    system: System prompt template containing `{workdir}` and `{mcp}`.
    session: Active `taskbridge.Session` providing `workdir`.
    prompt: Initial user message content.
    mcp: System-prompt section describing the task's MCP tools (`""` when the
      task offers none).

  Returns:
    A two-element list of OpenAI chat message dictionaries.
  """
  return [
      {
          "role": "system",
          "content": system.format(workdir=session.workdir, mcp=mcp),
      },
      {"role": "user", "content": prompt},
  ]


def _chars(messages):
  """Total characters across all message contents, for compaction."""
  total = 0
  for m in messages:
    content = m.get("content")
    if isinstance(content, str):
      total += len(content)
    elif isinstance(content, list):
      total += sum(
          len(p.get("text", "")) for p in content if isinstance(p, dict)
      )
  return total


async def agent(session, messages, tools, until, final=False, role="worker",
                evidence=None, costs=None, max_turns=80, mcp=None):
  """Runs the tool-calling loop on `messages` until the model answers.

  Enforces the per-role tool-turn cap, the per-task token budget and the
  wrap-up deadline; compacts the conversation when it grows past
  COMPACT_CHARS. When `final` is `True`, updates `session.final_message`
  with each non-empty response.

  Args:
    session: Active `taskbridge.Session` for the task.
    messages: Mutable conversation history list.
    tools: List of OpenAI tool definitions available to this role.
    until: Remaining task time threshold (in seconds) at which this step wraps
      up.
    final: Whether to record responses in `session.final_message`.
    role: Agent role, for evidence and turn-cap purposes.
    evidence: Evidence log to record tool calls and usage in.
    costs: CostTracker for the task token budget.
    max_turns: Hard cap on tool-call turns for this role.
    mcp: Registry mapping `<server>__<tool>` names to `(server, tool
      definition)` for the task's MCP servers, or `None` when the task
      offers none.

  Returns:
    The final text response produced by this agent step.
  """
  # head: the start of an answer cut off at the output limit
  nudges, head, waits = 0, "", iter(RETRY_WAITS)
  turns = 0
  while True:
    left = session.time_left()  # None when the task has no time limit
    over_budget = costs is not None and costs.exceeded
    capped = turns >= max_turns
    wrap_up = (left is not None and left < until) or over_budget or capped
    if over_budget and messages[-1:] != [BUDGET_MSG]:
      messages.append(dict(BUDGET_MSG))
    elif capped and messages[-1:] != [TURN_CAP_MSG]:
      messages.append(dict(TURN_CAP_MSG))
    elif left is not None and left < until and messages[-1:] != [WRAP_UP]:
      messages.append(dict(WRAP_UP))
    if _chars(messages) > COMPACT_CHARS:
      await compact(session, messages, evidence)
    # A call still running when the wrap-up starts is stopped. The wrap-up
    # call itself gets at most WRAP_UP_SEC and stops at the time limit; if it
    # takes longer, the step returns head, its answer so far.
    if left is None:
      timeout = None
    elif wrap_up:
      timeout = min(WRAP_UP_SEC, left)
    else:
      timeout = left - until
    try:
      reply = await asyncio.wait_for(
          client.chat.completions.create(
              model=MODEL,
              messages=marked(_with_reasoning(messages)),
              tools=tools,
              tool_choice="none" if wrap_up else "auto",
              reasoning_effort=EFFORT,
              max_completion_tokens=MAX_OUTPUT_TOKENS,
          ),
          timeout,
      )
    except asyncio.TimeoutError:
      if wrap_up:
        return head
      continue
    except openai.APIError as exc:
      if await recover(session, messages, exc, waits, until):
        continue
      raise
    waits = iter(RETRY_WAITS)
    if reply.usage:
      details = getattr(reply.usage, "prompt_tokens_details", None)
      cached = getattr(details, "cached_tokens", 0) or 0
      if costs is not None:
        costs.add(reply.usage.prompt_tokens, reply.usage.completion_tokens)
        if evidence is not None:
          evidence.record(
              "usage", role=role, model=MODEL,
              input_tokens=reply.usage.prompt_tokens,
              output_tokens=reply.usage.completion_tokens,
              cached_tokens=cached, cumulative_tokens=costs.total,
          )
      await session.usage(
          MODEL,
          reply.usage.prompt_tokens,
          reply.usage.completion_tokens,
          cached,
      )
    message, cut = (
        reply.choices[0].message,
        reply.choices[0].finish_reason == "length",
    )
    text = head + (message.content or "")
    if final and text:
      session.final_message = text
    # sent back as returned: Gemini refuses a tool call without its thought
    # signature, and DeepSeek refuses a turn without its reasoning_content
    turn = message.model_dump(exclude_none=True)
    if NEED_REASONING and not turn.get("reasoning_content"):
      # the model answered this turn without reasoning; the next call still
      # needs the field present on the message we send back
      turn["reasoning_content"] = REASONING_PLACEHOLDER
    if wrap_up or not message.tool_calls:
      turn.pop("tool_calls", None)  # calls made at the wrap-up are not run
      if message.content:
        # a fix round, or the request to go on after a cut-off reply,
        # continues from this answer
        messages.append(turn)
      if wrap_up or (message.content and not cut) or nudges == MAX_NUDGES:
        return text
      nudges, head = nudges + 1, text
      messages.append({
          "role": "user",
          "content": (
              "Your reply was cut off; continue where it stopped."
              if cut
              else "Continue, and finish with your answer."
          ),
      })
      continue
    nudges, head = 0, ""
    turns += 1
    messages.append(turn)
    for call in turn["tool_calls"]:
      await session.log(
          f"{call['function']['name']} {call['function']['arguments'][:200]}"
      )
      content = await run_tool(session, call["function"], role, evidence,
                               mcp=mcp)
      messages.append(
          {"role": "tool", "tool_call_id": call["id"], "content": content}
      )
    if left is not None:
      messages[-1]["content"] += (
          f"\n[{int((session.time_left() - until) // 60)} minutes left for"
          " this step]"
      )


async def solve(session):
  """Executes the planner, worker, and verifier loop for `session`.

  Args:
    session: Active `taskbridge.Session` for the task.

  Returns:
    The final summary message from the worker.
  """
  evidence = Evidence(session.task.get("session"))
  costs = CostTracker(TOKEN_BUDGET)
  total = session.time_left()
  mcp, read_tools, write_tools, mcp_section = await connect_mcp(
      session, evidence
  )

  def until(share):
    """Computes the remaining seconds when `share` of total task time is left.

    Args:
      share: Fraction of total task time remaining (`0.0` to `1.0`).

    Returns:
      Remaining seconds threshold, or `0` if the task has no time limit.
    """
    return 0 if total is None else total * share

  def time_for(wrap_up):
    """Checks whether at least `MIN_STEP_SEC` remains before `wrap_up`.

    Args:
      wrap_up: Remaining seconds threshold at which the next step wraps up.

    Returns:
      `True` if there is no deadline or at least `MIN_STEP_SEC` before
      `wrap_up`.
    """
    left = session.time_left()
    return left is None or left - wrap_up >= MIN_STEP_SEC

  task = session.instruction
  try:
    plan = await agent(
        session,
        conversation(PLANNER, session, task, mcp=mcp_section),
        read_tools,
        until(1 - PLAN_SHARE),
        role="planner",
        evidence=evidence,
        costs=costs,
        max_turns=MAX_TOOL_TURNS["planner"],
        mcp=mcp,
    )
    evidence.record("plan", text=plan)
    await session.log(f"plan: {plan[:1000]}")
    work = conversation(WORKER, session, f"{task}\n\n## Plan\n{plan}",
                        mcp=mcp_section)
    summary = await agent(
        session, work, write_tools, until(CHECK_SHARE), final=True,
        role="worker", evidence=evidence, costs=costs,
        max_turns=MAX_TOOL_TURNS["worker"], mcp=mcp,
    )
    for round_no in range(1, MAX_FIX_ROUNDS + 1):
      if not time_for(until(CHECK_SHARE / 2)):
        break
      review = (
          f"## Task\n{task}\n\n## Plan\n{plan}\n\n## The engineer's"
          f" summary\n{summary}"
      )
      verdict = await agent(
          session,
          conversation(VERIFIER, session, review, mcp=mcp_section),
          read_tools,
          until(CHECK_SHARE / 2),
          role="verifier",
          evidence=evidence,
          costs=costs,
          max_turns=MAX_TOOL_TURNS["verifier"],
          mcp=mcp,
      )
      evidence.record("verdict", round=round_no, text=verdict)
      await session.log(f"verifier: {verdict[:1000]}")
      # PASS: a standalone PASS line and no FAIL anywhere, or an empty reply,
      # names no gaps
      passed = bool(
          re.search(r"(?m)^\s*PASS\s*$", verdict)
          and "FAIL" not in verdict.upper()
      ) or verdict.strip("*_. \r\n\t").upper() in ("", "PASS")
      if passed or not time_for(WRAP_UP_SEC):
        break
      work.append({
          "role": "user",
          "content": (
              "A reviewer checked your files and found these"
              f" gaps:\n{verdict}\n\nFix them, then reply with your updated"
              " summary."
          ),
      })
      summary = (
          await agent(
              session, work, write_tools, WRAP_UP_SEC, final=True, role="fix",
              evidence=evidence, costs=costs,
              max_turns=MAX_TOOL_TURNS["fix"], mcp=mcp,
          )
          or summary
      )
    evidence.record("summary", text=summary or session.final_message)
    try:
      inv = await session.exec(
          "find . -type f -not -path './.git/*' -exec sha256sum {} + "
          "2>/dev/null | sort"
      )
      evidence.record("files", inventory=inv["output"][:4000])
    except taskbridge.TaskError:
      pass
    evidence.record(
        "cost", input_tokens=costs.input_tokens,
        output_tokens=costs.output_tokens, total_tokens=costs.total,
    )
    return summary or session.final_message
  finally:
    evidence.close()


if __name__ == "__main__":
  taskbridge.run(solve)
