# attest

**Evidence-grounded evaluation for AI agent trajectories.** Judge an agent by checking
its claims against the *actual tool outputs* — not by asking another LLM "did this look
good?"

```bash
uv tool install agent-attest    # distribution name; the CLI + import are `attest`
attest run your-trajectory.json
```

## Why

Evaluating AI agents usually means **LLM-as-judge** — one model grading another. Two
problems attest tackles directly:

1. **It grades the story, not the work.** A holistic "is this good?" judge reads the
   agent's confident narrative and can wave through specific ungrounded claims buried in
   an otherwise-solid answer. *(See [Gaming the Judge, arXiv:2601.14691](https://arxiv.org/pdf/2601.14691).)*
2. **The scores have no error bars.** Most tools report a bare pass rate, so teams chase
   differences that are pure noise.

**attest's approach:** never trust what the model *says* it did. Extract the answer's
claims and verify **each one against the recorded tool outputs**, report with confidence
intervals, and back every verdict with the exact evidence span. The same "verify against
real state, not narrative" primitive underpins the strongest prompt-injection defenses
(AgentDojo, CaMeL) — so it's also the foundation for security checks later.

## What it does

attest evaluates a **trajectory** (an agent run: tool calls, their real outputs, the
final answer) across dimensions and returns one combined report:

- **Faithfulness** — extracts atomic claims from the answer and verifies each against the
  tool outputs (`supported` / `unsupported` / `unverifiable`), with a quoted evidence
  span. The verifier never sees the agent's reasoning, so a reworded narrative can't move
  the verdict.
- **Tool-use correctness** — were the right tools called, with no unhandled errors?
  Deterministic by default (no API key); an optional LLM check judges tool *choice*.
- **Prompt-injection flag** — scans untrusted tool outputs for injection payloads
  (deterministic) and, with `--deep`, an *effect-based* check for whether the agent took
  an action the principal never authorized — catching **novel** injections, not just known
  phrasings like "ignore previous instructions".
- **Role adherence** — did the agent stay within the role and scope its system prompt
  defines? Catches **jailbreaks** the user types directly (out-of-scope requests, "ignore
  your instructions", attempts to change role or leak the policy) — the *principal-side*
  threat that injection detection deliberately ignores.
- **One report** — an `overall_score`, per-dimension scores, and Wilson 95% confidence
  intervals, all serializable to JSON.
- **Framework-agnostic** — a LangChain/LangGraph adapter turns any agent run into a
  trajectory; bring your own.
- **Read-only & safe** — attest only reads a *recorded* trajectory. It never executes
  tools, calls the agent, or needs your tools' credentials.

## How it works

```
final_answer ──extract claims──▶ [atomic claims]
each claim   ──verify against──▶ supported · unsupported · unverifiable   (evidence = tool outputs only)
                  evidence

tool calls   ──allowed? error-handled? appropriate?──▶ tool-use score
tool outputs ──payload scan + authorization check────▶ injection findings (suspicious / compromised)
                              │
                              ▼
              one TrajectoryReport  (overall + per-dimension + 95% CIs)
```

The key design choice: the verifier sees **only the claim and the evidence — never the
agent's reasoning.** That's what keeps it grounded.

## Usage

**CLI**

```bash
attest stats 41 50                # a pass rate with its Wilson 95% CI (no API key)
attest tools trajectory.json      # tool-use correctness — deterministic, no API key
attest injection trajectory.json  # prompt-injection scan — deterministic, no API key
attest role  trajectory.json      # role-adherence / jailbreak check (needs an API key)
attest run   trajectory.json      # full report: faithfulness + tool-use + overall
attest demo  trajectory.json      # naive LLM-judge vs attest, side by side
attest models openai              # list a provider's models (live if its key is set)

attest run trajectory.json --provider openai --model gpt-4o-mini   # any provider
```

**Library**

```python
from attest import Attest

judge = Attest(key="sk-ant-...")   # or Attest() to read ANTHROPIC_API_KEY from the env
report = judge.evaluate(traj)      # traj: a Trajectory (e.g. from the LangGraph adapter)
print(report.overall_score)
print(report.model_dump_json(indent=2))

judge.tool_use(traj)               # tool-use correctness
judge.injection(traj, deep=True)   # prompt-injection scan
judge.role_adherence(traj)         # role-adherence / jailbreak check
judge.stats(41, 50)                # pass rate + Wilson 95% CI (no API call)
```

Configure the provider, key, and model once, then evaluate many trajectories. Prefer
dependency injection? The functional API is still there — `from attest import evaluate,
check_tool_use`.

### Providers

attest runs on **Anthropic, OpenAI, or Gemini** behind one interface (via
[instructor](https://github.com/567-labs/instructor) for reliable structured output):

```python
Attest(provider="openai", model="gpt-4o-mini")    # key from OPENAI_API_KEY
Attest(provider="gemini")                          # key from GEMINI_API_KEY / GOOGLE_API_KEY
Attest.providers()                                 # ['anthropic', 'openai', 'gemini']
Attest.models("openai")                            # live list if OPENAI_API_KEY is set, else curated
```

The base install ships Anthropic. OpenAI and Gemini are optional extras:

```bash
pip install agent-attest             # base (Anthropic), exposes `import attest`
pip install "agent-attest[openai]"   # adds the OpenAI SDK
pip install "agent-attest[gemini]"   # adds the Google GenAI SDK
pip install "agent-attest[all]"      # both
```

Each provider reads its own key (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or
`GEMINI_API_KEY`) — a local `.env` is picked up automatically. Verification defaults to a
small/fast model per provider: cents, not dollars.

## Develop

```bash
uv run pytest                   # 66 tests, no API key needed (the LLM is mocked/injected)
```

Running the CLI from source before install: prefix with `uv run` (e.g. `uv run attest stats 41 50`).

## Layout

```
src/attest/
├── trajectory.py        # core data model — the thought-vs-tool-output distinction
├── _llm.py              # Anthropic wrapper: call(output=PydanticModel) -> validated
├── cli.py               # attest stats / tools / injection / role / run / demo / models
├── checks/              # the evaluation dimensions
│   ├── verify.py          # faithfulness: extract_claims + grounded_verifier
│   ├── tool_use.py        # tool-use correctness (deterministic + optional LLM)
│   ├── injection.py       # prompt-injection: payload scan + authorization check
│   ├── role.py            # role adherence / jailbreak resistance
│   └── judge_baseline.py  # the naive LLM-as-judge attest is built to beat
├── scoring/
│   ├── report.py          # evaluate() -> combined TrajectoryReport + overall_score
│   └── stats.py           # Wilson CI + two-proportion significance
└── adapters/
    └── langgraph.py       # LangChain/LangGraph run -> Trajectory
tests/                   # all offline (the LLM is mocked/injected)
examples/                # sample trajectories (clean, gamed, injection, jailbreak)
```

## Status

Early but working. **Faithfulness**, **tool-use correctness**, and a **prompt-injection
flag** (deterministic scan + effect-based authorization check) are built, tested, and
validated live against a real LangGraph agent. Next up: an answer-type-aware verifier and
self-contradiction. Not yet on PyPI.
