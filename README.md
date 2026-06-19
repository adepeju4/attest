# attest

**Evidence-grounded evaluation for AI agent trajectories.** Judge an agent by
verifying its claims against the *actual tool outputs* — not by asking another LLM
"did this look good?"

```bash
uv tool install attest   # (once published)
attest run trajectory.json
```

## Why this exists

2026 research showed the tools we use to measure AI are broken in two ways this
project attacks directly:

1. **LLM-as-judge can be gamed.** Rewording an agent's chain-of-thought inflates a
   judge's false-positive rate by up to ~90% — because the judge grades the agent's
   *story*, not what it did. *([Gaming the Judge, arXiv:2601.14691](https://arxiv.org/pdf/2601.14691))*
2. **Eval scores have no error bars.** Most tools report a bare pass rate, so teams
   chase differences that are pure noise.

**attest's bet:** stop trusting what the model *says* it did; verify each claim
against the recorded tool outputs, and report results with confidence intervals.
The same "verify against real state, not narrative" primitive is what powers the
strongest prompt-injection defenses (AgentDojo, CaMeL) — so this is also the
foundation for security work later.

## How it works

```
final_answer ──extract_claims──▶ [atomic claims]
each claim   ──verify vs────────▶ SUPPORTED · UNSUPPORTED · UNVERIFIABLE
                  evidence()                 (evidence = REAL tool outputs only)
score = supported / checkable,  reported with a Wilson 95% CI
```

The key design choice (in [`verify.py`](src/attest/verify.py)): the verifier sees
**only the claim and the evidence — never the agent's reasoning.** That's what makes
it resistant to chain-of-thought gaming.

## Status & build order

The core is built and tested:

- [x] **Step 0 — scaffold:** trajectory schema, stats (Wilson CI + significance),
  aggregation core, CLI, tests.
- [ ] **Step 1 — audit existing tools** (DeepEval, Inspect, Ragas, Braintrust): does
  anyone already do trajectory/evidence grounding? Avoid reinventing. *(half a day)*
- [x] **Step 2 — `extract_claims()`**: LLM structured output → atomic claims.
- [x] **Step 3 — `grounded_verifier()`**: the core entailment check, NLI-style
  (entailment→supported, contradiction→unsupported, neutral→unverifiable), on Haiku
  via forced tool-use.
- [x] **Step 4 — the demo:** `attest demo` runs a naive LLM-judge and attest side by
  side. On `examples/trajectory.json` (which claims "Paris is larger than Berlin"
  while the tool outputs show Berlin is bigger), the naive judge tends to *pass* the
  confident answer while attest flags the claim `UNSUPPORTED`.
- [ ] **Step 5 — ship:** README GIF, publish to PyPI, write the blog post
  ("The tools we use to measure AI are broken — here's the evidence").

## Develop

```bash
uv run pytest                                  # 15 tests, no API key needed

# CLI — in this workspace the path has a space, which makes uv drop the installed
# entry point between runs; the --reinstall-package flag makes each run reliable:
uv run --reinstall-package attest attest stats 41 50
uv run --reinstall-package attest attest run examples/trajectory.json
uv run --reinstall-package attest attest demo examples/trajectory.json
```

(Once installed from PyPI — `uv tool install attest` — it's just `attest demo …`.)

`run` and `demo` need a real `ANTHROPIC_API_KEY` in the shared workspace-root `.env`
(`load_dotenv()` finds it automatically). Verification runs on Haiku — cents, not dollars.

## Layout

```
src/attest/
├── trajectory.py        # core data model — the thought-vs-tool-output distinction
├── _llm.py              # Anthropic wrapper: call(output=PydanticModel) -> validated
├── cli.py               # attest stats / tools / run / demo
├── checks/              # the evaluation dimensions
│   ├── verify.py          # faithfulness: extract_claims + grounded_verifier
│   ├── tool_use.py        # tool-use correctness (deterministic + optional LLM)
│   └── judge_baseline.py  # the naive LLM-as-judge attest is built to beat
├── scoring/
│   ├── report.py          # evaluate() -> combined TrajectoryReport + overall_score
│   └── stats.py           # Wilson CI + two-proportion significance
└── adapters/
    └── langgraph.py       # LangChain/LangGraph run -> Trajectory
tests/                   # all offline (LLM mocked/injected)
examples/                # sample trajectories + codesprint_to_attest.py (live integration)
```
