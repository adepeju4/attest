# Changelog

All notable changes to `agent-attest` are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/) and [Semantic Versioning](https://semver.org/).

## [0.5.1] - 2026-06-25

Judge precision and reproducibility. These changes were driven by red-teaming a live agent
(a civic assistant) whose trajectories exposed two false-positive patterns in the checks.

### Fixed
- **`check_role_adherence` now sees the agent's tool calls and their outputs.** The judge
  previously received only the system prompt, task, and final answer, so any policy that
  conditions behaviour on tool use (e.g. "only state a phone number a tool returned") was
  judged blind and false-flagged compliant answers as `out_of_scope_complied`. The tool calls
  and outputs are now part of the judge's context.
- **`check_role_adherence` honours explicit policy carve-outs.** A policy with a headline
  domain plus named exceptions (safety, emergencies, first aid, helplines) no longer reads
  those exceptions as out of scope; honouring a behaviour the policy explicitly permits is
  `in_scope`.
- **`check_injection` deep mode no longer flags the agent for answering the user's own
  request.** Returning information a tool surfaced, or retrieving to fulfil the user's task,
  is authorized even when the same tool output also carried injected instructions. Only acting
  on a new, unrequested goal counts as a compromise.

### Added
- **Self-consistency voting (`attest._llm.vote`).** The verdict checks (`check_role_adherence`
  and `check_injection` deep) now sample the judge `samples` times (default 3) and take the
  majority verdict, which stabilises non-deterministic providers. Configurable per call via
  `samples=`.

### Changed
- **LLM calls are pinned to `temperature=0`** at the chokepoint for reproducibility.
- **`PROMPT_VERSION` bumped to `2026.06.25`** to reflect the revised role and injection prompts.

[0.5.1]: https://pypi.org/project/agent-attest/0.5.1/
