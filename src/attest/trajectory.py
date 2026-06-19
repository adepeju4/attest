"""
The data model for an agent run.

The whole philosophy of attest lives in one distinction:

  - `thought` / `final_answer`  = what the agent *says* it did   ->NOT trusted as truth
  - `ToolCall.output`           = what *actually happened*       ->the evidence we trust

LLM-as-judge fails because it grades the agent's narrative. attest grades the
agent's claims against the recorded tool outputs (`evidence()`), which is why
rewording the chain-of-thought can't fool it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A single tool invocation and its REAL recorded output (our ground truth)."""

    name: str
    arguments: dict = Field(default_factory=dict)
    output: str


class Step(BaseModel):
    """One step of the agent loop: optional reasoning + an optional tool call."""

    thought: str | None = None
    tool_call: ToolCall | None = None


class Trajectory(BaseModel):
    """A full agent run for one task."""

    task: str
    # The agent's legitimate authority (trusted): role, rules, tool descriptions.
    # Needed to judge tool-use correctness and to detect prompt injection.
    system_prompt: str | None = None
    # The tool names the agent is permitted to call (declarative metadata only —
    # attest never executes tools). Used by the deterministic allowed-tool check.
    allowed_tools: list[str] | None = None
    steps: list[Step] = Field(default_factory=list)
    final_answer: str

    def evidence(self) -> str:
        """
        The only thing attest trusts: the concatenated REAL tool outputs.
        A claim is 'grounded' only if it's supported by this text.
        """
        blocks: list[str] = []
        for i, step in enumerate(self.steps):
            if step.tool_call is not None:
                tc = step.tool_call
                blocks.append(f"[{i}] {tc.name}({tc.arguments}) -> {tc.output}")
        return "\n".join(blocks)

    def has_evidence(self) -> bool:
        return any(s.tool_call is not None for s in self.steps)

    def tool_calls(self) -> list[tuple[int, ToolCall]]:
        """(step index, ToolCall) pairs — the calls to review for tool-use correctness."""
        return [(i, s.tool_call) for i, s in enumerate(self.steps) if s.tool_call is not None]
