"""The data model for an agent run. Tool outputs are the evidence attest trusts."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    name: str
    arguments: dict = Field(default_factory=dict)
    output: str


class Step(BaseModel):
    thought: str | None = None
    tool_call: ToolCall | None = None


class Trajectory(BaseModel):
    task: str
    system_prompt: str | None = None
    allowed_tools: list[str] | None = None
    steps: list[Step] = Field(default_factory=list)
    final_answer: str

    def evidence(self) -> str:
        """The concatenated real tool outputs — the only text attest trusts."""
        blocks: list[str] = []
        for i, step in enumerate(self.steps):
            if step.tool_call is not None:
                tc = step.tool_call
                blocks.append(f"[{i}] {tc.name}({tc.arguments}) -> {tc.output}")
        return "\n".join(blocks)

    def has_evidence(self) -> bool:
        return any(s.tool_call is not None for s in self.steps)

    def tool_calls(self) -> list[tuple[int, ToolCall]]:
        return [(i, s.tool_call) for i, s in enumerate(self.steps) if s.tool_call is not None]
