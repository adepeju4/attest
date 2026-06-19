"""
Adapters — turn another framework's agent run into an attest Trajectory.

attest only judges Trajectories, so it works with *any* agent framework once you
map that framework's run onto the schema. This module handles LangChain /
LangGraph: the message list you get from `agent.invoke(...)["messages"]`.

Deliberately duck-typed — no langchain import — so attest stays dependency-light
and this keeps working across langchain versions.
"""

from __future__ import annotations

from ..trajectory import Step, ToolCall, Trajectory


def _flatten(content) -> str:
    """LangChain/Gemini content is sometimes a list of blocks; flatten to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in content]
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def from_langgraph_messages(
    messages,
    *,
    task: str | None = None,
    final_answer: str | None = None,
    system_prompt: str | None = None,
    allowed_tools: list[str] | None = None,
    response_tool: str | None = None,
) -> Trajectory:
    """
    Convert a LangGraph/LangChain message list into an attest Trajectory.

    - `task` defaults to the first HumanMessage.
    - `final_answer` defaults to the last AIMessage's text. Pass it explicitly if
      your agent returns a structured response (e.g. ToolStrategy / `structured_response`).
    - `system_prompt` defaults to a SystemMessage in the list, if present. It's the
      agent's authority — needed for tool-use correctness and injection detection.
    - `allowed_tools` is the agent's permitted tool names (declarative metadata, never
      executed). Pass the agent's tool list; not reliably recoverable from messages alone.

    Each tool call in an AIMessage is paired (by id) with the ToolMessage that
    recorded its output — that output is the *evidence* attest will verify against.
    """
    outputs: dict[str, str] = {}
    for m in messages:
        tcid = getattr(m, "tool_call_id", None)
        if tcid is not None:  # it's a ToolMessage
            outputs[tcid] = _flatten(getattr(m, "content", ""))

    steps: list[Step] = []
    first_human: str | None = None
    detected_system: str | None = None
    last_ai_text = ""
    for m in messages:
        kind = type(m).__name__
        content = _flatten(getattr(m, "content", ""))
        if kind == "SystemMessage" and detected_system is None:
            detected_system = content
        if kind == "HumanMessage" and first_human is None:
            first_human = content
        tool_calls = getattr(m, "tool_calls", None) or []
        for tc in tool_calls:
            # Skip the structured-output synthetic tool (e.g. ToolStrategy's response
            # model) — it's how the agent returns its answer, not a tool-use decision.
            if response_tool and tc.get("name") == response_tool:
                continue
            steps.append(
                Step(
                    thought=content or None,
                    tool_call=ToolCall(
                        name=tc.get("name", "tool"),
                        arguments=tc.get("args", {}) or {},
                        output=outputs.get(tc.get("id", ""), ""),
                    ),
                )
            )
        if kind == "AIMessage" and not tool_calls and content:
            last_ai_text = content

    return Trajectory(
        task=task or first_human or "",
        system_prompt=system_prompt if system_prompt is not None else detected_system,
        allowed_tools=allowed_tools,
        steps=steps,
        final_answer=final_answer if final_answer is not None else last_ai_text,
    )
