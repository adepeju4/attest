"""Turn a LangGraph/LangChain message list into an attest Trajectory. Duck-typed."""

from __future__ import annotations

from ..trajectory import Step, ToolCall, Trajectory


def _flatten(content) -> str:
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
    Convert a LangGraph/LangChain message list into a Trajectory.

    `task` defaults to the first HumanMessage; `final_answer` to the last AIMessage's
    text (pass it for structured responses); `system_prompt` to a SystemMessage if
    present. `response_tool` names a structured-output synthetic tool to skip.
    """
    outputs: dict[str, str] = {}
    for m in messages:
        tcid = getattr(m, "tool_call_id", None)
        if tcid is not None:
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
