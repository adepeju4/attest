"""
Tests for the LangGraph adapter using fake message objects — no langchain needed.
We mimic the shape LangChain produces (HumanMessage / AIMessage.tool_calls /
ToolMessage.tool_call_id) so the adapter is verified offline.
"""

from attest.adapters.langgraph import from_langgraph_messages


class HumanMessage:
    def __init__(self, content):
        self.content = content


class AIMessage:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class ToolMessage:
    def __init__(self, content, tool_call_id, name=""):
        self.content = content
        self.tool_call_id = tool_call_id
        self.name = name


class SystemMessage:
    def __init__(self, content):
        self.content = content


def test_adapts_a_codesprint_like_run():
    messages = [
        HumanMessage("Review my binary search implementation"),
        AIMessage(tool_calls=[{"name": "understand_code_snippet",
                               "args": {"code_snippet": "def bsearch(...)"},
                               "id": "c1"}]),
        ToolMessage("It's an iterative binary search over a sorted list; O(log n).", "c1"),
        AIMessage(tool_calls=[{"name": "reason_through_design", "args": {}, "id": "c2"}]),
        ToolMessage("Sorted-array precondition; consider bounds handling.", "c2"),
        AIMessage("Here is my final review."),
    ]

    traj = from_langgraph_messages(messages, final_answer="A clean O(log n) binary search.")

    assert traj.task == "Review my binary search implementation"
    assert len(traj.steps) == 2
    assert traj.steps[0].tool_call.name == "understand_code_snippet"
    assert traj.steps[0].tool_call.output.startswith("It's an iterative")
    # evidence = the real tool outputs, NOT the final answer
    assert "O(log n)" in traj.evidence()
    assert "final review" not in traj.evidence()
    assert traj.final_answer == "A clean O(log n) binary search."


def test_defaults_final_answer_to_last_ai_message():
    messages = [HumanMessage("hi"), AIMessage("the answer")]
    traj = from_langgraph_messages(messages)
    assert traj.task == "hi"
    assert traj.final_answer == "the answer"


def test_handles_gemini_block_content():
    # Gemini returns content as a list of blocks; the adapter should flatten it.
    messages = [
        HumanMessage([{"type": "text", "text": "do a thing"}]),
        AIMessage([{"type": "text", "text": "done"}]),
    ]
    traj = from_langgraph_messages(messages)
    assert traj.task == "do a thing"
    assert traj.final_answer == "done"


def test_captures_system_prompt_and_allowed_tools():
    messages = [
        SystemMessage("You are a dev assistant. Tools: understand, reason."),
        HumanMessage("review this"),
        AIMessage("done"),
    ]
    traj = from_langgraph_messages(messages, allowed_tools=["understand", "reason"])
    assert traj.system_prompt.startswith("You are a dev assistant")  # auto-detected
    assert traj.allowed_tools == ["understand", "reason"]             # passed in


def test_defaults_authority_fields_to_none():
    traj = from_langgraph_messages([HumanMessage("hi"), AIMessage("yo")])
    assert traj.system_prompt is None
    assert traj.allowed_tools is None


def test_tool_calls_helper():
    messages = [
        HumanMessage("x"),
        AIMessage(tool_calls=[{"name": "search", "args": {"q": "a"}, "id": "1"}]),
        ToolMessage("result", "1"),
        AIMessage("final"),
    ]
    calls = from_langgraph_messages(messages).tool_calls()
    assert len(calls) == 1
    idx, tc = calls[0]
    assert idx == 0 and tc.name == "search"


def test_response_tool_is_excluded():
    # ToolStrategy/structured-output emits the final answer as a synthetic tool call;
    # it must NOT be counted as a real tool-use decision.
    messages = [
        HumanMessage("review this"),
        AIMessage(tool_calls=[{"name": "search", "args": {}, "id": "1"}]),
        ToolMessage("result", "1"),
        AIMessage(tool_calls=[{"name": "FinalAnswer", "args": {"x": 1}, "id": "2"}]),
    ]
    traj = from_langgraph_messages(messages, response_tool="FinalAnswer")
    assert [tc.name for _, tc in traj.tool_calls()] == ["search"]
