"""
End-to-end: run the codesprint-studio LangGraph agent, then judge its run with attest.

This is the integration in one file:
    codesprint agent  --run-->  LangGraph messages  --adapter-->  Trajectory  --attest-->  report

HOW TO RUN
----------
Needs an environment with BOTH this project's deps AND codesprint's
(langchain, langgraph, langchain-google-genai), plus two keys:
  - GOOGLE_API_KEY     (codesprint / Gemini)
  - ANTHROPIC_API_KEY  (attest's verifier)

  GOOGLE_API_KEY=... ANTHROPIC_API_KEY=... uv run --with langchain --with langgraph \
      --with langchain-google-genai examples/codesprint_to_attest.py "Plan a rate limiter."
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from attest import evaluate, from_langgraph_messages, naive_judge

CODESPRINT_MAIN = Path(
    "/Users/adepejuorefejo/Desktop/Everything/Programming/codesprint-studio/main.py"
)


def _load_codesprint():
    """Import codesprint-studio's main.py as a module (its `agent` is module-level)."""
    spec = importlib.util.spec_from_file_location("codesprint_main", CODESPRINT_MAIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # builds the Gemini agent (needs GOOGLE_API_KEY)
    return mod


def _final_answer(review) -> str:
    """Flatten codesprint's structured CodeReviewRequest into the answer attest checks."""
    return "\n".join(
        f"{label}: {getattr(review, field)}"
        for label, field in [
            ("Summary", "summary"),
            ("Components", "identified_components"),
            ("Data structures & algorithms", "data_structures_and_algorithms"),
            ("Implementation plan", "implementation_plan"),
        ]
    )


def main(user_input: str) -> None:
    from langchain_core.messages import HumanMessage  # codesprint dep

    cs = _load_codesprint()
    result = cs.agent.invoke(
        {"messages": [HumanMessage(user_input)]},
        {"configurable": {"thread_id": cs.generate_thread_id()}},
    )

    # LangGraph run  ->  attest Trajectory (incl. the agent's authority for tool-use checks)
    traj = from_langgraph_messages(
        result["messages"],
        task=user_input,
        final_answer=_final_answer(result["structured_response"]),
        system_prompt=cs.SYSTEM_PROMPT,
        allowed_tools=[t.name for t in cs.tools],
        response_tool=cs.CodeReviewRequest.__name__,  # ToolStrategy output, not a real tool
    )
    print(f"Captured trajectory: {len(traj.steps)} tool step(s).\n")

    # naive LLM-judge (for contrast) vs attest's combined report
    verdict = naive_judge(traj)
    print(f"Naive LLM-judge: {'PASS' if verdict.passed else 'FAIL'} - {verdict.reason}\n")

    report = evaluate(traj)
    f, t = report.faithfulness, report.tool_use
    print(f"attest overall: {report.overall_score:.0%}")
    print(f"  faithfulness {f.grounding_rate:.0%} ({f.supported}/{f.checkable} checkable claims)")
    for r in f.results:
        if r.verdict.value != "supported":
            print(f"    {r.verdict.value.upper()}: {r.claim}")
            if r.reason:
                print(f"      reason: {r.reason}")
    print(f"  tool-use {t.correct_rate:.0%} - {t.summary}")


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Plan a rate limiter for an API gateway."
    main(prompt)
