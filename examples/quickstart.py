"""
Minimal attest example: build a trajectory by hand and evaluate it.

The agent was asked which city is larger. It looked up both populations — those
tool outputs are the real evidence — then wrote an answer with one TRUE claim and
one FALSE claim. attest verifies each claim against the tool outputs (not the
agent's wording), so it marks the false one UNSUPPORTED.

Run it (faithfulness needs an API key):

    export ANTHROPIC_API_KEY=sk-ant-...        # default provider
    uv run python examples/quickstart.py

Other providers: Attest(provider="openai") / Attest(provider="gemini").
"""

from attest import Attest, Step, ToolCall, Trajectory

trajectory = Trajectory(
    task="Which city is larger by population, Paris or Berlin?",
    allowed_tools=["get_population"],
    steps=[
        Step(
            thought="Look up Paris.",
            tool_call=ToolCall(
                name="get_population",
                arguments={"city": "Paris"},
                output="Paris: 2,103,000 residents",
            ),
        ),
        Step(
            thought="Look up Berlin.",
            tool_call=ToolCall(
                name="get_population",
                arguments={"city": "Berlin"},
                output="Berlin: 3,677,000 residents",
            ),
        ),
    ],
    final_answer=(
        "Berlin has about 3.7 million residents and Paris about 2.1 million, "
        "so Paris is the larger city."
    ),
)


def main() -> None:
    judge = Attest()
    report = judge.evaluate(trajectory)

    print(f"overall score  : {report.overall_score:.0%}")
    print(f"grounding rate : {report.faithfulness.grounding_rate:.0%}")
    print(f"tool-use rate  : {report.tool_use.correct_rate:.0%}")
    print()

    for r in report.faithfulness.results:
        print(f"[{r.verdict.value.upper():>12}]  {r.claim}")
        if r.reason:
            print(f"               {r.reason}")


if __name__ == "__main__":
    main()
