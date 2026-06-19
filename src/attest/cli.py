"""
CLI for attest.

  attest stats 41 50                          # pass rate + Wilson 95% CI (no API)
  attest tools examples/trajectory.json       # tool-use correctness (deterministic, NO API)
  attest injection examples/trajectory.json   # prompt-injection scan (deterministic, NO API)
  attest run examples/trajectory.json         # full report: faithfulness + tool-use
  attest demo examples/trajectory.json        # naive LLM-judge vs attest, side by side

`stats`, `tools`, and `injection` (without --appropriate / --deep) need no API key.
`run` and `demo` call Claude, reading ANTHROPIC_API_KEY from a local .env.
"""

from __future__ import annotations

import os
from pathlib import Path

import typer
from dotenv import load_dotenv

from .checks.judge_baseline import naive_judge
from .scoring.report import evaluate
from .scoring.stats import wilson_interval
from .checks.tool_use import ToolUseVerdict, check_tool_use
from .checks.injection import check_injection
from .trajectory import Trajectory
from .checks.verify import Verdict, extract_claims, grounded_verifier, judge_trajectory

load_dotenv()  # finds the shared workspace-root .env by walking up the tree
app = typer.Typer(help="Evidence-grounded evaluation for AI agent trajectories.")


def _require_key() -> None:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key or key.startswith("sk-ant-your"):
        typer.echo("Set a real ANTHROPIC_API_KEY in the workspace .env to run this.")
        raise typer.Exit(code=1)


def _load(path: Path) -> list[Trajectory]:
    raw = path.read_text().strip()
    lines = [raw] if path.suffix == ".json" else raw.splitlines()
    return [Trajectory.model_validate_json(ln) for ln in lines if ln.strip()]


def _show_flagged(r) -> None:
    typer.echo(f"  UNSUPPORTED: {r.claim}")
    if r.reason:
        typer.echo(f"    reason:   {r.reason}")
    if r.evidence_quote:
        typer.echo(f"    evidence: {r.evidence_quote!r}")


def _render_report(traj: Trajectory, report) -> None:
    f, t = report.faithfulness, report.tool_use
    typer.echo(f"- {traj.task[:70]!r}")
    typer.echo(f"  overall {report.overall_score:.0%}  "
               f"(faithfulness {f.grounding_rate:.0%}, tool-use {t.correct_rate:.0%})")
    typer.echo(f"  agent answer: {f.final_answer[:200]}")
    for r in f.results:
        if r.verdict is Verdict.UNSUPPORTED:
            _show_flagged(r)
    typer.echo(f"  tool use: {t.summary}")
    typer.echo("")


@app.command()
def stats(successes: int, n: int) -> None:
    """Report a pass rate with a Wilson 95% confidence interval."""
    typer.echo(wilson_interval(successes, n).pct())


@app.command()
def tools(
    path: Path,
    appropriate: bool = typer.Option(False, "--appropriate",
                                      help="Also run the LLM tool-choice check (needs a key)."),
) -> None:
    """Tool-use correctness only — deterministic, NO API key (unless --appropriate)."""
    if appropriate:
        _require_key()
    for traj in _load(path):
        score = check_tool_use(traj, appropriate=appropriate)
        typer.echo(f"- {traj.task[:70]!r}")
        typer.echo(f"  tool-use {score.correct_rate:.0%}  "
                   f"({score.correct}/{score.total} calls correct)")
        typer.echo(f"  {score.summary}")
        for r in score.reviews:
            if r.verdict is not ToolUseVerdict.CORRECT:
                typer.echo(f"    {r.verdict.value.upper()}: step {r.step} ({r.tool}) — {r.reason}")
        typer.echo("")


@app.command()
def injection(
    path: Path,
    deep: bool = typer.Option(False, "--deep",
                              help="Also LLM-check whether the agent FOLLOWED each payload (needs a key)."),
) -> None:
    """Flag prompt-injection payloads in tool outputs — deterministic, NO API key (unless --deep)."""
    if deep:
        _require_key()
    for traj in _load(path):
        report = check_injection(traj, deep=deep)
        typer.echo(f"- {traj.task[:70]!r}")
        typer.echo(f"  {'CLEAN' if report.clean else 'FLAGGED'}: {report.summary}")
        for f in report.findings:
            where = f" ({f.tool})" if f.tool else ""
            typer.echo(f"    {f.verdict.value.upper()}: step {f.step}{where} — {f.detail[:90]!r}")
            if f.reason:
                typer.echo(f"      reason: {f.reason}")
        typer.echo("")


@app.command()
def run(
    path: Path,
    appropriate: bool = typer.Option(False, "--appropriate",
                                     help="Also run the LLM tool-choice check (1 call per tool call)."),
) -> None:
    """Full report on a trajectory (or JSONL): faithfulness + tool-use + overall score."""
    _require_key()
    trajectories = _load(path)
    typer.echo(f"Loaded {len(trajectories)} trajectory(ies) from {path}\n")

    overalls: list[float] = []
    for traj in trajectories:
        report = evaluate(traj, appropriate=appropriate)
        _render_report(traj, report)
        overalls.append(report.overall_score)

    if overalls:
        typer.echo(f"Mean overall score: {sum(overalls) / len(overalls):.0%}")


@app.command()
def demo(path: Path = typer.Argument(Path("examples/trajectory.json"))) -> None:
    """Run the naive LLM-judge and attest side by side — the headline comparison."""
    _require_key()
    for traj in _load(path):
        typer.echo("-" * 72)
        typer.echo(f"TASK: {traj.task}\n")
        typer.echo("Agent's answer (the initial LLM's claim):")
        typer.echo(f"  {traj.final_answer}\n")

        verdict = naive_judge(traj)
        typer.echo("Naive LLM-judge (reads the agent's reasoning):")
        typer.echo(f"  {'PASS' if verdict.passed else 'FAIL'} - {verdict.reason}\n")

        score = judge_trajectory(traj, extract_claims, grounded_verifier)
        typer.echo("attest (checks claims against real tool outputs only):")
        typer.echo(f"  grounding {score.grounding_rate:.0%} "
                   f"({score.supported}/{score.checkable} checkable claims)")
        flagged = [r for r in score.results if r.verdict is Verdict.UNSUPPORTED]
        for r in flagged:
            _show_flagged(r)

        typer.echo("")
        if verdict.passed and flagged:
            typer.echo("=> attest caught a false claim the LLM-judge waved through.")
        elif not verdict.passed and flagged:
            typer.echo("=> both caught the problem.")
        elif verdict.passed and not flagged:
            typer.echo("=> both agree this answer is sound.")
        else:
            typer.echo("=> the LLM-judge flagged something attest's grounding didn't.")
    typer.echo("-" * 72)


if __name__ == "__main__":
    app()
