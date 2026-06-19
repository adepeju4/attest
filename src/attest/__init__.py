"""attest — evidence-grounded evaluation for AI agent trajectories."""

from .trajectory import Trajectory, Step, ToolCall
from .checks.verify import (
    Verdict,
    ClaimResult,
    TrajectoryScore,
    judge_trajectory,
    extract_claims,
    grounded_verifier,
)
from .checks.judge_baseline import naive_judge, JudgeVerdict
from .checks.tool_use import check_tool_use, ToolUseScore, ToolCallReview, ToolUseVerdict
from .scoring.report import evaluate, TrajectoryReport
from .adapters.langgraph import from_langgraph_messages
from .scoring.stats import wilson_interval, difference_is_real, Proportion

__all__ = [
    "Trajectory", "Step", "ToolCall",
    "Verdict", "ClaimResult", "TrajectoryScore",
    "judge_trajectory", "extract_claims", "grounded_verifier",
    "naive_judge", "JudgeVerdict",
    "check_tool_use", "ToolUseScore", "ToolCallReview", "ToolUseVerdict",
    "evaluate", "TrajectoryReport",
    "from_langgraph_messages",
    "wilson_interval", "difference_is_real", "Proportion",
]

__version__ = "0.1.0"
