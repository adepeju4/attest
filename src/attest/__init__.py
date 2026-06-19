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
from .checks.injection import check_injection, InjectionReport, InjectionFinding, InjectionVerdict
from .checks.role import check_role_adherence, RoleReport, RoleVerdict
from .scoring.report import evaluate, TrajectoryReport
from .adapters.langgraph import from_langgraph_messages
from .scoring.stats import wilson_interval, difference_is_real, Proportion
from .providers import providers as list_providers, list_models, default_model
from .api import Attest

__all__ = [
    "Attest",
    "list_providers", "list_models", "default_model",
    "Trajectory", "Step", "ToolCall",
    "Verdict", "ClaimResult", "TrajectoryScore",
    "judge_trajectory", "extract_claims", "grounded_verifier",
    "naive_judge", "JudgeVerdict",
    "check_tool_use", "ToolUseScore", "ToolCallReview", "ToolUseVerdict",
    "check_injection", "InjectionReport", "InjectionFinding", "InjectionVerdict",
    "check_role_adherence", "RoleReport", "RoleVerdict",
    "evaluate", "TrajectoryReport",
    "from_langgraph_messages",
    "wilson_interval", "difference_is_real", "Proportion",
]

__version__ = "0.3.0"
