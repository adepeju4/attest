"""attest — evidence-grounded evaluation for AI agent trajectories."""

from .trajectory import Trajectory, Step, ToolCall
from .results import Report, CheckResult, Finding, Severity, PROMPT_VERSION
from .checks.verify import judge_trajectory, extract_claims, grounded_verifier
from .checks.judge_baseline import naive_judge, JudgeVerdict
from .checks.tool_use import check_tool_use
from .checks.injection import check_injection
from .checks.role import check_role_adherence
from .scoring.report import evaluate
from .adapters.langgraph import from_langgraph_messages
from .scoring.stats import wilson_interval, difference_is_real, Proportion
from .providers import providers as list_providers, list_models, default_model
from .api import Attest

__all__ = [
    "Attest",
    "list_providers", "list_models", "default_model",
    "Trajectory", "Step", "ToolCall",
    "Report", "CheckResult", "Finding", "Severity", "PROMPT_VERSION",
    "judge_trajectory", "extract_claims", "grounded_verifier",
    "naive_judge", "JudgeVerdict",
    "check_tool_use", "check_injection", "check_role_adherence",
    "evaluate",
    "from_langgraph_messages",
    "wilson_interval", "difference_is_real", "Proportion",
]

__version__ = "0.4.0"
