"""Plan compilation, execution, and deterministic reproduction."""

from .executor import execute_plan, execute_plan_to_target
from .plan import CompiledPlan, compile_plan
from .verifier import (
    VerificationOutcome,
    reproduce_output,
    reproduce_output_to_target,
)

__all__ = [
    "CompiledPlan",
    "VerificationOutcome",
    "compile_plan",
    "execute_plan",
    "execute_plan_to_target",
    "reproduce_output",
    "reproduce_output_to_target",
]
