from typing import Any

from model_library.agent import AgentResult, TurnSummary
from model_library.base import LLMConfig, TokenRetryParams
from .get_agent import Parameters, build_input, get_agent


def create_override_config(**kwargs: object) -> LLMConfig:
    if "max_output_tokens" in kwargs and "max_tokens" not in kwargs:
        kwargs["max_tokens"] = kwargs["max_output_tokens"]

    return LLMConfig.model_validate(kwargs, strict=False)


def _build_output_context(result: AgentResult) -> dict[str, Any]:
    """Build a trimmed output context that keeps tool usage metadata but drops per-turn cost/token details."""
    trimmed_turns = []
    for turn in result.turns:
        if not isinstance(turn, TurnSummary):
            continue
        trimmed_turns.append([tc.tool_name for tc in turn.tool_calls])

    return {
        "turns": trimmed_turns,
        "total_turns": result.total_turns,
        "success": result.success,
        "error_count": result.error_count,
        "tool_calls_count": result.tool_calls_count,
        "tool_usage": result.tool_usage,
        "final_duration_seconds": result.final_duration_seconds,
    }


async def get_custom_model(
    model_name: str,
    parameters: dict[str, Any],
    *_args: object,
    **_kwargs: object,
):
    from vals.sdk.types import OutputObject  # pyright: ignore

    token_retry_params = parameters.pop("token_retry_params", None)

    params = Parameters(
        model_name=model_name,
        llm_config=create_override_config(**parameters),
        token_retry_params=TokenRetryParams.model_validate(token_retry_params) if token_retry_params else None,
    )

    async def custom_call(test_input: str, files: dict, context: dict, question_id: str, run_id: str):
        agent = await get_agent(params)
        result = await agent.run(
            build_input(test_input),
            question_id=question_id,
            run_id=run_id,
        )

        if not result.success and result.final_error:
            print(f"\nFAIL {question_id} failed: [{result.final_error.type}] {result.final_error.message}\n")
        return OutputObject.from_agent_result(
            result,
            output_context=_build_output_context(result),
            count_tool_metadata=True,
        )

    return custom_call
