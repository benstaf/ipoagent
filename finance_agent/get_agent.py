from pathlib import Path
import os

from dotenv import load_dotenv

# Load .env before reading any environment variables
load_dotenv(override=True)

#import logging
#import http.client

## Add this right after load_dotenv() at the top of get_agent.py
#http.client.HTTPSConnection.debuglevel = 1
#logging.basicConfig()
#logging.getLogger().setLevel(logging.DEBUG)
#requests_log = logging.getLogger("urllib3")
#requests_log.setLevel(logging.DEBUG)
#requests_log.propagate = True


#import httpx

#original_send = httpx.AsyncClient.send

#async def patched_send(self, request, **kwargs):
 #   response = await original_send(self, request, **kwargs)
  #  if "z.ai" in str(request.url):
   #     body = await response.aread()
    #    print(f"[RAW RESPONSE] status={response.status_code} body={body}")
  #  return response

#httpx.AsyncClient.send = patched_send

# Tested model credentials — set env vars immediately so model_library picks them up
TESTED_MODEL     = os.getenv("TESTED_MODEL")
TESTED_MODEL_URL = os.getenv("TESTED_MODEL_URL")
TESTED_API_KEY   = os.getenv("TESTED_API_KEY")

if TESTED_MODEL_URL:
    os.environ["OPENAI_BASE_URL"] = TESTED_MODEL_URL
if TESTED_API_KEY:
    os.environ["OPENAI_API_KEY"] = TESTED_API_KEY

from model_library.agent import (
    Agent,
    AgentConfig,
    AgentHooks,
    TimeLimit,
    ToolCallRecord,
    TurnLimit,
    TurnResult,
    default_before_query,
    truncate_oldest,
)

from model_library.base import (
    LLM,
    LLMConfig,
    RawResponse,
    TextInput,
)

from model_library.base.input import (
    InputItem,
    SystemInput,
)

from model_library.exceptions import MaxContextWindowExceededError
from model_library.registry_utils import get_registry_model
from pydantic import BaseModel

from .prompt import (
    BASE_SYSTEM_PROMPT,
    QUESTION_PROMPT,
    DEFAULT_CONTEXT,
)

from .exceptions import RetryExhaustedError

from .tools import (
    VALID_TOOLS,
    Calculator,
    EDGARSearch,
    SubmitFinalResult,
    TavilyWebSearch,
    Tool,
    PriceHistory,
)

from .tools_ingestion import ParseHtmlPage
from .tools_retrieval import RetrieveInformation


MAX_TIME_SECONDS = 60 * 600

VALID_TOOLS = [
    "web_search",
    "retrieve_information",
    "parse_html_page",
    "edgar_search",
    "calculator",
    "price_history",
]

class Parameters(BaseModel):
    model_name: str
    max_time_seconds: int = MAX_TIME_SECONDS
    max_turns: int | None = None
    tools: list[str] = VALID_TOOLS
    llm_config: LLMConfig
    benchmark_context: str | None = None


def build_input(
    question: str,
    benchmark_context: str | None = None,
) -> list[InputItem]:

    system_text = BASE_SYSTEM_PROMPT

    context = benchmark_context or DEFAULT_CONTEXT

    if context:
        system_text += "\n\n---\n\n" + context

    return [
        SystemInput(text=system_text),
        TextInput(
            text=QUESTION_PROMPT.format(
                question=question
            )
        ),
    ]


def create_llm(parameters: Parameters) -> LLM:
    """Create an LLM instance from parameters using the model registry."""
    try:
        # Try to load the model normally first
        return get_registry_model(parameters.model_name, parameters.llm_config)
    except Exception as e:
        # If the library rejects our custom model string, we bypass the registry
        if "not found in registry" in str(e):
            print(f"\n[INFO] Bypassing strict registry for custom model: {parameters.model_name}")

            # 1. Ask the registry for a standard OpenAI client to pass the strict check
            llm = get_registry_model("openai/gpt-4o", parameters.llm_config)

            # 2. Clean the target model name (strip "openai/" prefix if you passed it)
            target_model = parameters.model_name
            if target_model.startswith("openai/"):
                target_model = target_model[7:]

            # 3. Overwrite the properties on the instantiated LLM object
            # so the actual payload uses the correct model string.
            if hasattr(llm, "model_name"):
                llm.model_name = target_model
            if hasattr(llm, "model"):
                llm.model = target_model

            # 4. Force chat completions endpoint
            if hasattr(llm, "use_completions"):
                llm.use_completions = True

# 5. Disable reasoning_effort — not supported by GLM/Z.AI,
            # causes silent empty responses
            if hasattr(llm, "reasoning_effort"):
                llm.reasoning_effort = None
            if hasattr(llm, "reasoning"):
                llm.reasoning = None

            return llm

        # If it failed for a different reason, raise the error normally
        raise e


def get_agent(
    parameters: Parameters,
    llm: LLM | None = None,
    log_dir: Path | None = None,
) -> Agent:

    if llm is None:
        llm = create_llm(parameters)

    available_tools: dict[str, type[Tool]] = {
        "web_search": TavilyWebSearch,
        "retrieve_information": RetrieveInformation,
        "parse_html_page": ParseHtmlPage,
        "edgar_search": EDGARSearch,
        "calculator": Calculator,
        "price_history": PriceHistory,
    }

    selected_tools: list[Tool] = []

    for tool_name in parameters.tools:

        if tool_name not in available_tools:
            raise Exception(
                f"Tool {tool_name} not found. "
                f"Available tools: {available_tools.keys()}"
            )

        tool_cls = available_tools[tool_name]

        if tool_name == "retrieve_information":
            selected_tools.append(tool_cls(llm=llm))
        else:
            selected_tools.append(tool_cls())

    selected_tools.append(SubmitFinalResult())

    def _before_query(
        history: list[InputItem],
        last_error: Exception | None,
    ) -> list[InputItem]:

        if isinstance(
            last_error,
            MaxContextWindowExceededError,
        ):
            return truncate_oldest(history)

        if history and isinstance(history[-1], RawResponse):
            history.append(
                TextInput(
                    text=(
                        "Your last response produced no tool call. "
                        "Call submit_final_result if finished, "
                        "otherwise continue with tool use."
                    )
                )
            )

        return default_before_query(
            history,
            last_error,
        )

    def _on_tool_result(
        record: ToolCallRecord,
        state: dict,
    ) -> None:

        if (
            record.error
            and record.error.type == "RetryExhaustedError"
        ):
            raise RetryExhaustedError(
                record.error.message
            )

    def _should_stop(
        turn_result: TurnResult,
    ) -> bool:
        return False

    return Agent(
        llm=llm,
        tools=selected_tools,
        name="finance",
        log_dir=log_dir or Path("logs"),
        config=AgentConfig(
            turn_limit=(
                TurnLimit(
                    max_turns=parameters.max_turns
                )
                if parameters.max_turns
                else None
            ),
            time_limit=TimeLimit(
                max_seconds=parameters.max_time_seconds
            ),
        ),
        hooks=AgentHooks(
            before_query=_before_query,
            should_stop=_should_stop,
            on_tool_result=_on_tool_result,
        ),
    )
