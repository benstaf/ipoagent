# IPO Finance Agent: Benchmark of LLM Financial Analysts Beyond Finance Agent v2, with Automated Rubric Generation, on the SpaceX (SPCX) IPO

## Paper: [https://arxiv.org/abs/2606.23032](https://arxiv.org/abs/2606.23032)

## Discord: [![](https://dcbadge.limes.pink/api/server/ekrySuRBf4)](https://discord.gg/ekrySuRBf4)


## Running IPO Finance Agent Benchmark

The IPO Finance Agent Benchmark evaluates LLM agents on their ability to analyze IPO filings and perform due diligence using public registration statements.

Unlike traditional finance benchmarks that primarily focus on historical SEC filings and financial statements, this benchmark emphasizes reasoning over IPO registration documents (S-1, F-1, prospectuses, amendments, exhibits, and related filings), requiring agents to synthesize information across lengthy filings similarly to real-world investment research and venture due diligence.

The agent has access to the following tools:

- `web_search`: Search the web for public information (via Tavily)
- `edgar_search`: Search the SEC EDGAR database for IPO filings and related documents
- `parse_html_page`: Parse and extract content from web pages
- `retrieve_information`: Access information collected during previous reasoning steps
- `price_history`: Fetch historical daily OHLCV price data for supported equities, ETFs, crypto, and FX

The benchmark measures an agent's ability to:

- analyze IPO registration statements (S-1, F-1, etc.)
- reason across multiple sections of long filings
- extract financial, operational, governance, and risk information
- answer quantitative and qualitative due diligence questions
- synthesize evidence from SEC filings and external sources

## Set up

### Dependencies

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management.

Then run:

```bash
make install
source .venv/bin/activate
```

### Environment Variables

Create a `.env` file in the project root:

```text
# LLM API Keys
OPENAI_API_KEY=<openai_api_key>
ANTHROPIC_API_KEY=<anthropic_api_key>
GOOGLE_API_KEY=<google_api_key>

# Tool API Keys
TAVILY_API_KEY=<tavily_api_key>
SEC_EDGAR_API_KEY=<sec_api_key>   # Multiple keys may be separated by semicolons
PRICING_DATA_API_KEY=<pricing_data_api_key>
```

You can obtain:

- Tavily API key from https://tavily.com
- SEC API key from https://sec-api.io

The `.env` file takes precedence over existing environment variables.

## Running the benchmark

Display all available options:

```bash
finance-agent --help
```

Run a single question:

```bash
finance-agent \
  --questions "What customer concentration risks are disclosed in CoreWeave's S-1?" \
  --model openai/gpt-5
```

Run multiple questions:

```bash
finance-agent \
  --questions \
  "What customer concentration risks are disclosed in CoreWeave's S-1?" \
  "How does Circle generate revenue according to its registration statement?"
```

Run questions from a file:

```bash
finance-agent \
  --question-file data/questions.txt
```

The default configuration reproduces the benchmark settings used in our experiments.

## Models

Any supported LLM can be evaluated.

You may also integrate your own inference harness by modifying the `get_custom_model` function.

## Logs

Each benchmark run produces detailed logs under the `logs/` directory.

Logs include:

- model reasoning trace
- tool calls
- retrieved documents
- token usage
- execution timing
- errors
- final answers

These logs are useful for debugging, reproducibility, and agent analysis.

## Forked from [Finance Agent v2](https://github.com/vals-ai/finance-agent-v2/tree/main) by [Vals AI](https://www.vals.ai/home)
