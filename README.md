# Simple Quote Agent Demo

A minimal example demonstrating how to build, instrument, and evaluate an AI agent on the Domino platform. The agent responds to questions with random philosophy or science quotes using pydantic-ai tool calls. No RAG, no vector store — just the simplest possible agent that still exercises the full Domino tracing, evaluation, and deployment workflow.

For a more advanced example with RAG and ChromaDB, see: [rag-agent-demo](https://github.com/dominodatalab/rag-agent-demo)

For full platform documentation see: [Build and Evaluate Agentic Systems](https://docs.dominodatalab.com/en/cloud/user_guide/e437a3/build-and-evaluate-agentic-systems/)

---

## Project Structure

```
.
├── static/                        # Frontend assets for the chat UI
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── ai_system_config.yaml          # Central config: model, prompts, retries
├── simplest_agent.py              # Core agent definition (pydantic-ai)
├── evaluation_library.py          # Evaluation logic: toxicity, relevancy, accuracy
├── sample_questions.csv           # Test dataset for batch evaluation
├── dev_eval_simplest_agent.py     # Dev workflow: batch eval with Domino tracing
├── prod_eval_simplest_agent.py    # Prod workflow: evaluate live traces retroactively
├── chat_app.py                    # FastAPI chat UI with production tracing
├── app.sh                         # Domino App launcher
└── requirements.txt               # Python package dependencies
```

### File Descriptions

| File | Purpose |
|------|---------|
| `ai_system_config.yaml` | Single source of truth for model provider, base_url, retries, and system prompt. See [Configuration](#1-configure-the-agent) for details. |
| `simplest_agent.py` | Defines the pydantic-ai `Agent` with two tool functions (`science_quote` and `philosophy_quote`) that return random quotes. The agent is created with `instrument=True` so pydantic-ai steps are auto-traced. |
| `evaluation_library.py` | `AgentEvaluator` class that scores responses on toxicity, relevancy, and accuracy. Currently uses placeholder random scoring — replace each `evaluate_*` method with an LLM judge or external scorer for real use. |
| `sample_questions.csv` | Test questions (with categories like `deductions`, `space`, `biology`) used as the dev evaluation dataset. |
| `dev_eval_simplest_agent.py` | **Development evaluation script.** The main example of how to instrument agent traces and attach evaluation metrics. See [How Tracing and Evaluation Work](#how-tracing-and-evaluation-work-dev_eval_simplest_agentpy) for a detailed walkthrough. **Must run as a Domino Job** (not a workspace) to create an experiment run. |
| `prod_eval_simplest_agent.py` | **Production evaluation script.** Uses `search_agent_traces` to fetch new traces from the deployed app, scores them, and writes scores back via `log_evaluation`. **Run as a Domino Scheduled Job.** |
| `chat_app.py` | FastAPI server that serves the chat UI. Each request is wrapped with `@add_tracing` and `DominoRun`, so every real user conversation is automatically captured as a trace in Domino. |
| `app.sh` | Launches `chat_app.py` on port 8888 — used as the entry point for a Domino App. |

---

## Key Domino Platform Concepts

### How Tracing and Evaluation Work (`dev_eval_simplest_agent.py`)

This is the primary pattern for instrumenting an agent and attaching evaluation metrics to traces. It has two parts: a **traced function** and a **judge function**.

#### The traced function (`process_single_question`)

This is the function that actually runs the agent. It is decorated with `@add_tracing`, which:

- Creates a Domino trace capturing all inputs, outputs, and internal agent steps
- Auto-instruments pydantic-ai via `autolog_frameworks=["pydantic_ai"]`
- Calls the `evaluator=` function after each invocation, passing it the completed trace span

```python
@add_tracing(name='single_question_agent', autolog_frameworks=["pydantic_ai"], evaluator=judge_single_question)
def process_single_question(data_point: Dict[str, Any]) -> Dict[str, Any]:
    agent = create_agent()
    result = agent.run_sync(data_point['question'])
    return {"answer": result.output}
```

The function receives a data point (from `sample_questions.csv`) and returns a dict. Both the input and output are captured in the trace.

#### The judge function (`judge_single_question`)

This is where evaluation metrics are computed and attached to the trace. It receives the trace `span` object, from which it reads `.inputs` and `.outputs`:

```python
def judge_single_question(span):
    inputs = span.inputs    # What was passed to process_single_question
    output = span.outputs   # What process_single_question returned

    question = inputs['data_point']['question']
    agent_output = output['answer']

    evaluator = AgentEvaluator()
    evaluation_result = evaluator.evaluate_response(query=question, agent_output=agent_output)

    # Return a dict of metrics — these become the evaluation scores on the trace
    return {
        "toxicity_score": evaluation_result.toxicity_score,
        "relevancy_score": evaluation_result.relevancy_score,
        "accuracy_score": evaluation_result.accuracy_score,
        "overall_score": evaluation_result.overall_score,
    }
```

The dict returned by the judge becomes the evaluation metrics visible on that trace in the Domino UI.

#### Wrapping it in a DominoRun

The outer loop wraps everything in `DominoRun`, which groups all traces into a single experiment run and attaches config metadata:

```python
with DominoRun(agent_config_path=config_path) as run:
    for row in csv_reader:
        process_single_question(row)
```

### 2. Production Evaluation (`search_agent_traces` + `log_evaluation`)

Once the chat app is deployed, `prod_eval_simplest_agent.py` evaluates live user conversations retroactively:

```python
from domino.agents.tracing import search_agent_traces
from domino.agents.logging import log_evaluation

traces = search_agent_traces(agent_id=AGENT_ID, start_time=last_run_dt)
for trace in traces.data:
    question = trace.spans[0].inputs.get("question")
    agent_output = trace.spans[0].outputs.get("output")
    score = evaluator.evaluate_response(query=question, agent_output=agent_output)
    log_evaluation(trace_id=trace.id, name='relevancy_score', value=score.relevancy_score)
```

The `AGENT_ID` is found on the Domino agent dashboard ("Evaluation setup" button). The script persists a timestamp to `/mnt/data/<project>/last_trace_time.txt` so it only processes new traces on each run.

---

## Usage

### 0. Install dependencies

```bash
pip install -r requirements.txt
```

In a Domino workspace you may need to run this manually. For Apps, Agent Deploys, and Jobs, these may be installed automatically depending on your Domino environment configuration.

**Note:** The `dominodatalab[agents]` package (which provides `domino.agents.tracing`, `domino.agents.logging`, etc.) is pre-installed in Domino compute environments. If running outside Domino for local testing, install it separately:

```bash
pip install dominodatalab[agents]
```

### 1. Configure the agent

Edit `ai_system_config.yaml` to set your model provider, base_url, and system prompt.

**Using OpenAI directly:**

```yaml
model:
  provider: "openai"
  full_name: "openai:gpt-4.1-mini"
```

Requires `OPENAI_API_KEY` environment variable.

**Using a Domino-hosted LLM (vLLM):**

Domino hosts LLMs via Model APIs, which use vLLM under the hood. To use one:

1. Deploy a model via **Domino Model APIs** (see [Domino docs on hosting LLMs](https://docs.dominodatalab.com/en/cloud/user_guide/e437a3/build-and-evaluate-agentic-systems/))
2. Copy the endpoint URL from the Model API dashboard
3. Configure:

```yaml
model:
  provider: "vllm"
  name: "qwen-3-4b"
  full_name: "qwen-3-4b"
  base_url: "https://<your-domino-instance>/endpoints/<endpoint-id>/v1"
```

The agent code automatically fetches a short-lived access token from `http://localhost:8899/access-token` (provided by the Domino runtime) for authentication.

**System prompt:** The `prompts.simple_agent_system` field controls the agent's behavior. The default instructs it to respond only with quotes from the `science_quote` or `philosophy_quote` tool functions.

### 2. Run development evaluation (as a Domino Job)

**Important:** Run this as a **Domino Job**, not from a workspace. Running as a Job creates an **experiment run** in Domino, which is the mechanism that lets you compare different configurations and deploy the agent.

```bash
python dev_eval_simplest_agent.py
```

This loops through `sample_questions.csv`, runs the agent on each question, and attaches evaluation scores (toxicity, relevancy, accuracy) to each trace via the judge function. Adjust `MAX_ROWS_TO_PROCESS` at the top of the file to limit the batch size.

After the Job completes:

- View traces in the **Agent Traces** tab in your project
- View the experiment run in the **Experiments** tab
- **Deploy the agent** directly from the experiment run UI by clicking "Deploy"

### 3. Deploy as a Domino App (production)

In Domino, create an App with:

- **Entry point**: `app.sh`

The app serves the chat UI on port 8888. Every user conversation is automatically traced via `@add_tracing` in `chat_app.py` and visible in the Domino agent dashboard.

### 4. Run production evaluation (as a Domino Scheduled Job)

Set up `prod_eval_simplest_agent.py` as a **Domino Scheduled Job** to continuously evaluate traces from the live app:

```bash
python prod_eval_simplest_agent.py
```

Before running, update `AGENT_ID` in the file to match your deployed agent's ID (found in the Domino agent dashboard under "Evaluation setup").

The script:

1. Reads the last-processed timestamp from `/mnt/data/<project>/last_trace_time.txt`
2. Fetches all new traces since that time via `search_agent_traces`
3. Scores each trace with `AgentEvaluator`
4. Writes scores back to each trace via `log_evaluation`
5. Updates the timestamp file

---

## Customizing the Evaluator

`evaluation_library.py` has three `evaluate_*` methods stubbed with random scores. Replace them with real logic — for example, an LLM-as-judge call:

```python
def evaluate_relevancy(self, query: str, agent_output: str) -> float:
    # Call an LLM to score relevancy, return 0.0-1.0
    ...
```

The overall score is computed as: `(relevancy + accuracy) / 2 * (1 - toxicity)`.

---

## Production Best Practices

### Add conversation history to the chat app

The chat app is currently stateless — each request is handled independently with no memory of previous messages. A follow-up like "Can you give me another?" after "Give me a science quote" won't have context. To make it conversational, store message history keyed by `conversation_id` and pass it to the agent:

```python
from pydantic_ai.messages import ModelRequest, ModelResponse

# In-memory conversation store (use Redis or a DB for production)
conversations: dict[str, list[ModelRequest | ModelResponse]] = {}

@add_tracing(name='single_question_agent', autolog_frameworks=["pydantic_ai"])
async def ask_agent(question: str, conversation_id: str):
    agent = create_agent()
    history = conversations.get(conversation_id, [])
    result = await agent.run(question, message_history=history)
    conversations[conversation_id] = result.all_messages()
    return result

# Add a clear endpoint
@app.post("/clear")
async def clear_conversation(request: ChatMessage):
    conversations.pop(request.conversation_id, None)
    return {"status": "cleared"}
```

The key change is `message_history=history` — pydantic-ai replays the prior messages so the LLM sees the full conversation. Add a "New Chat" button in the UI that calls `/clear` to reset.

---

## How This Differs from the RAG Agent Demo

This project is intentionally minimal. It strips away RAG, vector stores, and document ingestion to focus purely on the Domino instrumentation patterns:

| | This Demo | [RAG Agent Demo](https://github.com/dominodatalab/rag-agent-demo) |
|---|---|---|
| **Agent logic** | Returns random quotes via tool calls | Retrieves answers from documents via ChromaDB |
| **Vector store** | None | ChromaDB with SentenceTransformers |
| **Document ingestion** | None | `init_chromadb.py` indexes PDFs/TXT/MD |
| **Custom span attributes** | None | MLflow span attributes for retrieval distances |
| **Domino tracing** | Same pattern | Same pattern |
| **Evaluation** | Same pattern (random placeholder scores) | Same pattern (random placeholder scores) |
| **Production eval** | Same pattern | Same pattern |
| **Chat app** | Same pattern | Same pattern |

If you already understand Domino tracing and evaluation from this demo, the RAG demo adds retrieval-augmented generation on top of the same foundation.
