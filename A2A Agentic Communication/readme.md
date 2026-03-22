
<!-- Press Shift + Command + V to open the Markdown Preview. -->
# A2A Agentic Communication — Research Pipeline


A hands-on learning project demonstrating Google's **Agent-to-Agent (A2A) protocol** through a working multi-agent research pipeline. Built as a deep dive for technical product managers learning about AI agent architectures.

---

## What This Project Does

A user asks a question. Instead of one monolithic AI answering it, **three specialized agents collaborate**:

1. The **Coordinator** receives the question and orchestrates the workflow
2. The **Research Agent** fetches and returns raw research notes on the topic
3. The **Summarizer Agent** distills those notes into a clean executive summary

The user only interacts with the Coordinator. The two specialist agents are fully autonomous HTTP services that could run on different servers, be written in different languages, or be maintained by different teams — A2A is the contract between them.

---

## Project Structure

```
A2A Agentic Communication/
├── research_agent.py       Agent 1 — returns raw research notes (port 8001)
├── summarizer_agent.py     Agent 2 — distills notes into a summary (port 8002)
├── coordinator.py          Orchestrator — discovers agents, chains tasks
├── requirements.txt        Python dependencies
└── readme.md               This file
```

---

## Architecture

### High-Level Flow

```
User
  │
  ▼
coordinator.py
  │
  ├── GET /.well-known/agent.json ──────► research_agent.py   (Agent Card)
  ├── GET /.well-known/agent.json ──────► summarizer_agent.py (Agent Card)
  │
  ├── Task: "What are the key trends?" ─► research_agent.py
  │                                             │
  │◄── Artifact: raw research notes ────────────┘
  │
  ├── Task: "Summarize this: [notes]" ──► summarizer_agent.py
  │                                             │
  │◄── Artifact: executive summary ─────────────┘
  │
  └── Prints final answer to user
```

### A2A Protocol Concepts Used

| Concept | Role in This Project |
|---|---|
| **Agent Card** | Each agent publishes a JSON manifest at `/.well-known/agent.json` declaring its name, URL, skills, and supported input/output modes. The coordinator reads these to discover agents at runtime. |
| **Task** | The unit of work sent between agents. Has a unique ID, a status lifecycle, and produces Artifacts as output. |
| **Message** | The input to a Task. Contains one or more Parts (text, file, or structured data). Every message requires a globally unique `message_id`. |
| **Artifact** | The output of a completed Task. The coordinator reads artifacts from one agent and passes them as input to the next. |
| **EventQueue** | The raw event bus inside each agent server. The agent publishes status updates and artifacts to this queue during execution. |
| **TaskUpdater** | A convenience wrapper around EventQueue. Provides named methods (`start_work()`, `add_artifact()`, `complete()`) instead of raw event construction. |
| **A2ACardResolver** | Client-side utility that fetches an agent's card from `/.well-known/agent.json`. Used by the coordinator for discovery. |
| **ClientFactory** | Creates the right transport client (JSONRPC, REST, gRPC) based on the agent's card. Keeps the coordinator decoupled from transport details. |
| **Task Chaining** | The coordinator takes the Artifact from Agent 1 and passes it as the Message input to Agent 2. Neither agent knows about the other — the coordinator owns the pipeline logic. |

---

## Component Details

### Research Agent (`research_agent.py`) — Port 8001

**Responsibility:** Receive a research question, return structured notes.

**A2A role:** A2A Server

**Key implementation:**
- Subclasses `AgentExecutor` and implements `execute(context, event_queue)`
- Uses `context.get_user_input()` to extract the question from the incoming message
- Wraps the `EventQueue` in a `TaskUpdater` for clean status reporting
- Calls `task_updater.add_artifact()` to attach notes as the output
- Calls `task_updater.complete()` to signal task completion
- In production: replace the mock notes with a real search API call (Tavily, Perplexity, etc.)

**Agent Card (served at `http://localhost:8001/.well-known/agent.json`):**
```json
{
  "name": "Research Agent",
  "url": "http://localhost:8001",
  "skills": [{ "id": "research_topic", "name": "Research Topic", "tags": ["research", "notes"] }]
}
```

---

### Summarizer Agent (`summarizer_agent.py`) — Port 8002

**Responsibility:** Receive raw text, return a clean executive summary.

**A2A role:** A2A Server

**Key implementation:** Identical structure to the Research Agent. The only differences are the port number, skill definition, and the executor logic.

This is intentional — **every A2A agent looks the same from the outside.** The coordinator doesn't need to know anything about internals.

- In production: replace the mock summarization with a real LLM call (Claude, GPT-4, Gemini, etc.)

---

### Coordinator (`coordinator.py`)

**Responsibility:** Discover agents, send tasks, chain results, return final answer.

**A2A role:** A2A Client (pure client — no server, no agent card)

**Key implementation:**

```
Step 1 — Discovery
  A2ACardResolver fetches /.well-known/agent.json from each agent URL
  → Parses the AgentCard (name, skills, transport preference)

Step 2 — Client Creation
  ClientFactory reads the agent card and creates the right transport client
  → Handles JSONRPC formatting, retries, and error wrapping automatically

Step 3 — Send Task to Research Agent
  Builds a Message with message_id + user question
  Calls client.send_message(message) → returns an async event stream
  Iterates events → collects final Task → extracts artifact text

Step 4 — Chain: Feed research output to Summarizer Agent
  Takes artifact text from Step 3
  Builds a new Message with that text as input
  Sends to summarizer → collects summary artifact

Step 5 — Output
  Prints final executive summary
```

---

## Task Lifecycle

Every task in A2A goes through a defined state machine:

```
submitted ──► working ──► completed
                │
                ├──► failed
                ├──► canceled
                ├──► input_required
                └──► auth_required
```

In this project:
- `working` — set by `task_updater.start_work()` when the agent begins processing
- `completed` — set by `task_updater.complete()` when the artifact is ready

---

## Event Stream Model

When the coordinator calls `client.send_message()`, it receives an **async event stream** (not a single response). Each event is one of:

| Event Type | Meaning |
|---|---|
| `(Task, TaskStatusUpdateEvent)` | Agent reported a status change (e.g., working → completed) |
| `(Task, TaskArtifactUpdateEvent)` | Agent attached a new artifact chunk |
| `(Task, None)` | Task reached a terminal state |
| `Message` | Agent replied directly without creating a task |

The coordinator iterates through all events and reads artifacts from the final `Task` object.

This design supports **streaming agents** (that send partial results as they work) and **non-streaming agents** (that send one final artifact) with the same client code.

---

## SDK Version Notes

Built and tested against **`a2a-sdk 0.3.25`**.

Key API differences vs. earlier versions (0.2.x):

| 0.2.x API | 0.3.25 API | Reason for Change |
|---|---|---|
| `A2AStarlette` | `A2AStarletteApplication` + `.build()` | Separated config from instantiation |
| `AgentExecutor.execute(context, task_updater)` | `execute(context, event_queue)` | Gives agents direct access to the event bus for advanced use cases |
| `A2AClient.get_client_from_agent_card_url()` | `A2ACardResolver` + `ClientFactory` | Separated discovery from transport creation |
| `Message` (no required ID) | `Message(message_id=str(uuid4()), ...)` | Every message must be uniquely identifiable in a distributed system |
| `defaultInputModes` (camelCase) | `default_input_modes` (snake_case) | Python API enforces snake_case; camelCase is only for the JSON wire format |
| `AgentSkill` (no tags) | `AgentSkill(tags=[...], ...)` | Tags are required for agent discovery and skill routing |

---

## How to Run

### 1. Install dependencies (once)

```bash
cd "A2A Agentic Communication"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the agents (two separate terminals)

```bash
# Terminal 1
source venv/bin/activate
python3 research_agent.py
# Expected: INFO: Uvicorn running on http://0.0.0.0:8001

# Terminal 2
source venv/bin/activate
python3 summarizer_agent.py
# Expected: INFO: Uvicorn running on http://0.0.0.0:8002
```

### 3. Run the coordinator (third terminal)

```bash
source venv/bin/activate
python3 coordinator.py
```

### Expected Output

```
[Coordinator] Question: 'What are the key trends in AI agents in 2025?'
[Coordinator] Discovering agents via Agent Cards...

  Discovered: [Research Agent] @ http://localhost:8001
    Skills: ['Research Topic']
  Discovered: [Summarizer Agent] @ http://localhost:8002
    Skills: ['Summarize Notes']

[Coordinator] → Sending task to Research Agent...
[Research Agent] ← Task complete. Received 621 chars.

[Coordinator] → Sending task to Summarizer Agent...
[Summarizer Agent] ← Task complete.

============================================================
FINAL ANSWER
============================================================
Executive Summary
────────────────────────────────────────
  • Multi-agent architectures are the dominant pattern for complex AI tasks in 2025.
  • A2A protocol provides a vendor-neutral standard for inter-agent calls.
  • Reasoning models outperform base models on multi-step tasks.
  • Key open problems: persistent memory, long-horizon planning, and cost efficiency.
  • Deployment patterns are shifting to "agent mesh" architectures.
  • Tool use has matured across file I/O, code execution, and web browsing.
============================================================
```

---

## What to Extend Next

| Extension | What it teaches |
|---|---|
| Replace mock research with Tavily API | Real tool use inside an A2A agent |
| Replace mock summarization with Claude API | LLM integration inside an A2A agent |
| Add a third agent (e.g., Translator) | Deeper task chaining |
| Enable `streaming=True` on an agent | SSE streaming and partial artifact delivery |
| Deploy agents to separate servers | True distributed agent communication |
| Add authentication to the agent card | A2A security schemes |
