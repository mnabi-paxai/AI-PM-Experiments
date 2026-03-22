"""
Research Agent — Port 8001

Receives a research question, returns raw notes as an Artifact.

A2A Concepts:
  - AgentCard:              Public identity & capability manifest (served at /.well-known/agent.json)
  - AgentExecutor:          Subclass this to write your agent's logic
  - RequestContext:         Holds the incoming task — use get_user_input() to read the question
  - EventQueue:             The raw event bus passed to execute() — we wrap it in a TaskUpdater
  - TaskUpdater:            Convenience wrapper around EventQueue for publishing progress/artifacts
  - A2AStarletteApplication: Wraps everything into a standards-compliant HTTP server
"""

import os
import uvicorn
from dotenv import load_dotenv
from tavily import TavilyClient
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCard,
    AgentCapabilities,
    AgentSkill,
    Part,
    TextPart,
)

load_dotenv()
_tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


# ── Agent Logic ───────────────────────────────────────────────────────────────

class ResearchExecutor(AgentExecutor):
    """
    The brain of the Research Agent.

    NOTE: In v0.3.x the signature is execute(context, event_queue) — not task_updater.
    We wrap the raw EventQueue in a TaskUpdater for a cleaner API.
    """

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Wrap the raw event queue in a TaskUpdater for convenience
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        # get_user_input() extracts plain text from the incoming A2A Message parts
        question = context.get_user_input()

        # Signal we're actively working (moves task status to TaskState.working)
        await task_updater.start_work(
            message=task_updater.new_agent_message(
                parts=[Part(root=TextPart(text=f"Researching: {question}..."))]
            )
        )

        # Call Tavily to get real search results for the question
        response = _tavily.search(question, max_results=5)
        snippets = "\n\n".join(
            f"- {r['title']}\n  {r['content']}"
            for r in response["results"]
        )
        notes = f"Research Notes — Topic: {question}\n\n{snippets}"

        # Attach the result as an Artifact — this is the agent's "return value"
        await task_updater.add_artifact(
            parts=[Part(root=TextPart(text=notes))],
            name="research_notes",
        )

        # Mark task complete (TaskState.completed, final=True)
        await task_updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("Cancel not supported by this agent")


# ── Agent Card ────────────────────────────────────────────────────────────────

AGENT_CARD = AgentCard(
    name="Research Agent",
    description="Given a topic or question, returns structured research notes.",
    url="http://localhost:8001",
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=False),
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    skills=[
        AgentSkill(
            id="research_topic",
            name="Research Topic",
            description="Returns raw research notes on any topic.",
            tags=["research", "notes"],
            input_modes=["text/plain"],
            output_modes=["text/plain"],
        )
    ],
)


# ── Server Entry Point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    handler = DefaultRequestHandler(
        agent_executor=ResearchExecutor(),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=AGENT_CARD, http_handler=handler).build()
    uvicorn.run(app, host="0.0.0.0", port=8001)
