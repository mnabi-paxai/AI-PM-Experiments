"""
Summarizer Agent — Port 8002

Receives raw research notes, returns a clean executive summary as an Artifact.
In production: replace the summarization logic with a real LLM call.

A2A Concept Highlighted:
  - Task chaining: This agent doesn't know it's part of a pipeline.
    It just processes whatever text it receives. The coordinator owns
    the routing logic — agents stay decoupled.
"""

import uvicorn
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


# ── Agent Logic ───────────────────────────────────────────────────────────────

class SummarizerExecutor(AgentExecutor):
    """
    Receives raw notes (from any source), returns a clean executive summary.
    """

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        # The input here is the raw notes produced by the Research Agent.
        # This agent has no idea where the text came from — it just processes it.
        raw_notes = context.get_user_input()

        await task_updater.start_work(
            message=task_updater.new_agent_message(
                parts=[Part(root=TextPart(text="Summarizing notes..."))]
            )
        )

        # Extract the title line from each Tavily result block (lines starting with "- ")
        # Replace this block with a real LLM call for production use.
        bullets = []
        blocks = raw_notes.strip().split("\n\n")
        for block in blocks:
            lines = [l.strip() for l in block.splitlines() if l.strip()]
            for line in lines:
                if line.startswith("- "):
                    bullets.append(f"  • {line[2:].strip()}")
                    break

        bullet_points = "\n".join(bullets) if bullets else raw_notes[:500]
        summary = f"Executive Summary\n{'─' * 40}\n{bullet_points}\n"

        await task_updater.add_artifact(
            parts=[Part(root=TextPart(text=summary))],
            name="summary",
        )
        await task_updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("Cancel not supported by this agent")


# ── Agent Card ────────────────────────────────────────────────────────────────

AGENT_CARD = AgentCard(
    name="Summarizer Agent",
    description="Takes raw research notes and returns a clean executive summary.",
    url="http://localhost:8002",
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=False),
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    skills=[
        AgentSkill(
            id="summarize_notes",
            name="Summarize Notes",
            description="Condenses raw notes into a clean executive summary.",
            tags=["summarization", "notes"],
            input_modes=["text/plain"],
            output_modes=["text/plain"],
        )
    ],
)


# ── Server Entry Point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    handler = DefaultRequestHandler(
        agent_executor=SummarizerExecutor(),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=AGENT_CARD, http_handler=handler).build()
    uvicorn.run(app, host="0.0.0.0", port=8002)
