"""
Coordinator — Orchestrates the Research → Summarize pipeline

Acts as an A2A CLIENT. It:
  1. Uses A2ACardResolver to fetch each agent's AgentCard (discovery)
  2. Uses ClientFactory to create a proper Client for each agent
  3. Sends a research task → collects events → extracts raw notes
  4. Sends those notes to the summarizer → collects events → extracts summary
  5. Prints the final result

A2A Concepts:
  - A2ACardResolver:  Fetches /.well-known/agent.json to discover an agent
  - ClientFactory:    Creates the right transport client based on the agent's card
  - ClientConfig:     Configuration for the client (http client, transport prefs)
  - Client.send_message: Sends a Message and returns an async event stream
  - Event stream:     Each event is either a (Task, update) tuple or a Message
  - Task chaining:    Output artifact of Agent 1 becomes the input to Agent 2
"""

import asyncio
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientFactory, ClientConfig
from a2a.types import (
    Message,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    TextPart,
)

RESEARCH_AGENT_URL = "http://localhost:8001"
SUMMARIZER_AGENT_URL = "http://localhost:8002"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _user_message(text: str) -> Message:
    """Build a user-role A2A Message with plain text.
    message_id is required — every message must have a unique ID.
    """
    return Message(
        role=Role.user,
        message_id=str(uuid4()),
        parts=[Part(root=TextPart(text=text))],
    )


async def _send_and_get_result(client, message: Message) -> str:
    """
    Send a message to an agent and collect the final artifact text.

    Client.send_message() returns an async event stream. Each event is either:
      - (Task, TaskStatusUpdateEvent)  — a status change (working, completed, etc.)
      - (Task, TaskArtifactUpdateEvent) — a new artifact chunk was added
      - (Task, None)                   — task reached a terminal state
      - Message                        — agent replied directly with a message

    We iterate through all events and return the text from the last artifact
    on the final Task object.
    """
    final_task = None

    async for event in client.send_message(message):
        if isinstance(event, tuple):
            task, update = event
            final_task = task   # keep updating — last one is the complete task
        # isinstance(event, Message) would be a direct reply with no task

    if not final_task or not final_task.artifacts:
        return ""

    for artifact in final_task.artifacts:
        for part in artifact.parts:
            if hasattr(part, "root") and hasattr(part.root, "text"):
                return part.root.text
            if hasattr(part, "text"):
                return part.text
    return ""


# ── Main Orchestration Logic ──────────────────────────────────────────────────

async def run(question: str) -> None:
    async with httpx.AsyncClient() as http_client:

        # ── PHASE 1: Agent Discovery ─────────────────────────────────────────
        # A2ACardResolver fetches /.well-known/agent.json from each agent URL.
        # This is how the coordinator learns what each agent can do —
        # without hardcoding it. Agents can update their cards independently.
        print(f"\n[Coordinator] Question: {question!r}")
        print("[Coordinator] Discovering agents via Agent Cards...\n")

        research_card = await A2ACardResolver(http_client, RESEARCH_AGENT_URL).get_agent_card()
        summarizer_card = await A2ACardResolver(http_client, SUMMARIZER_AGENT_URL).get_agent_card()

        print(f"  Discovered: [{research_card.name}] @ {research_card.url}")
        print(f"    Skills: {[s.name for s in research_card.skills]}")
        print(f"  Discovered: [{summarizer_card.name}] @ {summarizer_card.url}")
        print(f"    Skills: {[s.name for s in summarizer_card.skills]}\n")

        # ── PHASE 2: Create Clients via ClientFactory ─────────────────────────
        # ClientFactory reads the agent card to pick the right transport (JSONRPC, REST, etc.)
        # ClientConfig holds shared settings — here we pass the httpx client.
        config = ClientConfig(httpx_client=http_client)
        factory = ClientFactory(config)

        research_client = factory.create(research_card)
        summarizer_client = factory.create(summarizer_card)

        # ── PHASE 3: Send Task to Research Agent ─────────────────────────────
        # send_message() takes a Message directly (not a SendMessageRequest wrapper).
        # It returns an async event stream we iterate to get the final result.
        print("[Coordinator] → Sending task to Research Agent...")

        raw_notes = await _send_and_get_result(
            research_client,
            _user_message(question),
        )
        print(f"[Research Agent] ← Task complete. Received {len(raw_notes)} chars.\n")

        # ── PHASE 4: Chain — feed research output into Summarizer Agent ──────
        # The coordinator takes the artifact from Agent 1 and passes it as
        # the input to Agent 2. Neither agent knows about the other.
        print("[Coordinator] → Sending task to Summarizer Agent...")

        final_summary = await _send_and_get_result(
            summarizer_client,
            _user_message(raw_notes),
        )
        print("[Summarizer Agent] ← Task complete.\n")

        # ── PHASE 5: Output ───────────────────────────────────────────────────
        print("=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)
        print(final_summary)
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run("What are the key trends in AI agents in 2025?"))
