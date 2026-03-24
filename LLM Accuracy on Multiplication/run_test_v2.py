"""
run_test_v2.py — Three-way multiplication accuracy comparison

For each test config, this script measures accuracy across three approaches:

  1. Python           — exact arithmetic, always correct (ground truth)
  2. LLM Direct       — Claude answers from memory/pattern-matching, no tools
  3. LLM + Tool       — Claude acts as an agent: it calls a calculator tool,
                        gets the exact result back, then returns the answer

This demonstrates the core value of tool use in AI systems:
the LLM's reasoning ability + exact computation = reliable answers.

Usage:
    python3 run_test_v2.py
"""

import os
import csv
import re
import random
import time
from dotenv import load_dotenv
from anthropic import AnthropicBedrock

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

CLAUDE_MODEL = "us.anthropic.claude-sonnet-4-20250514-v1:0"

TEST_CONFIGS = [
    {"digits": 2, "samples": 30},
    {"digits": 5, "samples": 30},
]

RESULTS_DIR            = os.path.join(os.path.dirname(__file__), "results")
DELAY_BETWEEN_SAMPLES  = 0.5   # seconds between samples (2 API calls each with tool)


# ── Calculator Tool Definition ────────────────────────────────────────────────
# This is the tool we expose to Claude. It tells Claude:
# "you have a calculator available — use it for arithmetic."

CALCULATOR_TOOL = [
    {
        "name": "calculator",
        "description": (
            "Performs exact arithmetic. "
            "Always use this tool when asked to multiply two numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "num1":      {"type": "number", "description": "First number"},
                "num2":      {"type": "number", "description": "Second number"},
                "operation": {
                    "type": "string",
                    "enum": ["multiply"],
                    "description": "Arithmetic operation to perform",
                },
            },
            "required": ["num1", "num2", "operation"],
        },
    }
]


# ── Number Generation ─────────────────────────────────────────────────────────

def random_n_digit_number(digits: int) -> int:
    low  = 10 ** (digits - 1)
    high = (10 ** digits) - 1
    return random.randint(low, high)


# ── Response Parsing ──────────────────────────────────────────────────────────

def parse_number(text: str) -> int | None:
    """Extract an integer from LLM response text."""
    cleaned = text.replace(",", "").replace(".", "").strip()
    match = re.search(r"\d+", cleaned)
    return int(match.group()) if match else None


# ── Approach 1: LLM Direct ────────────────────────────────────────────────────

def ask_llm_direct(client: AnthropicBedrock, num1: int, num2: int) -> tuple[str, int | None]:
    """
    Ask Claude to multiply without any tools.
    Claude must rely purely on its own internal computation.
    """
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=64,
        messages=[{
            "role": "user",
            "content": (
                f"What is {num1} × {num2}? "
                "Respond with only the final number. "
                "No explanation, no working — just the number."
            ),
        }],
    )
    raw = response.content[0].text.strip()
    return raw, parse_number(raw)


# ── Approach 2: LLM + Tool (Agentic) ─────────────────────────────────────────

def ask_llm_with_tool(client: AnthropicBedrock, num1: int, num2: int) -> tuple[str, int | None, bool]:
    """
    Ask Claude to multiply using the calculator tool.

    This is the agentic pattern — a two-turn conversation:
      Turn 1: Claude decides to call the calculator tool
      Turn 2: We return the exact Python result; Claude gives the final answer

    Returns:
        raw           — Claude's final text response
        parsed_answer — integer extracted from the response
        tool_called   — whether Claude actually used the tool
    """
    user_message = {
        "role": "user",
        "content": (
            f"What is {num1} × {num2}? "
            "Use the calculator tool to compute this. "
            "Respond with only the final number."
        ),
    }

    # ── Turn 1: Claude decides to call the tool ──────────────────────────────
    response1 = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=256,
        tools=CALCULATOR_TOOL,
        messages=[user_message],
    )

    tool_called = False

    if response1.stop_reason == "tool_use":
        # Claude decided to use the calculator — extract its inputs
        tool_block = next(b for b in response1.content if b.type == "tool_use")
        tool_input = tool_block.input

        # Execute the tool in Python (exact, never wrong)
        tool_result = int(tool_input["num1"]) * int(tool_input["num2"])
        tool_called = True

        # ── Turn 2: Return the result, let Claude give the final answer ──────
        response2 = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=64,
            tools=CALCULATOR_TOOL,
            messages=[
                user_message,
                {"role": "assistant", "content": response1.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": str(tool_result),
                        }
                    ],
                },
            ],
        )
        raw = response2.content[0].text.strip()
        return raw, parse_number(raw), tool_called

    else:
        # Claude answered without calling the tool — treat as direct answer
        raw = next(
            (b.text for b in response1.content if hasattr(b, "text")), ""
        ).strip()
        return raw, parse_number(raw), tool_called


# ── Single Test Run ───────────────────────────────────────────────────────────

def run_test(config: dict, client: AnthropicBedrock) -> dict:
    digits  = config["digits"]
    samples = config["samples"]
    low     = 10 ** (digits - 1)
    high    = (10 ** digits) - 1

    print(f"\n{'='*65}")
    print(f"TEST: {digits}-digit multiplication  ({low}–{high})  ×  {samples} samples")
    print(f"{'='*65}")
    print(f"  {'#':>3}  {'num1':>8}  {'num2':>8}  {'correct':>12}  {'direct':>12}  {'tool':>12}  D  T")
    print(f"  {'-'*63}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"results_v2_{digits}digit.csv")

    direct_correct = 0
    tool_correct   = 0
    tool_call_count = 0
    rows = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample", "num1", "num2",
            "python_answer",
            "llm_direct_answer", "llm_direct_correct",
            "llm_tool_answer",   "llm_tool_correct", "tool_was_called",
        ])

        for i in range(1, samples + 1):
            num1    = random_n_digit_number(digits)
            num2    = random_n_digit_number(digits)
            correct = num1 * num2   # Python ground truth

            # LLM Direct
            try:
                _, direct_ans = ask_llm_direct(client, num1, num2)
            except Exception as e:
                direct_ans = None

            # LLM + Tool
            try:
                _, tool_ans, tool_called = ask_llm_with_tool(client, num1, num2)
            except Exception as e:
                tool_ans, tool_called = None, False

            d_ok = (direct_ans == correct)
            t_ok = (tool_ans   == correct)

            if d_ok: direct_correct  += 1
            if t_ok: tool_correct    += 1
            if tool_called: tool_call_count += 1

            d_sym = "✓" if d_ok else ("?" if direct_ans is None else "✗")
            t_sym = "✓" if t_ok else ("?" if tool_ans   is None else "✗")

            print(
                f"  [{i:>3}]  {num1:>8}  {num2:>8}  {correct:>12}  "
                f"{str(direct_ans):>12}  {str(tool_ans):>12}  {d_sym}  {t_sym}"
            )

            writer.writerow([
                i, num1, num2, correct,
                direct_ans, d_ok,
                tool_ans,   t_ok, tool_called,
            ])

            time.sleep(DELAY_BETWEEN_SAMPLES)

    direct_acc = (direct_correct / samples) * 100
    tool_acc   = (tool_correct   / samples) * 100

    summary = {
        "digits":          digits,
        "samples":         samples,
        "range":           f"{low}–{high}",
        "direct_correct":  direct_correct,
        "tool_correct":    tool_correct,
        "direct_accuracy": direct_acc,
        "tool_accuracy":   tool_acc,
        "tool_call_rate":  (tool_call_count / samples) * 100,
        "csv_path":        csv_path,
    }

    print(f"\n  Direct accuracy : {direct_correct}/{samples} = {direct_acc:.1f}%")
    print(f"  Tool accuracy   : {tool_correct}/{samples}  = {tool_acc:.1f}%")
    print(f"  Tool call rate  : {tool_call_count}/{samples} ({summary['tool_call_rate']:.0f}% of samples used the tool)")
    print(f"  Saved to: {csv_path}")

    return summary


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    client = AnthropicBedrock()
    summaries = []

    for config in TEST_CONFIGS:
        summary = run_test(config, client)
        summaries.append(summary)

    print(f"\n{'='*65}")
    print("FINAL SUMMARY")
    print(f"{'='*65}")
    print(f"{'Test':<25} {'Samples':>8} {'Python':>8} {'Direct':>10} {'Tool':>10}")
    print(f"{'-'*65}")
    for s in summaries:
        label = f"{s['digits']}-digit ({s['range']})"
        print(
            f"{label:<25} {s['samples']:>8} {'100.0%':>8} "
            f"{s['direct_accuracy']:>9.1f}% {s['tool_accuracy']:>9.1f}%"
        )
    print(f"{'='*65}")
    print("\nDetailed results saved to the results/ folder.")
