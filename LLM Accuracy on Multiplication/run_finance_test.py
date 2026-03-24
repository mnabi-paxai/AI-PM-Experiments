"""
run_finance_test.py — Compound Interest Accuracy: Python vs LLM Direct vs LLM + Tool

Tests how accurately an LLM computes compound interest across 30 random scenarios.

Formula:  A = P × (1 + r/n)^(n×t)
  P = principal ($100–$10,000)
  r = annual interest rate (1%–20%)
  n = 12 (compounded monthly, fixed)
  t = 1–30 years

Three approaches compared:
  1. Python     — exact (ground truth)
  2. LLM Direct — Claude answers from memory, no tools
  3. LLM + Tool — Claude orchestrates a step-by-step calculator (multi-tool call loop)

Why LLM + Tool can still fail here:
  - Claude must decompose the formula correctly into the right sequence of operations
  - It must use r as a decimal (0.05), not a percentage (5)
  - It must apply the correct order: divide → add → power → multiply
  - Each individual tool call is exact — but the orchestration can be wrong

Usage:
    python3 run_finance_test.py
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

CLAUDE_MODEL           = "us.anthropic.claude-sonnet-4-20250514-v1:0"
SAMPLES                = 30
RESULTS_DIR            = os.path.join(os.path.dirname(__file__), "results")
DELAY_BETWEEN_SAMPLES  = 0.5   # seconds
TOLERANCE              = 0.01  # answers within $0.01 are considered correct
MAX_TOOL_TURNS         = 10    # max back-and-forth turns in the agentic loop


# ── Calculator Tool ───────────────────────────────────────────────────────────
# A general arithmetic tool. Claude must call it multiple times in sequence
# to decompose the compound interest formula into steps.

CALCULATOR_TOOL = [
    {
        "name": "calculate",
        "description": (
            "Performs a single arithmetic operation. "
            "Call this multiple times to break a complex formula into steps. "
            "Operations: add (a+b), subtract (a-b), multiply (a×b), "
            "divide (a÷b), power (a^b)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide", "power"],
                },
                "a": {"type": "number", "description": "First operand"},
                "b": {"type": "number", "description": "Second operand"},
            },
            "required": ["operation", "a", "b"],
        },
    }
]


# ── Problem Generation ────────────────────────────────────────────────────────

def generate_problem() -> dict:
    """Generate a random compound interest problem."""
    P = round(random.uniform(100, 10000), 2)
    r = round(random.uniform(0.01, 0.20), 4)   # 1%–20% as decimal
    n = 12                                      # monthly compounding
    t = random.randint(1, 30)
    correct = round(P * (1 + r / n) ** (n * t), 2)
    return {"P": P, "r": r, "n": n, "t": t, "correct": correct}


def problem_prompt(p: dict, use_tool: bool = False) -> str:
    r_pct = round(p["r"] * 100, 2)
    base = (
        f"I invest ${p['P']:.2f} at an annual interest rate of {r_pct}%, "
        f"compounded monthly, for {p['t']} year(s). "
        f"What is the final balance using the formula A = P × (1 + r/n)^(n×t) "
        f"where r is the annual rate as a decimal and n=12? "
        f"Respond with only the final dollar amount as a number rounded to 2 decimal places. "
        f"No dollar sign, no explanation — just the number."
    )
    if use_tool:
        base += " Use the calculator tool to compute each step."
    return base


# ── Tool Execution ────────────────────────────────────────────────────────────

def execute_tool(operation: str, a: float, b: float) -> float:
    """Execute a single arithmetic operation. Always exact."""
    if operation == "add":      return a + b
    if operation == "subtract": return a - b
    if operation == "multiply": return a * b
    if operation == "divide":   return a / b if b != 0 else float("inf")
    if operation == "power":    return a ** b
    raise ValueError(f"Unknown operation: {operation}")


# ── Response Parsing ──────────────────────────────────────────────────────────

def parse_amount(text: str) -> float | None:
    """Extract a dollar amount from LLM response text."""
    cleaned = text.replace(",", "").replace("$", "").strip()
    match = re.search(r"\d+(\.\d+)?", cleaned)
    return round(float(match.group()), 2) if match else None


def is_correct(llm_answer: float | None, correct: float) -> bool:
    if llm_answer is None:
        return False
    return abs(llm_answer - correct) <= TOLERANCE


# ── Approach 1: LLM Direct ────────────────────────────────────────────────────

def ask_llm_direct(client: AnthropicBedrock, problem: dict) -> tuple[str, float | None]:
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=64,
        messages=[{"role": "user", "content": problem_prompt(problem, use_tool=False)}],
    )
    raw = response.content[0].text.strip()
    return raw, parse_amount(raw)


# ── Approach 2: LLM + Tool (Agentic Loop) ────────────────────────────────────

def ask_llm_with_tool(client: AnthropicBedrock, problem: dict) -> tuple[str, float | None, int]:
    """
    Agentic loop: Claude can make multiple tool calls before giving a final answer.

    Unlike the multiplication test (1 tool call), compound interest requires
    Claude to call the tool 4–5 times in the correct sequence:
      Step 1: divide(r, n)         → monthly rate
      Step 2: add(result, 1)       → 1 + monthly rate
      Step 3: power(result, n*t)   → growth factor
      Step 4: multiply(P, result)  → final balance

    If Claude gets the sequence wrong, the answer is wrong even though
    every individual tool call returned an exact result.

    Returns: (raw_text, parsed_answer, number_of_tool_calls_made)
    """
    messages = [{"role": "user", "content": problem_prompt(problem, use_tool=True)}]
    tool_call_count = 0

    for _ in range(MAX_TOOL_TURNS):
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=512,
            tools=CALCULATOR_TOOL,
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            # Process all tool calls in this response turn
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_call_count += 1
                    try:
                        result = execute_tool(
                            block.input["operation"],
                            float(block.input["a"]),
                            float(block.input["b"]),
                        )
                    except Exception as e:
                        result = f"error: {e}"

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    })

            # Append assistant turn + tool results, then loop back
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user",      "content": tool_results})

        else:
            # Claude gave its final answer (stop_reason == "end_turn")
            raw = next(
                (b.text for b in response.content if hasattr(b, "text")), ""
            ).strip()
            return raw, parse_amount(raw), tool_call_count

    # Exceeded max turns without a final answer
    return "max_turns_exceeded", None, tool_call_count


# ── Main Test Runner ──────────────────────────────────────────────────────────

def run_test(client: AnthropicBedrock) -> dict:
    print(f"\n{'='*75}")
    print("TEST: Compound Interest  A = P × (1 + r/n)^(n×t)  |  n=12 (monthly)")
    print(f"Samples: {SAMPLES}  |  Tolerance: ±${TOLERANCE}")
    print(f"{'='*75}")
    print(f"  {'#':>3}  {'P':>8}  {'r%':>5}  {'t':>3}  {'correct':>12}  {'direct':>12}  {'tool':>12}  D  T  calls")
    print(f"  {'-'*73}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "results_finance.csv")

    direct_correct_count = 0
    tool_correct_count   = 0
    total_tool_calls     = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample", "principal", "annual_rate_pct", "years", "n",
            "python_answer",
            "llm_direct_answer", "llm_direct_correct",
            "llm_tool_answer",   "llm_tool_correct",
            "tool_calls_made",   "llm_direct_raw", "llm_tool_raw",
        ])

        for i in range(1, SAMPLES + 1):
            problem = generate_problem()
            correct = problem["correct"]

            # LLM Direct
            try:
                direct_raw, direct_ans = ask_llm_direct(client, problem)
            except Exception as e:
                direct_raw, direct_ans = str(e), None

            # LLM + Tool
            try:
                tool_raw, tool_ans, n_calls = ask_llm_with_tool(client, problem)
            except Exception as e:
                tool_raw, tool_ans, n_calls = str(e), None, 0

            d_ok = is_correct(direct_ans, correct)
            t_ok = is_correct(tool_ans,   correct)

            if d_ok: direct_correct_count += 1
            if t_ok: tool_correct_count   += 1
            total_tool_calls += n_calls

            d_sym = "✓" if d_ok else ("?" if direct_ans is None else "✗")
            t_sym = "✓" if t_ok else ("?" if tool_ans   is None else "✗")

            r_pct = round(problem["r"] * 100, 2)
            print(
                f"  [{i:>3}]  {problem['P']:>8.2f}  {r_pct:>5}  {problem['t']:>3}  "
                f"{correct:>12.2f}  {str(direct_ans):>12}  {str(tool_ans):>12}  "
                f"{d_sym}  {t_sym}  {n_calls}"
            )

            writer.writerow([
                i, problem["P"], r_pct, problem["t"], problem["n"],
                correct,
                direct_ans, d_ok,
                tool_ans,   t_ok,
                n_calls, direct_raw, tool_raw,
            ])

            time.sleep(DELAY_BETWEEN_SAMPLES)

    direct_acc = (direct_correct_count / SAMPLES) * 100
    tool_acc   = (tool_correct_count   / SAMPLES) * 100
    avg_calls  = total_tool_calls / SAMPLES

    summary = {
        "direct_correct": direct_correct_count,
        "tool_correct":   tool_correct_count,
        "direct_acc":     direct_acc,
        "tool_acc":       tool_acc,
        "avg_tool_calls": avg_calls,
        "csv_path":       csv_path,
    }

    print(f"\n  Direct accuracy : {direct_correct_count}/{SAMPLES} = {direct_acc:.1f}%")
    print(f"  Tool accuracy   : {tool_correct_count}/{SAMPLES}  = {tool_acc:.1f}%")
    print(f"  Avg tool calls  : {avg_calls:.1f} per sample")
    print(f"  Saved to        : {csv_path}")
    return summary


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    client = AnthropicBedrock()
    summary = run_test(client)

    print(f"\n{'='*75}")
    print("FINAL SUMMARY — Compound Interest (30 samples)")
    print(f"{'='*75}")
    print(f"  {'Approach':<20} {'Correct':>8} {'Accuracy':>10}")
    print(f"  {'-'*40}")
    print(f"  {'Python (ground truth)':<20} {'30/30':>8} {'100.0%':>10}")
    print(f"  {'LLM Direct':<20} {summary['direct_correct']}/30{'':<4} {summary['direct_acc']:>9.1f}%")
    print(f"  {'LLM + Tool':<20} {summary['tool_correct']}/30{'':<4} {summary['tool_acc']:>9.1f}%")
    print(f"\n  Average tool calls per sample: {summary['avg_tool_calls']:.1f}")
    print(f"{'='*75}")
    print(f"\n  Full results: {summary['csv_path']}")
