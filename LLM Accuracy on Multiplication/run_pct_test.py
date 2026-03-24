"""
run_pct_test.py — Percentage Change Chains: Python vs LLM Direct vs LLM + Tool

Tests whether LLMs correctly compute sequential percentage changes.

The core question: when a value changes by multiple percentages in sequence,
does the LLM apply each percentage to the RUNNING TOTAL (correct) or to
the ORIGINAL value (wrong)?

WHY THIS IS INTERESTING:
  Unlike the mortgage experiment (architectural failure — can't loop), this
  problem is STRUCTURALLY solvable with a calculator. The LLM needs 3–6 tool
  calls, one per step. If it understands the problem correctly, it will get
  the right answer.

  The failure here is not architectural. It is conceptual:
    - Correct approach:  value × (1+r1) × (1+r2) × (1+r3) ...
    - Common LLM error:  value × (1 + r1 + r2 + r3 ...)   ← percentages added, not chained

  Every tool call in the wrong approach returns correct arithmetic.
  The tool is not to blame. The LLM fed it the wrong inputs.

EXAMPLE:
  Start $10,000. Changes: +25%, −20%, +10%
  Correct:  $10,000 × 1.25 × 0.80 × 1.10 = $11,000.00
  Wrong:    $10,000 × (1 + 0.25 − 0.20 + 0.10) = $11,500.00  [+$500 error]

Usage:
    python3 run_pct_test.py
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

CLAUDE_MODEL          = "us.anthropic.claude-sonnet-4-20250514-v1:0"
SAMPLES               = 30
RESULTS_DIR           = os.path.join(os.path.dirname(__file__), "results")
DELAY_BETWEEN_SAMPLES = 0.5
TOLERANCE             = 0.01   # within $0.01 — this is pure arithmetic
MAX_TOOL_TURNS        = 15     # at most 6 steps, so ~8 tool calls needed


# ── Calculator Tool ────────────────────────────────────────────────────────────

CALCULATOR_TOOL = [
    {
        "name": "calculate",
        "description": (
            "Performs a single arithmetic operation. "
            "Call this once per step to chain percentage changes. "
            "Operations: add (a+b), subtract (a-b), multiply (a×b), divide (a÷b)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                },
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["operation", "a", "b"],
        },
    }
]


# ── Python Ground Truth ────────────────────────────────────────────────────────

def compute_chain(start: float, changes: list[float]) -> float:
    """
    Correct chain multiplication. Always exact.
    Each change is a decimal (e.g., 0.25 for +25%, -0.15 for -15%).
    """
    value = start
    for r in changes:
        value *= (1 + r)
    return round(value, 2)


# ── Problem Generation ─────────────────────────────────────────────────────────

def generate_problem() -> dict:
    """
    Generate a random percentage chain problem.
    - Starting value: $1,000–$50,000
    - Chain length: 3–6 steps
    - Each change: -30% to +50%
    - At least one positive and one negative to avoid trivial cases
    """
    start = round(random.uniform(1_000, 50_000), 2)
    n_steps = random.randint(3, 6)

    # Ensure mix of positive and negative changes
    changes = []
    for _ in range(n_steps):
        pct = round(random.uniform(-0.30, 0.50), 4)
        changes.append(pct)

    # Ensure at least one negative change
    if all(c >= 0 for c in changes):
        idx = random.randint(0, n_steps - 1)
        changes[idx] = round(random.uniform(-0.30, -0.05), 4)

    correct = compute_chain(start, changes)
    return {"start": start, "changes": changes, "correct": correct}


# ── Prompt Builder ─────────────────────────────────────────────────────────────

def build_prompt(p: dict, use_tool: bool) -> str:
    steps = []
    for i, r in enumerate(p["changes"], 1):
        direction = "increases" if r >= 0 else "decreases"
        pct_str = f"{abs(round(r * 100, 2))}%"
        steps.append(f"  Step {i}: value {direction} by {pct_str}")

    steps_text = "\n".join(steps)

    prompt = (
        f"A portfolio starts at ${p['start']:,.2f}. "
        f"It then goes through the following sequential changes:\n\n"
        f"{steps_text}\n\n"
        f"Each percentage change applies to the CURRENT value at that step "
        f"(not the original starting value).\n\n"
        f"What is the final value after all changes?\n\n"
        f"Respond with only the final dollar amount rounded to 2 decimal places. "
        f"No dollar sign, no explanation — just the number."
    )

    if use_tool:
        prompt += (
            "\n\nUse the calculator tool to work through each step. "
            "Multiply the current value by (1 + rate) for each step in sequence."
        )

    return prompt


# ── Tool Execution ─────────────────────────────────────────────────────────────

def execute_tool(op: str, a: float, b: float) -> float:
    if op == "add":      return a + b
    if op == "subtract": return a - b
    if op == "multiply": return a * b
    if op == "divide":   return a / b if b != 0 else float("inf")
    raise ValueError(f"Unknown operation: {op}")


# ── Response Parsing ───────────────────────────────────────────────────────────

def parse_amount(text: str) -> float | None:
    cleaned = text.replace(",", "").replace("$", "").strip()
    match = re.search(r"\d+(\.\d+)?", cleaned)
    return round(float(match.group()), 2) if match else None


def is_correct(llm: float | None, correct: float) -> bool:
    return llm is not None and abs(llm - correct) <= TOLERANCE


# ── Failure Classification ─────────────────────────────────────────────────────

def classify_error(llm: float | None, p: dict) -> str:
    """Identify which error strategy the LLM used."""
    if llm is None:
        return "no_answer"

    start = p["start"]
    changes = p["changes"]

    # Strategy: sum all percentages, apply once
    additive = round(start * (1 + sum(changes)), 2)
    if abs(llm - additive) <= 1.0:
        return "additive_pct"

    # Strategy: apply each % to original (not running total)
    orig_base = round(start + sum(c * start for c in changes), 2)
    if abs(llm - orig_base) <= 1.0:
        return "original_base"

    # Strategy: average percentages
    avg_rate = sum(changes) / len(changes)
    averaged = round(start * (1 + avg_rate), 2)
    if abs(llm - averaged) <= 1.0:
        return "averaged_pct"

    return "other"


# ── LLM Direct ────────────────────────────────────────────────────────────────

def ask_direct(client, problem: dict):
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=64,
        messages=[{"role": "user", "content": build_prompt(problem, use_tool=False)}],
    )
    raw = response.content[0].text.strip()
    return raw, parse_amount(raw)


# ── LLM + Tool (Agentic Loop) ─────────────────────────────────────────────────

def ask_with_tool(client, problem: dict):
    messages = [{"role": "user", "content": build_prompt(problem, use_tool=True)}]
    tool_calls = 0

    for _ in range(MAX_TOOL_TURNS):
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            tools=CALCULATOR_TOOL,
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_calls += 1
                    try:
                        result = execute_tool(
                            block.input["operation"],
                            float(block.input["a"]),
                            float(block.input["b"]),
                        )
                    except Exception as e:
                        result = f"error: {e}"
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    })
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user",      "content": results})
        else:
            raw = next((b.text for b in response.content if hasattr(b, "text")), "").strip()
            return raw, parse_amount(raw), tool_calls

    return "max_turns_exceeded", None, tool_calls


# ── Main Test ──────────────────────────────────────────────────────────────────

def run_test(client):
    print(f"\n{'='*85}")
    print("TEST: Sequential Percentage Change Chains  |  30 scenarios  |  3–6 steps each")
    print(f"Tolerance: ±${TOLERANCE:.2f}  |  Each % applies to CURRENT value (chain multiply)")
    print(f"{'='*85}")
    print(f"  {'#':>3}  {'start':>10}  {'steps':>5}  {'correct':>12}  {'direct':>12}  {'tool':>12}  D  T  calls  d_error")
    print(f"  {'-'*83}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "results_pct.csv")

    direct_correct = tool_correct = 0
    total_calls = 0
    rows = []

    error_counts = {"additive_pct": 0, "original_base": 0, "averaged_pct": 0, "other": 0, "no_answer": 0}

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample", "start", "n_steps", "changes_pct",
            "correct",
            "llm_direct", "direct_correct",
            "llm_tool",   "tool_correct",
            "tool_calls", "direct_error_type",
            "direct_raw", "tool_raw",
        ])

        for i in range(1, SAMPLES + 1):
            p = generate_problem()
            correct = p["correct"]
            n_steps = len(p["changes"])
            changes_pct = [round(c * 100, 2) for c in p["changes"]]

            try:
                d_raw, d_ans = ask_direct(client, p)
            except Exception as e:
                d_raw, d_ans = str(e), None

            try:
                t_raw, t_ans, n_calls = ask_with_tool(client, p)
            except Exception as e:
                t_raw, t_ans, n_calls = str(e), None, 0

            d_ok = is_correct(d_ans, correct)
            t_ok = is_correct(t_ans, correct)

            if d_ok: direct_correct += 1
            if t_ok: tool_correct   += 1
            total_calls += n_calls

            err_type = classify_error(d_ans, p) if not d_ok else "correct"
            if err_type != "correct":
                error_counts[err_type] = error_counts.get(err_type, 0) + 1

            d_sym = "✓" if d_ok else ("?" if d_ans is None else "✗")
            t_sym = "✓" if t_ok else ("?" if t_ans is None else "✗")

            print(
                f"  [{i:>3}]  ${p['start']:>9,.2f}  {n_steps:>5}  ${correct:>11,.2f}  "
                f"{str(d_ans):>12}  {str(t_ans):>12}  {d_sym}  {t_sym}  {n_calls:>5}  {err_type}"
            )

            writer.writerow([
                i, p["start"], n_steps, str(changes_pct),
                correct,
                d_ans, d_ok,
                t_ans, t_ok,
                n_calls, err_type,
                d_raw, t_raw,
            ])
            rows.append({"p": p, "correct": correct, "direct": d_ans, "tool": t_ans,
                         "d_ok": d_ok, "t_ok": t_ok, "err_type": err_type})

            time.sleep(DELAY_BETWEEN_SAMPLES)

    direct_acc = direct_correct / SAMPLES * 100
    tool_acc   = tool_correct   / SAMPLES * 100
    avg_calls  = total_calls / SAMPLES

    print(f"\n{'='*85}")
    print("FINAL SUMMARY — Sequential Percentage Change Chains (30 samples)")
    print(f"{'='*85}")
    print(f"  {'Metric':<35} {'Direct':>10} {'LLM+Tool':>10} {'Python':>10}")
    print(f"  {'-'*65}")
    print(f"  {'Final value accuracy (±$0.01)':<35} {direct_acc:>9.1f}% {tool_acc:>9.1f}% {'100.0%':>10}")
    print(f"\n  Avg tool calls per sample: {avg_calls:.1f}")

    # Error breakdown for LLM Direct failures
    total_d_failures = SAMPLES - direct_correct
    if total_d_failures > 0:
        print(f"\n  LLM Direct failure breakdown ({total_d_failures} failures):")
        for etype, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                label = {
                    "additive_pct":  "Summed percentages, applied once",
                    "original_base": "Applied each % to original value",
                    "averaged_pct":  "Averaged percentages, applied once",
                    "no_answer":     "No parseable answer",
                    "other":         "Other / unclassified error",
                }.get(etype, etype)
                print(f"    {count:>3}  {label}")

    print(f"\n  Full results: {csv_path}")
    print(f"{'='*85}")


if __name__ == "__main__":
    client = AnthropicBedrock()
    run_test(client)
