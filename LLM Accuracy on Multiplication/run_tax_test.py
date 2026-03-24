"""
run_tax_test.py — Tax Bracket Calculation Accuracy: Python vs LLM Direct vs LLM + Tool

Tests how accurately an LLM computes US federal income tax using tiered brackets
across 30 random income levels.

2024 US Federal Tax Brackets (Single Filer):
  10%  on $0           – $11,600
  12%  on $11,601      – $47,150
  22%  on $47,151      – $100,525
  24%  on $100,526     – $191,950
  32%  on $191,951     – $243,725
  35%  on $243,726     – $609,350
  37%  on above $609,350

WHY TOOL USE CAN FAIL HERE:
  Unlike compound interest (one clean formula), tax requires the LLM to:
    1. Understand the tiered structure — each bracket applies only to the SLICE
       of income within that range, not the full income
    2. Identify which brackets the income crosses
    3. Call the tool once per bracket (variable number of calls)
    4. Sum all the bracket taxes at the end

  Common failure mode: LLM applies the highest applicable rate as a FLAT rate
  on the entire income — even when using tools. Every tool call returns a correct
  number, but the LLM computed the wrong thing.

  Example: Income = $60,000
    Correct (tiered):  $1,160 + $4,266 + $2,805.50 = $8,231.50
    Wrong (flat 22%):  $60,000 × 0.22 = $13,200  ← all tool calls correct, answer wrong

Usage:
    python3 run_tax_test.py
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
TOLERANCE             = 1.00   # within $1.00 is considered correct
MAX_TOOL_TURNS        = 15     # tax can require up to 7 bracket calls + summing


# ── 2024 Tax Brackets ─────────────────────────────────────────────────────────

BRACKETS = [
    (11_600,    0.10),
    (47_150,    0.12),
    (100_525,   0.22),
    (191_950,   0.24),
    (243_725,   0.32),
    (609_350,   0.35),
    (float("inf"), 0.37),
]

BRACKET_TEXT = """2024 US Federal Income Tax Brackets (Single Filer):
  - 10%  on the first $11,600 of income
  - 12%  on income from $11,601  to $47,150
  - 22%  on income from $47,151  to $100,525
  - 24%  on income from $100,526 to $191,950
  - 32%  on income from $191,951 to $243,725
  - 35%  on income from $243,726 to $609,350
  - 37%  on income above $609,350

IMPORTANT: Each rate applies ONLY to the portion of income within that bracket,
not to the full income. Calculate the tax for each bracket slice separately,
then sum them."""


# ── Calculator Tool ───────────────────────────────────────────────────────────

CALCULATOR_TOOL = [
    {
        "name": "calculate",
        "description": (
            "Performs a single arithmetic operation. "
            "Call this once per bracket to compute each slice of tax, "
            "then call it again to sum the results."
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


# ── Python Ground Truth ───────────────────────────────────────────────────────

def calculate_tax(income: float) -> float:
    """Exact tiered tax calculation. Always correct."""
    tax = 0.0
    prev_limit = 0
    for limit, rate in BRACKETS:
        if income <= prev_limit:
            break
        slice_top = min(income, limit)
        taxable_slice = slice_top - prev_limit
        tax += taxable_slice * rate
        prev_limit = limit
    return round(tax, 2)


# ── Problem Generation ────────────────────────────────────────────────────────

def generate_income() -> float:
    """
    Generate a random income that spans multiple tax brackets.
    Spread across ranges to test different bracket combinations.
    """
    ranges = [
        (20_000,  50_000),   # crosses brackets 1-2
        (50_000,  110_000),  # crosses brackets 1-3
        (110_000, 200_000),  # crosses brackets 1-4
        (200_000, 260_000),  # crosses brackets 1-5
    ]
    low, high = random.choice(ranges)
    return round(random.uniform(low, high), 2)


# ── Prompt Builder ────────────────────────────────────────────────────────────

def build_prompt(income: float, use_tool: bool) -> str:
    base = (
        f"{BRACKET_TEXT}\n\n"
        f"Calculate the total federal income tax owed on an annual income of "
        f"${income:,.2f}.\n"
        f"Respond with only the final dollar amount rounded to 2 decimal places. "
        f"No dollar sign, no explanation — just the number."
    )
    if use_tool:
        base += "\nUse the calculator tool to compute each bracket's tax separately, then sum them."
    return base


# ── Tool Execution ────────────────────────────────────────────────────────────

def execute_tool(op: str, a: float, b: float) -> float:
    if op == "add":      return a + b
    if op == "subtract": return a - b
    if op == "multiply": return a * b
    if op == "divide":   return a / b if b != 0 else float("inf")
    raise ValueError(f"Unknown operation: {op}")


# ── Response Parsing ──────────────────────────────────────────────────────────

def parse_amount(text: str) -> float | None:
    cleaned = text.replace(",", "").replace("$", "").strip()
    match = re.search(r"\d+(\.\d+)?", cleaned)
    return round(float(match.group()), 2) if match else None


def is_correct(llm_ans: float | None, correct: float) -> bool:
    return llm_ans is not None and abs(llm_ans - correct) <= TOLERANCE


# ── Approach 1: LLM Direct ────────────────────────────────────────────────────

def ask_direct(client: AnthropicBedrock, income: float) -> tuple[str, float | None]:
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=64,
        messages=[{"role": "user", "content": build_prompt(income, use_tool=False)}],
    )
    raw = response.content[0].text.strip()
    return raw, parse_amount(raw)


# ── Approach 2: LLM + Tool (Agentic Loop) ────────────────────────────────────

def ask_with_tool(client: AnthropicBedrock, income: float) -> tuple[str, float | None, int]:
    messages = [{"role": "user", "content": build_prompt(income, use_tool=True)}]
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


# ── Main Test ─────────────────────────────────────────────────────────────────

def run_test(client: AnthropicBedrock):
    print(f"\n{'='*80}")
    print("TEST: US Federal Income Tax (Tiered Brackets)  |  30 random incomes")
    print(f"Tolerance: ±${TOLERANCE:.2f}")
    print(f"{'='*80}")
    print(f"  {'#':>3}  {'income':>12}  {'correct tax':>12}  {'direct':>12}  {'tool':>12}  D  T  calls")
    print(f"  {'-'*76}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "results_tax.csv")

    direct_correct = 0
    tool_correct   = 0
    total_calls    = 0
    rows           = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample", "income", "correct_tax",
            "llm_direct",  "direct_correct",
            "llm_tool",    "tool_correct",
            "tool_calls",  "direct_raw", "tool_raw",
        ])

        for i in range(1, SAMPLES + 1):
            income  = generate_income()
            correct = calculate_tax(income)

            try:
                d_raw, d_ans = ask_direct(client, income)
            except Exception as e:
                d_raw, d_ans = str(e), None

            try:
                t_raw, t_ans, n_calls = ask_with_tool(client, income)
            except Exception as e:
                t_raw, t_ans, n_calls = str(e), None, 0

            d_ok = is_correct(d_ans, correct)
            t_ok = is_correct(t_ans, correct)

            if d_ok: direct_correct += 1
            if t_ok: tool_correct   += 1
            total_calls += n_calls

            d_sym = "✓" if d_ok else ("?" if d_ans is None else "✗")
            t_sym = "✓" if t_ok else ("?" if t_ans is None else "✗")

            print(
                f"  [{i:>3}]  ${income:>11,.2f}  ${correct:>11,.2f}  "
                f"{str(d_ans):>12}  {str(t_ans):>12}  {d_sym}  {t_sym}  {n_calls}"
            )

            writer.writerow([
                i, income, correct,
                d_ans, d_ok,
                t_ans, t_ok,
                n_calls, d_raw, t_raw,
            ])
            rows.append({"income": income, "correct": correct,
                         "direct": d_ans, "tool": t_ans,
                         "d_ok": d_ok, "t_ok": t_ok})

            time.sleep(DELAY_BETWEEN_SAMPLES)

    direct_acc = (direct_correct / SAMPLES) * 100
    tool_acc   = (tool_correct   / SAMPLES) * 100
    avg_calls  = total_calls / SAMPLES

    # ── Failure Analysis ──────────────────────────────────────────────────────
    tool_failures = [r for r in rows if not r["t_ok"] and r["tool"] is not None]

    print(f"\n  Direct accuracy : {direct_correct}/{SAMPLES} = {direct_acc:.1f}%")
    print(f"  Tool accuracy   : {tool_correct}/{SAMPLES}  = {tool_acc:.1f}%")
    print(f"  Avg tool calls  : {avg_calls:.1f} per sample")

    if tool_failures:
        print(f"\n  Tool failure analysis ({len(tool_failures)} cases):")
        for r in tool_failures:
            error    = r["tool"] - r["correct"]
            flat_tax = round(r["income"] * 0.22, 2)
            flat_err = round(abs(flat_tax - r["correct"]), 2)
            print(
                f"    income=${r['income']:>10,.2f}  correct=${r['correct']:>9,.2f}  "
                f"llm=${r['tool']:>9,.2f}  error=${abs(error):>8,.2f}  "
                f"(flat-22% would give error=${flat_err:,.2f})"
            )

    print(f"\n  Saved to: {csv_path}")

    print(f"\n{'='*80}")
    print("FINAL SUMMARY — US Federal Tax Brackets (30 samples)")
    print(f"{'='*80}")
    print(f"  {'Approach':<25} {'Correct':>8} {'Accuracy':>10}")
    print(f"  {'-'*45}")
    print(f"  {'Python (ground truth)':<25} {'30/30':>8} {'100.0%':>10}")
    print(f"  {'LLM Direct':<25} {f'{direct_correct}/30':>8} {direct_acc:>9.1f}%")
    print(f"  {'LLM + Tool':<25} {f'{tool_correct}/30':>8} {tool_acc:>9.1f}%")
    print(f"\n  Avg tool calls per sample : {avg_calls:.1f}")
    print(f"  Full results              : {csv_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    client = AnthropicBedrock()
    run_test(client)
