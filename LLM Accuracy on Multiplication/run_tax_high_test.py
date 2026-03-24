"""
run_tax_high_test.py — Tax Bracket Calculation: High-Income Focus

Same experiment as run_tax_test.py but with income distribution weighted
heavily toward higher brackets where the original test showed failures.

Original distribution (run_tax_test.py): equal weight across 4 ranges up to $260k
  → Only ~7–8 samples crossed 5 brackets; both failures were in that tier.

This distribution: 22 of 30 samples in 5-bracket territory or above
  → Stress-tests whether 93.3% accuracy holds at higher orchestration depth.

Income distribution (30 samples):
  4 samples:  $20k  – $110k   (brackets 1–3)  ← baseline control
  4 samples:  $110k – $191k   (brackets 1–4)
  10 samples: $191k – $243k   (brackets 1–5, 32% marginal)
  8 samples:  $243k – $609k   (brackets 1–6, 35% marginal)
  4 samples:  $609k – $900k   (brackets 1–7, 37% marginal)

Usage:
    python3 run_tax_high_test.py
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
RESULTS_DIR           = os.path.join(os.path.dirname(__file__), "results")
DELAY_BETWEEN_SAMPLES = 0.5
TOLERANCE             = 1.00   # within $1.00 is considered correct
MAX_TOOL_TURNS        = 15


# ── 2024 Tax Brackets ─────────────────────────────────────────────────────────

BRACKETS = [
    (11_600,      0.10),
    (47_150,      0.12),
    (100_525,     0.22),
    (191_950,     0.24),
    (243_725,     0.32),
    (609_350,     0.35),
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


# ── Income distribution ────────────────────────────────────────────────────────

# (range_low, range_high, bracket_label, n_samples)
INCOME_PLAN = [
    (20_000,    110_000,  "brackets 1-3",  4),
    (110_000,   191_950,  "brackets 1-4",  4),
    (191_951,   243_725,  "brackets 1-5", 10),
    (243_726,   609_350,  "brackets 1-6",  8),
    (609_351,   900_000,  "brackets 1-7",  4),
]

def build_income_list() -> list[tuple[float, str]]:
    """Return 30 (income, bracket_label) pairs per INCOME_PLAN."""
    items = []
    for low, high, label, n in INCOME_PLAN:
        for _ in range(n):
            items.append((round(random.uniform(low, high), 2), label))
    random.shuffle(items)
    return items


# ── Calculator Tool ────────────────────────────────────────────────────────────

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


# ── Python Ground Truth ────────────────────────────────────────────────────────

def calculate_tax(income: float) -> float:
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


# ── Prompt Builder ─────────────────────────────────────────────────────────────

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


# ── LLM Direct ────────────────────────────────────────────────────────────────

def ask_direct(client, income: float):
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=64,
        messages=[{"role": "user", "content": build_prompt(income, use_tool=False)}],
    )
    raw = response.content[0].text.strip()
    return raw, parse_amount(raw)


# ── LLM + Tool (Agentic Loop) ─────────────────────────────────────────────────

def ask_with_tool(client, income: float):
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


# ── Main Test ──────────────────────────────────────────────────────────────────

def run_test(client):
    income_list = build_income_list()
    SAMPLES = len(income_list)

    print(f"\n{'='*90}")
    print("TEST: US Federal Tax — High-Income Focus  |  30 samples  |  22 in 5–7 bracket tiers")
    print(f"Tolerance: ±${TOLERANCE:.2f}  |  Distribution: 4 / 4 / 10 / 8 / 4 across bracket tiers")
    print(f"{'='*90}")
    print(f"  {'#':>3}  {'income':>12}  {'tier':<15}  {'correct':>10}  {'direct':>10}  {'tool':>10}  D  T  calls")
    print(f"  {'-'*84}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "results_tax_high.csv")

    direct_correct = tool_correct = total_calls = 0
    rows = []

    # Track accuracy by tier
    tier_stats: dict[str, dict] = {}
    for _, _, label, _ in INCOME_PLAN:
        tier_stats[label] = {"n": 0, "d_ok": 0, "t_ok": 0}

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample", "income", "bracket_tier", "correct_tax",
            "llm_direct", "direct_correct",
            "llm_tool",   "tool_correct",
            "tool_calls", "direct_raw", "tool_raw",
        ])

        for i, (income, tier) in enumerate(income_list, 1):
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

            tier_stats[tier]["n"]    += 1
            tier_stats[tier]["d_ok"] += int(d_ok)
            tier_stats[tier]["t_ok"] += int(t_ok)

            d_sym = "✓" if d_ok else ("?" if d_ans is None else "✗")
            t_sym = "✓" if t_ok else ("?" if t_ans is None else "✗")

            print(
                f"  [{i:>3}]  ${income:>11,.2f}  {tier:<15}  ${correct:>9,.2f}  "
                f"{str(d_ans):>10}  {str(t_ans):>10}  {d_sym}  {t_sym}  {n_calls}"
            )

            writer.writerow([
                i, income, tier, correct,
                d_ans, d_ok, t_ans, t_ok,
                n_calls, d_raw, t_raw,
            ])
            rows.append({"income": income, "tier": tier, "correct": correct,
                         "direct": d_ans, "tool": t_ans, "d_ok": d_ok, "t_ok": t_ok})

            time.sleep(DELAY_BETWEEN_SAMPLES)

    direct_acc = direct_correct / SAMPLES * 100
    tool_acc   = tool_correct   / SAMPLES * 100
    avg_calls  = total_calls / SAMPLES

    print(f"\n{'='*90}")
    print("FINAL SUMMARY — High-Income Tax Distribution (30 samples)")
    print(f"{'='*90}")
    print(f"  {'Metric':<35} {'Direct':>10} {'LLM+Tool':>10} {'Python':>10}")
    print(f"  {'-'*65}")
    print(f"  {'Overall accuracy':<35} {direct_acc:>9.1f}% {tool_acc:>9.1f}% {'100.0%':>10}")

    print(f"\n  Accuracy by bracket tier:")
    print(f"  {'Tier':<20} {'n':>4}  {'Direct':>10}  {'LLM+Tool':>10}")
    print(f"  {'-'*50}")
    for _, _, label, _ in INCOME_PLAN:
        s = tier_stats[label]
        if s["n"] > 0:
            d_pct = s["d_ok"] / s["n"] * 100
            t_pct = s["t_ok"] / s["n"] * 100
            print(f"  {label:<20} {s['n']:>4}  {d_pct:>9.1f}%  {t_pct:>9.1f}%")

    print(f"\n  Avg tool calls per sample: {avg_calls:.1f}")

    # Tool failures
    tool_failures = [r for r in rows if not r["t_ok"] and r["tool"] is not None]
    if tool_failures:
        print(f"\n  Tool failures ({len(tool_failures)}):")
        for r in tool_failures:
            error = abs(r["tool"] - r["correct"])
            print(f"    income=${r['income']:>10,.2f}  tier={r['tier']:<15}  "
                  f"correct=${r['correct']:>9,.2f}  llm=${r['tool']:>9,.2f}  error=${error:,.2f}")

    print(f"\n  Full results: {csv_path}")
    print(f"{'='*90}")

    # Compare with original run
    print(f"\n  Comparison vs. original run_tax_test.py (equal distribution):")
    print(f"  {'':35} {'Original':>10} {'High-Focus':>10}")
    print(f"  {'-'*55}")
    print(f"  {'LLM Direct accuracy':<35} {'0.0%':>10} {direct_acc:>9.1f}%")
    print(f"  {'LLM + Tool accuracy':<35} {'93.3%':>10} {tool_acc:>9.1f}%")


if __name__ == "__main__":
    client = AnthropicBedrock()
    run_test(client)
