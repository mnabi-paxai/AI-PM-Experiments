"""
run_mortgage_test.py — Mortgage with Extra Payments: Python vs LLM Direct vs LLM + Tool

Tests whether LLMs can correctly compute the effect of extra monthly mortgage payments.

Standard mortgage payment has a closed-form formula.
Mortgage WITH extra payments does NOT — it requires month-by-month simulation.

For each of 30 random mortgage scenarios the script computes:
  Q1: How many months until payoff (with extra payment)?
  Q2: How much total interest is saved vs. making no extra payment?

WHY THIS BREAKS TOOL USE:
  A calculator tool performs one operation at a time. Computing a mortgage with
  extra payments requires simulating up to 360 monthly iterations where each
  month's interest depends on the prior month's remaining balance.

  The LLM cannot loop. It will resort to one of three strategies — all wrong:
    Strategy A: Use the standard formula (ignores extra payments entirely)
    Strategy B: Simulate a few months manually, then extrapolate
    Strategy C: Apply an approximate rule-of-thumb

  Unlike the tax experiment (where tool calls were correct but inputs were wrong),
  here the failure is ARCHITECTURAL — the problem structure requires iteration
  that a single-operation tool fundamentally cannot provide.

Usage:
    python3 run_mortgage_test.py
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

CLAUDE_MODEL            = "us.anthropic.claude-sonnet-4-20250514-v1:0"
SAMPLES                 = 30
RESULTS_DIR             = os.path.join(os.path.dirname(__file__), "results")
DELAY_BETWEEN_SAMPLES   = 0.5
MONTHS_TOLERANCE        = 2      # within 2 months is correct
INTEREST_TOLERANCE_PCT  = 0.02   # within 2% of correct interest saved is correct
MAX_TOOL_TURNS          = 25     # allow LLM to attempt many steps


# ── Calculator Tool ───────────────────────────────────────────────────────────

CALCULATOR_TOOL = [
    {
        "name": "calculate",
        "description": (
            "Performs a single arithmetic operation. "
            "Use this to compute mortgage payments, interest charges, and balance updates. "
            "Operations: add (a+b), subtract (a-b), multiply (a×b), divide (a÷b), power (a^b)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide", "power"],
                },
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["operation", "a", "b"],
        },
    }
]


# ── Python Ground Truth ───────────────────────────────────────────────────────

def simulate_mortgage(principal: float, annual_rate: float,
                      term_months: int, extra_payment: float) -> dict:
    """
    Exact month-by-month simulation. Always correct.

    Returns months_to_payoff, interest_saved, and supporting values.
    """
    monthly_rate = annual_rate / 12

    # Standard monthly payment (no extra)
    M = principal * (monthly_rate * (1 + monthly_rate) ** term_months) / \
        ((1 + monthly_rate) ** term_months - 1)

    # Simulate WITHOUT extra payment (full term)
    balance = principal
    total_interest_standard = 0.0
    for _ in range(term_months):
        interest = balance * monthly_rate
        total_interest_standard += interest
        balance += interest - M
    total_interest_standard = round(total_interest_standard, 2)

    # Simulate WITH extra payment (month by month until paid off)
    balance = principal
    total_interest_extra = 0.0
    months = 0
    while balance > 0.01 and months < term_months * 2:
        interest = balance * monthly_rate
        total_interest_extra += interest
        payment = min(M + extra_payment, balance + interest)
        balance = balance + interest - payment
        months += 1

    total_interest_extra = round(total_interest_extra, 2)
    interest_saved       = round(total_interest_standard - total_interest_extra, 2)
    months_saved         = term_months - months

    return {
        "monthly_payment":           round(M, 2),
        "months_to_payoff":          months,
        "months_saved":              months_saved,
        "total_interest_standard":   total_interest_standard,
        "total_interest_with_extra": total_interest_extra,
        "interest_saved":            interest_saved,
    }


# ── Problem Generation ────────────────────────────────────────────────────────

def generate_scenario() -> dict:
    principal     = round(random.uniform(150_000, 750_000), -3)   # round to nearest $1k
    annual_rate   = round(random.uniform(0.035, 0.08), 4)         # 3.5%–8%
    term_months   = random.choice([180, 360])                     # 15 or 30 years
    extra_payment = random.choice([100, 150, 200, 250, 300, 400, 500])

    result = simulate_mortgage(principal, annual_rate, term_months, extra_payment)
    return {
        "principal":     principal,
        "annual_rate":   annual_rate,
        "term_months":   term_months,
        "extra_payment": extra_payment,
        **result,
    }


# ── Prompt Builder ────────────────────────────────────────────────────────────

def build_prompt(s: dict, use_tool: bool) -> str:
    rate_pct  = round(s["annual_rate"] * 100, 3)
    term_yrs  = s["term_months"] // 12

    prompt = (
        f"I have a ${s['principal']:,.0f} mortgage at {rate_pct}% annual interest rate "
        f"({s['term_months']} months / {term_yrs} years). "
        f"I make the standard monthly payment PLUS an extra ${s['extra_payment']} every month.\n\n"
        f"Answer both questions with only numbers, one per line:\n"
        f"Line 1: How many months until the mortgage is fully paid off?\n"
        f"Line 2: How much total interest is saved compared to making no extra payment?\n\n"
        f"No labels, no dollar signs, no explanation — just two numbers."
    )
    if use_tool:
        prompt += (
            "\n\nUse the calculator tool to work through this step by step. "
            "Note: the standard monthly payment formula is M = P×[r(1+r)^n]/[(1+r)^n-1] "
            "where r = annual_rate/12 and n = term_months."
        )
    return prompt


# ── Response Parsing ──────────────────────────────────────────────────────────

def parse_two_numbers(text: str) -> tuple[int | None, float | None]:
    """Extract months (integer) and interest saved (float) from LLM response."""
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", text.replace("$", ""))
    cleaned = [float(n.replace(",", "")) for n in numbers]
    if len(cleaned) >= 2:
        return int(cleaned[0]), round(cleaned[1], 2)
    if len(cleaned) == 1:
        return int(cleaned[0]), None
    return None, None


def is_months_correct(llm: int | None, correct: int) -> bool:
    return llm is not None and abs(llm - correct) <= MONTHS_TOLERANCE


def is_interest_correct(llm: float | None, correct: float) -> bool:
    if llm is None or correct == 0:
        return False
    return abs(llm - correct) / correct <= INTEREST_TOLERANCE_PCT


# ── Tool Execution ────────────────────────────────────────────────────────────

def execute_tool(op: str, a: float, b: float) -> float:
    if op == "add":      return a + b
    if op == "subtract": return a - b
    if op == "multiply": return a * b
    if op == "divide":   return a / b if b != 0 else float("inf")
    if op == "power":    return a ** b
    raise ValueError(f"Unknown op: {op}")


# ── LLM Direct ────────────────────────────────────────────────────────────────

def ask_direct(client, scenario):
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=64,
        messages=[{"role": "user", "content": build_prompt(scenario, use_tool=False)}],
    )
    raw = response.content[0].text.strip()
    return raw, parse_two_numbers(raw)


# ── LLM + Tool (Agentic Loop) ─────────────────────────────────────────────────

def ask_with_tool(client, scenario):
    messages = [{"role": "user", "content": build_prompt(scenario, use_tool=True)}]
    tool_calls = 0

    for _ in range(MAX_TOOL_TURNS):
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
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
            return raw, parse_two_numbers(raw), tool_calls

    return "max_turns", (None, None), tool_calls


# ── Main Test ─────────────────────────────────────────────────────────────────

def run_test(client):
    print(f"\n{'='*90}")
    print("TEST: Mortgage with Extra Payments  |  30 scenarios  |  15 & 30-year terms")
    print(f"Tolerances: months ±{MONTHS_TOLERANCE}  |  interest ±{INTEREST_TOLERANCE_PCT*100:.0f}%")
    print(f"{'='*90}")
    hdr = f"  {'#':>3}  {'principal':>10}  {'rate':>5}  {'extra':>5}  {'mo_correct':>10}  {'mo_direct':>9}  {'mo_tool':>7}  {'int_saved':>11}  {'d_int':>11}  {'t_int':>11}  Dm Tm Di Ti calls"
    print(hdr)
    print(f"  {'-'*len(hdr)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "results_mortgage.csv")

    dm_correct = dt_correct = di_correct = ti_correct = 0
    total_calls = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample", "principal", "annual_rate_pct", "term_months", "extra_payment",
            "correct_months", "correct_interest_saved",
            "direct_months", "direct_months_correct",
            "direct_interest", "direct_interest_correct",
            "tool_months", "tool_months_correct",
            "tool_interest", "tool_interest_correct",
            "tool_calls", "direct_raw", "tool_raw",
        ])

        for i in range(1, SAMPLES + 1):
            s = generate_scenario()
            c_mo  = s["months_to_payoff"]
            c_int = s["interest_saved"]

            try:
                d_raw, (d_mo, d_int) = ask_direct(client, s)
            except Exception as e:
                d_raw, d_mo, d_int = str(e), None, None

            try:
                t_raw, (t_mo, t_int), n_calls = ask_with_tool(client, s)
            except Exception as e:
                t_raw, t_mo, t_int, n_calls = str(e), None, None, 0

            dm_ok = is_months_correct(d_mo, c_mo)
            dt_ok = is_months_correct(t_mo, c_mo)
            di_ok = is_interest_correct(d_int, c_int)
            ti_ok = is_interest_correct(t_int, c_int)

            if dm_ok: dm_correct += 1
            if dt_ok: dt_correct += 1
            if di_ok: di_correct += 1
            if ti_ok: ti_correct += 1
            total_calls += n_calls

            rate_pct = round(s["annual_rate"] * 100, 2)
            print(
                f"  [{i:>3}]  ${s['principal']:>9,.0f}  {rate_pct:>4}%  "
                f"${s['extra_payment']:>4}  "
                f"{c_mo:>10}  {str(d_mo):>9}  {str(t_mo):>7}  "
                f"${c_int:>10,.0f}  ${str(d_int):>10}  ${str(t_int):>10}  "
                f"{'✓' if dm_ok else '✗'}  {'✓' if dt_ok else '✗'}  "
                f"{'✓' if di_ok else '✗'}  {'✓' if ti_ok else '✗'}  {n_calls}"
            )

            writer.writerow([
                i, s["principal"], rate_pct, s["term_months"], s["extra_payment"],
                c_mo, c_int,
                d_mo, dm_ok, d_int, di_ok,
                t_mo, dt_ok, t_int, ti_ok,
                n_calls, d_raw, t_raw,
            ])

            time.sleep(DELAY_BETWEEN_SAMPLES)

    avg_calls = total_calls / SAMPLES
    print(f"\n{'='*90}")
    print("FINAL SUMMARY — Mortgage with Extra Payments (30 samples)")
    print(f"{'='*90}")
    print(f"  {'Metric':<35} {'Direct':>10} {'LLM+Tool':>10} {'Python':>10}")
    print(f"  {'-'*65}")
    print(f"  {'Months to payoff accuracy':<35} {dm_correct/SAMPLES*100:>9.1f}% {dt_correct/SAMPLES*100:>9.1f}% {'100.0%':>10}")
    print(f"  {'Interest saved accuracy (±2%)':<35} {di_correct/SAMPLES*100:>9.1f}% {ti_correct/SAMPLES*100:>9.1f}% {'100.0%':>10}")
    print(f"\n  Avg tool calls per sample: {avg_calls:.1f}")
    print(f"  Full results: {csv_path}")
    print(f"{'='*90}")


if __name__ == "__main__":
    client = AnthropicBedrock()
    run_test(client)
