# Experiment: Mortgage with Extra Payments — LLM vs LLM + Tool

A numerical accuracy experiment testing whether tool use fixes LLM failures on
iterative amortization calculations. The core question: **can an LLM correctly
compute a mortgage payoff timeline when given a calculator tool?**

---

## Why This Experiment

The tax bracket experiment showed tool use achieving 93.3% accuracy — because
the problem had conditional but finite logic (3–14 fixed steps per income level).

Mortgage amortization with extra payments breaks that assumption entirely.
There is **no closed-form formula** for this problem. The only correct approach
is month-by-month simulation: up to 360 iterations where each month's interest
depends on the previous month's remaining balance.

This isolates a specific and fundamental failure mode:
> **The tool can compute. The LLM cannot loop. The problem requires a loop.**

---

## The Problem

For a given mortgage scenario, compute two values:

1. **Months to payoff** — how many months until the balance reaches zero when
   making the standard payment PLUS an extra fixed amount each month
2. **Interest saved** — total interest paid with extra payments vs. without

### Why This Cannot Be Solved With a One-Operation Tool

A calculator tool performs one arithmetic operation per call. Solving this
problem correctly requires simulating the amortization schedule month by month:

```
For each month until balance = 0:
    interest  = balance × monthly_rate
    payment   = min(standard_payment + extra, balance + interest)
    balance   = balance + interest - payment
```

For a 30-year mortgage at $400k, this loop runs roughly 260 times (with extra
payments) to 360 times (without). The LLM cannot issue 260+ sequential dependent
tool calls — each call depends on the result of the previous one, and there is
no mechanism for the LLM to loop.

### What the LLM Does Instead

Without the ability to iterate, the LLM falls back to one of three approximations:

| Strategy | Description | Why it fails |
|---|---|---|
| **A — Standard formula** | Uses the closed-form formula for no-extra-payment scenario | Ignores extra payments entirely |
| **B — Manual simulation + extrapolation** | Simulates 3–5 months, then extrapolates linearly | Non-linear amortization makes extrapolation inaccurate |
| **C — Rule of thumb** | Applies an approximate "each $100 extra saves X months" heuristic | Highly sensitive to rate and principal; wildly off |

Unlike the tax experiment (where tool calls were correct but inputs were wrong),
here the failure is **architectural** — the problem structure requires iteration
that a single-operation tool fundamentally cannot provide.

---

## Three Approaches Tested

### 1. Python (Ground Truth)
Exact month-by-month simulation. Always correct. Computes both the standard
amortization (full term, no extra) and the accelerated amortization (with extra),
then subtracts to get interest saved.

### 2. LLM Direct
Claude is given the principal, rate, term, and extra payment and asked to
return two numbers: months to payoff and interest saved. No tools.

**Expected failure:** Claude applies the standard formula, ignores the iterative
nature of the extra payment, and produces answers that are roughly plausible
but consistently wrong.

### 3. LLM + Tool (Agentic Loop)
Claude is given a single-operation calculator (add, subtract, multiply, divide,
power) and up to 25 turns to work through the problem.

**Expected failure:** Even 25 tool calls cannot replicate 260+ iterations of
a dependent loop. Claude either:
- Hits the turn limit and returns nothing (timeout)
- Simulates a few steps and extrapolates (wrong)
- Falls back to an approximation (wrong)

---

## Experiment Design

| Parameter | Value |
|---|---|
| Samples | 30 |
| Principal range | $150,000 – $750,000 |
| Annual rate range | 3.5% – 8.0% |
| Loan terms | 15 years (180 months) or 30 years (360 months) |
| Extra payment | $100, $150, $200, $250, $300, $400, or $500/month |
| Months tolerance | ±2 months |
| Interest tolerance | ±2% of correct answer |
| Model | Claude Sonnet 4 (AWS Bedrock) |
| Max tool turns | 25 per sample |

---

## Results

| Approach | Months Accuracy | Interest Saved Accuracy |
|---|---|---|
| Python (ground truth) | 100.0% | 100.0% |
| LLM Direct | **6.7%** (2/30) | **0.0%** (0/30) |
| LLM + Tool | **0.0%** (0/30) | **0.0%** (0/30) |

**Average tool calls per sample: 14.1**

---

## Failure Analysis

### LLM Direct — 6.7% months, 0% interest

Two samples (#6, #10) happened to land within 2 months on the months question.
These were not correct by reasoning — the LLM applied an approximation that
accidentally fell inside the tolerance window. Interest saved was 0/30.

**Characteristic error patterns:**

- **Months error:** LLM returns a value 10–40 months away from correct. It
  applies the standard formula or a rough rule-of-thumb, producing a number
  in the right ballpark but not within ±2.
- **Interest saved error:** LLM answers are often 2–5× off from correct.
  Common error: it estimates total interest on the original loan and subtracts
  a flat approximation for the extra payment's effect, rather than simulating
  the compounding difference.

### LLM + Tool — 0% on both metrics

Tool use made accuracy *worse* than going direct on the months metric (0% vs 6.7%).

**What happened with tool calls (14.1 avg):**

The LLM used its 25 turns to:
1. Compute the monthly payment (correct — this has a formula)
2. Simulate 3–8 months of amortization manually (each month: one multiply for
   interest, one subtract for payment, one add/subtract for new balance)
3. Run out of turns, or extrapolate the remaining balance using an approximation

The interest saved answers from the tool are wildly wrong — often off by
$50,000–$300,000. The tool computed real arithmetic on each call, but the
extrapolation step produces catastrophically wrong interest totals.

**Timeout cases (samples 2, 4, 5, 8, 11, 18, 21):**
Seven samples hit MAX_TOOL_TURNS=25 without a final answer, returning None.
These were counted as incorrect.

**The compounding damage:** Each month simulated requires 3–4 tool calls
(interest calc, payment calc, balance update). Simulating 8 months takes ~32
calls — already exceeding the turn limit for a 360-step problem. The LLM
cannot get past the first few months.

---

## The Core Insight

```
Compound Interest (5 fixed steps):      Tool accuracy = 100%
Tax Brackets (3–14 variable steps):     Tool accuracy = 93.3%
Mortgage Amortization (260–360 loops):  Tool accuracy =   0.0%
```

The failure is not about intelligence. The failure is about **architecture**.

| Problem type | Tool use fixes it? | Why |
|---|---|---|
| Fixed formula (compound interest) | Yes | Same steps every time |
| Conditional/variable (tax brackets) | Mostly | More steps = more orchestration risk |
| Iterative/looping (mortgage) | No | Tool can't loop; problem requires 360 dependent steps |
| Ambiguous logic (% change on wrong base) | No | Error is in understanding, not execution |

The calculator tool is perfectly capable of computing `balance * monthly_rate`.
The problem is that you need to call it 360 times in sequence, where each input
depends on the previous output. That is a program, not a prompt.

---

## What This Means for AI Product Design

**1. Distinguish "compute" problems from "simulate" problems.**
Compound interest is a compute problem: plug into a formula, get an answer.
Mortgage amortization is a simulate problem: run a loop until a condition is met.
A tool that performs single operations can fix the former but not the latter.

**2. Tool use does not solve architectural mismatches.**
Before adding a tool, ask: does this problem require iteration with state that
depends on previous results? If yes, the tool won't help. You need code execution
(a Python REPL, a function call, a server-side calculator) — not a single-op tool.

**3. Giving the LLM more turns does not fix a loop.**
25 turns is more than most agentic tasks need. It still wasn't enough to simulate
a mortgage. The issue is not token budget — it is that dependent iteration cannot
be unrolled into a fixed number of tool calls.

**4. The right tool for iterative simulation is code execution.**
```
Wrong:  LLM + single-op calculator  →  fails at step 10 of 360
Right:  LLM + Python REPL           →  writes the loop, executes it, gets correct answer
```
The LLM's job in this architecture shifts from "orchestrate arithmetic" to
"write correct code." That is a very different capability — and one where
modern LLMs perform much better.

**5. When accuracy is required, verify against Python.**
The pattern from the tax experiment applies here even more strongly:
```
LLM + Tool  →  output  →  Python verifier  →  if mismatch: flag for review
```
For any numerical task that could be run in Python deterministically, run it
in Python and use the LLM for interpretation, not computation.

---

## Experiment Progression

| Experiment | Task | LLM Direct | LLM + Tool | Key Finding |
|---|---|---|---|---|
| `run_test.py` | 2-digit multiplication | ~85–95% | — | LLM has arithmetic limits |
| `run_test.py` | 5-digit multiplication | ~2–10% | — | Accuracy collapses with complexity |
| `run_test_v2.py` | 2-digit multiplication | ~85–95% | ~100% | Tool use fixes simple arithmetic |
| `run_test_v2.py` | 5-digit multiplication | ~2–10% | ~100% | Tool use fixes complex arithmetic |
| `run_finance_test.py` | Compound interest | 0% | 100% | Tool use fixes fixed-formula math |
| `run_tax_test.py` | Tax brackets | 0% | 93.3% | Tool use fails on variable orchestration |
| **`run_mortgage_test.py`** | **Mortgage amortization** | **6.7% / 0%** | **0% / 0%** | **Tool use fails on iterative problems** |

---

## Files

```
LLM Accuracy on Multiplication/
├── run_mortgage_test.py              ← this experiment
├── results/results_mortgage.csv     ← full 30-sample results
└── readme_mortgage.md               ← this file
```

## How to Run

```bash
source venv/bin/activate
python3 run_mortgage_test.py
```
