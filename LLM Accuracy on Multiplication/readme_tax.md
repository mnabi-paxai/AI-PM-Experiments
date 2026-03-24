# Experiment: Tax Bracket Calculation — LLM vs LLM + Tool

A numerical accuracy experiment testing whether tool use fixes LLM failures on
multi-step conditional calculations. The core question: **can an LLM correctly
apply tiered tax brackets when given a calculator tool?**

---

## Why This Experiment

The compound interest experiment showed tool use achieving 100% accuracy —
because the formula is fixed, deterministic, and always requires the same 5 steps.

Tax brackets break that assumption. The number of steps is **variable** (depends
on how many brackets the income crosses), the LLM must apply **conditional logic**
(which slices apply to this income?), and the boundary values must be applied
with **exact precision**.

This isolates a specific tool use failure mode:
> **The tool executes correctly. The LLM decides incorrectly what to feed the tool.**

---

## The Problem

Compute the total US federal income tax for a given income using 2024 brackets
for a single filer:

| Bracket | Rate | Income Range |
|---|---|---|
| 1 | 10% | $0 – $11,600 |
| 2 | 12% | $11,601 – $47,150 |
| 3 | 22% | $47,151 – $100,525 |
| 4 | 24% | $100,526 – $191,950 |
| 5 | 32% | $191,951 – $243,725 |
| 6 | 35% | $243,726 – $609,350 |
| 7 | 37% | above $609,350 |

**Critical rule:** Each rate applies only to the *slice* of income within that
bracket — not the full income. This is what makes the calculation hard.

**Example — Income $60,000:**
```
Bracket 1:  $11,600            × 10% =  $1,160.00
Bracket 2: ($47,150 - $11,600) × 12% =  $4,266.00
Bracket 3: ($60,000 - $47,150) × 22% =  $2,827.00
                                        ──────────
Total tax:                               $8,253.00

Wrong (flat 22%):  $60,000 × 0.22 = $13,200.00  ← $4,947 overestimate
```

---

## Three Approaches Tested

### 1. Python (Ground Truth)
Exact tiered calculation. Loops through each bracket, computes the taxable slice,
sums them. Always correct.

### 2. LLM Direct
Claude is given the bracket table and asked to compute the tax. No tools.
Must perform the entire tiered calculation in its forward pass.

**Expected failure:** Claude applies a flat effective rate or uses approximate
bracket values — close but never exactly right.

### 3. LLM + Tool (Agentic Loop)
Claude is given a single-operation calculator tool and must:
1. Identify which brackets the income crosses
2. Compute each slice's boundaries
3. Call the tool once per bracket: `multiply(slice_amount, rate)`
4. Call the tool repeatedly to sum the results
5. Return the final total

**Variable tool calls:** 3–4 calls for low incomes (2 brackets), 13–14 calls
for high incomes crossing 5+ brackets.

**Where it can still fail:** If Claude uses the wrong boundary value for a bracket
(e.g., $191,950 instead of $191,951), or miscalculates a slice width, every tool
call executes perfectly — but on the wrong input.

---

## Experiment Design

| Parameter | Value |
|---|---|
| Samples | 30 |
| Income range | $20,000 – $260,000 (spread across bracket tiers) |
| Tolerance | ±$1.00 |
| Model | Claude Sonnet 4 (AWS Bedrock) |
| Max tool turns | 15 per sample |

### Income distribution
Samples were drawn from four ranges to ensure coverage of different bracket depths:
- $20,000–$50,000 → crosses 2 brackets
- $50,000–$110,000 → crosses 3 brackets
- $110,000–$200,000 → crosses 4 brackets
- $200,000–$260,000 → crosses 5 brackets

---

## Results

| Approach | Correct | Accuracy |
|---|---|---|
| Python (ground truth) | 30/30 | 100.0% |
| LLM Direct | 0/30 | **0.0%** |
| LLM + Tool | 28/30 | **93.3%** |

**Average tool calls per sample: 7.3**

---

## Failure Analysis

### LLM Direct — 0% (30 failures)
Every single answer was wrong. The errors fell into two patterns:

**Pattern 1 — Flat rate applied to full income**
The most common error. Claude applies the marginal bracket rate to the entire
income rather than just the slice within that bracket.
```
Income $80,035:  correct = $12,660  |  LLM = $13,467  |  error = +$807
                                       ↑ 22% applied to full income above $47,150
```

**Pattern 2 — Small offset errors**
Occasionally close but off by small amounts, suggesting Claude partially
applied the tiered logic but used rounded or approximate boundary values.
```
Income $34,198:  correct = $3,871.86  |  LLM = $3,863.86  |  error = -$8.00
```

### LLM + Tool — 93.3% (2 failures)

Both failures occurred on **high-income samples crossing 5 brackets**
(where tool call count reached 10–14):

| Income | Correct Tax | LLM + Tool | Error | Tool calls |
|---|---|---|---|---|
| $245,123.74 | $56,168.06 | $56,126.10 | $41.96 | 14 |
| $221,147.97 | $48,453.85 | $46,118.01 | $2,335.84 | 10 |

**What went wrong:**
- Sample 19 ($245k): Minor boundary precision error at the 32% bracket cutoff ($243,725).
  The error is small ($41.96), suggesting a rounding issue in the slice calculation.
- Sample 29 ($221k): Larger error ($2,335), likely from applying the wrong rate
  to the 32% bracket slice — the tool calls were correct, but Claude fed the wrong
  slice width into one of them.

**The pattern:** Failures concentrate in higher brackets. The more brackets crossed,
the more orchestration steps, and the higher the probability of one step going wrong.

---

## Follow-Up: High-Income Distribution Test (`run_tax_high_test.py`)

The original 93.3% result used equal sampling across incomes up to $260k —
meaning only ~7 samples crossed 5 brackets, and none crossed 6 or 7.

A follow-up test weighted 22 of 30 samples toward higher brackets:

| Tier | n | LLM+Tool accuracy |
|---|---|---|
| Brackets 1–3 ($20k–$110k) | 4 | **100%** |
| Brackets 1–4 ($110k–$191k) | 4 | **100%** |
| Brackets 1–5 ($191k–$243k) | 10 | **90%** |
| Brackets 1–6 ($243k–$609k) | 8 | **12.5%** |
| Brackets 1–7 ($609k+) | 4 | **0%** |
| **Overall** | **30** | **60.0%** |

**The 93.3% figure was an artifact of the income distribution.** The tool hits
a hard ceiling as bracket depth increases because 6-bracket calculations require
~14 tool calls and 7-bracket calculations require ~16+. With MAX_TOOL_TURNS=15,
most high-income samples exhaust the turn limit before completing.

This reveals a **second failure mode** beyond wrong inputs:

| Failure mode | What happens | Income range |
|---|---|---|
| Wrong slice boundary | Tool call succeeds, LLM fed wrong input | $191k–$260k |
| Turn limit exhaustion | LLM uses correct algorithm, runs out of turns | $243k+ |

The tool-use accuracy by bracket depth:
```
2–3 brackets:   100%   (3–7 tool calls)
4 brackets:     100%   (10–11 tool calls)
5 brackets:      90%   (13–14 tool calls)
6 brackets:    12.5%   (14–15 calls needed, limit hit)
7 brackets:       0%   (16+ calls needed, limit always hit)
```

---

## The Core Insight

```
Compound Interest (5 fixed steps):    Tool accuracy = 100%
Tax Brackets — balanced dist.:        Tool accuracy = 93.3%
Tax Brackets — high-income dist.:     Tool accuracy = 60.0%
```

The difference is not the tool. The tool executed every calculation perfectly.
The difference is **orchestration complexity**:

- Fixed formula → fixed steps → LLM always knows exactly what to call
- Conditional/variable formula → LLM must decide steps → errors compound
- More brackets → more steps → turn limit becomes the binding constraint

This reveals the boundary of what tool use can fix:

| Problem type | Tool use fixes it? | Why |
|---|---|---|
| Fixed formula (compound interest) | ✓ Yes | Same steps every time |
| Variable steps, low depth (2–4 brackets) | ✓ Yes | Steps fit within turn budget |
| Variable steps, high depth (6–7 brackets) | ✗ No | Steps exceed turn budget |
| Iterative/looping (mortgage amortization) | ✗ No | Tool can't loop; LLM would need 360 calls |
| Arithmetic errors (pct chains) | ✓ Yes | Tool computes what LLM knows to call |

---

## What This Means for AI Product Design

**1. Tool use reliability degrades with orchestration complexity.**
The more decisions the LLM must make about *what to compute* (vs. just *how to compute it*),
the more failure modes exist. A 7-step orchestration is not 7× safer than a 1-step one.

**2. Variable-depth reasoning is harder than fixed-depth reasoning.**
Tax calculation requires a different number of tool calls depending on the input.
This variability means the LLM is doing more work than just "call the right tools in order."

**3. Turn budget is a hard constraint, not a soft one.**
The 93.3% accuracy observed with balanced sampling dropped to 60% when sampling
weighted toward high-income brackets — not because the LLM got worse, but because
those brackets require more tool calls than MAX_TOOL_TURNS=15 allows. In production,
any agentic tool-use system has a turn budget; knowing which inputs exhaust it is
as important as knowing the average case accuracy.

**4. The failure rate on tax matters more than the 0% rate on multiplication.**
In production, a tax calculation that is wrong by $2,335 is a serious problem — even
if 93% of calculations are correct. Accuracy requirements depend on the stakes of the task.

**4. Verification is the missing layer.**
The right architecture for high-stakes numerical tasks is:
```
LLM + Tool  →  output  →  Python verifier  →  if mismatch: flag for review
```
Not just delegating to the LLM to orchestrate tool calls, but independently
verifying the output against a deterministic reference.

---

## Experiment Progression

| Experiment | Task | LLM Direct | LLM + Tool | Key Finding |
|---|---|---|---|---|
| `run_test.py` | 2-digit multiplication | ~85–95% | — | LLM has arithmetic limits |
| `run_test.py` | 5-digit multiplication | ~2–10% | — | Accuracy collapses with complexity |
| `run_test_v2.py` | 2-digit multiplication | ~85–95% | ~100% | Tool use fixes simple arithmetic |
| `run_test_v2.py` | 5-digit multiplication | ~2–10% | ~100% | Tool use fixes complex arithmetic |
| `run_finance_test.py` | Compound interest | 0% | 100% | Tool use fixes fixed-formula math |
| `run_tax_test.py` | Tax brackets (balanced) | 0% | 93.3% | Tool use degrades with orchestration depth |
| **`run_tax_high_test.py`** | **Tax brackets (high-income)** | **0%** | **60.0%** | **Turn budget becomes the binding constraint** |

---

## Files

```
LLM Accuracy on Multiplication/
├── run_tax_test.py               ← balanced income distribution
├── run_tax_high_test.py          ← high-income stress test
├── results/results_tax.csv       ← balanced run results
├── results/results_tax_high.csv  ← high-income run results
└── readme_tax.md                 ← this file
```

## How to Run

```bash
source venv/bin/activate
python3 run_tax_test.py
```
