# Experiment: Sequential Percentage Change Chains — LLM vs LLM + Tool

A numerical accuracy experiment testing whether tool use fixes LLM failures on
multi-step percentage chain calculations. The core question: **can an LLM
correctly compute the final value after 3–6 sequential percentage changes when
given a calculator tool?**

---

## Why This Experiment

The mortgage experiment showed tool use achieving 0% accuracy — because the
problem required 260–360 iterations that cannot be unrolled into tool calls.

Percentage chains are structurally different. The problem requires exactly
N multiplications, one per step. With 3–6 steps and MAX_TOOL_TURNS=15, the LLM
has more than enough room to solve this correctly with a calculator.

This experiment isolates whether LLM Direct failures on this problem are:
- **Conceptual** — applying percentages to the wrong base (adding instead of chaining)
- **Arithmetic** — understanding the algorithm correctly but failing on the multiplication

---

## The Problem

For a given starting value and chain of percentage changes, compute the final value.

**Rule:** Each percentage applies to the **current running value**, not the original.

**Example — 3-step chain, start $10,000:**
```
Step 1: +25%   →  $10,000 × 1.25 = $12,500.00
Step 2: −20%   →  $12,500 × 0.80 = $10,000.00
Step 3: +10%   →  $10,000 × 1.10 = $11,000.00

Correct final value:  $11,000.00

Wrong (additive):   $10,000 × (1 + 0.25 − 0.20 + 0.10) = $11,500.00  ← $500 error
Wrong (returns start): $10,000.00  ← no changes applied
```

---

## Three Approaches Tested

### 1. Python (Ground Truth)
Exact chain multiplication:
```python
value = start
for r in changes:
    value *= (1 + r)
```
Always correct.

### 2. LLM Direct
Claude is given starting value and chain of percentages, asked for the final
value. No tools. Must perform all multiplications in its forward pass.

**Predicted failure:** Arithmetic errors on multi-step chain multiplication.
The numbers involve 2–5 decimal places and 3–6 sequential multiplications —
beyond reliable in-context arithmetic.

### 3. LLM + Tool (Agentic Loop)
Claude is given an arithmetic tool (add, subtract, multiply, divide) and must:
1. Compute `(1 + r1)` for step 1
2. Multiply starting value by that factor
3. Compute `(1 + r2)` for step 2
4. Multiply running total by that factor
5. Repeat for all steps

**This is structurally solvable.** Unlike the mortgage problem, there is no
loop dependency — the LLM knows all N steps upfront and can call the tool
exactly N times in sequence. With 3–6 steps, it needs 3–6 tool calls, well
within MAX_TOOL_TURNS=15.

---

## Experiment Design

| Parameter | Value |
|---|---|
| Samples | 30 |
| Starting value | $1,000 – $50,000 |
| Chain length | 3 – 6 steps |
| Per-step range | −30% to +50% |
| Guaranteed negative | At least 1 decrease per chain |
| Tolerance | ±$0.01 (strict — this is pure arithmetic) |
| Model | Claude Sonnet 4 (AWS Bedrock) |
| Max tool turns | 15 per sample |

---

## Results

| Approach | Correct | Accuracy |
|---|---|---|
| Python (ground truth) | 30/30 | 100.0% |
| LLM Direct | 0/30 | **0.0%** |
| LLM + Tool | 29/30 | **96.7%** |

**Average tool calls per sample: 4.8**

---

## Failure Analysis

### LLM Direct — 0% (30 failures)

The predicted failure mode was the classic "additive percentages" error —
summing all rates and applying once. The actual error pattern was more varied:

**Pattern 1 — Returned starting value (8 samples)**
Samples 1, 8, 11, 12, 17, 22, 28, 30 all returned the original starting value
unchanged. The LLM appeared to either:
- Return the input instead of computing the output
- Compute a net change of zero and apply it

Examples:
```
Sample  1:  start=$48,041  correct=$74,050  direct=$48,041  (returned start)
Sample  8:  start=$30,626  correct=$40,284  direct=$30,626  (returned start)
Sample 11:  start=$31,887  correct=$30,328  direct=$31,887  (returned start)
```

**Pattern 2 — Arithmetic errors in chain (22 samples)**
Most failures show the LLM attempting the correct algorithm (chain multiply)
but producing wrong intermediate values. The error is not conceptual — the
LLM knows each step should apply to the current value. The error is
**arithmetic**: multiplying irrational decimals across 4–6 steps in context
produces accumulated rounding errors.

Examples:
```
Sample  5:  correct=$46,102.40  direct=$46,499.42  (error: $397, 3-step chain)
Sample 10:  correct=$51,138.74  direct=$51,210.68  (error: $72, 3-step chain)
Sample 13:  correct=$50,886.95  direct=$40,668.90  (error: $10,218, 4-step chain)
Sample 29:  correct=$87,778.28  direct=$73,588.69  (error: $14,190, 4-step chain)
```

**The key finding:** LLM Direct does not primarily fail because it misunderstands
the problem (wrong conceptual base). It fails because it cannot reliably perform
multi-step floating-point multiplication in-context. The errors grow with chain
length and with the magnitude of the rates.

### LLM + Tool — 96.7% (1 failure)

Sample 2 (5-step chain, start $13,219.70) returned `None` after only 4 tool
calls. The turn limit was not reached — the tool loop terminated early without
a final answer. This is likely a tool call parsing issue or an early `end_turn`
before all steps were completed. Not a conceptual failure.

The 29 correct answers show that when the LLM correctly chains the tool calls,
every step is exact. The tool does what it's supposed to.

---

## How This Fits the Progression

```
Multiplication (1 step):              Direct ~90%,  Tool ~100%
Compound interest (5 fixed steps):    Direct   0%,  Tool  100%
Tax brackets (3–14 variable steps):   Direct   0%,  Tool  93.3%
Mortgage amortization (260–360 loop): Direct 6.7%,  Tool    0%
Pct change chains (3–6 steps):        Direct   0%,  Tool  96.7%
```

Percentage chains sit between compound interest and tax brackets. Like compound
interest, the problem has a fixed, known sequence of steps. Like tax brackets,
the number of steps varies by input. The LLM handles it at ~97% with tools.

---

## The Core Insight

This experiment was designed to test **conceptual** failure (wrong base for
each % change). The actual failure was **arithmetic** (correct algorithm,
wrong multiplication results).

This is an important distinction:

| Failure type | LLM does | With tool |
|---|---|---|
| **Architectural** (needs a loop) | Cannot structure the calls | Tool still can't loop |
| **Conceptual** (wrong algorithm) | Calls tool with wrong inputs | Tool returns correct answer to wrong question |
| **Arithmetic** (correct algorithm, wrong computation) | Calls tool incorrectly in some steps | Tool corrects the arithmetic → high accuracy |

Percentage chain errors are arithmetic, not conceptual. The LLM understands
that `new_value = old_value × (1 + rate)`. It just cannot evaluate
`$14,798.34 × 1.1217` reliably in-context. The tool removes that burden.

---

## What This Means for AI Product Design

**1. Distinguish arithmetic failure from conceptual failure.**
If the LLM is failing because it cannot compute a precise decimal multiplication,
a calculator tool will fix it. If the LLM is failing because it is applying the
wrong formula, the tool will faithfully compute the wrong answer.

**2. Chain length matters even when the structure is known.**
3-step chains had lower direct errors than 6-step chains. Each additional step
compounds rounding errors. For tasks with more than 2–3 floating-point operations,
tools become increasingly valuable.

**3. The 96.7% tool accuracy confirms a pattern across experiments.**
Every structurally solvable problem (fixed or variable steps, but no looping)
sees tool accuracy between 93–100%. The LLM's orchestration is reliable when
the step count is small.

**4. "Returned starting value" is a distinct failure mode.**
In 8/30 direct cases, the model returned the input unchanged. This is not
arithmetic error — it is a response failure, possibly from a format misparse or
overconfident shortcutting. Monitoring for "output equals input" is a useful
production check for calculation tasks.

---

## Experiment Progression

| Experiment | Task | LLM Direct | LLM + Tool | Key Finding |
|---|---|---|---|---|
| `run_test.py` | 2-digit multiplication | ~85–95% | — | LLM has arithmetic limits |
| `run_test.py` | 5-digit multiplication | ~2–10% | — | Accuracy collapses with complexity |
| `run_test_v2.py` | 2-digit multiplication | ~85–95% | ~100% | Tool use fixes simple arithmetic |
| `run_test_v2.py` | 5-digit multiplication | ~2–10% | ~100% | Tool use fixes complex arithmetic |
| `run_finance_test.py` | Compound interest | 0% | 100% | Tool use fixes fixed-formula math |
| `run_tax_test.py` | Tax brackets | 0% | 93.3% | Tool use degrades with orchestration depth |
| `run_mortgage_test.py` | Mortgage amortization | 6.7% / 0% | 0% / 0% | Tool use fails on iterative problems |
| **`run_pct_test.py`** | **Pct change chains** | **0%** | **96.7%** | **Tool use fixes arithmetic, not conceptual, failures** |

---

## Files

```
LLM Accuracy on Multiplication/
├── run_pct_test.py              ← this experiment
├── results/results_pct.csv     ← full 30-sample results
└── readme_pct.md               ← this file
```

## How to Run

```bash
source venv/bin/activate
python3 run_pct_test.py
```
