# LLM Accuracy on Multiplication

A benchmarking experiment that measures how accurately a large language model (Claude Sonnet 4 via AWS Bedrock) performs multiplication compared to Python's exact arithmetic — across two levels of numerical complexity.

---

## The Problem

LLMs are not calculators. They don't perform arithmetic — they *predict* the most likely next token based on patterns learned from training data. This means:

- For simple multiplications they've seen many times (e.g., `12 × 7`), they may answer correctly because the answer appears in training data
- For larger or more unusual numbers, they are essentially pattern-matching on digit sequences — and the accuracy degrades rapidly

This raises a practical question for product managers and AI builders:

> **At what point does an LLM's arithmetic become unreliable — and by how much?**

Understanding this boundary matters for any product that involves numerical reasoning, financial calculations, code generation with math, or data analysis.

---

## Why This Is Hard for LLMs

### How LLMs "do math"

LLMs tokenize numbers into subword pieces. The number `84732` might be split into tokens like `847` and `32` — the model never sees it as a single numeric value. Multiplication then requires the model to:

1. Understand the positional relationship between all digits
2. Mentally simulate carrying operations across multiple digit positions
3. Produce a sequence of output tokens that, concatenated, form the correct product

This is fundamentally different from how a CPU computes `84732 × 61409` — which is a direct binary operation with zero ambiguity.

### Why accuracy drops with digit size

| Digit count | What the model must track |
|---|---|
| 2-digit | Up to 4 digit positions in the product |
| 5-digit | Up to 10 digit positions in the product |
| 7-digit | Up to 14 digit positions — effectively intractable |

Each additional digit multiplies the number of intermediate steps the model must simulate in its forward pass. There is no scratchpad, no memory — just attention over tokens.

---

## The Experiment

### Design

Two separate tests are run using the same methodology:

| Test | Number range | Samples |
|---|---|---|
| 2-digit | 10 – 99 | 100 |
| 5-digit | 10,000 – 99,999 | 100 |

### Per sample, the script:
1. Randomly generates two numbers of the target digit size
2. Computes the **correct answer** using Python (`num1 * num2`)
3. Asks Claude: *"What is {num1} × {num2}? Respond with only the number."*
4. Parses the LLM's response and extracts the numeric answer
5. Compares the two — records correct (✓), wrong (✗), or unparseable (?)

### Ground truth
Python's integer multiplication is exact by definition. There is no floating point error, no rounding. It is the unambiguous reference.

---

## Architecture

```
run_test.py
    │
    ├── TEST_CONFIGS  ← add any digit size here to extend the experiment
    │   ├── {digits: 2, samples: 100}
    │   └── {digits: 5, samples: 100}
    │
    ├── For each sample:
    │   ├── random_n_digit_number()  → generates num1, num2
    │   ├── Python: num1 * num2      → correct_answer
    │   ├── ask_llm()                → sends to Claude via AWS Bedrock
    │   ├── parse_number()           → extracts integer from LLM response
    │   └── compare                 → is_correct = (llm_answer == correct_answer)
    │
    ├── Saves results/results_{N}digit.csv
    │   columns: sample, num1, num2, correct_answer,
    │            llm_answer, llm_raw_response, is_correct
    │
    └── Prints final accuracy summary table
```

### Key design decision — strict equality
The comparison is exact: `llm_answer == correct_answer`. There is no partial credit, no tolerance window. Either the full product is correct down to the last digit, or it isn't. This is intentional — in any real application, a multiplication that is "close" is still wrong.

### LLM prompt design
```
What is {num1} × {num2}?
Respond with only the final number and nothing else.
No explanation, no working, no punctuation — just the number.
```

The prompt is engineered to minimize the chance of the model adding text around the answer (e.g., "The answer is 12345"). A regex parser also handles cases where the model does add surrounding text, commas, or punctuation.

---

## Extensibility

To add a new digit size, add one line to `TEST_CONFIGS` in `run_test.py`:

```python
TEST_CONFIGS = [
    {"digits": 2, "samples": 100},
    {"digits": 5, "samples": 100},
    {"digits": 3, "samples": 100},   # ← new test
    {"digits": 7, "samples":  50},   # ← new test, fewer samples
]
```

No other changes needed. Each test produces its own CSV file in `results/`.

---

## How to Run

```bash
# First time setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the experiment (~3–4 minutes for 200 total API calls)
python3 run_test.py
```

Results are saved to:
```
results/
├── results_2digit.csv
└── results_5digit.csv
```

---

## Expected Findings

Based on published research and general knowledge of LLM arithmetic behavior:

| Test | Expected accuracy | Why |
|---|---|---|
| 2-digit | 80 – 95% | Common multiplications appear frequently in training data |
| 5-digit | 5 – 20% | Rare in training data; requires tracking 10 digit positions |

The gap between 2-digit and 5-digit accuracy is the core finding — it quantifies how quickly LLM arithmetic degrades with problem complexity.

---

## Actual Results

*(Run `python3 run_test.py` and fill in below)*

| Test | Range | Samples | Correct | Wrong | Accuracy |
|---|---|---|---|---|---|
| 2-digit | 10–99 | 100 | — | — | —% |
| 5-digit | 10,000–99,999 | 100 | — | — | —% |

---

## Key Takeaways for Product Managers

1. **Never trust LLMs for exact arithmetic in production.** Even high accuracy on 2-digit numbers is not 100% — and users will find the failures.

2. **The fix is tool use, not prompting.** Asking the LLM to "show its work" or "be careful" does not reliably fix arithmetic errors. The correct architecture is to give the LLM access to a calculator tool (code interpreter, function call) and let Python do the math.

3. **Accuracy degradation is steep, not gradual.** The jump from 2-digit to 5-digit accuracy is not linear — it collapses. This is important when scoping what kinds of calculations an AI feature can handle.

4. **This pattern generalizes.** The same degradation applies to other multi-step reasoning tasks — long chains of logic, complex SQL, multi-hop reasoning. The more steps required, the more the model is pattern-matching rather than computing.

---

## Project Structure

```
LLM Accuracy on Multiplication/
├── run_test.py         ← main experiment script
├── requirements.txt    ← dependencies
├── .env                ← AWS credentials (git-ignored)
├── .gitignore
├── readme.md
└── results/            ← auto-created when you run the test
    ├── results_2digit.csv
    └── results_5digit.csv
```
